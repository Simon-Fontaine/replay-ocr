import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
import cv2
import numpy as np
import re
from typing import Dict, List, Tuple
import uvicorn
from paddleocr import PaddleOCR
import uuid
from supabase import create_client, Client
import glob
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

load_dotenv()

# Environment variables for Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://<your-id>.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "<your-anon-key>")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "overwatch-replays")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "matches")
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", os.path.join("app", "models", "best.pt"))

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

app = FastAPI()

KNOWN_MODES = [
    "COMPETITIVE ROLE QUEUE",
    "COMPETITIVE OPEN QUEUE",
    "OVERWATCH: CLASSIC",
    "UNRANKED",
    "QUICK PLAY",
    "ARCADE",
    "CUSTOM GAME",
    "MYSTERY HEROES",
    "NO LIMITS",
    "TOTAL MAYHEM",
]

KNOWN_RESULTS = ["VICTORY!", "DEFEAT!", "DRAW!"]

KNOWN_MAPS = [
    # Control
    "Antarctic Peninsula",
    "Busan",
    "Ilios",
    "Lijiang Tower",
    "Nepal",
    "Oasis",
    "Samoa",
    # Escort
    "Circuit Royal",
    "Dorado",
    "Havana",
    "Junkertown",
    "Rialto",
    "Route 66",
    "Shambali Monastery",
    "Watchpoint: Gibraltar",
    # Flashpoint
    "New Junk City",
    "Suravasa",
    # Hybrid
    "Blizzard World",
    "Eichenwalde",
    "Hollywood",
    "King's Row",
    "Midtown",
    "Numbani",
    "Paraíso",
    # Push
    "Colosseo",
    "Esperança",
    "New Queen Street",
    "Runasapi",
    # Clash
    "Hanaoka",
    "Throne of Anubis",
]


class ReplayAnalyzer:
    def __init__(self, model_path=MODEL_PATH):
        logger.info("Loading YOLO model from %s", model_path)
        self.model = YOLO(model_path)
        logger.info("YOLO model loaded successfully.")

        logger.info("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(lang="en", use_angle_cls=True, show_log=False)
        logger.info("PaddleOCR initialized.")

        self.confidence_threshold = 0.5

    def extract_text(self, image: np.ndarray, bbox: List[int]) -> str:
        x1, y1, x2, y2 = bbox
        logger.debug("Extracting text from bbox: %s", bbox)
        roi = image[y1:y2, x1:x2]

        try:
            result = self.ocr.ocr(roi, cls=True)
        except Exception as e:
            logger.warning("OCR failed on ROI: %s", e)
            return ""

        if not result or not isinstance(result, list) or len(result) == 0:
            logger.debug("No OCR results for given ROI.")
            return ""

        lines = result[0]
        if lines is None:
            logger.debug("OCR returned None lines.")
            return ""

        extracted_text = ""
        for line in lines:
            if len(line) == 2 and isinstance(line[1], tuple):
                text_line = line[1][0]
                extracted_text += text_line + " "

        extracted_text = re.sub(r"\s+", " ", extracted_text.strip())
        logger.debug("Extracted text: '%s'", extracted_text)
        return extracted_text

    def format_match_result(self, result_text: str) -> tuple:
        if not result_text:
            return None, None

        parts = result_text.split("|")
        if len(parts) != 2:
            # If no pipe, try last token as score
            tokens = result_text.split()
            if len(tokens) > 1 and re.match(r"\d+-\d+", tokens[-1]):
                result = " ".join(tokens[:-1])
                score = tokens[-1]
                return self.validate_result(result), score
            return None, None

        result = parts[0].strip()
        score = parts[1].strip()
        return self.validate_result(result), score

    def validate_result(self, result: str) -> str:
        if result in KNOWN_RESULTS:
            return result

        upper_res = result.upper()
        if "VICT" in upper_res:
            return "VICTORY!"
        elif "DEFE" in upper_res:
            return "DEFEAT!"
        elif "DRAW" in upper_res:
            return "DRAW!"
        return result

    def sanitize_replay_code(self, code: str) -> str:
        code = code.upper()
        code = re.sub(r"[^A-Z0-9]", "", code)
        return code

    def best_map_match(self, text: str) -> str:
        if not text:
            return ""
        upper_text = text.upper()
        candidates = [(m, self.levenshtein(upper_text, m.upper())) for m in KNOWN_MAPS]
        candidates.sort(key=lambda x: x[1])
        best_map, dist = candidates[0]
        if dist < len(text) / 2:
            return best_map
        return text

    def levenshtein(self, a, b):
        if a == b:
            return 0
        if len(a) == 0:
            return len(b)
        if len(b) == 0:
            return len(a)
        v0 = list(range(len(b) + 1))
        v1 = [0] * (len(b) + 1)
        for i in range(len(a)):
            v1[0] = i + 1
            for j in range(len(b)):
                cost = 0 if a[i] == b[j] else 1
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            v0, v1 = v1, v0
        return v0[len(b)]

    def fix_time_format(self, time_str: str) -> str:
        time_str = re.sub(r"[.;]", ":", time_str.strip())
        if ":" not in time_str and len(time_str) == 4 and time_str.isdigit():
            hh, mm = time_str[:2], time_str[2:]
            if 0 <= int(hh) < 24 and 0 <= int(mm) < 60:
                time_str = f"{hh}:{mm}"
        return time_str

    def best_mode_match(self, mode: str) -> str:
        if not mode:
            return "COMPETITIVE ROLE QUEUE"
        for m in KNOWN_MODES:
            if m.upper() == mode.upper():
                return m

        upper_mode = mode.upper()
        if "COMPETITIVE" in upper_mode and "ROLE" in upper_mode:
            return "COMPETITIVE ROLE QUEUE"
        elif "OPEN" in upper_mode and "QUEUE" in upper_mode:
            return "COMPETITIVE OPEN QUEUE"
        elif "UNRANKED" in upper_mode:
            return "UNRANKED"
        elif "CLASSIC" in upper_mode:
            return "OVERWATCH: CLASSIC"
        elif "QUICK" in upper_mode and "PLAY" in upper_mode:
            return "QUICK PLAY"
        elif "ARCADE" in upper_mode:
            return "ARCADE"
        elif "CUSTOM" in upper_mode and "GAME" in upper_mode:
            return "CUSTOM GAME"

        return "COMPETITIVE ROLE QUEUE"

    def parse_duration_to_datetime(self, time_ago: str, duration: str) -> str:
        """
        Convert 'time_ago' and 'duration' into an actual datetime string.

        For example:
        time_ago = "10:02"
        duration = "1 DAY AGO"

        This would mean the match occurred 1 day ago at 10:02 today.

        If duration does not contain days, or if time_ago is invalid, we return them as is.
        """
        # Parse the number of days from duration if present
        days_ago = 0
        match_days = re.search(r"(\d+)\s+DAY", duration.upper())
        if match_days:
            days_ago = int(match_days.group(1))

        # Parse time_ago as HH:MM if possible
        match_time = re.match(r"(\d{1,2}):(\d{2})", time_ago)
        if match_time:
            hours = int(match_time.group(1))
            minutes = int(match_time.group(2))
        else:
            # If we can't parse time, just return original strings
            return time_ago

        # Compute the datetime
        # Use current local time as reference. If you want UTC, use datetime.utcnow()
        now = datetime.now()
        final_time = (now - timedelta(days=days_ago)).replace(
            hour=hours, minute=minutes, second=0, microsecond=0
        )

        # Return as ISO format string
        return final_time.isoformat()

    async def process_image(
        self, image: np.ndarray, run_id: str
    ) -> Tuple[List[Dict], str]:
        logger.info("Running YOLO inference on the provided image.")
        results = self.model(image, save=True)

        if results is None or len(results) == 0:
            logger.info("No results returned from YOLO model.")
            return [], ""

        result = results[0]
        if (
            result is None
            or not hasattr(result, "boxes")
            or result.boxes is None
            or result.boxes.data is None
        ):
            logger.info("No boxes detected by YOLO model.")
            return [], str(result.save_dir)

        boxes_data = result.boxes.data.tolist() if len(result.boxes.data) > 0 else []
        if not boxes_data:
            logger.info("No detections found in YOLO results.")
            return [], str(result.save_dir)

        rows = {}
        logger.info("Processing detected objects...")
        for r in boxes_data:
            if r is None:
                continue
            x1, y1, x2, y2, score, class_id = r
            if score < self.confidence_threshold:
                continue

            if result.names is None or class_id not in result.names:
                logger.warning("Class ID %s not found in result.names.", class_id)
                continue

            class_name = result.names[int(class_id)]
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            text = self.extract_text(image, bbox)

            # Approximate row indexing
            row_index = int(y1) // 73
            if row_index not in rows:
                rows[row_index] = {}
            if class_name == "match_result":
                res, match_score = self.format_match_result(text)
                rows[row_index]["result"] = res
                rows[row_index]["score"] = match_score
            else:
                rows[row_index][class_name] = text

        formatted_matches = []
        logger.info("Formatting detected match rows...")
        for _, row in sorted(rows.items()):
            if "map_name" in row:
                map_name = self.best_map_match(row.get("map_name", ""))
                mode = self.best_mode_match(row.get("game_mode", ""))
                replay_code = self.sanitize_replay_code(row.get("replay_code", ""))

                raw_time_ago = row.get("time_ago", "").strip("- ").strip()
                time_ago = self.fix_time_format(raw_time_ago)

                duration = row.get("time_duration", "").strip("- ").strip()

                result_text = row.get("result", "")
                score_text = row.get("score", "")

                # Convert time_ago + duration to actual datetime
                # If either is missing or doesn't parse, it will just return time_ago as is.
                final_time = self.parse_duration_to_datetime(time_ago, duration)

                match_dict = {
                    "run_id": run_id,
                    "map": map_name,
                    "mode": mode,
                    "replay_code": replay_code,
                    "time_ago": final_time,  # now a computed datetime
                    "duration": duration,
                    "result": result_text,
                    "score": score_text,
                }
                logger.debug("Formatted match: %s", match_dict)
                formatted_matches.append(match_dict)

        logger.info(
            "Processing complete. Total matches found: %d", len(formatted_matches)
        )
        return formatted_matches, str(result.save_dir)


analyzer = ReplayAnalyzer()


@app.post("/analyze_replay")
async def analyze_replay(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(400, "File must be an image (PNG or JPEG)")

    logger.info("Received file: %s", file.filename)
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.error("Failed to decode image.")
            raise HTTPException(400, "Invalid image file")

        run_id = str(uuid.uuid4())
        matches, save_dir = await analyzer.process_image(image, run_id)

        # Find annotated image from save_dir
        annotated_image_path = None
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            found = glob.glob(os.path.join(save_dir, ext))
            if found:
                annotated_image_path = found[0]
                break

        if annotated_image_path:
            with open(annotated_image_path, "rb") as img_file:
                filename = f"{run_id}_{os.path.basename(annotated_image_path)}"
                try:
                    res = supabase.storage.from_(SUPABASE_BUCKET).upload(
                        filename, img_file
                    )
                    if isinstance(res, dict) and "path" in res:
                        logger.info(
                            "Uploaded image to Supabase bucket: %s", res["path"]
                        )
                    else:
                        logger.error("Unexpected response from upload: %s", res)
                except Exception as e:
                    logger.error("Failed to upload image: %s", e)

        # Insert matches into Supabase table if any
        if matches:
            try:
                res = supabase.table(SUPABASE_TABLE).insert(matches).execute()
                if isinstance(res, dict) and "data" in res:
                    logger.info("Inserted matches into Supabase table successfully.")
                else:
                    logger.warning("Unexpected response structure from insert: %s", res)
            except Exception as e:
                logger.error("Error inserting matches into Supabase: %s", e)
        return matches

    except Exception as e:
        logger.exception("Error processing image:")
        raise HTTPException(500, f"Error processing image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
