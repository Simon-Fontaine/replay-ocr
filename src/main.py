import logging
import os
import re
import uuid
import glob
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from ultralytics import YOLO
from paddleocr import PaddleOCR
from supabase import create_client, Client
import uvicorn
import Levenshtein

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://<your-id>.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "<your-anon-key>")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "overwatch-replays")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "matches")
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "src/best.pt")

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

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger("uvicorn.error")


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def sanitize_replay_code(code: str) -> str:
    """Sanitize a given replay code by removing invalid characters and uppercasing."""
    code = code.upper()
    return re.sub(r"[^A-Z0-9]", "", code)


def fix_time_format(time_str: str) -> str:
    """
    Attempts to fix the time format.
    E.g., convert `1002` -> `10:02`.
    Replace '.' or ';' with ':'.
    """
    time_str = re.sub(r"[.;]", ":", time_str.strip())
    if ":" not in time_str and len(time_str) == 4 and time_str.isdigit():
        hh, mm = time_str[:2], time_str[2:]
        if 0 <= int(hh) < 24 and 0 <= int(mm) < 60:
            time_str = f"{hh}:{mm}"
    return time_str


def parse_duration_to_datetime(time_ago: str, duration: str) -> str:
    """
    Convert 'time_ago' + 'duration' into an ISO datetime string.

    Supported formats for `time_ago`:
    - "HH:MM" (e.g., "10:02" means 10 hours and 2 minutes ago)
    - "<X> hour(s) ago", "<X> minute(s) ago", "<X> day(s) ago"
    - If parsing fails, we return current time in ISO format.

    The `duration` can add extra days if it's like "1 DAY AGO".
    """
    now = datetime.now()

    # Extract additional days from duration if present (e.g. "1 DAY AGO" => days_ago = 1)
    days_ago = 0
    day_match = re.search(r"(\d+)\s+DAY", duration.upper())
    if day_match:
        days_ago = int(day_match.group(1))

    # Initialize offsets
    add_days = 0
    add_hours = 0
    add_minutes = 0

    ta = time_ago.strip().lower()

    # Check if time_ago matches HH:MM format
    hhmm_match = re.match(r"(\d{1,2}):(\d{2})", ta)
    if hhmm_match:
        # Interpreted as HH:MM ago
        add_hours = int(hhmm_match.group(1))
        add_minutes = int(hhmm_match.group(2))
    else:
        # Try natural language durations: "<number> <unit> ago"
        nl_match = re.match(r"(\d+)\s*(day|days|hour|hours|minute|minutes)\s*ago", ta)
        if nl_match:
            quantity = int(nl_match.group(1))
            unit = nl_match.group(2)

            if "day" in unit:
                add_days = quantity
            elif "hour" in unit:
                add_hours = quantity
            elif "minute" in unit:
                add_minutes = quantity
        else:
            # If we don't recognize the format, just return now
            return now.isoformat()

    # Combine durations: days_ago from duration + add_days from time_ago
    total_days = days_ago + add_days

    final_time = now - timedelta(days=total_days, hours=add_hours, minutes=add_minutes)
    return final_time.isoformat()


def validate_result(result: str) -> str:
    """Validate and normalize the result string into a known result."""
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


def best_map_match(text: str) -> str:
    """Find the closest known map to a given text using Levenshtein distance."""
    if not text:
        return ""
    upper_text = text.upper()
    candidates = [(m, Levenshtein.distance(upper_text, m.upper())) for m in KNOWN_MAPS]
    candidates.sort(key=lambda x: x[1])
    best_map, dist = candidates[0]
    # Heuristic to ensure best match is reasonably close
    if dist < len(text) / 2:
        return best_map
    return text


def best_mode_match(mode: str) -> str:
    """
    Match a given mode text to a known mode, falling back gracefully.
    """
    if not mode:
        return "COMPETITIVE ROLE QUEUE"

    mode_upper = mode.upper()
    # Direct exact matches
    for m in KNOWN_MODES:
        if m.upper() == mode_upper:
            return m

    # Heuristics
    if "COMPETITIVE" in mode_upper and "ROLE" in mode_upper:
        return "COMPETITIVE ROLE QUEUE"
    if "OPEN" in mode_upper and "QUEUE" in mode_upper:
        return "COMPETITIVE OPEN QUEUE"
    if "UNRANKED" in mode_upper:
        return "UNRANKED"
    if "CLASSIC" in mode_upper:
        return "OVERWATCH: CLASSIC"
    if "QUICK" in mode_upper and "PLAY" in mode_upper:
        return "QUICK PLAY"
    if "ARCADE" in mode_upper:
        return "ARCADE"
    if "CUSTOM" in mode_upper and "GAME" in mode_upper:
        return "CUSTOM GAME"

    return "COMPETITIVE ROLE QUEUE"


def format_match_result(result_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse and format a match result text into a standardized result and score.
    """
    if not result_text:
        return None, None

    parts = result_text.split("|")
    if len(parts) != 2:
        # If no pipe, try last token as score
        tokens = result_text.split()
        if len(tokens) > 1 and re.match(r"\d+-\d+", tokens[-1]):
            result = " ".join(tokens[:-1])
            score = tokens[-1]
            return validate_result(result), score
        return None, None

    result = parts[0].strip()
    score = parts[1].strip()
    return validate_result(result), score


# -----------------------------------------------------------------------------
# Classes for OCR, YOLO, and Database interactions
# -----------------------------------------------------------------------------
class ReplayTextExtractor:
    """Handles text extraction from images using OCR."""

    def __init__(self):
        logger.info("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(lang="en", use_angle_cls=True, show_log=False)
        logger.info("PaddleOCR initialized.")

    def extract_text(self, image: np.ndarray, bbox: List[int]) -> str:
        """Extract text from a given bounding box in the image."""
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        try:
            result = self.ocr.ocr(roi, cls=True)
        except Exception as e:
            logger.warning("OCR failed on ROI: %s", e)
            return ""

        if (
            not result
            or not isinstance(result, list)
            or len(result) == 0
            or result[0] is None
        ):
            return ""

        extracted_text = []
        for line in result[0]:
            if len(line) == 2 and isinstance(line[1], tuple):
                text_line = line[1][0]
                extracted_text.append(text_line)

        text_str = " ".join(extracted_text)
        return re.sub(r"\s+", " ", text_str.strip())


class MatchDataFormatter:
    """Formats YOLO detections into structured match data."""

    def __init__(self):
        pass

    def format_rows(self, rows: Dict[int, Dict], run_id: str) -> List[Dict]:
        """
        Convert the collected rows dictionary into a list of structured match dictionaries.
        """
        formatted_matches = []

        for _, row in sorted(rows.items()):
            if "map_name" in row:
                map_name = best_map_match(row.get("map_name", ""))
                mode = best_mode_match(row.get("game_mode", ""))
                replay_code = sanitize_replay_code(row.get("replay_code", ""))

                raw_time_ago = row.get("time_ago", "").strip("- ").strip()
                time_ago = fix_time_format(raw_time_ago)
                duration = row.get("time_duration", "").strip("- ").strip()

                result_text = row.get("result", "")
                score_text = row.get("score", "")

                final_time = parse_duration_to_datetime(time_ago, duration)

                match_dict = {
                    "run_id": run_id,
                    "map": map_name,
                    "mode": mode,
                    "replay_code": replay_code,
                    "time_ago": final_time,
                    "duration": duration,
                    "result": result_text,
                    "score": score_text,
                }
                formatted_matches.append(match_dict)

        return formatted_matches


class YOLOAnalyzer:
    """Handles YOLO model inference and post-processing of detections."""

    def __init__(self, model_path: str, text_extractor: ReplayTextExtractor):
        logger.info("Loading YOLO model from %s", model_path)
        self.model = YOLO(model_path)
        logger.info("YOLO model loaded successfully.")
        self.confidence_threshold = 0.5
        self.text_extractor = text_extractor

    def process_image(self, image: np.ndarray, run_id: str) -> Tuple[List[Dict], str]:
        """Run YOLO inference and process results."""
        logger.info("Running YOLO inference on the provided image.")
        results = self.model(image, save=True)
        if not results or len(results) == 0:
            logger.info("No results from YOLO model.")
            return [], ""

        result = results[0]
        if (
            not hasattr(result, "boxes")
            or result.boxes is None
            or result.boxes.data is None
        ):
            logger.info("No boxes detected.")
            return [], str(result.save_dir)

        boxes_data = result.boxes.data.tolist() if len(result.boxes.data) > 0 else []
        if not boxes_data:
            logger.info("No detections found in YOLO results.")
            return [], str(result.save_dir)

        rows = {}
        for r in boxes_data:
            if r is None:
                continue
            x1, y1, x2, y2, score, class_id = r
            if score < self.confidence_threshold:
                continue

            if not result.names or class_id not in result.names:
                logger.warning("Class ID %s not found in names map.", class_id)
                continue

            class_name = result.names[int(class_id)]
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            extracted_text = self.text_extractor.extract_text(image, bbox)

            row_index = int(y1) // 73
            if row_index not in rows:
                rows[row_index] = {}

            if class_name == "match_result":
                res, match_score = format_match_result(extracted_text)
                rows[row_index]["result"] = res
                rows[row_index]["score"] = match_score
            else:
                rows[row_index][class_name] = extracted_text

        formatter = MatchDataFormatter()
        formatted_matches = formatter.format_rows(rows, run_id)
        logger.info(
            "Processing complete. Total matches found: %d", len(formatted_matches)
        )
        return formatted_matches, str(result.save_dir)


class DatabaseClient:
    """Handles Supabase database operations."""

    def __init__(self, url: str, anon_key: str, bucket: str, table: str):
        self.client: Client = create_client(url, anon_key)
        self.bucket = bucket
        self.table = table

    def upload_image(self, local_path: str, remote_name: str) -> None:
        """Upload an image to Supabase storage."""
        try:
            with open(local_path, "rb") as img_file:
                res = self.client.storage.from_(self.bucket).upload(
                    remote_name, img_file
                )
                if hasattr(res, "path") and res.path:
                    logger.info("Uploaded image to storage: %s", res.path)
                else:
                    logger.warning("Unexpected upload response: %s", res)
        except Exception as e:
            logger.error("Failed to upload image: %s", e)

    def insert_matches(self, matches: List[Dict]) -> None:
        """Insert match records into the Supabase table."""
        if not matches:
            return
        try:
            res = self.client.table(self.table).insert(matches).execute()
            if res.data:
                logger.info("Inserted %d matches into table.", len(res.data))
            else:
                logger.warning("Unexpected insert response: %s", res)
        except Exception as e:
            logger.error("Error inserting matches: %s", e)


# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------
app = FastAPI()
text_extractor = ReplayTextExtractor()
analyzer = YOLOAnalyzer(MODEL_PATH, text_extractor)
db_client = DatabaseClient(
    SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_BUCKET, SUPABASE_TABLE
)


@app.post("/analyze_replay")
async def analyze_replay(file: UploadFile = File(...)):
    """Analyze an uploaded Overwatch replay image."""
    start_time = datetime.now()

    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="File must be an image (PNG/JPEG)")

    logger.info("Received file: %s", file.filename)
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.error("Failed to decode image.")
            raise HTTPException(status_code=400, detail="Invalid image file")

        run_id = str(uuid.uuid4())
        matches, save_dir = analyzer.process_image(image, run_id)

        # Find annotated image and upload it
        annotated_image_path = None
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            found = glob.glob(os.path.join(save_dir, ext))
            if found:
                annotated_image_path = found[0]
                break

        if annotated_image_path:
            remote_name = f"{run_id}_{os.path.basename(annotated_image_path)}"
            db_client.upload_image(annotated_image_path, remote_name)

        # Insert matches into DB
        db_client.insert_matches(matches)

        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info("Processing complete in %.2f seconds.", elapsed_time)

        return matches

    except Exception as e:
        logger.exception("Error processing image:")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
