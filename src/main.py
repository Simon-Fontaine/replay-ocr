import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from ultralytics import YOLO
from paddleocr import PaddleOCR
import uvicorn
import Levenshtein

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import JSONResponse

from src.config import Config
from src.model_manager import ModelManager

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

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


def parse_time_ago(time_ago: str) -> Optional[str]:
    """
    Convert 'time_ago' into an ISO datetime string.

    Supported formats for `time_ago`:
    - "<X> day(s) ago"
    - "<X> hour(s) ago"
    - "<X> minute(s) ago"

    This function handles cases with or without spaces between the number and the unit,
    such as "2days ago" or "2 days ago".

    Returns:
        ISO formatted datetime string if parsing is successful, else None.
    """
    now = datetime.now()
    ta = time_ago.strip().lower()

    # Insert a space between digits and letters if missing (e.g., "2days ago" -> "2 days ago")
    ta = re.sub(r"(\d+)([a-z]+)", r"\1 \2", ta)

    # Regular expression to match patterns like "2 days ago", "1 hour ago", "30 minutes ago"
    pattern = r"^(?P<value>\d+)\s*(?P<unit>day|days|hour|hours|minute|minutes)\s+ago$"
    match = re.match(pattern, ta)

    if not match:
        logger.warning("Unrecognized time_ago format: '%s'. Unable to parse.", time_ago)
        return None

    value = int(match.group("value"))
    unit = match.group("unit")

    if unit.startswith("day"):
        delta = timedelta(days=value)
    elif unit.startswith("hour"):
        delta = timedelta(hours=value)
    elif unit.startswith("minute"):
        delta = timedelta(minutes=value)
    else:
        logger.warning("Unrecognized time unit in time_ago: '%s'.", unit)
        return None

    final_time = now - delta
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
# Classes for OCR, YOLO, and Data Formatting
# -----------------------------------------------------------------------------
class ReplayTextExtractor:
    """Handles text extraction from images using OCR."""

    def __init__(self):
        self._model_manager = ModelManager()

    def extract_text(self, image: np.ndarray, bbox: List[int]) -> str:
        """Extract text from a given bounding box in the image."""
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        try:
            result = self._model_manager.ocr.ocr(roi, cls=True)
        except Exception as e:
            logger.warning(f"OCR failed on ROI: {e}")
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
                final_time = parse_time_ago(raw_time_ago)

                result_text = row.get("result", "")
                score_text = row.get("score", "")

                if final_time is None:
                    logger.warning(
                        "Failed to parse time_ago: '%s'. Skipping match.", raw_time_ago
                    )
                    continue  # Skip this match if time parsing failed

                match_dict = {
                    "run_id": run_id,
                    "map": map_name,
                    "mode": mode,
                    "replay_code": replay_code,
                    "time_ago": final_time,
                    "result": result_text,
                    "score": score_text,
                }
                formatted_matches.append(match_dict)

        return formatted_matches


class YOLOAnalyzer:
    """Handles YOLO model inference and post-processing of detections."""

    def __init__(self, text_extractor: ReplayTextExtractor):
        self._model_manager = ModelManager()
        self.confidence_threshold = 0.5
        self.text_extractor = text_extractor

    def process_image(self, image: np.ndarray, run_id: str) -> Tuple[List[Dict], str]:
        """Run YOLO inference and process results."""
        logger.info("Running YOLO inference on the provided image.")
        results = self._model_manager.yolo(image)
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

            row_index = int(y1) // 73  # Assuming each row is 73 pixels high
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


# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------
app = FastAPI()
text_extractor = ReplayTextExtractor()
analyzer = YOLOAnalyzer(text_extractor)


# -----------------------------------------------------------------------------
# Rate Limiting Configuration
# -----------------------------------------------------------------------------
# Construct Redis URL with authentication if password is provided
if Config.REDIS_PASSWORD:
    redis_url = (
        f"rediss://:{Config.REDIS_PASSWORD}@{Config.REDIS_HOST}:{Config.REDIS_PORT}"
    )
else:
    redis_url = f"rediss://{Config.REDIS_HOST}:{Config.REDIS_PORT}"

limiter = Limiter(key_func=get_remote_address, storage_uri=redis_url)

# Register SlowAPI's exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add SlowAPI middleware to FastAPI
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    user_ip = get_remote_address(request)
    logger.warning(f"Rate limit exceeded for IP: {user_ip}. Details: {exc.detail}")
    return JSONResponse(
        status_code=429,
        content={
            "message": "Too many requests. Please try again later.",
            "rate_limit": exc.detail,
        },
        headers=exc.headers,
    )


# -----------------------------------------------------------------------------
# FastAPI Endpoints
# -----------------------------------------------------------------------------
@app.post("/analyze_replay")
@limiter.limit(Config.RATE_LIMITS)  # Apply all rate limits from config
async def analyze_replay(request: Request, file: UploadFile = File(...)):
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
        matches, _ = analyzer.process_image(image, run_id)

        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info("Processing complete in %.2f seconds.", elapsed_time)

        return {"matches": matches}

    except Exception as e:
        logger.exception("Error processing image:")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
