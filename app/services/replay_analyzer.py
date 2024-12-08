import numpy as np
from typing import Dict, List, Tuple
from app.logging_config import logger
from app.services.ocr_service import OCRService
from app.services.yolo_service import YOLOService
from app.services.text_utils import (
    format_match_result,
    sanitize_replay_code,
    best_map_match,
    best_mode_match,
    fix_time_format,
    parse_duration_to_datetime,
)
from app.config import KNOWN_MAPS


class ReplayAnalyzer:
    def __init__(self):
        self.yolo_service = YOLOService()
        self.ocr_service = OCRService(lang="en")
        self.confidence_threshold = 0.5

    def extract_text(self, image: np.ndarray, bbox: List[int]) -> str:
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        return self.ocr_service.extract_text_from_image(roi)

    async def process_image(
        self, image: np.ndarray, run_id: str
    ) -> Tuple[List[Dict], str]:
        results = self.yolo_service.run_inference(image)
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

            row_index = int(y1) // 73
            if row_index not in rows:
                rows[row_index] = {}

            if class_name == "match_result":
                res, match_score = format_match_result(text)
                rows[row_index]["result"] = res
                rows[row_index]["score"] = match_score
            else:
                rows[row_index][class_name] = text

        formatted_matches = []
        logger.info("Formatting detected match rows...")
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

                # Convert to actual datetime if possible
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

        logger.info(
            "Processing complete. Total matches found: %d", len(formatted_matches)
        )
        return formatted_matches, str(result.save_dir)
