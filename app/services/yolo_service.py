from ultralytics import YOLO
from app.logging_config import logger
from app.config import MODEL_PATH


class YOLOService:
    def __init__(self, model_path=MODEL_PATH):
        logger.info("Loading YOLO model from %s", model_path)
        self.model = YOLO(model_path)
        logger.info("YOLO model loaded successfully.")

    def run_inference(self, image):
        """Run YOLO inference on the provided image."""
        logger.info("Running YOLO inference on the provided image.")
        results = self.model(image, save=True)
        return results
