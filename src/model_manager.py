import os
import logging
from ultralytics import YOLO
from paddleocr import PaddleOCR

logger = logging.getLogger("uvicorn.error")


class ModelManager:
    """
    Singleton class to manage ML model instances with lazy loading.
    """

    _instance = None
    _ocr = None
    _yolo = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            logger.info("Initializing ModelManager singleton")
            self._initialized = True

    @property
    def ocr(self) -> PaddleOCR:
        """Lazy loading for PaddleOCR model."""
        if self._ocr is None:
            logger.info("Initializing PaddleOCR model...")
            try:
                self._ocr = PaddleOCR(
                    lang="en",
                    use_angle_cls=True,
                    rec_model_dir=os.path.join("models", "rec"),
                    cls_model_dir=os.path.join("models", "cls"),
                    det_model_dir=os.path.join("models", "det"),
                    show_log=False,
                )
                logger.info("PaddleOCR model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR model: {e}")
                raise
        return self._ocr

    @property
    def yolo(self) -> YOLO:
        """Lazy loading for YOLO model."""
        if self._yolo is None:
            logger.info("Loading YOLO model...")
            try:
                model_path = os.path.join("models", "best.pt")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"YOLO model not found at {model_path}")
                self._yolo = YOLO(model_path)
                logger.info("YOLO model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                raise
        return self._yolo

    def cleanup(self):
        """Cleanup model resources."""
        logger.info("Cleaning up model resources...")
        self._ocr = None
        self._yolo = None
