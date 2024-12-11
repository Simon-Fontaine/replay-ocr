import re
import os
import logging
import numpy as np
from typing import Optional
from ultralytics import YOLO
from paddleocr import PaddleOCR

logger = logging.getLogger("uvicorn.error")


class ModelManager:
    """
    Singleton class to manage ML model instances with lazy loading.
    """

    _instance: Optional["ModelManager"] = None
    _ocr: Optional[PaddleOCR] = None
    _yolo: Optional[YOLO] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only run initialization once
        if not self._initialized:
            self._initialized = True
            logger.info("Initializing ModelManager singleton")

    @property
    def ocr(self) -> PaddleOCR:
        """
        Lazy loading for PaddleOCR model.
        """
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
        """
        Lazy loading for YOLO model.
        """
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
