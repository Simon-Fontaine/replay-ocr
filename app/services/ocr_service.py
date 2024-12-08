from paddleocr import PaddleOCR
from app.logging_config import logger


class OCRService:
    def __init__(self, lang="en"):
        logger.info("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)
        logger.info("PaddleOCR initialized.")

    def extract_text_from_image(self, roi):
        """Perform OCR on the given ROI and return extracted text."""
        try:
            result = self.ocr.ocr(roi, cls=True)
        except Exception as e:
            logger.warning("OCR failed on ROI: %s", e)
            return ""

        if not result or not isinstance(result, list) or len(result) == 0:
            return ""

        lines = result[0]
        if lines is None:
            return ""

        extracted_text = ""
        for line in lines:
            if len(line) == 2 and isinstance(line[1], tuple):
                text_line = line[1][0]
                extracted_text += text_line + " "

        return extracted_text.strip()
