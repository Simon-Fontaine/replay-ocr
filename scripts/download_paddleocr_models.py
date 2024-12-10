import os
from paddleocr import PaddleOCR


def download_models(model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)

    # Initialize PaddleOCR with specified directories
    ocr = PaddleOCR(
        lang="en",
        use_angle_cls=True,
        rec_model_dir=os.path.join(model_dir, "rec"),
        cls_model_dir=os.path.join(model_dir, "cls"),
        show_log=False,
    )
    print(f"PaddleOCR models downloaded to '{model_dir}'.")


if __name__ == "__main__":
    download_models()
