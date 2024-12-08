import os
import uuid
import glob
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.database.supabase_client import upload_image_to_supabase, insert_matches
from app.services.replay_analyzer import ReplayAnalyzer
from app.logging_config import logger

router = APIRouter()
analyzer = ReplayAnalyzer()


@router.post("/analyze_replay")
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

        # Find annotated image
        annotated_image_path = None
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            found = glob.glob(os.path.join(save_dir, ext))
            if found:
                annotated_image_path = found[0]
                break

        if annotated_image_path:
            with open(annotated_image_path, "rb") as img_file:
                filename = f"{run_id}_{os.path.basename(annotated_image_path)}"
                upload_image_to_supabase(filename, img_file)

        # Insert matches into Supabase
        if matches:
            insert_matches(matches)

        return matches

    except Exception as e:
        logger.exception("Error processing image:")
        raise HTTPException(500, f"Error processing image: {str(e)}")
