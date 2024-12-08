from supabase import create_client, Client
from app.config import SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_BUCKET, SUPABASE_TABLE
from app.logging_config import logger

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def upload_image_to_supabase(filename: str, image_file):
    """Upload image to Supabase bucket."""
    try:
        res = supabase.storage.from_(SUPABASE_BUCKET).upload(filename, image_file)
        if isinstance(res, dict) and "path" in res:
            logger.info("Uploaded image to Supabase bucket: %s", res["path"])
        else:
            logger.error("Unexpected response from upload: %s", res)
    except Exception as e:
        logger.error("Failed to upload image: %s", e)


def insert_matches(matches: list):
    """Insert match records into Supabase table."""
    if matches:
        try:
            res = supabase.table(SUPABASE_TABLE).insert(matches).execute()
            if isinstance(res, dict) and "data" in res:
                logger.info("Inserted matches into Supabase table successfully.")
            else:
                logger.warning("Unexpected response structure from insert: %s", res)
        except Exception as e:
            logger.error("Error inserting matches into Supabase: %s", e)
