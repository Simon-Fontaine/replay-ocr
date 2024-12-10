import argparse
import asyncio
import json
import logging
import os
from typing import List, Optional, Dict

import aiofiles
from aiohttp import ClientSession, FormData
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def setup_requests_session(
    retries: int = 3, backoff_factor: float = 0.3
) -> requests.Session:
    """
    Set up a requests.Session with retry logic.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(500, 502, 504),
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


async def async_test_analyze_replay(
    image_paths: List[str], url: str, output_file: Optional[str] = None
):
    """
    Asynchronously send multiple image files to the /analyze_replay endpoint and handle responses.
    """
    async with ClientSession() as session:
        tasks = []
        for image_path in image_paths:
            tasks.append(send_image(session, image_path, url))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        results = []
        for response in responses:
            if isinstance(response, Exception):
                logger.error(f"An error occurred: {response}")
            else:
                results.append(response)

        if output_file:
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Responses saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to write to output file: {e}")


async def send_image(session: ClientSession, image_path: str, url: str) -> Dict:
    """
    Send a single image to the /analyze_replay endpoint asynchronously.
    """
    if not os.path.isfile(image_path):
        logger.error(f"File not found: {image_path}")
        return {"image": image_path, "error": "File not found"}

    _, ext = os.path.splitext(image_path.lower())
    if ext not in [".png", ".jpg", ".jpeg"]:
        logger.error(f"Unsupported file type ({ext}) for file: {image_path}")
        return {"image": image_path, "error": "Unsupported file type"}

    logger.info(f"Sending file: {image_path} to {url}")
    try:
        async with aiofiles.open(image_path, "rb") as f:
            file_content = await f.read()

        form = FormData()
        form.add_field(
            "file",
            file_content,
            filename=os.path.basename(image_path),
            content_type=f"image/{ext.strip('.')}",
        )

        async with session.post(url, data=form) as resp:
            if resp.status == 200:
                try:
                    data = await resp.json()
                    logger.info(f"Success: {image_path}")
                    return {"image": image_path, "matches": data.get("matches", [])}
                except json.JSONDecodeError:
                    text = await resp.text()
                    logger.error(f"Non-JSON response for {image_path}: {text}")
                    return {"image": image_path, "error": "Non-JSON response"}
            else:
                text = await resp.text()
                logger.error(f"Failed ({resp.status}) for {image_path}: {text}")
                return {"image": image_path, "error": f"Status {resp.status}: {text}"}
    except Exception as e:
        logger.exception(f"Exception occurred while processing {image_path}: {e}")
        return {"image": image_path, "error": str(e)}


def send_image_sync(session: requests.Session, image_path: str, url: str) -> Dict:
    """
    Send a single image to the /analyze_replay endpoint synchronously.
    """
    if not os.path.isfile(image_path):
        logger.error(f"File not found: {image_path}")
        return {"image": image_path, "error": "File not found"}

    _, ext = os.path.splitext(image_path.lower())
    if ext not in [".png", ".jpg", ".jpeg"]:
        logger.error(f"Unsupported file type ({ext}) for file: {image_path}")
        return {"image": image_path, "error": "Unsupported file type"}

    logger.info(f"Sending file: {image_path} to {url}")
    try:
        with open(image_path, "rb") as f:
            files = {
                "file": (os.path.basename(image_path), f, f"image/{ext.strip('.')}")
            }
            response = session.post(url, files=files)
            if response.status_code == 200:
                try:
                    data = response.json()
                    logger.info(f"Success: {image_path}")
                    return {"image": image_path, "matches": data.get("matches", [])}
                except json.JSONDecodeError:
                    logger.error(f"Non-JSON response for {image_path}: {response.text}")
                    return {"image": image_path, "error": "Non-JSON response"}
            else:
                logger.error(
                    f"Failed ({response.status_code}) for {image_path}: {response.text}"
                )
                return {
                    "image": image_path,
                    "error": f"Status {response.status_code}: {response.text}",
                }
    except Exception as e:
        logger.exception(f"Exception occurred while processing {image_path}: {e}")
        return {"image": image_path, "error": str(e)}


def test_analyze_replay(
    image_paths: List[str],
    url: str = "http://localhost:8080/analyze_replay",
    output_file: Optional[str] = None,
    async_mode: bool = True,
):
    """
    Send one or multiple image files to the /analyze_replay endpoint and handle responses.

    Args:
        image_paths (List[str]): List of paths to image files.
        url (str): API endpoint URL.
        output_file (Optional[str]): Path to save the responses as JSON.
        async_mode (bool): Whether to send requests asynchronously.
    """
    if async_mode:
        asyncio.run(async_test_analyze_replay(image_paths, url, output_file))
    else:
        session = setup_requests_session()
        results = []
        for image_path in image_paths:
            result = send_image_sync(session, image_path, url)
            results.append(result)

        if output_file:
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Responses saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to write to output file: {e}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Test the /analyze_replay API endpoint by sending image files."
    )
    parser.add_argument(
        "images",
        metavar="IMAGE_PATH",
        type=str,
        nargs="+",
        help="Path(s) to the image file(s) to be analyzed.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080/analyze_replay",
        help="API endpoint URL. Default is http://localhost:8080/analyze_replay",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the responses as a JSON file.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous requests instead of asynchronous.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    test_analyze_replay(
        image_paths=args.images,
        url=args.url,
        output_file=args.output,
        async_mode=not args.sync,
    )
