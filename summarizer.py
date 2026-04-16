import requests
import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_URL = "https://router.huggingface.co/hf-inference/models/sshleifer/distilbart-cnn-12-6"
HF_TOKEN = os.getenv('HF_TOKEN')
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

MIN_WORDS = 100
MAX_WORDS = 2000

def validate_text(text):
    if not text or not isinstance(text, str):
        raise ValueError("Input must be a non-empty string.")
    text = text.strip()
    if len(text) == 0:
        raise ValueError("Input cannot be blank or whitespace only.")
    word_count = len(text.split())
    if word_count < MIN_WORDS:
        raise ValueError(f"Text too short. Minimum {MIN_WORDS} words, you sent {word_count}.")
    if word_count > MAX_WORDS:
        raise ValueError(f"Text too long. Maximum {MAX_WORDS} words, you sent {word_count}.")
    return text


def call_api(text, retries=3, wait=25):
    # ── Diagnostic logs ──
    logger.info(f"HF_TOKEN present: {bool(HF_TOKEN)}")
    logger.info(f"HF_TOKEN prefix: {HF_TOKEN[:6] if HF_TOKEN else 'MISSING'}")
    logger.info(f"Calling API with {len(text.split())} words")

    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 150,
            "min_length": 40,
            "do_sample": False
        }
    }

    for attempt in range(retries):
        logger.info(f"API attempt {attempt + 1}/{retries}")
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response body: {response.text[:300]}")  # first 300 chars

            if response.status_code == 200:
                result = response.json()[0]["summary_text"]
                logger.info(f"Success — summary length: {len(result)} chars")
                return result
            elif response.status_code == 503:
                logger.warning(f"503 — model loading, waiting {wait}s before retry")
                if attempt < retries - 1:
                    time.sleep(wait)
                    continue
                raise RuntimeError("Model is loading, please try again in 30 seconds.")
            elif response.status_code == 401:
                logger.error("401 Unauthorized — HF_TOKEN is invalid or missing")
                raise RuntimeError("Invalid HuggingFace token.")
            elif response.status_code == 429:
                logger.error("429 Rate limited by HuggingFace")
                raise RuntimeError("Rate limited. Please wait and try again.")
            else:
                logger.error(f"Unexpected status {response.status_code}: {response.text}")
                raise RuntimeError(f"API error {response.status_code}: {response.text}")

        except requests.exceptions.Timeout:
            logger.error(f"Attempt {attempt + 1} timed out after 60s")
            if attempt < retries - 1:
                time.sleep(5)
                continue
            raise RuntimeError("HuggingFace API timed out after all retries.")

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {str(e)}")
            raise RuntimeError(f"Could not connect to HuggingFace API: {str(e)}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {str(e)}")
            raise RuntimeError(f"Network error: {str(e)}")


def chunk_text(text, chunk_size=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def summarize(text):
    text = validate_text(text)
    word_count = len(text.split())
    logger.info(f"Summarizing {word_count} words")

    if word_count <= 400:
        return call_api(text)

    chunks = chunk_text(text, chunk_size=400)
    logger.info(f"Split into {len(chunks)} chunks")
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
        chunk_summary = call_api(chunk)
        chunk_summaries.append(chunk_summary)

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    combined = " ".join(chunk_summaries)
    logger.info("Running final summary pass on combined chunks")
    return call_api(combined)