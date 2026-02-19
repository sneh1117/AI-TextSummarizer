import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
HEADERS = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

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


def call_api(text):
    """Send a single chunk to the HuggingFace API."""
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 150,
            "min_length": 40,
            "do_sample": False
        }
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code == 503:
        raise RuntimeError("Model is loading on HuggingFace servers, try again in 20 seconds.")
    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")

    return response.json()[0]["summary_text"]


def chunk_text(text, chunk_size=400):
    """Split text into word-based chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def summarize(text):
    text = validate_text(text)
    word_count = len(text.split())

    # Short enough — send directly
    if word_count <= 400:
        return call_api(text)

    # Too long — chunk it, summarize each, then summarize the summaries
    chunks = chunk_text(text, chunk_size=400)
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        chunk_summary = call_api(chunk)
        chunk_summaries.append(chunk_summary)

    # If only one chunk came back just return it
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    # Final pass — combine all chunk summaries into one
    combined = " ".join(chunk_summaries)
    final_summary = call_api(combined)
    return final_summary


