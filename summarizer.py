import requests
import os
import time
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


def call_api(text, retries=3, wait=25):
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 150,
            "min_length": 40,
            "do_sample": False
        }
    }
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)

            if response.status_code == 200:
                return response.json()[0]["summary_text"]
            elif response.status_code == 503:
                if attempt < retries - 1:
                    time.sleep(wait)
                    continue
                raise RuntimeError("Model is loading, please try again in 30 seconds.")
            else:
                raise RuntimeError(f"API error {response.status_code}: {response.text}")

        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(5)
                continue
            raise RuntimeError("HuggingFace API timed out. Please try again.")

        except requests.exceptions.RequestException as e:
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

    if word_count <= 400:
        return call_api(text)

    chunks = chunk_text(text, chunk_size=400)
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        chunk_summary = call_api(chunk)
        chunk_summaries.append(chunk_summary)

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    combined = " ".join(chunk_summaries)
    final_summary = call_api(combined)
    return final_summary