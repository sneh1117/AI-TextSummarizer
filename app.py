from flask import Flask, request, jsonify, render_template
from summarizer import summarize
from database import init_db, save_summary, get_history

import os
import logging
import time
from dotenv import load_dotenv

# ── Logging setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("app.log"),   # saves to app.log file
        logging.StreamHandler()            # also prints to terminal
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
init_db()

load_dotenv()
HEADERS = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

MIN_WORDS = 100
MAX_WORDS = 2000

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize_route():
    data = request.get_json()

    # ── Log incoming request ──
    logger.info("Summarize request received")

    if not data or "text" not in data:
        logger.warning("Request rejected — missing 'text' field in body")
        return jsonify({"error": "Request body must include a 'text' field."}), 400

    text = data["text"]
    char_count = len(text.strip())
    logger.info(f"Input received — {char_count} characters")

    try:
        start_time = time.time()

        summary = summarize(text)

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Summarization successful — completed in {elapsed}s | Input: {char_count} chars | Output: {len(summary)} chars")

        save_summary(text.strip(), summary)
        logger.info("Summary saved to database")

        return jsonify({"summary": summary})

    except ValueError as e:
        logger.warning(f"Validation error — {str(e)}")
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        logger.error(f"Unexpected error during summarization — {str(e)}", exc_info=True)
        return jsonify({"error": "Something went wrong during summarization."}), 500

@app.route("/history")
def history():
    logger.info("History endpoint called")
    rows = get_history()
    logger.info(f"Returned {len(rows)} history records")
    return jsonify([
        {"id": r[0], "summary": r[1], "char_count": r[2], "created_at": r[3]}
        for r in rows
    ])

if __name__ == "__main__":
    logger.info("Starting Distill Flask app...")
    app.run(debug=True)