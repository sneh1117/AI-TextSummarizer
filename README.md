# Distill — AI Text Summarizer

A web app that summarizes long-form text using HuggingFace's BART model.

## Built With
- Python & Flask
- HuggingFace Inference API (BART)
- SQLite
- Vanilla HTML/CSS/JS

## Features
- Word count validation (100–2000 words)
- Multi-pass chunking for long texts
- Summary history stored in SQLite
- Deployed on Render

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your HuggingFace token: `HF_TOKEN=your_token_here`
4. Run: `python app.py`
