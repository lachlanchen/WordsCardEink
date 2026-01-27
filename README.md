# Eink Words GPT

A Raspberry Pi + Waveshare e‑ink project that displays dynamically selected vocabulary with phonetics and multilingual synonyms. The system can fetch words from local datasets or OpenAI, render them into a layout, and push the result to supported e‑paper panels. It also exposes a small HTTP service for triggering word updates and retrieving rendered images.

## Highlights
- E‑ink rendering pipeline with multiple content modes (kanji, Japanese, Arabic, Chinese, emoji).
- Local word database (`words_phonetics.db`) with CSV-backed word lists in `data/`.
- OpenAI-backed word selection and phonetic enrichment with structured JSON outputs.
- HTTP service for external triggers and image retrieval.

## Project Layout
- `app.py`: Tornado web server that drives word selection + display updates.
- `words_gpt.py`: Main standalone runner for rendering to hardware or virtual images.
- `words_data*.py`, `words_data_utils.py`, `words_database.py`: Data access, word selection, and OpenAI helpers.
- `openai_request_json.py`: Structured output helper + caching.
- `data/`: CSV word lists and curated datasets.
- `font/`, `pic/`, `words_card_temp/`: Fonts and images.
- `waveshare/`, `lib/`: Waveshare e‑paper drivers and examples.

## Requirements
- Python 3.9+ recommended.
- Hardware: Raspberry Pi and a supported Waveshare e‑paper panel (examples in `waveshare/examples/`).
- Python packages used throughout the project include: `openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.

Install the Waveshare driver on device:
```bash
python setup.py install
```

If you use the NLTK word list, you may need to download it once:
```bash
python -m nltk.downloader words
```

## Configuration (.env)
This repo loads environment variables from `.env` at import time and **overrides** any existing shell values. This makes local overrides deterministic even when you have values in `.bashrc` or `.profile`.

Create or update `.env` with your keys:
```
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

## Running the Server
Start the HTTP service (default port 8082):
```bash
python app.py
```
Key routes:
- `GET /current_word` (returns JSON + base64 image)
- `POST /display_word` (send a word or list of words)
- `GET /next_random_word` (triggers selection + display)

## Running the Renderer
Basic run with CSV-based word list:
```bash
python words_gpt.py --use_csv
```
Enable OpenAI:
```bash
python words_gpt.py --enable_openai --use_csv
```
Emoji rendering + simplified CJK:
```bash
python words_gpt.py --make_emoji --simplify
```

## Data & Logs
- Word lists live in `data/` (e.g., `data/words_list.csv`).
- Logs are stored in `logs/` and `logs-word-phonetics/`.
- Generated card images may be written under `words_card_temp/` depending on mode.

## Hardware Tests
Use a display-specific demo script (match your panel model):
```bash
python epd_7in3f_test.py
```
More examples live in `waveshare/examples/`.

## Notes on OpenAI Usage
OpenAI access is optional but recommended for fresh word generation and phonetic enrichment. The structured JSON helper in `openai_request_json.py` caches results under `cache/` to reduce repeated calls.

## Contributing
See `AGENTS.md` for contributor guidelines, coding style, and PR expectations.
