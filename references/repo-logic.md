# WordsGPT Repository Logic Overview

This document summarizes how the current codebase works and how to refactor toward a clean, structured-output AI layer.

## 1) High-Level Flow
1. **Word selection**
   - Local word lists live in `data/*.csv`.
   - `WordsDatabase` (`words_database.py`) stores canonical word details in SQLite (`words_phonetics.db`).
   - `AdvancedWordFetcher` (`words_data.py`) can pull from OpenAI (or local fallback) and enrich words.
2. **Word rendering**
   - `EPaperDisplay` (`words_gpt.py`) renders word cards (layout, fonts, multi-language synonyms, phonetics).
   - The hardware driver is Waveshare (`lib/` and `waveshare/`); `EPaperHardware` abstracts device operations.
3. **Serving updates**
   - `app.py` exposes Tornado HTTP routes to request a new word or fetch the current rendered image.

## 2) Core Modules
- `app.py`: Tornado web server + orchestration; triggers selection and render.
- `words_gpt.py`: Standalone renderer and hardware control. Also builds “virtual images”.
- `words_data.py`: Primary data layer, OpenAI word generation/enrichment, and helpers.
- `words_data_utils.py`: Utility functions and chooser logic (randomization, cleanup).
- `words_database.py`: SQLite interactions.
- `openai_request_json.py`: Structured JSON request helper + caching.

## 3) Data Model (Current)
`WordsDatabase` stores details like:
- `word`, `syllable_word`, `phonetic`
- multilingual synonyms (Japanese, Arabic, Chinese, French)
- various “cleaned” fields and heuristics

## 4) API Surface (Current)
`app.py` routes (typical):
- `GET /current_word`: returns JSON + base64 image
- `POST /display_word`: accepts word(s) and renders
- `GET /next_random_word`: selects next and updates display

## 5) AI Structured Outputs (Recommended)
Goal: use modern structured outputs consistently.

### OpenAI (structured outputs)
Use `response_format: { type: "json_schema", json_schema: { ... } }` and parse `message.content`.
This is already supported by `openai_request_json.py`, but it can be swapped with a unified client.

### DeepSeek (JSON object output)
DeepSeek doesn’t support JSON Schema. Use:
- `response_format: { type: "json_object" }`
- Provide a concrete example object in the system prompt.
This is implemented in `references/echomind_ai/deepseek_requests.py`.

### Mixed Mode (OpenAI + DeepSeek)
Use a priority list and fallback semantics (OpenAI -> DeepSeek).
Reference: `references/echomind_ai/mixed_ai_request.py`.

## 6) Refactor Direction (Suggested)
Create a single “AI client” interface that supports:
- `send_request_with_json_schema()`
- `send_simple_request()`
- provider selection: `openai` | `deepseek` | `mixed`

Suggested consolidation:
- Move provider configs to `ai_config.py`
- Add a factory (like EchoMind’s `ai_client_factory.py`)
- Replace direct `OpenAI()` creation in `words_data*.py` with the factory output

## 7) Caching Behavior
Current cache is file-based in `cache/`:
- Prompt hashes -> JSON responses
- Keep this for stability across retries.

## 8) Hardware + Rendering Notes
- `EPaperDisplay` builds a composite image and pushes it to the driver buffer.
- “Virtual image” modes write results to disk for previews.
- Fonts live in `font/`, test assets in `pic/`.

## 9) Next Refactor Steps (Tactical)
- Introduce `ai/` package:
  - `ai/client_factory.py`
  - `ai/providers/openai_client.py`
  - `ai/providers/deepseek_client.py`
  - `ai/providers/mixed_client.py`
- Add a single config file (JSON or env) describing:
  - mode (openai/deepseek/mixed)
  - primary/secondary providers
  - per-provider models
- Update `words_data.py` to call `ai_client.send_request_with_json_schema()` exclusively.

## 10) EchoMind Code Imported for Reference
Copied into:
- `references/echomind_ai/ai_client_factory.py`
- `references/echomind_ai/ai_config.py`
- `references/echomind_ai/mixed_ai_request.py`
- `references/echomind_ai/openai_request.py`
- `references/echomind_ai/deepseek_requests.py`

Use these as a baseline for your refactor.
