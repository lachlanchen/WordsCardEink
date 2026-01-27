# Repository Guidelines

## Project Structure & Module Organization
- `app.py` runs the Tornado HTTP service and orchestrates word selection + e-ink rendering.
- `words_gpt.py` is the main standalone runner for rendering word cards to the display or virtual images.
- `words_data*.py` and `words_database.py` contain word fetching, phonetics, and DB helpers; `words_phonetics.db` is the SQLite store.
- `data/` holds CSV word lists and curated datasets; `font/` and `pic/` contain font assets and e-ink reference images.
- `waveshare/` and `lib/` vendor the Waveshare e-paper drivers and examples.
- `utilities/` contains ad-hoc scripts (OpenAI experiments, IPA helpers). `notebooks/` is for exploration.

## Build, Test, and Development Commands
- `python app.py` starts the local server on port 8082 for word updates and image rendering.
- `python words_gpt.py --enable_openai --use_csv` runs the main renderer with OpenAI-backed word selection and CSV lists.
- `python words_gpt.py --make_emoji --simplify` enables emoji generation and simplified CJK output.
- `python setup.py install` installs the Waveshare driver package (run on the Raspberry Pi / target device).

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and UTF-8 source files; keep imports grouped (stdlib, third-party, local).
- Prefer `snake_case` for functions and modules; keep script entrypoints in the repo root (e.g., `words_gpt.py`).
- Avoid adding new large assets to the root; put datasets in `data/` and images in `pic/` or `words_card_temp/`.

## Testing Guidelines
- No formal test runner is configured. Hardware smoke tests live in `epd_*_test.py` and `waveshare/examples/*_test.py`.
- Run a device-specific demo, e.g. `python epd_7in3f_test.py`, and document the panel model and outcome in your PR.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and lower-case (e.g., “add french and arabic phonetics”).
- PRs should include: summary, device model (if hardware-related), commands run, and screenshots/photos for UI/e-ink changes.
- If you modify datasets, describe the source and scope (file name + row/column impact).

## Security & Configuration Tips
- OpenAI access uses environment configuration (`OPENAI_API_KEY`); avoid hardcoding keys or org IDs.
- Logs and generated images can grow quickly—prefer adding new outputs under `logs/` or `words_card_temp/`.
