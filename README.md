[English](README.md) · [العربية](i18n/README.ar.md) · [Español](i18n/README.es.md) · [Français](i18n/README.fr.md) · [日本語](i18n/README.ja.md) · [한국어](i18n/README.ko.md) · [Tiếng Việt](i18n/README.vi.md) · [中文 (简体)](i18n/README.zh-Hans.md) · [中文（繁體）](i18n/README.zh-Hant.md) · [Deutsch](i18n/README.de.md) · [Русский](i18n/README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 🖨️ Eink Words GPT

**Language of this draft:** English

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=for-the-badge&logo=githubactions&logoColor=white)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=for-the-badge&logo=python&logoColor=white)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=for-the-badge&logo=openai&logoColor=white)

A Raspberry Pi + Waveshare e-ink project that renders dynamically selected vocabulary cards with IPA phonetics and multilingual hints. It supports local CSV workflows, optional AI enrichment, e-paper rendering, and remote HTTP control.

| 🔎 At a Glance | Details |
|---|---|
| Core runtime | `app.py` (HTTP service) + `words_gpt.py` (renderer loop) |
| Data path | CSV datasets in `data/` + SQLite store `words_phonetics.db` |
| Output targets | Waveshare e-paper panels and virtual image outputs |
| AI dependency | Optional (`--enable_openai`) with cache in `cache/` |
| Main loop defaults | Server on `8082`, periodic refresh around 5 minutes |

## 📚 Table of Contents
- [Overview](#overview)
- [Highlights](#highlights)
- [Demos](#demos)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Examples](#examples)
- [Data, cache, and logs](#data-cache-and-logs)
- [Development notes](#development-notes)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Support](#support)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

`words_gpt` is a Python vocabulary card generation stack for e-ink displays. It combines data orchestration, phonetic enrichment, and rendering orchestration behind two runtime modes:

- A long-running Tornado service (`app.py`) for remote control and image serving
- A standalone renderer (`words_gpt.py`) that can run in polling, loop, or direct-render modes

Primary modules:

- `words_data.py` / `words_data_utils.py` for word and enrichment workflows
- `words_database.py` for SQLite interaction
- `openai_request_json.py` for structured OpenAI requests with on-disk cache
- `env_loader.py` for deterministic environment loading
- `words_update.py` for DB maintenance and recheck workflows
- `app.py` and `words_gpt.py` for service/render lifecycle

## Highlights

- E-ink render pipeline with multiple language/content modes:
  - Japanese variants, kanji mode, Arabic, Chinese, emoji mode
- Local and OpenAI word sourcing in one workflow
- Optional simplified Chinese output in renderer path
- Server endpoints for direct interaction (`/next_random_word`, `/display_word`, etc.)
- Caching and persistence that reduce repeated network calls
- Optional PWA assets in `pwa/` for lightweight preview/config flows

## Demos

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## Project structure

```text
words_gpt/
├─ README.md
├─ AGENTS.md
├─ app.py
├─ words_gpt.py
├─ words_data.py
├─ words_data_utils.py
├─ words_database.py
├─ openai_request_json.py
├─ env_loader.py
├─ words_update.py
├─ setup.py
├─ scripts/
│  ├─ setup_pi_wordscard.sh
│  ├─ start_wordscard.sh
│  ├─ stop_wordscard.sh
│  └─ install_wordscard_service.sh
├─ epd_7in3f_test.py
├─ epd_13in3k_test.py
├─ words_phonetics.db
├─ data/
├─ font/
├─ pic/
├─ demos/
├─ figs/
├─ cache/
├─ logs/
├─ logs-word-phonetics/
├─ words_card_temp/
├─ pwa/
├─ i18n/
├─ utilities/
├─ references/
└─ waveshare/
    ├─ setup.py
    ├─ lib/
    ├─ lib.old/
    ├─ examples/
    └─ pic/
```

Important runtime files:

- `app.py`: Tornado app on port `8082` + periodic `next_random_word` trigger.
- `words_gpt.py`: standalone renderer and display abstractions (`EPaperHardware`, `EPaperDisplay`).
- `words_data.py`: advanced fetch/enrichment workflow and helper utilities.
- `words_database.py`: SQLite helpers for stored metadata and word cache operations.
- `scripts/*.sh`: install/service lifecycle and Raspberry Pi bootstrap helpers.

## Prerequisites

- Python `3.9+` (recommended)
- Raspberry Pi (required for hardware mode)
- Supported Waveshare e-paper panel (e.g., 7.3F / 13K family)
- SPI enabled (`raspi-config`), correct wiring, and power-stable runtime
- NLTK corpus available when using `nltk` word sources

Common dependencies observed in the codebase:
`openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.

## Installation

### Option 1 — Minimal/manual install (desktop or Pi)

From repo root:

```bash
python setup.py install
```

If needed:

```bash
python -m nltk.downloader words
```

### Option 2 — Raspberry Pi automated setup (recommended on-device)

From repo root:

```bash
bash scripts/setup_pi_wordscard.sh
```

This performs:

- Pi-specific dependencies
- SPI enablement
- `wordscard` virtual environment setup
- Python/runtime package installation
- Waveshare package install
- `tmux` launch of app process

### Option 3 — Service installation

To register app lifecycle with `systemd`:

```bash
bash scripts/install_wordscard_service.sh
```

Then:

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## Configuration

### Environment variables (`.env`)

The repo uses `.env` loading that currently overrides pre-existing shell variables. Use this intentionally:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### Runtime flags (used by both `app.py` and `words_gpt.py`)

- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

The Pi startup scripts support argument passthrough through `APP_ARGS` (example):

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### Routing behavior in app mode

Observed routes in current code:

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (served from `words_card_temp/`)

Compatibility note: earlier docs referenced `GET /current_word`; current route is `GET /get_current_word`.

### Notes on OpenAI usage

OpenAI features are optional and controlled by CLI/env flags. Cached API payloads are useful for reproducibility and rate-limiting control. In constrained environments, run in CSV mode first (`--use_csv`) and enable OpenAI selectively (`--enable_openai`) when enrichment is desired.

## Usage

### Run HTTP server

```bash
python app.py
```

The process keeps an image in `words_card_temp/` and exposes HTTP endpoints used by front-end tools or simple scripts.

### Run renderer directly

CSV mode:

```bash
python words_gpt.py --use_csv
```

OpenAI mode:

```bash
python words_gpt.py --enable_openai --use_csv
```

Emoji + simplified CJK:

```bash
python words_gpt.py --make_emoji --simplify
```

### Run on Pi hardware

- Start via `tmux` script:

```bash
bash scripts/start_wordscard.sh
```

- Stop via `tmux` script:

```bash
bash scripts/stop_wordscard.sh
```

## Examples

Get next random card metadata:

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

Fetch current stored word:

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

Ask for rendered page image payload:

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

Submit an explicit word:

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

Trigger card rendering via form style endpoint:

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

## Data, cache, and logs

Typical artifacts used by the app:

- `data/`: curated CSV datasets
- `words_phonetics.db`: SQLite cache/source database
- `cache/`: OpenAI request/result cache
- `word_phonetics_processed.csv`: processed/derived dataset
- `logs/`, `logs-word-phonetics/`: runtime logs
- `words_card_temp/`: generated cards and temporary output

## Development notes

- Legacy/backup files exist (for example `words_gpt_old.py`, `lib.old`), so treat those as references unless you are specifically migrating or maintaining compatibility.
- `words_update.py` contains batch refresh/recheck helpers useful for DB data quality pass.
- Hardware validation is handled by `epd_*_test.py` and `waveshare/examples/*` demos.
- There is no `requirements.txt` or lockfile in the repository root; dependency setup is done through the setup script or direct installation.
- No automated test suite is configured in this repo.

## Troubleshooting

- `ImportError` from Raspberry Pi GPIO/SPI modules:
  - Install via Pi path (`setup_pi_wordscard.sh`), or check `python setup.py install` on compatible target.
- `403/404` from image/static endpoints:
  - Confirm `/get_current_word*` endpoint usage and that `words_card_temp/` is writable.
- Empty/invalid word payload from OpenAI mode:
  - Confirm `OPENAI_API_KEY` and optional org/model values are loaded; inspect `cache/` and logs.
- Bad rendering/text clipping:
  - Verify font path and panel resolution settings in the rendering path inside `words_gpt.py`.
- API returns stale data:
  - Call `POST /next_random_word` manually and review periodic callback interval in `app.py`.
- Hardware update appears frozen:
  - Check tmux session and systemd logs (`journalctl -u wordscard`).
- Missing dataset or dictionary entries:
  - Validate CSV files in `data/` and run `words_update.py` workflows for refresh/cleanup.

## Roadmap

- Add a minimal `requirements.txt` / reproducible install manifest.
- Add clearer runtime modes and explicit CLI `--help` docs.
- Expand rendering schema docs for each content mode (`japanese_synonym`, `arabic_synonym`, `film`, etc.).
- Standardize error handling and user-facing API response schemas.
- Add small smoke-test script stubs for non-hardware CI validation.

## Support

| Support option | Link | Purpose |
|---|---|---|
| GitHub Sponsors | https://github.com/sponsors/lachlanchen | Ongoing and one-time project support |
| Lazying Art | https://lazying.art | Brand & related resources |
| Chat | https://chat.lazying.art | Discussion and support |
| Only Ideas | https://onlyideas.art | Creative research and side projects |

## Contributing

Contributions are welcome. Suggested flow:

1. Keep changes scoped to one behavior area (render, data, API, scripts).
2. Update command usage/docs for user-facing behavior changes.
3. Preserve existing CLI flags and endpoint compatibility where possible.
4. If hardware scripts change, document tested device/model and exact commands run.

## License

No `LICENSE` file is present in the current repository root. The effective license is therefore undefined in-tree as of this draft. Please add one if you want explicit redistribution/reuse terms.
