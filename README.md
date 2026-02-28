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
![Stars](https://img.shields.io/github/stars/lachlanchen/words_gpt?style=for-the-badge&logo=github&logoColor=white&label=Stars)
![Issues](https://img.shields.io/github/issues/lachlanchen/words_gpt?style=for-the-badge&logo=github&logoColor=white&label=Open%20Issues)
![Issues](https://img.shields.io/github/license/lachlanchen/words_gpt?style=for-the-badge&logo=github&logoColor=white&label=License)

> A Raspberry Pi + Waveshare e-ink project that renders dynamic vocabulary cards with IPA phonetics and multilingual hints. It supports local CSV workflows, optional AI enrichment, e-paper rendering, and remote HTTP control.

Run modes at a glance:
`app.py` (service) and `words_gpt.py` (standalone renderer) can be operated independently or together.

| 🔎 At a Glance | Details |
|---|---|
| **Core runtime** | `app.py` (HTTP service) + `words_gpt.py` (renderer loop) |
| **Data path** | CSV datasets in `data/` + SQLite store `words_phonetics.db` |
| **Output targets** | Waveshare e-paper panels and virtual image outputs |
| **AI dependency** | Optional (`--enable_openai`) with request cache in `cache/` |
| **Default cadence** | Server on `8082`, periodic refresh around every 5 minutes |

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

`words_gpt` is a Python vocabulary card generation stack for e-ink displays. It combines data orchestration, phonetic enrichment, and rendering behind two runtime modes:

- A long-running Tornado service (`app.py`) for remote control and image serving.
- A standalone renderer (`words_gpt.py`) that can run in polling, loop, or direct-render modes.

Primary modules:

- `words_data.py` / `words_data_utils.py` for word and enrichment workflows.
- `words_data_with_legacy.py`, `words_data_without_legacy.py`, `words_data_workable*.py` for variant workflows and legacy compatibility.
- `words_database.py` for SQLite interaction.
- `openai_request_json.py` for structured OpenAI requests with on-disk cache/retry behavior.
- `env_loader.py` for deterministic environment loading.
- `words_update.py` for DB maintenance and re-check workflows.
- `app.py` and `words_gpt.py` for service/render lifecycle.
- `pwa/` for lightweight browser preview/config tooling.

## Highlights

- E-ink render pipeline with multilingual modes: Japanese variants, kanji mode, Arabic, Chinese, and emoji mode.
- Local-only and OpenAI-backed word sourcing in one workflow.
- Optional simplified Chinese rendering (`--simplify`).
- HTTP endpoints for remote control (`/next_random_word`, `/display_word`, `/get_current_word`, `/get_current_word_page`, `/get_words_card`).
- Caching and persistence to reduce repeated AI calls.
- Packaging and hardware examples via vendored `waveshare/` driver tree.

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
├─ words_data_with_legacy.py
├─ words_data_without_legacy.py
├─ words_data_workable.py
├─ words_data_workable_daniel.py
├─ words_data_back.py
├─ words_database.py
├─ openai_request_json.py
├─ phonetic_checker.py
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
├─ notebooks/
├─ waveshare/
│  ├─ setup.py
│  ├─ lib/
│  ├─ lib.old/
│  ├─ examples/
│  └─ pic/
└─ .auto-readme-work/
```

Important runtime files:

- `app.py`: Tornado app on port `8082` + periodic `next_random_word` trigger.
- `words_gpt.py`: standalone renderer and display abstractions (`EPaperHardware`, `EPaperDisplay`).
- `words_data.py`: fetch/enrichment workflow and chooser utilities.
- `words_database.py`: SQLite helpers for stored metadata and word cache operations.
- `scripts/*.sh`: install/service lifecycle and Raspberry Pi bootstrap helpers.
- `words_update.py`: batch DB refresh/recheck helper for data quality.

## Prerequisites

- Python `3.9+` (recommended)
- Raspberry Pi (required for hardware mode)
- Supported Waveshare e-paper panel (for example 7.3F or 13K family)
- SPI enabled (`raspi-config`), correct wiring, and stable power
- NLTK corpus when using NLTK word sources

Common dependencies in-tree and used by runtime paths:
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

- Pi-specific dependency installation
- SPI enablement helper checks
- `wordscard` virtual environment setup
- Python/runtime package installation
- Waveshare package install
- `tmux` launch of app process

### Option 3 — systemd service installation

To register app lifecycle under `systemd`:

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

`env_loader` reads environment keys and applies them in process startup context. Current docs and runtime usage commonly include:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

Assumption: keep secrets in local environment configuration and do not commit them to version control.

### Runtime flags (used by `app.py` and `words_gpt.py`)

| CLI flag | Purpose |
| --- | --- |
| `--enable_openai` | Enable optional OpenAI enrichment mode |
| `--make_emoji` | Render emoji-focused cards |
| `--ignore_list` | Skip words from configured ignore lists |
| `--simplify` | Produce simplified CJK output |
| `--use_csv` | Read words from CSV datasets |
| `--complete_csv` | Use the complete CSV source mode |
| `--filename <csv_file>` | Point to a specific CSV input file |

`APP_ARGS` can be passed through startup scripts. Example:

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### Routing behavior in app mode

Observed API routes:

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (served from `words_card_temp/`)

Compatibility note: older docs may reference `GET /current_word`; current route is `GET /get_current_word`.

### OpenAI usage notes

OpenAI features are optional and controlled via CLI/env flags. Cached requests in `cache/` help with reproducibility and rate-limit control. For deterministic offline-first runs, start with CSV mode (`--use_csv`) and enable OpenAI selectively.

## Usage

### Run HTTP server

```bash
python app.py
```

The process keeps the latest image in `words_card_temp/` and exposes endpoints used by front-end tools or scripts.

### Run renderer directly

CSV mode:

```bash
python words_gpt.py --use_csv
```

OpenAI mode:

```bash
python words_gpt.py --enable_openai --use_csv
```

Emoji + simplified CJK mode:

```bash
python words_gpt.py --make_emoji --simplify
```

### Run on Pi hardware

- Start with tmux script:

```bash
bash scripts/start_wordscard.sh
```

- Stop with tmux script:

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

Trigger card rendering via form-style endpoint:

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
- `pic/` and `figs/`: reference images and banners

## Development notes

- Legacy/backup modules and artifacts exist (for example `words_gpt_old.py`, `lib.old`), so treat those as references unless migration or compatibility is intentional.
- `words_update.py` contains batch refresh/recheck helpers useful for DB quality maintenance.
- Hardware validation is handled by `epd_*_test.py` and `waveshare/examples/*` demo scripts.
- There is no `requirements.txt` or lockfile in the repository root; dependencies are driven by setup scripts and direct install flows.
- No automated test suite is configured in this repo.

## Troubleshooting

- `ImportError` from Raspberry Pi GPIO/SPI modules:
  - Use Pi setup flow (`scripts/setup_pi_wordscard.sh`) or install dependencies explicitly on a compatible target.
- `403/404` from image/static endpoints:
  - Confirm route usage (`/get_current_word*`) and that `words_card_temp/` is writable.
- Empty/invalid word payload from OpenAI mode:
  - Verify `OPENAI_API_KEY` (and org/model values) are loaded, then inspect `cache/` and logs.
- Bad rendering or text clipping:
  - Verify font paths, display resolution constants, and the selected mode settings inside `words_gpt.py`.
- API returns stale data:
  - Call `POST /next_random_word` manually and review callback interval in `app.py`.
- Hardware rendering appears frozen:
  - Check tmux session and systemd logs with `journalctl -u wordscard`.
- Missing dataset/dictionary entries:
  - Validate CSV files in `data/` and run `words_update.py` maintenance tasks.

## Roadmap

- Add a minimal `requirements.txt` / reproducible install manifest.
- Add clearer runtime modes and explicit CLI `--help` docs.
- Expand rendering mode documentation (`japanese_synonym`, `arabic_synonym`, `film`, and other workflows).
- Standardize error handling and user-facing API response schemas.
- Add lightweight smoke-test stubs for non-hardware CI validation.

## Contributing

Contributions are welcome. Suggested workflow:

1. Keep changes scoped to one behavior area (rendering, data, API, scripts).
2. Update command usage/docs for user-facing behavior changes.
3. Preserve existing CLI flags and endpoint compatibility where possible.
4. If hardware scripts change, document tested device/model and exact commands run.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## License

No `LICENSE` file is present in the current repository root. The effective license is therefore undefined in-tree as of this draft. Please add one if you want explicit redistribution/reuse terms.
