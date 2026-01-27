# Eink Words GPT

A Raspberry Pi + Waveshare e‑ink project that displays dynamically selected vocabulary with phonetics and multilingual synonyms. The system can fetch words from local datasets or OpenAI, render them into a layout, and push the result to supported e‑paper panels. It also exposes a small HTTP service for triggering word updates and retrieving rendered images.

## Highlights
- E‑ink rendering pipeline with multiple content modes (kanji, Japanese, Arabic, Chinese, emoji).
- Local word database (`words_phonetics.db`) with CSV-backed word lists in `data/`.
- OpenAI-backed word selection and phonetic enrichment with structured JSON outputs.
- HTTP service for external triggers and image retrieval.

## Demos
<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

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

## Support

### What your support makes possible
- <b>Keep tools open</b>: hosting, inference, data storage, and community ops.  
- <b>Ship faster</b>: focused open‑source time on WordsCardEink and related learning tools.  
- <b>Prototype devices</b>: e‑ink hardware iterations and display layout research.  
- <b>Access for all</b>: subsidized deployments for students, creators, and community groups.

### Donate

<div align="center">
<table style="margin:0 auto; text-align:center; border-collapse:collapse;">
  <tr>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://chat.lazying.art/donate">https://chat.lazying.art/donate</a>
    </td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://chat.lazying.art/donate"><img src="figs/donate_button.svg" alt="Donate" height="44"></a>
    </td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://paypal.me/RongzhouChen">
        <img src="https://img.shields.io/badge/PayPal-Donate-003087?logo=paypal&logoColor=white" alt="Donate with PayPal">
      </a>
    </td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400">
        <img src="https://img.shields.io/badge/Stripe-Donate-635bff?logo=stripe&logoColor=white" alt="Donate with Stripe">
      </a>
    </td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;"><strong>WeChat</strong></td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;"><strong>Alipay</strong></td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;"><img alt="WeChat QR" src="figs/donate_wechat.png" width="240"/></td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;"><img alt="Alipay QR" src="figs/donate_alipay.png" width="240"/></td>
  </tr>
</table>
</div>

**支援 / Donate**

- ご支援は研究・開発と運用の継続に役立ち、より多くのオープンなプロジェクトを皆さんに届ける力になります。  
- 你的支持将用于研发与运维，帮助我持续公开分享更多项目与改进。  
- Your support sustains my research, development, and ops so I can keep sharing more open projects and improvements.

## Contributing
See `AGENTS.md` for contributor guidelines, coding style, and PR expectations.
