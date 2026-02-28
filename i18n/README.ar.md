[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Eink Words GPT

**خيارات اللغة:** العربية (هذه النسخة)

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=flat-square&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=flat-square)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=flat-square)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=flat-square&logo=openai&logoColor=white)

مشروع يعتمد على Raspberry Pi مع شاشة Waveshare e-ink لعرض مفردات يتم اختيارها ديناميكيًا مع النطق الصوتي (phonetics) ومرادفات متعددة اللغات. يمكن للنظام جلب الكلمات من بيانات محلية أو عبر OpenAI، ثم تنسيقها بصريًا ودفعها إلى شاشات e-paper المدعومة. كما يوفّر خدمة HTTP صغيرة لتشغيل تحديثات الكلمات واسترجاع الصور المولَّدة.

| 🔎 نظرة سريعة | التفاصيل |
|---|---|
| التشغيل الأساسي | `app.py` (خدمة HTTP) + `words_gpt.py` (حلقة التصيير) |
| مسار البيانات | ملفات CSV داخل `data/` + قاعدة SQLite باسم `words_phonetics.db` |
| مخرجات العرض | شاشات Waveshare e-paper ومخرجات صور افتراضية |
| اعتماد الذكاء الاصطناعي | اختياري (`--enable_openai`) مع كاش داخل `cache/` |

## 📚 Table of Contents
- [نظرة عامة](#overview)
- [أهم النقاط](#highlights)
- [بدء سريع](#quick-start)
- [العروض](#demos)
- [الميزات](#features)
- [هيكل المشروع](#project-structure)
- [المتطلبات المسبقة](#prerequisites)
- [التثبيت](#installation)
- [الإعداد](#configuration)
- [الاستخدام](#usage)
- [أمثلة](#examples)
- [البيانات والكاش والسجلات](#data-cache-and-logs)
- [ملاحظات التطوير](#development-notes)
- [استكشاف الأخطاء وإصلاحها](#troubleshooting)
- [ملاحظات حول استخدام OpenAI](#notes-on-openai-usage)
- [خارطة الطريق](#roadmap)
- [الدعم](#-support)
- [المساهمة](#contributing)
- [الترخيص](#license)

## Overview
`words_gpt` هو نظام مبني بلغة Python لتوليد بطاقات مفردات وعرضها على أجهزة e-ink.

ويجمع بين:
- جلب الكلمات من CSV/بيانات محلية مع توليد اختياري عبر OpenAI.
- الإثراء اللغوي (IPA + حقول مرادفات متعددة اللغات).
- مسارات تصيير لمخرجات العتاد والمخرجات الافتراضية.
- خدمة Tornado HTTP للتشغيل عن بُعد واسترجاع الصور.

يرتكز الكود الحالي بشكل أساسي على `app.py` و`words_gpt.py` و`words_data.py` و`words_database.py` و`openai_request_json.py`.

## Highlights
- 🖼️ مسار تصيير e-ink مع أوضاع محتوى متعددة (kanji، اليابانية، العربية، الصينية، emoji).
- 🗃️ قاعدة كلمات محلية (`words_phonetics.db`) مع قوائم كلمات CSV داخل `data/`.
- 🤖 اختيار كلمات وإثراء صوتي عبر OpenAI باستخدام مخرجات JSON منظَّمة.
- 🌐 خدمة HTTP للتشغيل الخارجي واسترجاع الصور.
- ⚡ طبقة كاش (`cache/`) لتقليل استدعاءات OpenAI المتكررة.

## Quick Start
| الهدف | الأمر |
|---|---|
| تشغيل خادم HTTP (المنفذ `8082`) | `python app.py` |
| تشغيل المصيّر المستقل (CSV) | `python words_gpt.py --use_csv` |
| تشغيل مع OpenAI + CSV | `python words_gpt.py --enable_openai --use_csv` |
| وضع Emoji + simplified CJK | `python words_gpt.py --make_emoji --simplify` |
| إعداد Raspberry Pi تلقائيًا | `bash scripts/setup_pi_wordscard.sh` |

## Demos
<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## Features
- تدفق تصيير عتادي وافتراضي (`EPaperHardware`, `EPaperDisplay`) من `words_gpt.py`.
- مسار إثراء متعدد اللغات في `words_data.py` (IPA، صيغ يابانية، عربية، فرنسية، وصينية).
- تخزين دائم عبر SQLite مع مساعدات تحديث حقول ديناميكية في `words_database.py`.
- مساعد طلبات OpenAI بصيغة JSON منظَّمة مع كاش ملفات في `openai_request_json.py`.
- أصول PWA اختيارية داخل `pwa/` لسير عمل خفيف للإعداد/المعاينة على الواجهة الأمامية.

## Project Structure
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
├─ epd_7in3f_test.py
├─ epd_13in3k_test.py
├─ words_phonetics.db
├─ word_phonetics_processed.csv
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
├─ scripts/
├─ utilities/
├─ references/
├─ i18n/
└─ waveshare/
```

ملفات التشغيل المهمة:
- `app.py`: خادم Tornado (المنفذ الافتراضي `8082`) وحلقة تحديث دورية.
- `words_gpt.py`: حلقة التصيير المستقلة وفئات العرض.
- `words_data.py`: تنسيق جلب/إثراء الكلمات الأساسي.
- `words_database.py`: مساعدات تخزين SQLite.
- `scripts/*.sh`: إعداد Raspberry Pi وتثبيت الخدمة وسكربتات دورة حياة tmux.

## Prerequisites
- Python `3.9+` (مُوصى به).
- جهاز Raspberry Pi مستهدف (لوضع العتاد).
- شاشة Waveshare e-paper مدعومة.
- تفعيل SPI على Pi (`raspi-config`) مع التوصيلات المناسبة لطراز الشاشة.

تشمل حزم Python المستخدمة في هذا المشروع:
- `openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.
- سكربت الإعداد يثبّت أيضًا: `json5`, `pandas`, `spidev`, `RPi.GPIO`, `gpiozero`, `lgpio`.

## Installation

### Option A: Minimal/manual install
تثبيت حزمة تعريف Waveshare:
```bash
python setup.py install
```

إذا كنت تستخدم قائمة كلمات NLTK، نزّلها مرة واحدة:
```bash
python -m nltk.downloader words
```

### Option B: Raspberry Pi automated setup (recommended on device)
من جذر المستودع:
```bash
bash scripts/setup_pi_wordscard.sh
```

يقوم هذا السكربت بما يلي:
- تثبيت اعتماديات apt.
- التأكد من تفعيل SPI.
- إنشاء وتفعيل بيئة `wordscard` الافتراضية.
- تثبيت اعتماديات Python التشغيلية.
- تثبيت حزمة Waveshare.
- تشغيل `app.py` داخل جلسة tmux.

## Configuration

### `.env` behavior
هذا المستودع يحمّل متغيرات البيئة من `.env` وقت الاستيراد ويقوم **بالكتابة فوق** أي قيم موجودة مسبقًا في shell. هذا يجعل تجاوزات الإعداد المحلي حتمية حتى عند تصدير القيم مسبقًا في ملفات إعداد shell.

أنشئ أو حدّث `.env`:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### App argument passthrough
تدعم سكربتات systemd/tmux ما يلي:
```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### CLI flags (server and renderer)
كل من `app.py` و`words_gpt.py` يدعمان:
- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

## Usage

### Running the HTTP server
شغّل الخدمة (المنفذ الافتراضي `8082`):
```bash
python app.py
```

المسارات الملاحظة في الكود:
- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (من `words_card_temp/`)

ملاحظة توافق: كانت وثائق أقدم تشير إلى `GET /current_word`، لكن المسار الحالي في `app.py` هو `GET /get_current_word`.

### Running standalone renderer
قائمة معتمدة على CSV:
```bash
python words_gpt.py --use_csv
```

تفعيل OpenAI:
```bash
python words_gpt.py --enable_openai --use_csv
```

تصيير Emoji + simplified CJK:
```bash
python words_gpt.py --make_emoji --simplify
```

### Service mode on Raspberry Pi
تثبيت وحدة الخدمة:
```bash
bash scripts/install_wordscard_service.sh
```

ثم:
```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## Examples

### Trigger next random word
```bash
curl "http://127.0.0.1:8082/next_random_word"
```

### Read current word payload
```bash
curl "http://127.0.0.1:8082/get_current_word"
```

### Submit explicit word
```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

### Hardware smoke tests
استخدم السكربت المخصص للشاشة:
```bash
python epd_7in3f_test.py
```

أو:
```bash
python epd_13in3k_test.py
```

يوجد المزيد من الأمثلة في `waveshare/examples/`.

## Data, Cache, and Logs
| المنطقة | المسار/المسارات | ملاحظات |
|---|---|---|
| قوائم الكلمات | `data/` | تتضمن `data/words_list.csv` وملفات CSV موضوعية |
| قاعدة البيانات الدائمة | `words_phonetics.db` | مخزن محلي للنطق والإثراء |
| عناصر OpenAI/الكاش | `cache/` | يقلل الطلبات المتكررة |
| السجلات | `logs/`, `logs-word-phonetics/` | سجلات التشغيل والتحديث |
| البطاقات المُولدة | `words_card_temp/` | مخرجات الصور ومصدر الخدمة الثابتة |

## Development Notes
- إدارة الاعتماديات تعتمد على السكربتات أولًا (`scripts/setup_pi_wordscard.sh`) + `setup.py`؛ لا يوجد `requirements.txt` أو `pyproject.toml` حتى الآن.
- توجد ملفات نسخ احتياطي/إرث متعددة (`words_data_*`, `words_gpt_old.py`)؛ مسار التشغيل النشط يتركز أساسًا في `app.py` + `words_gpt.py` + `words_data.py` + `words_database.py`.
- `env_loader.py` يكتب دائمًا فوق متغيرات البيئة من `.env` عند وجود المفاتيح.
- وضع الخادم يشغّل تدفق تحديث دوري (كل ~5 دقائق) ويمكنه استدعاء نقطة التحديث داخليًا.

## Troubleshooting
- `ModuleNotFoundError` أو مشاكل الاستيراد:
  - تأكد من تفعيل البيئة الافتراضية وتثبيت الاعتماديات.
  - أعد تشغيل `bash scripts/setup_pi_wordscard.sh` على Pi.
- أخطاء OpenAI (`401`، نموذج/مفتاح مفقود):
  - تحقّق من `OPENAI_API_KEY` و`OPENAI_MODEL` الاختياري داخل `.env`.
  - تأكد من اتصال الشبكة من الجهاز.
- الشاشة لا تتحدّث:
  - تحقّق من طراز الشاشة والتوصيلات وشغّل سكربت الاختبار المطابق (`epd_7in3f_test.py` أو `epd_13in3k_test.py`).
  - تأكد من تفعيل SPI (`sudo raspi-config nonint do_spi 0`).
  - على Pi 5، تأكد من وصلة التوافق `/dev/spidev0.0` إذا كان الجهاز يعرِض `/dev/spidev10.0`.
- مشاكل تثبيت OpenCC:
  - استخدم الحزمة المتوافقة مع توزيعتك (`libopencc1` أو `libopencc2`) كما في سكربت الإعداد.
- عدم تطابق مسارات API:
  - استخدم `/get_current_word` للحمولة الحالية وليس `/current_word`.

## Notes on OpenAI Usage
الوصول إلى OpenAI اختياري لكنه مفيد لتوليد كلمات جديدة وإثراء النطق. المساعد المعتمد على JSON المنظَّم في `openai_request_json.py` يخزن النتائج مؤقتًا داخل `cache/` لتقليل الاستدعاءات المتكررة.

## Roadmap
- إضافة ملف اعتماديات رسمي (`requirements.txt` أو `pyproject.toml`) لتثبيتات قابلة لإعادة الإنتاج.
- توسيع `i18n/` بنسخ README مترجمة ومُحافَظ عليها.
- توحيد نسخ السكربتات القديمة/الاحتياطية بعد تثبيت المسار القياسي.
- توثيق سير عمل PWA (`pwa/`) مع أمثلة نقاط النهاية ولقطات شاشة.
- إضافة اختبارات آلية قابلة للتكرار لسلوك البيانات والمسارات.

## ❤️ Support

إذا كان هذا المشروع مفيدًا لك، فهذه الروابط تدعم مباشرةً الصيانة المستمرة وتكرار تطوير العتاد.

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

### ما الذي يتيحه دعمك
- <b>إبقاء الأدوات مفتوحة</b>: الاستضافة، الاستدلال، تخزين البيانات، وعمليات المجتمع.
- <b>تطوير أسرع</b>: وقت أكثر تركيزًا للمصدر المفتوح على WordsCardEink وأدوات تعلم مرتبطة.
- <b>نمذجة الأجهزة</b>: تكرارات عتاد e-ink وأبحاث تنسيقات العرض.
- <b>وصول للجميع</b>: نشرات مدعومة للطلاب والمبدعين ومجموعات المجتمع.

### التبرع

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
اطّلع على `AGENTS.md` لإرشادات المساهمة وأسلوب كتابة الكود وتوقعات طلبات السحب (PR).

قائمة تحقق مقترحة للمساهمة:
- تضمين طراز الشاشة وملاحظات العتاد عند تغيير العرض.
- إدراج الأوامر الدقيقة التي استُخدمت للتحقق.
- إرفاق لقطات شاشة/صور لأي تغييرات على واجهة المستخدم أو مخرجات e-ink.
- وصف تعديلات البيانات (الملف + تأثير الصف/العمود).

## License
لا يوجد ملف `LICENSE` حاليًا في جذر المستودع (وفق الملاحظة في هذه المسودة). إلى أن يُضاف ملف ترخيص، لا توجد صلاحيات إعادة استخدام مصرح بها بشكل صريح.

افتراض: قد يضيف المشرفون ترخيصًا مفتوح المصدر بشكل صريح في تحديث لاحق.
