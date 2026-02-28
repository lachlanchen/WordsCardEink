[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 🖨️ Eink Words GPT

**لغة هذه المسودة:** العربية

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=for-the-badge&logo=githubactions&logoColor=white)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=for-the-badge&logo=python&logoColor=white)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=for-the-badge&logo=openai&logoColor=white)

مشروع يعمل على Raspberry Pi وشاشة Waveshare e-ink لعرض بطاقات مفردات يتم اختيارها ديناميكيًا مع علم الأصوات (IPA) وإشارات لغوية إضافية. يدعم سير عمل CSV محليًا، إثراءً اختياريًا عبر OpenAI، عرضًا على e-paper، وتحكمًا عن بُعد عبر HTTP.

| 🔎 نظرة سريعة | التفاصيل |
|---|---|
| وقت التشغيل الأساسي | `app.py` (خدمة HTTP) + `words_gpt.py` (حلقة التوليد) |
| مسار البيانات | مجموعات CSV في `data/` + قاعدة SQLite `words_phonetics.db` |
| مخرجات العرض | لوحات Waveshare e-paper ومخرجات صور افتراضية |
| اعتماد الذكاء الاصطناعي | اختياري (`--enable_openai`) مع ذاكرة مؤقتة في `cache/` |
| إعدادات الحلقة الأساسية | الخادم على `8082`، تحديث دوري يقارب كل 5 دقائق |

## 📚 جدول المحتويات
- [نظرة عامة](#overview)
- [أبرز الملامح](#highlights)
- [الاستعراضات](#demos)
- [بنية المشروع](#project-structure)
- [المتطلبات المسبقة](#prerequisites)
- [التثبيت](#installation)
- [الإعداد](#configuration)
- [الاستخدام](#usage)
- [الأمثلة](#examples)
- [البيانات والذاكرة المؤقتة والسجلات](#data-cache-and-logs)
- [ملاحظات التطوير](#development-notes)
- [استكشاف الأخطاء وإصلاحها](#troubleshooting)
- [خريطة الطريق](#roadmap)
- [الدعم](#support)
- [المساهمة](#contributing)
- [الترخيص](#license)

---

## Overview

`words_gpt` هو حزمة Python لتوليد بطاقات مفردات مخصصة لشاشات e-ink. يجمع بين تنظيم البيانات، وإثراء النطق، وتنسيق التصدير ضمن نمط تشغيلين أساسيين:

- خدمة Tornado طويلة التشغيل (`app.py`) للتحكم البعيد وخدمة الصور
- أداة مستقلة (`words_gpt.py`) يمكن تشغيلها في أوضاع polling أو loop أو عرض مباشر

الوحدات الرئيسية:

- `words_data.py` / `words_data_utils.py` لإدارة الكلمات ومسارات الإثراء
- `words_database.py` للتعامل مع SQLite
- `openai_request_json.py` لطلبات OpenAI المنظمة مع التخزين المؤقت على القرص
- `env_loader.py` لتحميل المتغيرات البيئية بطريقة متوقعة
- `words_update.py` لصيانة قاعدة البيانات وسير عمل إعادة الفحص
- `app.py` و`words_gpt.py` لدورة تشغيل الخدمة والعرض

## Highlights

- خط أنابيب عرض e-ink مع أوضاع متعددة للغة والمحتوى:
  - نسخ يابانية، وضع كانجي، العربية، الصينية، ووضع emoji
- مصدر كلمات محلي وOpenAI ضمن نفس سير العمل
- إخراج صيني مبسط اختياري في مسار الـ renderer
- نقاط نهاية خدمة للتفاعل المباشر (`/next_random_word`, `/display_word`، وغيرها)
- كاش وتخزين دائم يقلل استدعاءات الشبكة المكررة
- أصول PWA اختيارية داخل `pwa/` لتدفقات معاينة/ضبط خفيفة

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

ملفات التشغيل المهمة:

- `app.py`: تطبيق Tornado يعمل على المنفذ `8082` مع منبه دوري `next_random_word`.
- `words_gpt.py`: أداة عرض مستقلة وطبقات عرض (`EPaperHardware`, `EPaperDisplay`).
- `words_data.py`: سير عمل متقدم لجلب/إثراء الكلمات وأدوات مساعدة.
- `words_database.py`: أدوات SQLite للبيانات التعريفية المخزنة وعمليات cache للكلمات.
- `scripts/*.sh`: تسجيل/إدارة دورة حياة الخدمة ومهام bootstrap لجهاز Raspberry Pi.

## Prerequisites

- Python `3.9+` (موصى به)
- Raspberry Pi (مطلوب لوضع العتاد)
- لوحة Waveshare e-paper مدعومة (مثل عائلة 7.3F / 13K)
- تفعيل SPI (`raspi-config`) وتوصيل كهربائي صحيح وتشغيل مستقر للطاقة
- توفر NLTK corpus عند استخدام مصادر الكلمات عبر `nltk`

الاعتماديات الشائعة في الكود:
`openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.

## Installation

### Option 1 — تثبيت يدوي/مبسّط (سطح المكتب أو Pi)

من جذر المستودع:

```bash
python setup.py install
```

إذا لزم الأمر:

```bash
python -m nltk.downloader words
```

### Option 2 — إعداد Raspberry Pi تلقائي (موصى به على الجهاز)

من جذر المستودع:

```bash
bash scripts/setup_pi_wordscard.sh
```

هذا السكربت ينفّذ:

- تبعيات مخصوصة بـ Pi
- تفعيل SPI
- إعداد البيئة الافتراضية `wordscard`
- تثبيت حزم Python/runtime
- تثبيت حزمة Waveshare
- تشغيل التطبيق داخل `tmux`

### Option 3 — تثبيت كخدمة

لتسجيل دورة حياة التطبيق عبر `systemd`:

```bash
bash scripts/install_wordscard_service.sh
```

ثم:

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## Configuration

### متغيرات البيئة (`.env`)

المستودع يستخدم تحميل `.env` الذي يتجاوز متغيرات shell الموجودة حاليًا. استخدمه عن قصد:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### أعلام التشغيل (تستخدمها `app.py` و`words_gpt.py`)

- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

سكريبتات Pi تدعم تمرير المعاملات عبر `APP_ARGS` (مثال):

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### سلوك التوجيه في وضع app

المسارات المرصودة في الكود الحالي:

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (يُخدم من `words_card_temp/`)

ملاحظة توافق: الوثائق السابقة أشارت إلى `GET /current_word`، المسار الحالي هو `GET /get_current_word`.

### ملاحظات حول استخدام OpenAI

ميزات OpenAI اختيارية وتُضبط عبر وسوم CLI/بيئة. يتم حفظ payloads الـ API في كاش مفيد لإعادة الإنتاج والتحكم بمعدلات الطلب. في البيئات المقيدة، شغّل وضع CSV أولًا (`--use_csv`) وفعّل OpenAI عند الحاجة عبر (`--enable_openai`).

## Usage

### تشغيل خادم HTTP

```bash
python app.py
```

تحافظ العملية على صورة داخل `words_card_temp/` وتعرض نقاط النهاية HTTP التي تستخدمها أدوات الواجهة الأمامية أو سكربتات بسيطة.

### تشغيل المولّد مباشرة

وضع CSV:

```bash
python words_gpt.py --use_csv
```

وضع OpenAI:

```bash
python words_gpt.py --enable_openai --use_csv
```

Emoji + CJK مبسط:

```bash
python words_gpt.py --make_emoji --simplify
```

### تشغيل على عتاد Pi

- ابدأ عبر سكربت tmux:

```bash
bash scripts/start_wordscard.sh
```

- أوقف عبر سكربت tmux:

```bash
bash scripts/stop_wordscard.sh
```

## Examples

الحصول على بيانات البطاقة العشوائية التالية:

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

جلب الكلمة الحالية المخزّنة:

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

طلب payload صفحة معروضة:

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

إرسال كلمة صريحة:

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

إطلاق عرض البطاقة عبر endpoint بنمط form:

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

## Data, cache, and logs

الملفات النموذجية المستخدمة من التطبيق:

- `data/`: مجموعات CSV منظَّمة
- `words_phonetics.db`: قاعدة SQLite للكاش/مصدر البيانات
- `cache/`: كاش طلبات/نتائج OpenAI
- `word_phonetics_processed.csv`: ملف بيانات معالج/مشتق
- `logs/`, `logs-word-phonetics/`: سجلات التشغيل
- `words_card_temp/`: بطاقات مولّدة ومخرجات مؤقتة

## Development notes

- توجد ملفات قديمة/احتياطية مثل `words_gpt_old.py`, `lib.old`، فاعتبرها مراجع ما لم تكن مهتمًا بالهجرة أو الحفاظ على التوافق تحديدًا.
- يحتوي `words_update.py` على أدوات تحديث/إعادة فحص مجمّعة مفيدة لتنقية جودة قاعدة البيانات.
- التحقق من العتاد يتم عبر `epd_*_test.py` وعينات `waveshare/examples/*`.
- لا يوجد `requirements.txt` أو ملف lock في جذر المستودع؛ تثبيت الاعتماديات يتم عبر سكربت الإعداد أو التثبيت المباشر.
- لا يوجد إطار اختبار آلي مهيأ في هذا المستودع.

## Troubleshooting

- `ImportError` من وحدات Raspberry Pi GPIO/SPI:
  - ثبّت عبر مسار Pi (`setup_pi_wordscard.sh`)، أو تحقق من `python setup.py install` على الهدف المتوافق.
- أخطاء `403/404` من نقاط الصورة/الـ static:
  - تأكد من استخدام نقطة نهاية `/get_current_word*` وأن مجلد `words_card_temp/` قابل للكتابة.
- payload فارغ أو غير صالح في وضع OpenAI:
  - تأكد من تحميل `OPENAI_API_KEY` وموعدات النموذج/المنظمة، وفحص `cache/` والسجلات.
- عرض غير جيد/قص للنص:
  - تحقق من مسار الخط والـ panel resolution داخل مسار التوليد في `words_gpt.py`.
- API تعيد بيانات قديمة:
  - نفّذ `POST /next_random_word` يدويًا وراجع الفاصل الدوري داخل `app.py`.
- تحديث العتاد يبدو متوقفًا:
  - افحص جلسة tmux وسجلات systemd (`journalctl -u wordscard`).
- مجموعة بيانات أو إدخالات قاموس ناقصة:
  - تحقق من ملفات CSV داخل `data/` وشغّل `words_update.py` لتحديث/تنظيف.

## Roadmap

- إضافة `requirements.txt` أساسي / ملف تثبيت قابل لإعادة الإنتاج.
- توضيح أوضاع التشغيل وواجهة `--help` صريحة في CLI.
- توسيع وثائق مخطط العرض لكل وضع محتوى (`japanese_synonym`, `arabic_synonym`, `film`, وغيرها).
- توحيد معالجة الأخطاء ونُسق استجابة API الموجه للمستخدم.
- إضافة سكربتات smoke-test خفيفة للتحقق في CI غير المعتمد على العتاد.

## Support

| خيار الدعم | الرابط | الغرض |
|---|---|---|
| GitHub Sponsors | https://github.com/sponsors/lachlanchen | دعم المشروع المستمر أو لمرة واحدة |
| Lazying Art | https://lazying.art | العلامة التجارية والموارد المرتبطة |
| Chat | https://chat.lazying.art | نقاش ودعم |
| Only Ideas | https://onlyideas.art | أبحاث إبداعية ومشاريع جانبية |

## Contributing

المساهمات مرحب بها. مقترح سير العمل:

1. ابقِ التغييرات ضمن نطاق وظيفة واحدة (عرض، بيانات، API، سكربتات).
2. حدّث أوامر الاستخدام والوثائق لأي تغيير يطال واجهة المستخدم.
3. حافظ على توافق وسوم CLI ومسارات endpoints قدر الإمكان.
4. إذا تغيّرت سكربتات العتاد، وثّق جهاز الاختبار/الطراز والأوامر المنفّذة بدقة.

## License

لا يوجد ملف `LICENSE` حاليًا في جذر المستودع. وبالتالي فالرخصة المفعلة غير معرفة صراحة داخل الشجرة. أضف ملف ترخيص إذا أردت شروط إعادة استخدام/توزيع واضحة.
