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
![Stars](https://img.shields.io/github/stars/lachlanchen/words_gpt?style=for-the-badge&logo=github&logoColor=white&label=Stars)
![Issues](https://img.shields.io/github/issues/lachlanchen/words_gpt?style=for-the-badge&logo=github&logoColor=white&label=Open%20Issues)
![Issues](https://img.shields.io/github/license/lachlanchen/words_gpt?style=for-the-badge&logo=github&logoColor=white&label=License)

> مشروع Raspberry Pi + Waveshare e-ink يصنع بطاقات مفردات ديناميكية بصوتيات IPA وتلميحات متعددة اللغات. يدعم سير عمل CSV محلي، وتغذية OpenAI اختيارية، والعرض على e-paper، والتحكم البعيد عبر HTTP.

ملخص سريع عن أوضاع التشغيل:
`app.py` (الخدمة) و `words_gpt.py` (المُولّد المستقل) يمكن تشغيلهما بشكل منفصل أو معًا.

| 🔎 نظرة سريعة | التفاصيل |
|---|---|
| **وقت التشغيل الأساسي** | `app.py` (خدمة HTTP) + `words_gpt.py` (حلقة التوليد) |
| **مسار البيانات** | ملفات CSV في `data/` + قاعدة SQLite `words_phonetics.db` |
| **مخرجات العرض** | لوحات e-paper من Waveshare وصور مخرجات افتراضية |
| **اعتماد الذكاء الاصطناعي** | اختياري (`--enable_openai`) مع كاش مخزّن في `cache/` |
| **إيقاع العمل الافتراضي** | الخادم على المنفذ `8082`، تحديث دوري يقارب كل 5 دقائق |

## 📚 جدول المحتويات
- [نظرة عامة](#overview)
- [أبرز الملامح](#highlights)
- [العروض التوضيحية](#demos)
- [هيكل المشروع](#project-structure)
- [المتطلبات المسبقة](#prerequisites)
- [التثبيت](#installation)
- [الإعداد](#configuration)
- [الاستخدام](#usage)
- [الأمثلة](#examples)
- [البيانات، الكاش، والسجلات](#data-cache-and-logs)
- [ملاحظات التطوير](#development-notes)
- [استكشاف الأخطاء](#troubleshooting)
- [خريطة الطريق](#roadmap)
- [الدعم](#support)
- [المساهمة](#contributing)
- [الترخيص](#license)

---

<a id="overview"></a>
## نظرة عامة

`words_gpt` هي بنية توليد بطاقات مفردات بلغة Python لشاشات e-ink. تجمع بين تنسيق البيانات، إثراء الصوتيات، والعرض ضمن نمطي تشغيل:

- خدمة Tornado طويلة الأمد (`app.py`) للتحكم عن بعد وتقديم الصور.
- مولّد مستقل (`words_gpt.py`) يمكن تشغيله في أوضاع المسح الدوري أو الحلقة أو العرض المباشر.

الوحدات الأساسية:

- `words_data.py` / `words_data_utils.py` لتنسيق سير كلمات الإثراء.
- `words_data_with_legacy.py` و`words_data_without_legacy.py` و`words_data_workable*.py` للعمل بتدفقات بديلة والمحافظة على التوافق.
- `words_database.py` للتعامل مع SQLite.
- `openai_request_json.py` لطلبات OpenAI المنظمة مع سلوك كاش وإعادة المحاولة على القرص.
- `env_loader.py` لتحميل متغيرات البيئة بشكل حتمي.
- `words_update.py` لصيانة قاعدة البيانات وعمليات إعادة الفحص.
- `app.py` و`words_gpt.py` لدورة حياة الخدمة/العرض.
- `pwa/` لأدوات معاينة المتصفح الخفيفة وضبطها.

<a id="highlights"></a>
## أبرز الملامح

- مصدر كلمات محلي وOpenAI في سير عمل واحد.
- عرض مبسط للصيني عبر `--simplify` بشكل اختياري.
- مسارات HTTP للتحكم عن بعد (`/next_random_word`, `/display_word`, `/get_current_word`, `/get_current_word_page`, `/get_words_card`).
- الكاش ودوام البيانات لتقليل استدعاءات AI المتكررة.
- دعم التثبيت والتشغيل على العتاد عبر شجرة `waveshare/` المضمّنة.

<a id="demos"></a>
## العروض التوضيحية

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

<a id="project-structure"></a>
## هيكل المشروع

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

الملفات المهمة أثناء التشغيل:

- `app.py`: تطبيق Tornado على المنفذ `8082` + تشغيل دوري لـ `next_random_word`.
- `words_gpt.py`: مولّد مستقل وتجريدات العرض (`EPaperHardware`, `EPaperDisplay`).
- `words_data.py`: سير جلب/إثراء الكلمات وأدوات الاختيار.
- `words_database.py`: أدوات SQLite للبيانات التعريفية المخزنة وعمليات كاش الكلمات.
- `scripts/*.sh`: تثبيت وخدمة دورة حياة التطبيق ومساعدات إقلاع Raspberry Pi.
- `words_update.py`: أداة تحديث/إعادة فحص مجمّعة للحفاظ على جودة البيانات.

<a id="prerequisites"></a>
## المتطلبات المسبقة

- Python `3.9+` (موصى به)
- Raspberry Pi (مطلوب لوضع العتاد)
- لوحة e-paper مدعومة من Waveshare (مثل عائلة 7.3F أو 13K)
- تفعيل SPI (`raspi-config`) وتوصيلات صحيحة وتغذية طاقة مستقرة
- توفر حزمة NLTK عند استخدام مصادر كلمات NLTK

الاعتمادات الشائعة ضمن الكود ومسارات التشغيل:
`openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.

<a id="installation"></a>
## التثبيت

### الخيار 1 — تثبيت يدوي/أساسي (سطح المكتب أو Pi)

من جذر المستودع:

```bash
python setup.py install
```

إذا لزم الأمر:

```bash
python -m nltk.downloader words
```

### الخيار 2 — إعداد آلي على Raspberry Pi (موصى به على الجهاز)

من جذر المستودع:

```bash
bash scripts/setup_pi_wordscard.sh
```

ينفّذ ذلك:

- تثبيتات مخصصة لـ Pi
- فحوصات تمكين SPI
- إعداد البيئة الافتراضية `wordscard`
- تثبيت حزم Python/runtime
- تثبيت حزمة Waveshare
- تشغيل `app` عبر `tmux`

### الخيار 3 — تثبيت خدمة systemd

لتسجيل دورة حياة التطبيق في `systemd`:

```bash
bash scripts/install_wordscard_service.sh
```

ثم:

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

<a id="configuration"></a>
## الإعداد

### متغيرات البيئة (`.env`)

`env_loader` يقرأ مفاتيح البيئة ويطبّقها أثناء بدء العملية. التوثيق الشائع في التنفيذ الحالي عادةً يشمل:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

الافتراض: احتفظ بالمفاتيح السرية في إعداد البيئة المحلي ولا ترفعها إلى التحكم بالإصدار.

### أعلام زمن التشغيل (تُستخدم من `app.py` و`words_gpt.py`)

| علامة CLI | الغرض |
| --- | --- |
| `--enable_openai` | تفعيل وضع إغناء OpenAI الاختياري |
| `--make_emoji` | عرض بطاقات تركز على الإيموجي |
| `--ignore_list` | تخطي الكلمات المدرجة في قوائم التجاهل |
| `--simplify` | إنتاج مخرجات CJK مبسطة |
| `--use_csv` | قراءة الكلمات من مجموعات بيانات CSV |
| `--complete_csv` | استخدام وضع مصدر CSV الكامل |
| `--filename <csv_file>` | تحديد ملف CSV معيّن

`APP_ARGS` يمكن تمريره عبر سكربتات الإقلاع. مثال:

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### سلوك التوجيه في وضع الخدمة

مسارات API الملحوظة:

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (يتم تقديمه من `words_card_temp/`)

ملاحظة توافق: قد تشير بعض المراجع القديمة إلى `GET /current_word`، بينما المسار الحالي هو `GET /get_current_word`.

### ملاحظات استخدام OpenAI

ميزات OpenAI اختيارية وتُدار عبر أعلام CLI أو البيئة. تساعد الطلبات المخزنة في `cache/` على تحسين إمكانية إعادة الإنتاج والتحكم بمعدل الطلبات. لاختبارات تعمل دون اتصال مبدئيًا، ابدأ بـ CSV (`--use_csv`) ثم فعّل OpenAI بشكل انتقائي.

<a id="usage"></a>
## الاستخدام

### تشغيل خادم HTTP

```bash
python app.py
```

تحافظ العملية على آخر صورة داخل `words_card_temp/` وتعرض المسارات التي تُستخدم بواسطة أدوات الواجهة الأمامية أو السكربتات.

### تشغيل المولّد مباشرة

وضع CSV:

```bash
python words_gpt.py --use_csv
```

وضع OpenAI:

```bash
python words_gpt.py --enable_openai --use_csv
```

وضع الإيموجي + CJK المبسطة:

```bash
python words_gpt.py --make_emoji --simplify
```

### التشغيل على عتاد Pi

- البدء عبر سكربت tmux:

```bash
bash scripts/start_wordscard.sh
```

- الإيقاف عبر سكربت tmux:

```bash
bash scripts/stop_wordscard.sh
```

<a id="examples"></a>
## الأمثلة

الحصول على تعريف البطاقة العشوائية التالية:

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

جلب الكلمة المخزّنة حاليًا:

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

طلب payload صورة الصفحة المولّدة:

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

إرسال كلمة صريحة:

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

تفعيل عرض البطاقة عبر endpoint بصيغة form:

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

<a id="data-cache-and-logs"></a>
## البيانات، الكاش، والسجلات

الملحقات الشائعة التي يستخدمها التطبيق:

- `data/`: مجموعات CSV منظَّمة
- `words_phonetics.db`: مخزن/كاش SQLite
- `cache/`: كاش طلبات ونتائج OpenAI
- `word_phonetics_processed.csv`: مجموعة بيانات معالجة مشتقة
- `logs/`, `logs-word-phonetics/`: سجلات التشغيل
- `words_card_temp/`: البطاقات المولّدة والمخرجات المؤقتة
- `pic/` و`figs/`: صور مرجعية وبانرات

<a id="development-notes"></a>
## ملاحظات التطوير

- توجد وحدات وملفات تراثية (مثل `words_gpt_old.py`, `lib.old`)، فعاملها كمراجع ما لم يكن الهدف الهجرة أو التوافق مقصودًا.
- `words_update.py` يحتوي مساعدات تحديث/إعادة فحص مجمعة مفيدة للصيانة النوعية لقاعدة البيانات.
- يتحقق التحقق من العتاد عبر `epd_*_test.py` وسكربتات demo داخل `waveshare/examples/*`.
- لا يوجد `requirements.txt` أو ملف lock في جذر المشروع؛ تعتمد الاعتمادات على سكربتات التثبيت وتثبيت مباشر.
- لا يوجد نظام اختبارات آلي مضبوط في هذا المستودع.

<a id="troubleshooting"></a>
## استكشاف الأخطاء

- `ImportError` من وحدات Raspberry Pi GPIO/SPI:
  - استخدم تدفق إعداد Pi (`scripts/setup_pi_wordscard.sh`) أو ثبّت الاعتمادات صراحةً على هدف متوافق.
- أخطاء `403/404` من نقاط الصورة/الملفات الثابتة:
  - تأكد من طريقة استخدام المسار (`/get_current_word*`) وأن `words_card_temp/` قابلة للكتابة.
- payload فارغ/غير صالح من وضع OpenAI:
  - تأكد من تحميل `OPENAI_API_KEY` (وقيم org/model) ثم راجع `cache/` والسجلات.
- عرض سيّئ أو قص للنص:
  - تحقق من مسارات الخطوط، وثوابت دقة الشاشة، وإعدادات الوضع المختار داخل `words_gpt.py`.
- API تعيد بيانات قديمة:
  - استدعِ `POST /next_random_word` يدويًا وراجع فاصل الاتصال التلقائي في `app.py`.
- يبدو عرض العتاد متوقّفًا:
  - افحص جلسة tmux وسجلات systemd عبر `journalctl -u wordscard`.
- نقص في مجموعة البيانات/معاجم الكلمات:
  - تحقق من ملفات CSV داخل `data/` وشغّل مهام الصيانة في `words_update.py`.

<a id="roadmap"></a>
## خريطة الطريق

- إضافة `requirements.txt` مبسط / ملف تثبيت قابل لإعادة الإنتاج.
- توضيح أوضاع التشغيل وتوثيق أقوى لأعلام CLI ضمن `--help`.
- توسيع توثيق وضعيات العرض (`japanese_synonym`, `arabic_synonym`, `film`, وسائر الـ workflows).
- توحيد معالجة الأخطاء ومخططات استجابات API الموجهة للمستخدم النهائي.
- إضافة اختبارات smoke-light بسيطة للتحقق في CI بدون عتاد.

<a id="support"></a>
## المساهمة

المساهمات مرحّب بها. سلوك مقترح:

1. أبقِ التعديلات ضمن مجال سلوكي واحد (العرض، البيانات، API، السكربتات).
2. حدّث استخدام الأوامر والتوثيق عند أي تغييرات مرئية للمستخدم.
3. حافظ على توافق أعلام CLI وEndpoints قدر الإمكان.
4. إذا تغيّرت سكربتات العتاد، وثّق الطراز المجرّب والأوامر الفعلية مع النتائج.

<a id="license"></a>
## الترخيص

لا يوجد ملف `LICENSE` في جذر المستودع الحالي. وعليه فالرخصة الفعلية غير مذكورة ضمن الشجرة في هذا المسودة. أضف ملفًا إذا رغبت في شروط صريحة لإعادة الاستخدام وإعادة التوزيع.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
