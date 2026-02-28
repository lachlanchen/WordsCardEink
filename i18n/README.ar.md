[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

# Eink Words GPT

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi-green)
![Display](https://img.shields.io/badge/display-Waveshare%20e--Paper-black)
![Status](https://img.shields.io/badge/status-active%20prototype-orange)
![Server](https://img.shields.io/badge/http-Tornado-0A7EA4)
![Storage](https://img.shields.io/badge/storage-SQLite-003B57)
![AI](https://img.shields.io/badge/OpenAI-optional-412991)

مشروع Raspberry Pi + Waveshare e-ink يعرض مفردات يتم اختيارها ديناميكيًا مع النطق الصوتي ومرادفات متعددة اللغات. يستطيع النظام جلب الكلمات من مجموعات بيانات محلية أو من OpenAI، ثم تنسيقها ضمن تخطيط بصري ودفع النتيجة إلى شاشات e-paper المدعومة. كما يوفّر خدمة HTTP صغيرة لتشغيل تحديث الكلمات واسترجاع الصور المولّدة.

## نظرة عامة
`words_gpt` هو نظام مبني بلغة Python لتوليد بطاقات مفردات وعرضها على أجهزة e-ink.

ويجمع بين:
- جلب الكلمات من CSV/مجموعات بيانات محلية مع توليد اختياري عبر OpenAI.
- الإثراء (النطق IPA + حقول مرادفات متعددة اللغات).
- مسارات تصيير للمخرجات العتادية والافتراضية.
- خدمة Tornado HTTP للتشغيل عن بُعد واسترجاع الصور.

يرتكز الكود الحالي على `app.py` و`words_gpt.py` و`words_data.py` و`words_database.py` و`openai_request_json.py`.

## المزايا البارزة
- 🖼️ مسار تصيير e-ink مع أوضاع محتوى متعددة (kanji، اليابانية، العربية، الصينية، emoji).
- 🗃️ قاعدة كلمات محلية (`words_phonetics.db`) مع قوائم كلمات مدعومة بملفات CSV في `data/`.
- 🤖 اختيار كلمات وإثراء صوتي عبر OpenAI مع مخرجات JSON منظّمة.
- 🌐 خدمة HTTP للتشغيل الخارجي واسترجاع الصور.
- ⚡ طبقة تخزين مؤقت (`cache/`) لتقليل استدعاءات OpenAI المتكررة.

## البدء السريع
| الهدف | الأمر |
|---|---|
| تشغيل خادم HTTP (المنفذ `8082`) | `python app.py` |
| تشغيل المصيّر المستقل (CSV) | `python words_gpt.py --use_csv` |
| التشغيل مع OpenAI + CSV | `python words_gpt.py --enable_openai --use_csv` |
| وضع Emoji + simplified CJK | `python words_gpt.py --make_emoji --simplify` |
| إعداد Raspberry Pi تلقائيًا | `bash scripts/setup_pi_wordscard.sh` |

## عروض توضيحية
<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## الميزات
- تدفق تصيير عتادي + افتراضي (`EPaperHardware`, `EPaperDisplay`) من `words_gpt.py`.
- مسار إثراء متعدد اللغات في `words_data.py` (IPA، تنويعات يابانية، حقول عربية وفرنسية وصينية).
- تخزين دائم مبني على SQLite مع مساعدات تحديث ديناميكي للحقول في `words_database.py`.
- مساعد طلبات JSON منظّمة لـ OpenAI مع تخزين مؤقت للملفات في `openai_request_json.py`.
- أصول PWA اختيارية في `pwa/` لتدفقات إعداد/معاينة واجهة خفيفة.

## بنية المشروع
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
- `app.py`: خادم Tornado (المنفذ الافتراضي `8082`) وحلقة التحديث الدورية.
- `words_gpt.py`: حلقة التصيير المستقلة وفئات العرض.
- `words_data.py`: تنسيق جلب الكلمات وإثرائها.
- `words_database.py`: مساعدات تخزين SQLite.
- `scripts/*.sh`: إعداد Raspberry Pi، تثبيت الخدمة، وسكربتات دورة حياة tmux.

## المتطلبات المسبقة
- Python `3.9+` (موصى به).
- جهاز Raspberry Pi مستهدف (لوضع العتاد).
- شاشة Waveshare e-paper مدعومة.
- تفعيل SPI على Pi (`raspi-config`) بالإضافة إلى التوصيلات الخاصة بكل لوحة.

حزم Python المستخدمة في هذا المشروع تشمل:
- `openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.
- سكربت الإعداد يثبّت أيضًا: `json5`, `pandas`, `spidev`, `RPi.GPIO`, `gpiozero`, `lgpio`.

## التثبيت

### الخيار A: تثبيت يدوي/حد أدنى
ثبّت حزمة تعريفات Waveshare:
```bash
python setup.py install
```

إذا كنت تستخدم قائمة كلمات NLTK، نزّلها مرة واحدة:
```bash
python -m nltk.downloader words
```

### الخيار B: إعداد Raspberry Pi آليًا (موصى به على الجهاز)
من جذر المستودع:
```bash
bash scripts/setup_pi_wordscard.sh
```

هذا السكربت:
- يثبّت تبعيات apt.
- يتأكد من تفعيل SPI.
- ينشئ ويفعّل بيئة افتراضية `wordscard`.
- يثبّت تبعيات Python وقت التشغيل.
- يثبّت حزمة Waveshare.
- يشغّل `app.py` داخل جلسة tmux.

## الإعداد

### سلوك `.env`
هذا المستودع يحمّل متغيرات البيئة من `.env` وقت الاستيراد ويقوم **بالكتابة فوق** أي قيم موجودة مسبقًا في shell. هذا يجعل التجاوزات المحلية حتمية حتى لو كانت القيم مُصدّرة مسبقًا في ملفات profile.

أنشئ أو حدّث `.env`:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### تمرير معاملات التطبيق
سكربتات systemd/tmux تدعم:
```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### أعلام CLI (الخادم والمصيّر)
كل من `app.py` و`words_gpt.py` يدعمان:
- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

## الاستخدام

### تشغيل خادم HTTP
شغّل الخدمة (المنفذ الافتراضي `8082`):
```bash
python app.py
```

المسارات المرصودة في الكود:
- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (من `words_card_temp/`)

ملاحظة توافق: وثائق أقدم أشارت إلى `GET /current_word`؛ المسار الحالي في `app.py` هو `GET /get_current_word`.

### تشغيل المصيّر المستقل
قائمة مبنية على CSV:
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

### وضع الخدمة على Raspberry Pi
ثبّت وحدة الخدمة:
```bash
bash scripts/install_wordscard_service.sh
```

ثم:
```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## أمثلة

### تشغيل كلمة عشوائية تالية
```bash
curl "http://127.0.0.1:8082/next_random_word"
```

### قراءة حمولة الكلمة الحالية
```bash
curl "http://127.0.0.1:8082/get_current_word"
```

### إرسال كلمة محددة
```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

### اختبارات عتادية سريعة
استخدم السكربت المطابق للشاشة:
```bash
python epd_7in3f_test.py
```

أو:
```bash
python epd_13in3k_test.py
```

المزيد من الأمثلة موجود في `waveshare/examples/`.

## البيانات والتخزين المؤقت والسجلات
| المجال | المسار/المسارات | ملاحظات |
|---|---|---|
| قوائم الكلمات | `data/` | تتضمن `data/words_list.csv` وملفات CSV موضوعية |
| قاعدة بيانات دائمة | `words_phonetics.db` | مخزن محلي للنطق/الإثراء |
| مخرجات OpenAI/التخزين المؤقت | `cache/` | يقلّل الطلبات المتكررة |
| السجلات | `logs/`, `logs-word-phonetics/` | سجلات التشغيل والتحديث |
| البطاقات المولدة | `words_card_temp/` | مخرجات الصور ومصدر الخدمة الثابتة |

## ملاحظات تطوير
- إدارة التبعيات تعتمد على السكربتات أولًا (`scripts/setup_pi_wordscard.sh`) + `setup.py`؛ لا يوجد `requirements.txt` أو `pyproject.toml` بعد.
- توجد عدة ملفات احتياطية/قديمة (`words_data_*`, `words_gpt_old.py`)؛ مسار التشغيل الفعّال يعتمد أساسًا على `app.py` + `words_gpt.py` + `words_data.py` + `words_database.py`.
- `env_loader.py` يكتب دائمًا فوق متغيرات البيئة من `.env` عند وجود المفاتيح.
- وضع الخادم يشغّل تدفق تحديث دوري (كل ~5 دقائق) ويمكن أن يستدعي نقطة التحديث داخليًا.

## استكشاف الأخطاء وإصلاحها
- أخطاء `ModuleNotFoundError` أو الاستيراد:
  - تأكد من تفعيل البيئة الافتراضية وتثبيت التبعيات.
  - أعد تشغيل `bash scripts/setup_pi_wordscard.sh` على Pi.
- أخطاء OpenAI (`401`، نموذج/مفتاح مفقود):
  - تحقق من `OPENAI_API_KEY` و`OPENAI_MODEL` الاختياري في `.env`.
  - تأكد من اتصال الشبكة من الجهاز.
- الشاشة لا تتحدّث:
  - تحقق من نموذج الشاشة/التوصيلات وشغّل سكربت الاختبار المطابق (`epd_7in3f_test.py` أو `epd_13in3k_test.py`).
  - تأكد من تفعيل SPI (`sudo raspi-config nonint do_spi 0`).
  - على Pi 5، تأكد من وصلة التوافق `/dev/spidev0.0` إذا كان الجهاز يوفّر `/dev/spidev10.0`.
- مشكلات تثبيت OpenCC:
  - استخدم الحزمة المتوافقة مع التوزيعة (`libopencc1` أو `libopencc2`) كما في سكربت الإعداد.
- عدم تطابق مسارات API:
  - استخدم `/get_current_word` للحمولة الحالية، وليس `/current_word`.

## ملاحظات حول استخدام OpenAI
الوصول إلى OpenAI اختياري لكنه موصى به لتوليد كلمات جديدة وإثراء النطق. مساعد JSON المنظّم في `openai_request_json.py` يخزن النتائج تحت `cache/` لتقليل الاستدعاءات المتكررة.

## خارطة الطريق
- إضافة ملف تبعيات رسمي (`requirements.txt` أو `pyproject.toml`) لتثبيت قابل لإعادة الإنتاج.
- توسيع `i18n/` بنسخ README مترجمة تتم صيانتها.
- توحيد نسخ السكربتات القديمة/الاحتياطية بعد تثبيت المسار القياسي.
- توثيق تدفق PWA (`pwa/`) مع أمثلة نقاط نهاية ولقطات شاشة.
- إضافة اختبارات آلية قابلة للتكرار لسلوك البيانات ومسارات API.

## الدعم

### ما الذي يتيحه دعمك
- <b>إبقاء الأدوات متاحة</b>: الاستضافة، والاستدلال، وتخزين البيانات، وعمليات المجتمع.  
- <b>تسريع التطوير</b>: وقت مفتوح المصدر مركز على WordsCardEink وأدوات التعلم ذات الصلة.  
- <b>نمذجة الأجهزة</b>: تكرارات عتاد e-ink وأبحاث تخطيطات العرض.  
- <b>إتاحة أوسع للجميع</b>: نشرات مدعومة للطلاب والمبدعين ومجموعات المجتمع.

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

## المساهمة
راجع `AGENTS.md` لإرشادات المساهمة ونمط البرمجة وتوقعات PR.

قائمة تحقق مقترحة للمساهمة:
- أدرج نموذج الشاشة + ملاحظات العتاد عند تغييرات العرض.
- اذكر الأوامر الدقيقة التي شُغّلت للتحقق.
- أرفق لقطات شاشة/صور لتغييرات واجهة المستخدم أو مخرجات e-ink.
- صف تعديلات مجموعات البيانات (الملف + تأثير الصف/العمود).

## الترخيص
لا يوجد ملف `LICENSE` حاليًا في جذر المستودع (كما لوحظ في مسودة التمرير هذه). حتى تتم إضافة ملف ترخيص، لا يتم منح حقوق إعادة الاستخدام بشكل صريح.

افتراض: قد يضيف المشرفون ترخيصًا مفتوح المصدر صريحًا في تحديث لاحق.
