[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Eink Words GPT

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi-green)
![Display](https://img.shields.io/badge/display-Waveshare%20e--Paper-black)
![Status](https://img.shields.io/badge/status-active%20prototype-orange)
![Server](https://img.shields.io/badge/http-Tornado-0A7EA4)
![Storage](https://img.shields.io/badge/storage-SQLite-003B57)
![AI](https://img.shields.io/badge/OpenAI-optional-412991)

Проект для Raspberry Pi + Waveshare e-ink, который отображает динамически выбранную лексику с фонетикой и многоязычными синонимами. Система может получать слова из локальных наборов данных или OpenAI, рендерить их в карточки и отправлять результат на поддерживаемые e-paper панели. Также она предоставляет небольшой HTTP-сервис для запуска обновления слов и получения сгенерированных изображений.

## Обзор
`words_gpt` — это Python-система генерации и отображения словарных карточек для e-ink устройств.

Она объединяет:
- Источники слов из CSV/локальных наборов данных и опциональную генерацию через OpenAI.
- Обогащение данных (IPA-фонетика + многоязычные поля синонимов).
- Конвейеры рендеринга для аппаратного и виртуального вывода.
- HTTP-сервис на Tornado для удаленного запуска и получения изображений.

Текущая кодовая база сосредоточена вокруг `app.py`, `words_gpt.py`, `words_data.py`, `words_database.py` и `openai_request_json.py`.

## Ключевые возможности
- 🖼️ Конвейер рендеринга для e-ink с несколькими режимами контента (кандзи, японский, арабский, китайский, emoji).
- 🗃️ Локальная база слов (`words_phonetics.db`) со списками слов на основе CSV в `data/`.
- 🤖 Выбор слов и фонетическое обогащение через OpenAI со структурированным JSON-выводом.
- 🌐 HTTP-сервис для внешних триггеров и получения изображений.
- ⚡ Слой кэширования (`cache/`) для сокращения повторных вызовов OpenAI.

## Быстрый старт
| Цель | Команда |
|---|---|
| Запустить HTTP-сервер (порт `8082`) | `python app.py` |
| Запустить автономный рендерер (CSV) | `python words_gpt.py --use_csv` |
| Запуск с OpenAI + CSV | `python words_gpt.py --enable_openai --use_csv` |
| Режим emoji + упрощенный CJK | `python words_gpt.py --make_emoji --simplify` |
| Автонастройка Raspberry Pi | `bash scripts/setup_pi_wordscard.sh` |

## Демонстрации
<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## Возможности
- Поток аппаратного и виртуального рендеринга (`EPaperHardware`, `EPaperDisplay`) из `words_gpt.py`.
- Конвейер многоязычного обогащения в `words_data.py` (IPA, японские варианты, арабский, французский, китайские поля).
- Хранилище на SQLite с помощниками динамического обновления полей в `words_database.py`.
- Помощник структурированных JSON-запросов OpenAI с файловым кэшем в `openai_request_json.py`.
- Опциональные PWA-ресурсы в `pwa/` для легкой настройки фронтенда/предпросмотра.

## Структура проекта
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

Важные runtime-файлы:
- `app.py`: веб-сервер Tornado (порт по умолчанию `8082`) и цикл периодических обновлений.
- `words_gpt.py`: автономный цикл рендеринга и классы отображения.
- `words_data.py`: основная оркестрация получения/обогащения слов.
- `words_database.py`: помощники для SQLite-хранилища.
- `scripts/*.sh`: настройка Raspberry Pi, установка сервиса и скрипты жизненного цикла tmux.

## Предварительные требования
- Python `3.9+` (рекомендуется).
- Raspberry Pi в качестве целевого устройства (для аппаратного режима).
- Поддерживаемая панель Waveshare e-paper.
- Включенный SPI на Pi (`raspi-config`) и корректное подключение, зависящее от модели панели.

Используемые в проекте Python-пакеты:
- `openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.
- Скрипт установки дополнительно ставит: `json5`, `pandas`, `spidev`, `RPi.GPIO`, `gpiozero`, `lgpio`.

## Установка

### Вариант A: минимальная/ручная установка
Установите пакет драйверов Waveshare:
```bash
python setup.py install
```

Если используется список слов NLTK, скачайте его один раз:
```bash
python -m nltk.downloader words
```

### Вариант B: автоматическая настройка Raspberry Pi (рекомендуется на устройстве)
Из корня репозитория:
```bash
bash scripts/setup_pi_wordscard.sh
```

Этот скрипт:
- Устанавливает зависимости apt.
- Проверяет, что SPI включен.
- Создает и активирует виртуальное окружение `wordscard`.
- Устанавливает Python-зависимости для runtime.
- Устанавливает пакет Waveshare.
- Запускает `app.py` внутри сессии tmux.

## Конфигурация

### Поведение `.env`
Этот репозиторий загружает переменные окружения из `.env` во время импорта и **переопределяет** любые уже существующие значения shell. Это делает локальные переопределения детерминированными, даже если значения уже экспортированы в профилях shell.

Создайте или обновите `.env`:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### Проброс аргументов приложения
Скрипты systemd/tmux поддерживают:
```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### CLI-флаги (сервер и рендерер)
И `app.py`, и `words_gpt.py` поддерживают:
- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

## Использование

### Запуск HTTP-сервера
Запустите сервис (порт по умолчанию `8082`):
```bash
python app.py
```

Маршруты, обнаруженные в коде:
- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (из `words_card_temp/`)

Примечание по совместимости: в более ранней документации упоминался `GET /current_word`; в текущем `app.py` используется маршрут `GET /get_current_word`.

### Запуск автономного рендерера
Список на основе CSV:
```bash
python words_gpt.py --use_csv
```

Включить OpenAI:
```bash
python words_gpt.py --enable_openai --use_csv
```

Рендеринг emoji + упрощенный CJK:
```bash
python words_gpt.py --make_emoji --simplify
```

### Режим сервиса на Raspberry Pi
Установить unit сервиса:
```bash
bash scripts/install_wordscard_service.sh
```

Затем:
```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## Примеры

### Запустить следующее случайное слово
```bash
curl "http://127.0.0.1:8082/next_random_word"
```

### Получить текущий payload слова
```bash
curl "http://127.0.0.1:8082/get_current_word"
```

### Отправить явное слово
```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

### Быстрые аппаратные тесты
Используйте скрипт под конкретный дисплей:
```bash
python epd_7in3f_test.py
```

Или:
```bash
python epd_13in3k_test.py
```

Больше примеров находится в `waveshare/examples/`.

## Данные, кэш и логи
| Область | Путь(и) | Примечания |
|---|---|---|
| Списки слов | `data/` | Включает `data/words_list.csv` и тематические CSV-файлы |
| Постоянная БД | `words_phonetics.db` | Локальное хранилище фонетики/обогащения |
| Артефакты OpenAI/кэша | `cache/` | Снижает число повторных запросов |
| Логи | `logs/`, `logs-word-phonetics/` | Runtime- и update-логи |
| Сгенерированные карточки | `words_card_temp/` | Выходные изображения и источник для статической раздачи |

## Заметки по разработке
- Управление зависимостями сейчас основано на скриптах (`scripts/setup_pi_wordscard.sh`) + `setup.py`; `requirements.txt` или `pyproject.toml` пока нет.
- В репозитории есть несколько резервных/устаревших файлов (`words_data_*`, `words_gpt_old.py`); активный runtime-путь в основном `app.py` + `words_gpt.py` + `words_data.py` + `words_database.py`.
- `env_loader.py` всегда перезаписывает переменные окружения из `.env`, если ключи присутствуют.
- В режиме сервера выполняется периодический цикл обновления (примерно каждые 5 минут), который может внутренне вызывать endpoint обновления.

## Устранение неполадок
- `ModuleNotFoundError` или проблемы импорта:
  - Убедитесь, что виртуальное окружение активно и зависимости установлены.
  - На Pi повторно запустите `bash scripts/setup_pi_wordscard.sh`.
- Ошибки OpenAI (`401`, отсутствующая модель/ключ):
  - Проверьте `OPENAI_API_KEY` и опционально `OPENAI_MODEL` в `.env`.
  - Подтвердите сетевую доступность с устройства.
- Дисплей не обновляется:
  - Проверьте модель панели/подключение и запустите соответствующий тестовый скрипт (`epd_7in3f_test.py` или `epd_13in3k_test.py`).
  - Проверьте, что SPI включен (`sudo raspi-config nonint do_spi 0`).
  - На Pi 5 убедитесь в наличии compatibility symlink для `/dev/spidev0.0`, если устройство предоставляет `/dev/spidev10.0`.
- Проблемы установки OpenCC:
  - Используйте совместимый с дистрибутивом пакет (`libopencc1` или `libopencc2`), как в setup-скрипте.
- Несовпадение API-маршрутов:
  - Используйте `/get_current_word` для текущего payload, а не `/current_word`.

## Примечания по использованию OpenAI
Доступ к OpenAI опционален, но рекомендуется для генерации новых слов и фонетического обогащения. Помощник структурированного JSON в `openai_request_json.py` кэширует результаты в `cache/`, уменьшая количество повторных вызовов.

## Дорожная карта
- Добавить формальный манифест зависимостей (`requirements.txt` или `pyproject.toml`) для воспроизводимой установки.
- Расширить `i18n/` поддерживаемыми переводами README.
- Консолидировать устаревшие/резервные варианты скриптов после финализации канонического потока.
- Документировать PWA-процесс (`pwa/`) с примерами endpoint'ов и скриншотами.
- Добавить повторяемые автоматизированные тесты для данных и поведения маршрутов.

## Поддержка

### Что делает возможной ваша поддержка
- <b>Сохраняет инструменты открытыми</b>: хостинг, инференс, хранение данных и работа сообщества.  
- <b>Ускоряет разработку</b>: сфокусированное время на open-source для WordsCardEink и связанных учебных инструментов.  
- <b>Помогает прототипировать устройства</b>: итерации e-ink-железа и исследования layout'ов отображения.  
- <b>Дает доступ всем</b>: субсидируемые внедрения для студентов, авторов и сообществ.

### Пожертвовать

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

## Участие в разработке
См. `AGENTS.md` для правил участия, код-стиля и ожиданий по PR.

Рекомендуемый чеклист для вкладов:
- Указывайте модель панели и аппаратные примечания для изменений дисплея.
- Перечисляйте точные команды, использованные для проверки.
- Прикладывайте скриншоты/фото для изменений UI или вывода e-ink.
- Описывайте изменения в датасетах (файл + влияние на строки/столбцы).

## Лицензия
В корне репозитория сейчас отсутствует файл `LICENSE` (зафиксировано на этапе этой черновой ревизии). Пока файл лицензии не добавлен, права на повторное использование явно не предоставлены.

Предположение: мейнтейнеры могут добавить явную open-source лицензию в одном из следующих обновлений.
