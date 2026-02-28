[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 🖨️ Eink Words GPT


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

> Проект на базе Raspberry Pi и Waveshare e-paper, который рендерит динамические карточки слов с транскрипцией IPA и многоязычными подсказками. Он поддерживает локальные CSV‑процессы, опциональное обогащение AI, вывод на электронную бумагу и удалённое управление через HTTP.

Основные режимы работы:
`app.py` (сервис) и `words_gpt.py` (автономный рендерер) можно запускать независимо или вместе.

| 🔎 Кратко | Детали |
|---|---|
| **Основной runtime** | `app.py` (HTTP-сервис) + `words_gpt.py` (цикл рендеринга) |
| **Источник данных** | CSV-наборы в `data/` + SQLite-база `words_phonetics.db` |
| **Куда выводится** | Панели e-paper Waveshare и виртуальные изображения |
| **Зависимость от AI** | Опционально (`--enable_openai`) с кэшем запросов в `cache/` |
| **Период обновления по умолчанию** | Сервер на `8082`, периодическое обновление примерно каждые 5 минут |

## 📚 Оглавление
- [Обзор](#обзор)
- [Ключевые возможности](#ключевые-возможности)
- [Демо](#демо)
- [Структура проекта](#структура-проекта)
- [Требования](#требования)
- [Установка](#установка)
- [Настройка](#настройка)
- [Использование](#использование)
- [Примеры](#примеры)
- [Данные, кэш и логи](#данные-кэш-и-логи)
- [Заметки по разработке](#заметки-по-разработке)
- [Устранение неполадок](#устранение-неполадок)
- [Дорожная карта](#дорожная-карта)
- [Поддержка](#support)
- [Содействие проекту](#содействие-проекту)
- [Лицензия](#лицензия)

---

## Обзор<a id="обзор"></a>

`words_gpt` — это стек генерации вокабулярных карточек на Python для дисплеев e-ink. Он объединяет оркестрацию данных, фонетическое обогащение и рендеринг в двух рабочих режимах:

- Долговременный сервис Tornado (`app.py`) для удалённого управления и выдачи изображений.
- Автономный рендерер (`words_gpt.py`), который может работать в режимах polling, loop и прямого рендеринга.

Базовые модули:

- `words_data.py` / `words_data_utils.py` для работы со словами и расширением.
- `words_data_with_legacy.py`, `words_data_without_legacy.py`, `words_data_workable*.py` для альтернативных рабочих потоков и совместимости.
- `words_database.py` для взаимодействия с SQLite.
- `openai_request_json.py` для структурированных запросов OpenAI с дисковым кэшем и политикой повторных попыток.
- `env_loader.py` для детерминированной загрузки окружения.
- `words_update.py` для обслуживания БД и повторной проверки данных.
- `app.py` и `words_gpt.py` для жизненного цикла сервиса и рендера.
- `pwa/` для лёгкой браузерной предпросмотра/настройки.

## Ключевые возможности<a id="ключевые-возможности"></a>

- Конвейер рендера на e-ink с многоязычными режимами: японские варианты, режим кандзи, арабский, китайский и режим эмодзи.
- Локальное формирование слов и выбор через OpenAI в одном процессе.
- Опциональный рендер для упрощённого китайского (`--simplify`).
- HTTP-эндпоинты для удалённого управления (`/next_random_word`, `/display_word`, `/get_current_word`, `/get_current_word_page`, `/get_words_card`).
- Кэширование и персистентность для уменьшения повторных AI-вызовов.
- Пакетирование и аппаратные примеры через поставляемую структуру драйверов `waveshare/`.

## Демо<a id="демо"></a>

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## Структура проекта<a id="структура-проекта"></a>

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

Ключевые исполняемые файлы:

- `app.py`: Tornado-приложение на порту `8082` + периодический триггер `next_random_word`.
- `words_gpt.py`: автономный рендерер и абстракции вывода (`EPaperHardware`, `EPaperDisplay`).
- `words_data.py`: логика получения, обогащения и выборки слов.
- `words_database.py`: SQLite-утилиты для сохранённых метаданных и операций кэша слов.
- `scripts/*.sh`: скрипты установки, управления жизненным циклом сервиса и загрузки Pi.
- `words_update.py`: пакетная помощь по обновлению/перепроверке БД для контроля качества данных.

## Требования<a id="требования"></a>

- Python `3.9+` (рекомендуется)
- Raspberry Pi (обязательно для аппаратного режима)
- Поддерживаемая e-paper панель Waveshare (например семейства 7.3F или 13K)
- Включённый SPI (`raspi-config`), корректная проводка и стабильное питание
- NLTK-корпус при использовании источников слов из NLTK

Основные зависимости, используемые в коде:
`openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.

## Установка<a id="установка"></a>

### Вариант 1 — Минимальная/ручная установка (desktop или Pi)

Из корня репозитория:

```bash
python setup.py install
```

При необходимости:

```bash
python -m nltk.downloader words
```

### Вариант 2 — Автоматическая установка на Raspberry Pi (рекомендуется на устройстве)

Из корня репозитория:

```bash
bash scripts/setup_pi_wordscard.sh
```

Этот сценарий выполняет:

- Установку зависимостей под Pi
- Проверку и помощь включения SPI
- Настройку виртуального окружения `wordscard`
- Установку Python-пакетов и runtime-зависимостей
- Установку пакета Waveshare
- Запуск процесса в `tmux`

### Вариант 3 — Установка как сервис systemd

Для регистрации жизненного цикла приложения в `systemd`:

```bash
bash scripts/install_wordscard_service.sh
```

Затем:

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## Настройка<a id="настройка"></a>

### Переменные окружения (`.env`)

`env_loader` читает переменные окружения и применяет их при старте процесса. Текущая документация и runtime обычно используют:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

Рекомендация: хранить секреты только в локальной конфигурации окружения и не коммитить их в систему контроля версий.

### Флаги CLI (используются `app.py` и `words_gpt.py`)

| Флаг CLI | Назначение |
| --- | --- |
| `--enable_openai` | Включает необязательный режим обогащения OpenAI |
| `--make_emoji` | Рендерит карточки с акцентом на эмодзи |
| `--ignore_list` | Пропускает слова из заданных списков исключений |
| `--simplify` | Выводит упрощённый CJK |
| `--use_csv` | Загружает слова из CSV-наборов |
| `--complete_csv` | Использует режим полного CSV-источника |
| `--filename <csv_file>` | Указывает конкретный входной CSV-файл |

`APP_ARGS` можно передавать через скрипты запуска. Пример:

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### Поведение роутинга в режиме app

Текущие API-маршруты:

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (отдаётся из `words_card_temp/`)

Важно по совместимости: в старых описаниях может встречаться `GET /current_word`; текущий маршрут — `GET /get_current_word`.

### Примечания по OpenAI

Функции OpenAI опциональны и управляются флагами CLI/окружения. Кэшированные запросы в `cache/` повышают воспроизводимость и помогают с лимитами rate limit. Для детерминированного офлайн-режима лучше начать с CSV-режима (`--use_csv`) и включать OpenAI выборочно.

## Использование<a id="использование"></a>

### Запустить HTTP-сервер

```bash
python app.py
```

Процесс держит последнюю карточку в `words_card_temp/` и открывает эндпоинты, используемые фронтальными инструментами или скриптами.

### Запустить рендерер напрямую

CSV-режим:

```bash
python words_gpt.py --use_csv
```

Режим OpenAI:

```bash
python words_gpt.py --enable_openai --use_csv
```

Эмодзи + упрощённый CJK:

```bash
python words_gpt.py --make_emoji --simplify
```

### Запуск на оборудовании Pi

- Запуск через tmux:

```bash
bash scripts/start_wordscard.sh
```

- Остановка через tmux:

```bash
bash scripts/stop_wordscard.sh
```

## Примеры<a id="примеры"></a>

Получить следующий случайный метаданные карточки:

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

Получить текущее сохранённое слово:

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

Запросить payload отрендеренной страницы:

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

Передать конкретное слово:

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

Запустить отрисовку карточки через form-эндпоинт:

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

## Данные, кэш и логи<a id="данные-кэш-и-логи"></a>

Типичные артефакты, используемые приложением:

- `data/`: подготовленные CSV-наборы
- `words_phonetics.db`: кэш/исходная база SQLite
- `cache/`: кэш запросов/ответов OpenAI
- `word_phonetics_processed.csv`: обработанный/производный датасет
- `logs/`, `logs-word-phonetics/`: логи выполнения
- `words_card_temp/`: сгенерированные карточки и временные файлы
- `pic/` и `figs/`: эталонные изображения и баннеры

## Заметки по разработке<a id="заметки-по-разработке"></a>

- Существуют legacy/резервные модули и артефакты (например `words_gpt_old.py`, `lib.old`), поэтому относитесь к ним как к ссылкам, пока перенос или совместимость не запрошены намеренно.
- `words_update.py` содержит инструменты пакетного обновления/повторной проверки, полезные для поддержания качества БД.
- Проверка работы с железом выполняется скриптами `epd_*_test.py` и демо из `waveshare/examples/*`.
- В корне репозитория нет `requirements.txt` или lockfile; зависимости управляются установочными скриптами и прямой установкой.
- В репозитории не настроен автоматический test suite.

## Устранение неполадок<a id="устранение-неполадок"></a>

- `ImportError` из модулей Raspberry Pi GPIO/SPI:
  - Используйте путь настройки Pi (`scripts/setup_pi_wordscard.sh`) или установите зависимости вручную на совместимом целевом устройстве.
- `403/404` от эндпоинтов изображений/статических файлов:
  - Проверьте используемые маршруты (`/get_current_word*`) и доступность записи в `words_card_temp/`.
- Пустые/некорректные payload от OpenAI-режима:
  - Проверьте, что `OPENAI_API_KEY` (и org/model) загружены, затем изучите `cache/` и логи.
- Плохой рендер или обрезка текста:
  - Проверьте пути к шрифтам, константы разрешения дисплея и выбранные настройки режима в `words_gpt.py`.
- API отдаёт устаревшие данные:
  - Сделайте ручной `POST /next_random_word` и проверьте интервал обратного вызова в `app.py`.
- Рендер на железе «застыл»:
  - Проверьте tmux-сессию и системные логи командой `journalctl -u wordscard`.
- Отсутствуют записи словаря/словарные записи:
  - Проверьте CSV-файлы в `data/` и выполните задачи обслуживания через `words_update.py`.

## Дорожная карта<a id="дорожная-карта"></a>

- Добавить минимальный `requirements.txt` / воспроизводимый манифест установки.
- Добавить более явные режимы runtime и расширенную документацию CLI `--help`.
- Расширить документацию режимов рендера (`japanese_synonym`, `arabic_synonym`, `film` и другие workflows).
- Стандартизировать обработку ошибок и схемы API-ответов для пользователей.
- Добавить лёгкие smoke-test заглушки для проверки в CI без оборудования.

## Содействие проекту<a id="содействие-проекту"></a>

Патчи и идеи приветствуются. Рекомендуемый цикл:

1. Сфокусируйте изменения на одном функциональном блоке (рендеринг, данные, API, скрипты).
2. Обновите использование команд/документацию для изменений, затрагивающих поведение пользователя.
3. По возможности сохраняйте существующие CLI-флаги и совместимость с API.
4. Если меняются скрипты для оборудования, указывайте модель устройства и точный перечень запусков, которые тестировались.

## Лицензия<a id="лицензия"></a>

В текущем корне репозитория файл `LICENSE` отсутствует. Поэтому на данный момент формально лицензия в репозитории не определена. Добавьте его, если вам нужны явные условия распространения и повторного использования.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
