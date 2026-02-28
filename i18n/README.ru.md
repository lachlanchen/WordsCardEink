[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 🖨️ Eink Words GPT

**Language of this draft:** Русский

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=for-the-badge&logo=githubactions&logoColor=white)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=for-the-badge&logo=python&logoColor=white)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=for-the-badge&logo=openai&logoColor=white)

Проект для Raspberry Pi + e-ink Waveshare, который рендерит автоматически выбранные карточки слов с транскрипцией IPA и многоязычными подсказками. Поддерживаются локальные CSV-пайплайны, опциональное обогащение ИИ, вывод на e-paper и удалённое управление через HTTP.

| 🔎 На первый взгляд | Детали |
|---|---|
| Основной runtime | `app.py` (HTTP-сервис) + `words_gpt.py` (цикл рендера) |
| Путь данных | CSV-наборы в `data/` + SQLite-хранилище `words_phonetics.db` |
| Форматы вывода | Панели Waveshare e-paper и виртуальный рендер в изображения |
| Зависимость от ИИ | Необязательно (`--enable_openai`) с кешированием в `cache/` |
| Значения цикла по умолчанию | Сервер на `8082`, периодическое обновление примерно раз в 5 минут |

## 📚 Table of Contents
- [Обзор](#обзор)
- [Ключевые возможности](#ключевые-возможности)
- [Демо](#демо)
- [Структура проекта](#структура-проекта)
- [Требования](#требования)
- [Установка](#установка)
- [Настройка](#настройка)
- [Использование](#использование)
- [Примеры](#примеры)
- [Данные, кеш и логи](#данные-кеш-и-логи)
- [Замечания по разработке](#замечания-по-разработке)
- [Устранение неполадок](#устранение-неполадок)
- [Дорожная карта](#дорожная-карта)
- [Поддержка](#поддержка)
- [Участие](#участие)
- [Лицензия](#лицензия)

---

## Обзор

`words_gpt` — это стек генерации карточек слов для e-ink-дисплеев на Python. Он объединяет оркестрацию данных, фонетическое обогащение и рендер-процесс, предоставляя два режима выполнения:

- Долгоживущая служба Tornado (`app.py`) для удалённого управления и выдачи изображений
- Отдельный рендерер (`words_gpt.py`), работающий в режимах опроса, цикла или прямого рендера

Ключевые модули:

- `words_data.py` / `words_data_utils.py` для работы со словами и пайплайнами обогащения
- `words_database.py` для работы с SQLite
- `openai_request_json.py` для структурированных запросов к OpenAI с кешированием на диске
- `env_loader.py` для детерминированной загрузки окружения
- `words_update.py` для обслуживания БД и процедур перепроверки
- `app.py` и `words_gpt.py` для жизненного цикла сервиса/рендера

## Ключевые возможности

- Конвейер рендера для e-ink с несколькими языковыми/контентными режимами:
  - Японские варианты, режим канжи, арабский, китайский, emoji-режим
- Локальный и OpenAI-подход к поиску слов в одном workflow
- Опциональный вывод упрощённого китайского в пути рендера
- HTTP-роуты для прямого взаимодействия (`/next_random_word`, `/display_word` и др.)
- Кеширование и персистентность, уменьшающие повторные сетевые запросы
- Опциональные PWA-ресурсы в `pwa/` для лёгкого предпросмотра и конфигурации

## Демо

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

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

Важные runtime-файлы:

- `app.py`: Tornado-приложение на порту `8082` и периодический триггер `next_random_word`.
- `words_gpt.py`: самостоятельный рендерер и абстракции вывода (`EPaperHardware`, `EPaperDisplay`).
- `words_data.py`: продвинутый fetch/enrichment workflow и вспомогательные утилиты.
- `words_database.py`: SQLite-утилиты для метаданных и операций с кешем слов.
- `scripts/*.sh`: установка/сервисный lifecycle и helper-скрипты для bootstrap на Raspberry Pi.

## Требования

- Python `3.9+` (рекомендуется)
- Raspberry Pi (обязательно для аппаратного режима)
- Поддерживаемая e-paper панель Waveshare (например, семейства 7.3F / 13K)
- Включён SPI (`raspi-config`), корректная распайка и стабильное питание
- Корпус NLTK (`nltk`) должен быть доступен при использовании источников слов `nltk`

Распространённые зависимости, встречающиеся в кодовой базе:
`openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.

## Установка

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

Скрипт выполняет:

- Зависимости под Pi
- Включение SPI
- Настройку виртуального окружения `wordscard`
- Установку пакетов Python/runtime
- Установку пакета Waveshare
- Запуск процесса app через `tmux`

### Вариант 3 — Установка как служба

Для регистрации жизненного цикла приложения через `systemd`:

```bash
bash scripts/install_wordscard_service.sh
```

Далее:

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## Настройка

### Переменные окружения (`.env`)

В репозитории используется загрузка `.env`, которая в текущей реализации перезаписывает ранее установленные переменные shell. Используйте это осознанно:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### Флаги runtime (используются `app.py` и `words_gpt.py`)

- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

Скрипты запуска Pi поддерживают проброс аргументов через `APP_ARGS` (пример):

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### Поведение роутинга в режиме app

Наблюдаемые роуты в текущем коде:

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (доступен из `words_card_temp/`)

Важно: в более ранних документах встречался `GET /current_word`; текущий роут — `GET /get_current_word`.

### Примечания по использованию OpenAI

Функции OpenAI опциональны и управляются CLI/env-флагами. Кешированные ответы API полезны для воспроизводимости и контроля лимитов запросов. В ограниченных средах запускайте сначала CSV-режим (`--use_csv`), а OpenAI включайте выборочно (`--enable_openai`) только при необходимости обогащения.

## Использование

### Запуск HTTP-сервера

```bash
python app.py
```

Процесс хранит изображение в `words_card_temp/` и открывает HTTP-роуты для фронтендов или простых скриптов.

### Запуск рендера напрямую

CSV-режим:

```bash
python words_gpt.py --use_csv
```

OpenAI-режим:

```bash
python words_gpt.py --enable_openai --use_csv
```

Эмодзи + упрощённый китайский:

```bash
python words_gpt.py --make_emoji --simplify
```

### Запуск на оборудовании Pi

- Запуск через `tmux`-скрипт:

```bash
bash scripts/start_wordscard.sh
```

- Остановка через `tmux`-скрипт:

```bash
bash scripts/stop_wordscard.sh
```

## Примеры

Запрос следующей случайной карточки:

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

Получить текущую сохранённую карточку:

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

Запросать payload отрисованной страницы:

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

Отправить конкретное слово:

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

Запустить отрисовку карточки через endpoint в стиле form:

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

## Данные, кеш и логи

Типичные артефакты, используемые приложением:

- `data/`: подготовленные CSV-наборы
- `words_phonetics.db`: SQLite-кеш/источник базы данных
- `cache/`: кеш запросов и результатов OpenAI
- `word_phonetics_processed.csv`: обработанный/производный датасет
- `logs/`, `logs-word-phonetics/`: логи выполнения
- `words_card_temp/`: сгенерированные карточки и временные файлы

## Замечания по разработке

- В репозитории есть устаревшие/резервные файлы (например, `words_gpt_old.py`, `lib.old`), поэтому относитесь к ним как к справочнику, если вы не мигрируете и не поддерживаете обратную совместимость.
- `words_update.py` содержит batch-утилиты обновления/перепроверки, полезные для проходов очистки качества данных.
- Проверка аппаратной части выполняется через `epd_*_test.py` и демонстрации `waveshare/examples/*`.
- В корне репозитория нет `requirements.txt` или lockfile; зависимости устанавливаются через setup-скрипт или непосредственную установку.
- Автоматизированный test-suite в репозитории не настроен.

## Устранение неполадок

- `ImportError` от модулей Raspberry Pi GPIO/SPI:
  - Установить через путь Pi (`setup_pi_wordscard.sh`) или проверить `python setup.py install` на совместимом таргете.
- `403/404` с image/static endpoint-ов:
  - Проверьте использование `/get_current_word*` и права на запись `words_card_temp/`.
- Пустой/невалидный payload слова из OpenAI-режима:
  - Убедитесь, что загружены `OPENAI_API_KEY` и при необходимости org/model; проверьте `cache/` и логи.
- Плохой рендер/обрезка текста:
  - Проверьте путь к шрифтам и настройки разрешения панели внутри рендерного пути `words_gpt.py`.
- API отдает устаревшие данные:
  - Вручную вызовите `POST /next_random_word` и проверьте интервал периодического callback в `app.py`.
- Аппаратное обновление выглядит «зависшим»:
  - Проверьте tmux-сессию и логи systemd (`journalctl -u wordscard`).
- Отсутствует датасет или записи в словаре:
  - Проверьте CSV-файлы в `data/` и запустите workflow из `words_update.py` для обновления/очистки.

## Дорожная карта

- Добавить минимальный `requirements.txt` / воспроизводимый манифест зависимостей.
- Добавить более ясные runtime-режимы и явные CLI-документы `--help`.
- Расширить схему документации рендера для каждого content-режима (`japanese_synonym`, `arabic_synonym`, `film`, и т.д.).
- Стандартизировать обработку ошибок и схемы API-ответов для пользователя.
- Добавить небольшие шаблоны smoke-test для валидации без оборудования в CI.

## Поддержка

| Поддержка | Ссылка | Назначение |
|---|---|---|
| GitHub Sponsors | https://github.com/sponsors/lachlanchen | Постоянная или разовая поддержка проекта |
| Lazying Art | https://lazying.art | Бренд и связанные ресурсы |
| Chat | https://chat.lazying.art | Обсуждение и поддержка |
| Only Ideas | https://onlyideas.art | Креативные исследования и побочные проекты |

## Участие

Приветствуется вклад в проект. Рекомендуемый процесс:

1. Держите изменения в пределах одной области поведения (рендер, данные, API, скрипты).
2. Обновляйте usage-документацию и команды для изменений, затрагивающих пользовательское поведение.
3. Сохраняйте совместимость существующих CLI-флагов и маршрутов, где это возможно.
4. Если меняются hardware-скрипты, документируйте протестированное устройство/модель и точные команды.

## Лицензия

Файл `LICENSE` отсутствует в текущем корне репозитория. Эффективная лицензия на момент этого черновика в дереве кода не определена. Добавьте файл лицензии, если вам нужны явные условия распространения и повторного использования.
