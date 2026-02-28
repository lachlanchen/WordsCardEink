[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Eink Words GPT


![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=flat-square&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=flat-square)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=flat-square)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=flat-square&logo=openai&logoColor=white)

Проект для Raspberry Pi + Waveshare e-ink, который показывает динамически выбранную лексику с фонетикой и многоязычными синонимами. Система может получать слова из локальных датасетов или OpenAI, рендерить их в макет карточки и выводить результат на поддерживаемые e-paper панели. Также она предоставляет небольшой HTTP-сервис для запуска обновления слов и получения сгенерированных изображений.

| 🔎 Кратко | Детали |
|---|---|
| Основной runtime | `app.py` (HTTP-сервис) + `words_gpt.py` (цикл рендеринга) |
| Путь данных | CSV-датасеты в `data/` + SQLite-хранилище `words_phonetics.db` |
| Целевые выходы | Waveshare e-paper панели и виртуционные изображения |
| Зависимость от AI | Опционально (`--enable_openai`) с кэшем в `cache/` |

## 📚 Содержание
- [Обзор](#обзор)
- [Ключевые возможности](#ключевые-возможности)
- [Быстрый старт](#быстрый-старт)
- [Демо](#демо)
- [Функции](#функции)
- [Структура проекта](#структура-проекта)
- [Предварительные требования](#предварительные-требования)
- [Установка](#установка)
- [Конфигурация](#конфигурация)
- [Использование](#использование)
- [Примеры](#примеры)
- [Данные, кэш и логи](#данные-кэш-и-логи)
- [Заметки по разработке](#заметки-по-разработке)
- [Устранение неполадок](#устранение-неполадок)
- [Примечания по использованию OpenAI](#примечания-по-использованию-openai)
- [Дорожная карта](#дорожная-карта)
- [Support](#-support)
- [Участие в разработке](#участие-в-разработке)
- [Лицензия](#лицензия)

## Обзор
`words_gpt` — Python-система генерации и отображения словарных карточек для e-ink устройств.

Она объединяет:
- Получение слов из CSV/локальных датасетов и опциональную генерацию через OpenAI.
- Обогащение данных (IPA-фонетика + многоязычные поля синонимов).
- Конвейеры рендеринга для аппаратного и виртуального вывода.
- HTTP-сервис Tornado для удаленного запуска и получения изображений.

Текущая кодовая база сосредоточена вокруг `app.py`, `words_gpt.py`, `words_data.py`, `words_database.py` и `openai_request_json.py`.

## Ключевые возможности
- 🖼️ Конвейер e-ink рендеринга с несколькими режимами контента (канзи, японский, арабский, китайский, emoji).
- 🗃️ Локальная база слов (`words_phonetics.db`) со списками слов на базе CSV в `data/`.
- 🤖 Выбор слов и фонетическое обогащение через OpenAI со структурированными JSON-ответами.
- 🌐 HTTP-сервис для внешних триггеров и получения изображений.
- ⚡ Слой кэширования (`cache/`) для уменьшения повторных обращений к OpenAI.

## Быстрый старт
| Цель | Команда |
|---|---|
| Запустить HTTP-сервер (порт `8082`) | `python app.py` |
| Запустить автономный рендерер (CSV) | `python words_gpt.py --use_csv` |
| Запуск с OpenAI + CSV | `python words_gpt.py --enable_openai --use_csv` |
| Режим emoji + упрощенный CJK | `python words_gpt.py --make_emoji --simplify` |
| Автонастройка Raspberry Pi | `bash scripts/setup_pi_wordscard.sh` |

## Демо
<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## Функции
- Поток аппаратного и виртуального рендеринга (`EPaperHardware`, `EPaperDisplay`) из `words_gpt.py`.
- Конвейер многоязычного обогащения в `words_data.py` (IPA, японские варианты, арабский, французский, китайские поля).
- Хранилище на SQLite с помощниками динамического обновления полей в `words_database.py`.
- Помощник структурированных JSON-запросов OpenAI с файловым кэшем в `openai_request_json.py`.
- Опциональные PWA-ресурсы в `pwa/` для легкой настройки/предпросмотра frontend-потока.

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
- `words_database.py`: помощники SQLite-хранилища.
- `scripts/*.sh`: настройка Raspberry Pi, установка сервиса и скрипты жизненного цикла tmux.

## Предварительные требования
- Python `3.9+` (рекомендуется).
- Raspberry Pi как целевое устройство (для аппаратного режима).
- Поддерживаемая панель Waveshare e-paper.
- Включенный SPI на Pi (`raspi-config`) и корректное подключение под конкретную панель.

Python-пакеты, используемые в проекте:
- `openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.
- Setup-скрипт дополнительно устанавливает: `json5`, `pandas`, `spidev`, `RPi.GPIO`, `gpiozero`, `lgpio`.

## Установка

### Вариант A: минимальная/ручная установка
Установите пакет драйверов Waveshare:
```bash
python setup.py install
```

Если используете список слов NLTK, скачайте его один раз:
```bash
python -m nltk.downloader words
```

### Вариант B: автоматическая настройка Raspberry Pi (рекомендуется на устройстве)
Из корня репозитория:
```bash
bash scripts/setup_pi_wordscard.sh
```

Этот скрипт:
- Устанавливает apt-зависимости.
- Проверяет, что SPI включен.
- Создает и активирует виртуальное окружение `wordscard`.
- Устанавливает Python runtime-зависимости.
- Устанавливает пакет Waveshare.
- Запускает `app.py` внутри tmux-сессии.

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

Примечание по совместимости: в ранней документации упоминался `GET /current_word`; в текущем `app.py` маршрут — `GET /get_current_word`.

### Запуск автономного рендерера
Список на базе CSV:
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

### Прочитать текущий payload слова
```bash
curl "http://127.0.0.1:8082/get_current_word"
```

### Отправить конкретное слово
```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

### Быстрые аппаратные тесты
Используйте скрипт для конкретного дисплея:
```bash
python epd_7in3f_test.py
```

Или:
```bash
python epd_13in3k_test.py
```

Больше примеров есть в `waveshare/examples/`.

## Данные, кэш и логи
| Область | Путь(и) | Примечания |
|---|---|---|
| Списки слов | `data/` | Включает `data/words_list.csv` и тематические CSV-файлы |
| Постоянная БД | `words_phonetics.db` | Локальное хранилище фонетики/обогащения |
| Артефакты OpenAI/кэша | `cache/` | Уменьшает количество повторных запросов |
| Логи | `logs/`, `logs-word-phonetics/` | Runtime- и update-логи |
| Сгенерированные карточки | `words_card_temp/` | Выходные изображения и источник для статической раздачи |

## Заметки по разработке
- Управление зависимостями сейчас script-first (`scripts/setup_pi_wordscard.sh`) + `setup.py`; `requirements.txt` или `pyproject.toml` пока нет.
- В репозитории есть несколько резервных/legacy-файлов (`words_data_*`, `words_gpt_old.py`); активный runtime-путь в основном `app.py` + `words_gpt.py` + `words_data.py` + `words_database.py`.
- `env_loader.py` всегда перезаписывает переменные окружения из `.env`, когда ключи присутствуют.
- В серверном режиме работает периодический refresh-процесс (каждые ~5 минут), который может внутренне вызывать endpoint обновления.

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
- Консолидировать legacy/резервные варианты скриптов после финализации канонического потока.
- Документировать PWA-процесс (`pwa/`) с примерами endpoint'ов и скриншотами.
- Добавить повторяемые автоматизированные тесты для данных и поведения маршрутов.

## ❤️ Support

Если этот проект полезен вам, эти ссылки напрямую поддерживают дальнейшую разработку и итерации по железу.

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

### Что делает возможной ваша поддержка
- <b>Сохраняет инструменты открытыми</b>: хостинг, инференс, хранение данных и работа сообщества.  
- <b>Ускоряет разработку</b>: сфокусированное open-source время на WordsCardEink и связанных образовательных инструментах.  
- <b>Помогает прототипировать устройства</b>: итерации e-ink-аппаратуры и исследования макетов дисплея.  
- <b>Расширяет доступ</b>: субсидируемые внедрения для студентов, авторов и локальных сообществ.

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

## Участие в разработке
См. `AGENTS.md` для рекомендаций по участию, стилю кода и ожиданиям по PR.

Рекомендуемый чеклист для вкладов:
- Указывайте модель панели и аппаратные примечания для изменений дисплея.
- Перечисляйте точные команды, использованные для валидации.
- Прикладывайте скриншоты/фото для изменений UI или e-ink-вывода.
- Описывайте изменения в датасетах (файл + влияние на строки/столбцы).

## Лицензия
В корне репозитория сейчас отсутствует файл `LICENSE` (зафиксировано в этой версии чернового прохода). Пока файл лицензии не добавлен, права на повторное использование явно не предоставлены.

Предположение: мейнтейнеры могут добавить явную open-source лицензию в следующем обновлении.
