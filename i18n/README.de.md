[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 🖨️ Eink Words GPT

**Sprache dieses Entwurfs:** Deutsch

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

> Ein Raspberry-Pi + Waveshare-E-Ink-Projekt, das dynamische Wortkarten mit IPA-Phonetik und mehrsprachigen Hinweisen rendert. Es unterstützt lokale CSV-Workflows, optionales KI-Enhancement, E-Paper-Rendering und Fernsteuerung über HTTP.

Zur Laufzeit im Überblick:
`app.py` (Service) und `words_gpt.py` (Eigenständiger Renderer) können unabhängig oder gemeinsam betrieben werden.

| 🔎 Kurzüberblick | Details |
|---|---|
| **Hauptlaufzeit** | `app.py` (HTTP-Service) + `words_gpt.py` (Render-Schleife) |
| **Datenpfad** | CSV-Datasets in `data/` + SQLite-Speicher `words_phonetics.db` |
| **Ausgabeziele** | Waveshare-E-Paper-Panels und virtuelle Bildausgaben |
| **KI-Abhängigkeit** | optional (`--enable_openai`) mit Request-Cache in `cache/` |
| **Standardintervall** | Server auf `8082`, periodische Aktualisierung ca. alle 5 Minuten |

## 📚 Inhaltsverzeichnis
- [Übersicht](#übersicht)
- [Highlights](#highlights)
- [Demos](#demos)
- [Projektstruktur](#projektstruktur)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Konfiguration](#konfiguration)
- [Nutzung](#nutzung)
- [Beispiele](#beispiele)
- [Daten, Cache und Logs](#daten-cache-und-logs)
- [Entwicklungsnotizen](#entwicklungsnotizen)
- [Fehlerbehebung](#fehlerbehebung)
- [Roadmap](#roadmap)
- [Support](#support)
- [Mitwirken](#mitwirken)
- [Lizenz](#lizenz)

---

## Übersicht

`words_gpt` ist ein Python-Stack zur Erzeugung von Vokabelkarten für E-Ink-Anzeigen. Er kombiniert Datenorchestrierung, phonetische Anreicherung und Rendering in zwei Betriebsmodi:

- einen dauerhaft laufenden Tornado-Service (`app.py`) für Fernsteuerung und Bildausgabe,
- einen eigenständigen Renderer (`words_gpt.py`), der im Polling-, Loop- oder Direct-Render-Modus laufen kann.

Zentrale Module:

- `words_data.py` / `words_data_utils.py` für Wort- und Anreicherungs-Workflows.
- `words_data_with_legacy.py`, `words_data_without_legacy.py`, `words_data_workable*.py` für Varianten-Workflows und Legacy-Kompatibilität.
- `words_database.py` für die SQLite-Interaktion.
- `openai_request_json.py` für strukturierte OpenAI-Anfragen mit On-Disk-Cache und Retry-Handling.
- `env_loader.py` für deterministisches Laden der Umgebungsvariablen.
- `words_update.py` für Datenbankwartung und Re-Check-Workflows.
- `app.py` und `words_gpt.py` für Service-/Render-Lifecycle.
- `pwa/` für leichte Browser-Vorschau/ Konfiguration.

## Highlights

- E-Ink-Render-Pipeline mit mehrsprachigen Modi: japanische Varianten, Kanji-Modus, Arabisch, Chinesisch und Emoji-Modus.
- Lokale und OpenAI-gestützte Wortbeschaffung in einem gemeinsamen Ablauf.
- Optionales vereinfachtes Chinesisch-Rendering (`--simplify`).
- HTTP-Endpunkte für Fernsteuerung (`/next_random_word`, `/display_word`, `/get_current_word`, `/get_current_word_page`, `/get_words_card`).
- Caching und Persistenz zur Reduktion wiederholter KI-Anfragen.
- Packaging und Hardware-Beispiele über den ausgelieferten `waveshare/`-Treiberbaum.

## Demos

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## Projektstruktur

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

Wichtige Laufzeitdateien:

- `app.py`: Tornado-App auf Port `8082` + periodischer Trigger `next_random_word`.
- `words_gpt.py`: eigenständiger Renderer und Display-Abstraktionen (`EPaperHardware`, `EPaperDisplay`).
- `words_data.py`: Fetch-/Enrichment-Workflow und Hilfsfunktionen für die Wortauswahl.
- `words_database.py`: SQLite-Helfer für gespeicherte Metadaten und Wortcache-Operationen.
- `scripts/*.sh`: Installations- und Service-Lebenszyklus sowie Raspberry-Pi-Bootstrap-Skripte.
- `words_update.py`: Batch-Refresh-/Recheck-Helfer für die Datenbankqualität.

## Voraussetzungen

- Python `3.9+` (empfohlen)
- Raspberry Pi (für Hardwaremodus erforderlich)
- Unterstütztes Waveshare-E-Paper-Panel (z. B. 7.3F- oder 13K-Familie)
- SPI aktiviert (`raspi-config`), korrektes Verdrahtung und stabile Stromversorgung
- NLTK-Korpus bei Nutzung von NLTK-Wortquellen

Typische Abhängigkeiten im Repository und in Runtime-Pfaden:
`openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.

## Installation

### Option 1 — Minimale/Manuelle Installation (Desktop oder Pi)

Aus dem Repository-Root:

```bash
python setup.py install
```

Falls nötig:

```bash
python -m nltk.downloader words
```

### Option 2 — Automatisierte Raspberry-Pi-Installation (vor Ort empfohlen)

Aus dem Repository-Root:

```bash
bash scripts/setup_pi_wordscard.sh
```

Folgendes wird durchgeführt:

- Pi-spezifische Dependency-Installation
- Hilfen zur SPI-Aktivierung
- Einrichtung der `wordscard`-Virtuellen-Umgebung
- Installation der Python-/Runtime-Pakete
- Waveshare-Paket-Installation
- Starten des App-Prozesses über `tmux`

### Option 3 — Installation als systemd-Service

Um den App-Lifecycle unter `systemd` zu registrieren:

```bash
bash scripts/install_wordscard_service.sh
```

Danach:

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## Konfiguration

### Umgebungsvariablen (`.env`)

`env_loader` liest Umgebungsvariablen und wendet sie beim Prozessstartkontext an. Aktuelle Dokumentation und Laufzeitnutzung verwenden typischerweise:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

Annahme: Halte Geheimnisse in lokaler Umgebungs-Konfiguration und committe sie nicht ins Versionskontrollsystem.

### Laufzeit-Flags (verwendet von `app.py` und `words_gpt.py`)

| CLI-Flag | Zweck |
| --- | --- |
| `--enable_openai` | Optionalen OpenAI-Anreicherungstyp aktivieren |
| `--make_emoji` | Karten im Emoji-Modus rendern |
| `--ignore_list` | Wörter aus konfigurierten Ignore-Listen überspringen |
| `--simplify` | Vereinfachte CJK-Ausgabe erzeugen |
| `--use_csv` | Wörter aus CSV-Datasets lesen |
| `--complete_csv` | Vollständigen CSV-Quellmodus verwenden |
| `--filename <csv_file>` | Auf eine bestimmte CSV-Eingabedatei verweisen |

`APP_ARGS` kann über Startskripte übergeben werden. Beispiel:

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### Routing-Verhalten im App-Modus

Beobachtete API-Routen:

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (bereitgestellt aus `words_card_temp/`)

Kompatibilitätsnotiz: Ältere Dokumentation kann `GET /current_word` nennen; aktuelle Route ist `GET /get_current_word`.

### OpenAI-Hinweise

OpenAI-Funktionen sind optional und werden über CLI-/Env-Flags gesteuert. In `cache/` gecachte Requests verbessern Reproduzierbarkeit und helfen bei der Rate-Limit-Kontrolle. Für deterministische Offline-First-Ausführungen beginne mit CSV-Modus (`--use_csv`) und aktiviere OpenAI selektiv.

## Nutzung

### HTTP-Server starten

```bash
python app.py
```

Der Prozess hält das aktuellste Bild in `words_card_temp/` vor und stellt Endpunkte bereit, die von Frontend-Tools oder Skripten genutzt werden.

### Renderer direkt starten

CSV-Modus:

```bash
python words_gpt.py --use_csv
```

OpenAI-Modus:

```bash
python words_gpt.py --enable_openai --use_csv
```

Emoji + vereinfachtes CJK:

```bash
python words_gpt.py --make_emoji --simplify
```

### Auf Pi-Hardware ausführen

- Start über `tmux`-Skript:

```bash
bash scripts/start_wordscard.sh
```

- Stop über `tmux`-Skript:

```bash
bash scripts/stop_wordscard.sh
```

## Beispiele

Nächste zufällige Karten-Metadaten abrufen:

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

Aktuelles gespeichertes Wort abrufen:

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

Gerendertes Seitenbild-Payload anfordern:

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

Ein explizites Wort übermitteln:

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

Card-Rendering über Form-Endpoint auslösen:

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

## Daten, Cache und Logs

Typische Artefakte, die von der App verwendet werden:

- `data/`: kuratierte CSV-Datasets
- `words_phonetics.db`: SQLite-Cache/Quellen-Datenbank
- `cache/`: OpenAI-Request/Result-Cache
- `word_phonetics_processed.csv`: verarbeiteter/abgeleiteter Datensatz
- `logs/`, `logs-word-phonetics/`: Laufzeitlogs
- `words_card_temp/`: generierte Karten und temporäre Ausgabe
- `pic/` und `figs/`: Referenzbilder und Banner

## Entwicklungsnotizen

- Legacy-/Backup-Module und -Artefakte existieren (z. B. `words_gpt_old.py`, `lib.old`), daher bitte nur als Referenz verwenden, sofern keine beabsichtigte Migration/Kompatibilität vorliegt.
- `words_update.py` enthält Batch-Refresh-/Recheck-Helfer für die DB-Qualitätswartung.
- Hardware-Validierung erfolgt über `epd_*_test.py` und `waveshare/examples/*` Demo-Skripte.
- Es gibt weder `requirements.txt` noch Lockfile im Repository-Root; Abhängigkeiten werden über Setup-Skripte und direkte Installationsabläufe gesteuert.
- Es ist kein automatisierter Test-Suite im Repo konfiguriert.

## Fehlerbehebung

- `ImportError` von Raspberry-Pi GPIO/SPI-Modulen:
  - Nutze den Pi-Setup-Flow (`scripts/setup_pi_wordscard.sh`) oder installiere Abhängigkeiten explizit auf einem kompatiblen Zielsystem.
- `403/404` von Bild-/Static-Endpoints:
  - Bestätige die Routenverwendung (`/get_current_word*`) und dass `words_card_temp/` beschreibbar ist.
- Leere oder ungültige Wort-Payload im OpenAI-Modus:
  - Prüfe `OPENAI_API_KEY` (sowie Org-/Model-Werte), dann inspiziere `cache/` und Logs.
- Schlechte Darstellung oder Textbeschneidung:
  - Prüfe Font-Pfade, Display-Auflösungskonstanten und die gewählten Modi in `words_gpt.py`.
- API liefert veraltete Daten:
  - Rufe manuell `POST /next_random_word` auf und prüfe das Callback-Intervall in `app.py`.
- Hardware-Rendering wirkt eingefroren:
  - Prüfe tmux-Session und Systemd-Logs mit `journalctl -u wordscard`.
- Fehlende Datensätze/Wörterbuch-Einträge:
  - Validiere die CSV-Dateien in `data/` und führe Wartungsaufgaben in `words_update.py` aus.

## Roadmap

- Minimal `requirements.txt` / reproduzierbares Installationsmanifest hinzufügen.
- Klarere Laufzeitmodi und explizite CLI-`--help`-Dokumentation ergänzen.
- Rendering-Modi-Dokumentation ausbauen (`japanese_synonym`, `arabic_synonym`, `film` und weitere Workflows).
- Standardisiertes Fehlerhandling und benutzerseitige API-Response-Schemata einführen.
- Leichte Smoke-Test-Stubs für Nicht-Hardware-CI-Validierung ergänzen.

## Mitwirken

Beiträge sind willkommen. Empfohlener Ablauf:

1. Halte Änderungen auf einen Verhaltensbereich fokussiert (Rendering, Daten, API, Skripte).
2. Aktualisiere Kommandos und Dokumentation bei nutzerrelevanten Verhaltensänderungen.
3. Bewahre bestehende CLI-Flags und Endpoint-Kompatibilität soweit möglich.
4. Wenn Hardware-Skripte geändert werden, dokumentiere Modell und exakte ausgeführte Befehle.

## Lizenz

Es ist keine `LICENSE`-Datei im aktuellen Repository-Root vorhanden. Die effektive Lizenz ist daher im Tree nicht definiert. Bitte füge eine Datei hinzu, wenn du explizite Wiederverwendungs- und Redistribution-Bedingungen benötigst.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
