[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 🖨️ Eink Words GPT

**Sprache dieser Fassung:** Deutsch

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=for-the-badge&logo=githubactions&logoColor=white)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=for-the-badge&logo=python&logoColor=white)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=for-the-badge&logo=openai&logoColor=white)

Ein Raspberry-Pi + Waveshare-E-Ink-Projekt, das dynamisch ausgewählte Vokabelkarten mit IPA-Phonetik und mehrsprachigen Hinweisen rendert. Es unterstützt lokale CSV-Workflows, optionales KI-Enhancement, E-Paper-Rendering und Fernsteuerung über HTTP.

| 🔎 Kurzüberblick | Details |
|---|---|
| Zentrale Laufzeit | `app.py` (HTTP-Service) + `words_gpt.py` (Render-Loop) |
| Datenpfad | CSV-Datensätze in `data/` + SQLite-Store `words_phonetics.db` |
| Ausgabeziele | Waveshare-E-Paper-Panels und virtuelle Bilder |
| KI-Abhängigkeit | Optional (`--enable_openai`) mit Cache in `cache/` |
| Standard-Loop | Server auf `8082`, periodische Aktualisierung ca. alle 5 Minuten |

## 📚 Inhaltsverzeichnis
- [Überblick](#überblick)
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
- [Beiträge](#beiträge)
- [Lizenz](#lizenz)

---

## Überblick

`words_gpt` ist ein Python-Stack für die Erzeugung von Vokabelkarten für E-Ink-Anzeigen. Es kombiniert Datenorchestrierung, phonetische Anreicherung und Rendering-Kontrolle hinter zwei Betriebsmodi:

- Einen dauerhaft laufenden Tornado-Service (`app.py`) für Fernsteuerung und Bildausgabe
- Einen eigenständigen Renderer (`words_gpt.py`), der im Polling-, Loop- oder Direktrender-Modus laufen kann

Hauptmodule:

- `words_data.py` / `words_data_utils.py` für Wort- und Enrichment-Workflows
- `words_database.py` für SQLite-Interaktion
- `openai_request_json.py` für strukturierte OpenAI-Anfragen mit On-Disk-Cache
- `env_loader.py` für deterministisches Laden von Umgebungsvariablen
- `words_update.py` für DB-Wartung und Neuprüfungs-Workflows
- `app.py` und `words_gpt.py` für Service-/Render-Lifecycle

## Highlights

- E-Ink-Render-Pipeline mit mehreren Sprach-/Inhaltsmodi:
  - Japanische Varianten, Kanji-Modus, Arabisch, Chinesisch, Emoji-Modus
- Lokale und OpenAI-Wortbereitstellung in einem gemeinsamen Workflow
- Optional vereinfachtes Chinesisch im Renderpfad
- Server-Endpunkte für direkte Interaktion (`/next_random_word`, `/display_word`, etc.)
- Caching und Persistenz, die wiederholte Netzwerkaufrufe reduzieren
- Optionale PWA-Assets in `pwa/` für leichtgewichtige Vorschau-/Konfigurationsabläufe

## Demos

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabische Vokabelkarte" width="48%" />
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

Wichtige Runtime-Dateien:

- `app.py`: Tornado-App auf Port `8082` + periodischer `next_random_word`-Trigger.
- `words_gpt.py`: eigenständiger Renderer und Display-Abstraktionen (`EPaperHardware`, `EPaperDisplay`).
- `words_data.py`: erweiterter Fetch/Enrichment-Workflow und Hilfsfunktionen.
- `words_database.py`: SQLite-Helpers für gespeicherte Metadaten und Wortcache-Operationen.
- `scripts/*.sh`: Install-/Service-Lifecycle und Raspberry-Pi-Bootstrap-Helfer.

## Voraussetzungen

- Python `3.9+` (empfohlen)
- Raspberry Pi (erforderlich für Hardware-Modus)
- Unterstütztes Waveshare-E-Paper-Panel (z. B. 7.3F / 13K-Familie)
- SPI aktiviert (`raspi-config`), korrekte Verdrahtung und stabile Stromversorgung
- NLTK-Korpus verfügbar, wenn `nltk`-Wortquellen genutzt werden

Übliche Abhängigkeiten, wie im Code sichtbar:
`openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.

## Installation

### Option 1 — Minimale/Manuelle Installation (Desktop oder Pi)

Aus Repository-Root:

```bash
python setup.py install
```

Falls nötig:

```bash
python -m nltk.downloader words
```

### Option 2 — Automatisierte Raspberry-Pi-Installation (auf dem Gerät empfohlen)

Aus Repository-Root:

```bash
bash scripts/setup_pi_wordscard.sh
```

Das führt aus:

- Pi-spezifische Abhängigkeiten
- SPI-Aktivierung
- Einrichtung der `wordscard`-Virtualenv
- Installation von Python-/Runtime-Paketen
- Installation des Waveshare-Pakets
- Start des App-Prozesses über `tmux`

### Option 3 — Service-Installation

Zum Registrieren des App-Lifecycles mit `systemd`:

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

Das Repository nutzt `.env`-Laden, das aktuell vorhandene Shell-Variablen überschreibt. Das bewusst einsetzen:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### Laufzeit-Flags (verwendet von `app.py` und `words_gpt.py`)

- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

Die Pi-Startup-Skripte unterstützen Argument-Weitergabe über `APP_ARGS` (Beispiel):

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### Routing-Verhalten im App-Modus

Beobachtete Routen im aktuellen Code:

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (bereitgestellt aus `words_card_temp/`)

Hinweis zur Kompatibilität: Ältere Dokumentation nannte `GET /current_word`; aktuelle Route ist `GET /get_current_word`.

### Hinweise zur OpenAI-Nutzung

OpenAI-Funktionen sind optional und werden per CLI/Env-Flags gesteuert. Gepufferte API-Payloads sind nützlich für Reproduzierbarkeit und Rate-Limit-Steuerung. In eingeschränkten Umgebungen zuerst im CSV-Modus arbeiten (`--use_csv`) und OpenAI gezielt (`--enable_openai`) nur bei Bedarf aktivieren.

## Nutzung

### HTTP-Server starten

```bash
python app.py
```

Der Prozess hält ein Bild in `words_card_temp/` bereit und stellt HTTP-Endpunkte bereit, die von Frontend-Tools oder einfachen Skripten genutzt werden.

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

### Auf Pi-Hardware laufen lassen

- Start über `tmux`-Skript:

```bash
bash scripts/start_wordscard.sh
```

- Stopp über `tmux`-Skript:

```bash
bash scripts/stop_wordscard.sh
```

## Beispiele

Nächstes Zufallswort ermitteln:

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

Aktuelles gespeichertes Wort abrufen:

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

Gerenderte Seiten-Payload abrufen:

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

Explizites Wort senden:

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

Rendervorgang über Form-Endpoint auslösen:

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

## Daten, Cache und Logs

Typische Artefakte der App:

- `data/`: kuratierte CSV-Datensätze
- `words_phonetics.db`: SQLite-Cache/Quelldatenbank
- `cache/`: OpenAI-Request/Result-Cache
- `word_phonetics_processed.csv`: aufbereiteter/abgeleiteter Datensatz
- `logs/`, `logs-word-phonetics/`: Laufzeit-Logs
- `words_card_temp/`: generierte Karten und temporäre Ausgaben

## Entwicklungsnotizen

- Legacy-/Backup-Dateien sind vorhanden (z. B. `words_gpt_old.py`, `lib.old`), diese nur als Referenz nutzen, außer bei gezielter Migration oder Kompatibilitätswartung.
- `words_update.py` enthält Batch-Refresh/-Recheck-Helfer für die Datenqualitätsprüfung.
- Hardwarevalidierung erfolgt über `epd_*_test.py` und Demos in `waveshare/examples/*`.
- Es gibt keine `requirements.txt` oder Lock-Datei im Repository-Root; die Abhängigkeitsinstallation erfolgt über das Setup-Skript oder direkte Installation.
- Es ist keine automatisierte Test-Suite im Repo konfiguriert.

## Fehlerbehebung

- `ImportError` aus Raspberry-Pi GPIO/SPI-Modulen:
  - Installation über den Pi-Pfad (`setup_pi_wordscard.sh`) durchführen oder `python setup.py install` auf kompatibler Zielplattform prüfen.
- `403/404` von Bild-/Static-Endpoints:
  - Nutzung der `/get_current_word*`-Endpoints prüfen und sicherstellen, dass `words_card_temp/` schreibbar ist.
- Leere/ungültige OpenAI-Payload:
  - Prüfen, ob `OPENAI_API_KEY` sowie optionale Org-/Model-Werte geladen sind; `cache/` und Logs prüfen.
- Schlechtes Rendering/ Textüberlauf:
  - Font-Pfad und Panelauflösung in der Renderkette von `words_gpt.py` prüfen.
- API liefert veraltete Daten:
  - `POST /next_random_word` manuell aufrufen und das periodische Callback-Intervall in `app.py` prüfen.
- Hardware-Update wirkt eingefroren:
  - Tmux-Sitzung prüfen und Systemd-Logs prüfen (`journalctl -u wordcard`).
- Fehlende Datensätze oder Wörterbucheinträge:
  - CSV-Dateien in `data/` validieren und `words_update.py`-Workflows für Refresh/Cleanup ausführen.

## Roadmap

- Ein minimal `requirements.txt` / reproduzierbares Install-Manifest hinzufügen.
- Klarere Runtime-Modi und explizite CLI-`--help`-Dokumentation ergänzen.
- Render-Schema-Dokumentation je Inhaltsmodus erweitern (`japanese_synonym`, `arabic_synonym`, `film`, etc.).
- Standardisiertes Error-Handling und benutzernahe API-Response-Schemata einführen.
- Kleine Smoke-Test-Skripte für nicht-hardwarebasierte CI-Validierung ergänzen.

## Support

| Support-Option | Link | Zweck |
|---|---|---|
| GitHub Sponsors | https://github.com/sponsors/lachlanchen | Laufende und einmalige Projektunterstützung |
| Lazying Art | https://lazying.art | Marke und zugehörige Ressourcen |
| Chat | https://chat.lazying.art | Austausch und Unterstützung |
| Only Ideas | https://onlyideas.art | Kreative Forschung und Nebenprojekte |

## Beiträge

Beiträge sind willkommen. Empfohlener Ablauf:

1. Änderungen auf einen Verhaltensbereich konzentrieren (Rendern, Daten, API, Skripte).
2. Nutzungs- und Dokumentationsangaben bei sichtbaren Benutzeränderungen aktualisieren.
3. Bestehende CLI-Flags und Endpoint-Kompatibilität soweit möglich erhalten.
4. Bei Änderungen an Hardware-Skripten getestete Geräte/Modelle und exakte ausgeführte Befehle dokumentieren.

## Lizenz

Es ist keine `LICENSE`-Datei im aktuellen Repository-Root vorhanden. Die effektive Lizenz ist im Baum daher in diesem Stand nicht definiert. Bitte ergänzen Sie eine Lizenzdatei, wenn explizite Nutzungs- und Weiterverwendungsbedingungen gewünscht sind.
