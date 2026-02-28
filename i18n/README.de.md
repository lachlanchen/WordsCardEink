[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Eink Words GPT

**Sprachoptionen:** Deutsch (diese Fassung)

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=flat-square&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=flat-square)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=flat-square)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=flat-square&logo=openai&logoColor=white)

Ein Raspberry-Pi- + Waveshare-E-Ink-Projekt, das dynamisch ausgewählten Wortschatz mit Phonetik und mehrsprachigen Synonymen anzeigt. Das System kann Wörter aus lokalen Datensätzen oder über OpenAI beziehen, sie in ein Layout rendern und das Ergebnis auf unterstützte E-Paper-Panels ausgeben. Zusätzlich stellt es einen kleinen HTTP-Service bereit, um Wort-Updates auszulösen und gerenderte Bilder abzurufen.

| 🔎 Auf einen Blick | Details |
|---|---|
| Kernlaufzeit | `app.py` (HTTP-Service) + `words_gpt.py` (Renderer-Schleife) |
| Datenpfad | CSV-Datensätze in `data/` + SQLite-Speicher `words_phonetics.db` |
| Ausgabeziele | Waveshare-E-Paper-Panels und virtuelle Bildausgaben |
| AI-Abhängigkeit | Optional (`--enable_openai`) mit Cache in `cache/` |

## 📚 Inhaltsverzeichnis
- [Ueberblick](#ueberblick)
- [Highlights](#highlights)
- [Schnellstart](#schnellstart)
- [Demos](#demos)
- [Funktionen](#funktionen)
- [Projektstruktur](#projektstruktur)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Konfiguration](#konfiguration)
- [Verwendung](#verwendung)
- [Beispiele](#beispiele)
- [Daten, Cache und Logs](#daten-cache-und-logs)
- [Hinweise zur Entwicklung](#hinweise-zur-entwicklung)
- [Fehlerbehebung](#fehlerbehebung)
- [Hinweise zur OpenAI-Nutzung](#hinweise-zur-openai-nutzung)
- [Roadmap](#roadmap)
- [Support](#-support)
- [Mitwirken](#mitwirken)
- [Lizenz](#lizenz)

## Ueberblick
`words_gpt` ist ein Python-basiertes System zur Erstellung und Anzeige von Vokabelkarten auf E-Ink-Geräten.

Es kombiniert:
- Wortquellen aus CSV/lokalen Datensätzen und optionaler OpenAI-Generierung.
- Anreicherung (IPA-Phonetik + mehrsprachige Synonymfelder).
- Rendering-Pipelines für Hardware- und virtuelle Ausgaben.
- Einen Tornado-HTTP-Service für Remote-Auslösung und Bildabruf.

Die aktuelle Codebasis konzentriert sich auf `app.py`, `words_gpt.py`, `words_data.py`, `words_database.py` und `openai_request_json.py`.

## Highlights
- 🖼️ E-Ink-Rendering-Pipeline mit mehreren Inhaltsmodi (Kanji, Japanisch, Arabisch, Chinesisch, Emoji).
- 🗃️ Lokale Wortdatenbank (`words_phonetics.db`) mit CSV-basierten Wortlisten in `data/`.
- 🤖 OpenAI-gestützte Wortauswahl und phonetische Anreicherung mit strukturierten JSON-Ausgaben.
- 🌐 HTTP-Service für externe Trigger und Bildabruf.
- ⚡ Caching-Schicht (`cache/`) zur Reduktion wiederholter OpenAI-Aufrufe.

## Schnellstart
| Ziel | Befehl |
|---|---|
| HTTP-Server starten (Port `8082`) | `python app.py` |
| Standalone-Renderer ausfuehren (CSV) | `python words_gpt.py --use_csv` |
| Mit OpenAI + CSV ausfuehren | `python words_gpt.py --enable_openai --use_csv` |
| Emoji + vereinfachter CJK-Modus | `python words_gpt.py --make_emoji --simplify` |
| Raspberry-Pi-Auto-Setup | `bash scripts/setup_pi_wordscard.sh` |

## Demos
<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabische Vokabelkarte" width="48%" />
</p>

## Funktionen
- Hardware- + virtuelle Rendering-Pipeline (`EPaperHardware`, `EPaperDisplay`) aus `words_gpt.py`.
- Mehrsprachige Anreicherungspipeline in `words_data.py` (IPA, japanische Varianten, Arabisch, Franzoesisch, chinesische Felder).
- SQLite-basierte Persistenz mit Hilfsfunktionen fuer dynamische Feldaktualisierung in `words_database.py`.
- OpenAI-Helper fuer strukturierte JSON-Anfragen mit Dateicache in `openai_request_json.py`.
- Optionale PWA-Assets in `pwa/` fuer leichte Frontend-Konfigurations-/Vorschau-Workflows.

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

Wichtige Laufzeitdateien:
- `app.py`: Tornado-Webserver (Standardport `8082`) und periodische Update-Schleife.
- `words_gpt.py`: Standalone-Renderer-Schleife und Display-Klassen.
- `words_data.py`: Kern-Orchestrierung fuer Wortabruf und Anreicherung.
- `words_database.py`: SQLite-Hilfsfunktionen.
- `scripts/*.sh`: Raspberry-Pi-Setup, Service-Installation und tmux-Lifecycle-Skripte.

## Voraussetzungen
- Python `3.9+` (empfohlen).
- Raspberry Pi als Zielplattform (fuer Hardwaremodus).
- Unterstuetztes Waveshare-E-Paper-Panel.
- SPI auf dem Pi aktiviert (`raspi-config`) sowie panel-spezifische Verkabelung.

In diesem Projekt verwendete Python-Pakete:
- `openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.
- Das Setup-Skript installiert zusaetzlich: `json5`, `pandas`, `spidev`, `RPi.GPIO`, `gpiozero`, `lgpio`.

## Installation

### Option A: Minimale/manuelle Installation
Waveshare-Treiberpaket installieren:
```bash
python setup.py install
```

Wenn du die NLTK-Wortliste nutzt, einmalig herunterladen:
```bash
python -m nltk.downloader words
```

### Option B: Automatisches Raspberry-Pi-Setup (empfohlen auf dem Geraet)
Vom Repository-Root:
```bash
bash scripts/setup_pi_wordscard.sh
```

Dieses Skript:
- Installiert apt-Abhaengigkeiten.
- Stellt sicher, dass SPI aktiviert ist.
- Erstellt und aktiviert die virtuelle Umgebung `wordscard`.
- Installiert Python-Laufzeitabhaengigkeiten.
- Installiert das Waveshare-Paket.
- Startet `app.py` in einer tmux-Session.

## Konfiguration

### Verhalten von `.env`
Dieses Repository laedt Umgebungsvariablen beim Import aus `.env` und **ueberschreibt** vorhandene Shell-Werte. Dadurch bleiben lokale Overrides deterministisch, auch wenn Variablen bereits in Shell-Profilen exportiert sind.

`.env` erstellen oder aktualisieren:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### App-Argument-Passthrough
Die systemd-/tmux-Skripte unterstuetzen:
```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### CLI-Flags (Server und Renderer)
Sowohl `app.py` als auch `words_gpt.py` unterstuetzen:
- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

## Verwendung

### HTTP-Server starten
Service starten (Standardport `8082`):
```bash
python app.py
```

In Code beobachtete Routen:
- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (aus `words_card_temp/`)

Kompatibilitaetshinweis: Fruehere Dokus nennen `GET /current_word`; die aktuelle Route in `app.py` ist `GET /get_current_word`.

### Standalone-Renderer ausfuehren
CSV-basierte Liste:
```bash
python words_gpt.py --use_csv
```

OpenAI aktivieren:
```bash
python words_gpt.py --enable_openai --use_csv
```

Emoji-Rendering + vereinfachtes CJK:
```bash
python words_gpt.py --make_emoji --simplify
```

### Service-Modus auf Raspberry Pi
Service-Unit installieren:
```bash
bash scripts/install_wordscard_service.sh
```

Dann:
```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## Beispiele

### Naechstes Zufallswort ausloesen
```bash
curl "http://127.0.0.1:8082/next_random_word"
```

### Aktuelle Wort-Payload lesen
```bash
curl "http://127.0.0.1:8082/get_current_word"
```

### Explizites Wort senden
```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

### Hardware-Smoke-Tests
Panel-spezifisches Skript verwenden:
```bash
python epd_7in3f_test.py
```

Oder:
```bash
python epd_13in3k_test.py
```

Weitere Beispiele liegen in `waveshare/examples/`.

## Daten, Cache und Logs
| Bereich | Pfad(e) | Hinweise |
|---|---|---|
| Wortlisten | `data/` | Enthaelt `data/words_list.csv` und thematische CSV-Dateien |
| Persistente DB | `words_phonetics.db` | Lokaler Speicher fuer Phonetik/Anreicherung |
| OpenAI-/Cache-Artefakte | `cache/` | Reduziert wiederholte Anfragen |
| Logs | `logs/`, `logs-word-phonetics/` | Laufzeit- und Update-Logs |
| Generierte Karten | `words_card_temp/` | Bildausgaben und Quelle fuer statische Auslieferung |

## Hinweise zur Entwicklung
- Abhaengigkeitsverwaltung ist skriptzentriert (`scripts/setup_pi_wordscard.sh`) + `setup.py`; `requirements.txt` oder `pyproject.toml` fehlen derzeit.
- Es gibt mehrere Backup-/Legacy-Dateien (`words_data_*`, `words_gpt_old.py`); der aktive Laufzeitpfad ist hauptsaechlich `app.py` + `words_gpt.py` + `words_data.py` + `words_database.py`.
- `env_loader.py` ueberschreibt Umgebungsvariablen aus `.env` immer, wenn Schluessel vorhanden sind.
- Im Server-Modus laeuft ein periodischer Aktualisierungsfluss (ca. alle 5 Minuten), der den Update-Endpunkt intern aufrufen kann.

## Fehlerbehebung
- `ModuleNotFoundError` oder Import-Probleme:
  - Sicherstellen, dass die virtuelle Umgebung aktiv ist und Abhaengigkeiten installiert sind.
  - Auf dem Pi `bash scripts/setup_pi_wordscard.sh` erneut ausfuehren.
- OpenAI-Fehler (`401`, fehlendes Modell/Key):
  - `OPENAI_API_KEY` und optional `OPENAI_MODEL` in `.env` pruefen.
  - Netzwerkverbindung vom Geraet bestaetigen.
- Display aktualisiert sich nicht:
  - Panel-Modell/Verkabelung pruefen und passendes Testskript ausfuehren (`epd_7in3f_test.py` oder `epd_13in3k_test.py`).
  - Sicherstellen, dass SPI aktiviert ist (`sudo raspi-config nonint do_spi 0`).
  - Auf Pi 5 ggf. Kompatibilitaets-Symlink fuer `/dev/spidev0.0` setzen, falls das Geraet `/dev/spidev10.0` bereitstellt.
- OpenCC-Installationsprobleme:
  - Distributionskompatibles Paket verwenden (`libopencc1` oder `libopencc2`) wie im Setup-Skript.
- API-Routen-Mismatch:
  - Fuer aktuelle Payload `/get_current_word` verwenden, nicht `/current_word`.

## Hinweise zur OpenAI-Nutzung
Der OpenAI-Zugriff ist optional, aber fuer frische Wortgenerierung und phonetische Anreicherung empfohlen. Der strukturierte JSON-Helper in `openai_request_json.py` cached Ergebnisse unter `cache/`, um wiederholte Aufrufe zu reduzieren.

## Roadmap
- Formales Abhaengigkeits-Manifest (`requirements.txt` oder `pyproject.toml`) fuer reproduzierbare Installationen hinzufuegen.
- `i18n/` um gepflegte uebersetzte README-Varianten erweitern.
- Legacy-/Backup-Skriptvarianten konsolidieren, sobald der kanonische Ablauf final ist.
- PWA-Workflow (`pwa/`) mit Endpunkt-Beispielen und Screenshots dokumentieren.
- Reproduzierbare automatisierte Tests fuer Daten- und Routenverhalten hinzufuegen.

## ❤️ Support

Wenn dieses Projekt fuer dich nuetzlich ist, unterstuetzen diese Links direkt die laufende Wartung und Hardware-Iteration.

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

### Was deine Unterstuetzung ermoeglicht
- <b>Tools offen halten</b>: Hosting, Inferenz, Datenspeicherung und Community-Betrieb.  
- <b>Schneller liefern</b>: fokussierte Open-Source-Zeit fuer WordsCardEink und verwandte Lernwerkzeuge.  
- <b>Geraete prototypisieren</b>: E-Ink-Hardware-Iterationen und Forschung zu Display-Layouts.  
- <b>Zugang fuer alle</b>: gefoerderte Deployments fuer Studierende, Kreative und Community-Gruppen.

### Spenden

<div align="center">
<table style="margin:0 auto; text-align:center; border-collapse:collapse;">
  <tr>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://chat.lazying.art/donate">https://chat.lazying.art/donate</a>
    </td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://chat.lazying.art/donate"><img src="figs/donate_button.svg" alt="Spenden" height="44"></a>
    </td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://paypal.me/RongzhouChen">
        <img src="https://img.shields.io/badge/PayPal-Donate-003087?logo=paypal&logoColor=white" alt="Mit PayPal spenden">
      </a>
    </td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400">
        <img src="https://img.shields.io/badge/Stripe-Donate-635bff?logo=stripe&logoColor=white" alt="Mit Stripe spenden">
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

## Mitwirken
Siehe `AGENTS.md` fuer Richtlinien zu Beitragsarbeit, Coding-Style und PR-Erwartungen.

Empfohlene Checkliste fuer Beitraege:
- Bei Display-Aenderungen Panel-Modell + Hardware-Hinweise angeben.
- Exakt ausgefuehrte Validierungsbefehle auflisten.
- Screenshots/Fotos fuer UI- oder E-Ink-Ausgabeaenderungen anhaengen.
- Datensatz-Aenderungen beschreiben (Datei + Zeilen-/Spaltenauswirkung).

## Lizenz
Im Repository-Root ist derzeit keine `LICENSE`-Datei vorhanden (in diesem Draft-Durchlauf beobachtet). Bis eine Lizenzdatei hinzugefuegt wird, sind Wiederverwendungsrechte nicht explizit gewaehrt.

Annahme: Maintainer koennen in einem Folge-Update eine explizite Open-Source-Lizenz hinzufuegen.
