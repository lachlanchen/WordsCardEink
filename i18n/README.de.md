[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Eink Words GPT

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi-green)
![Display](https://img.shields.io/badge/display-Waveshare%20e--Paper-black)
![Status](https://img.shields.io/badge/status-active%20prototype-orange)
![Server](https://img.shields.io/badge/http-Tornado-0A7EA4)
![Storage](https://img.shields.io/badge/storage-SQLite-003B57)
![AI](https://img.shields.io/badge/OpenAI-optional-412991)

Ein Raspberry-Pi- plus Waveshare-E-Ink-Projekt, das dynamisch ausgewählten Wortschatz mit Phonetik und mehrsprachigen Synonymen anzeigt. Das System kann Wörter aus lokalen Datensätzen oder OpenAI beziehen, in ein Layout rendern und das Ergebnis auf unterstützte E-Paper-Panels ausgeben. Zusätzlich stellt es einen kleinen HTTP-Service bereit, um Wort-Updates auszulösen und gerenderte Bilder abzurufen.

## Überblick
`words_gpt` ist ein Python-basiertes System zur Erzeugung und Anzeige von Vokabelkarten auf E-Ink-Geräten.

Es kombiniert:
- Wortquellen aus CSV/lokalen Datensätzen und optionale OpenAI-Generierung.
- Anreicherung (IPA-Phonetik + mehrsprachige Synonymfelder).
- Rendering-Pipelines für Hardware- und virtuelle Ausgaben.
- Einen Tornado-HTTP-Service für Remote-Auslösung und Bildabruf.

Die aktuelle Codebasis konzentriert sich auf `app.py`, `words_gpt.py`, `words_data.py`, `words_database.py` und `openai_request_json.py`.

## Highlights
- 🖼️ E-Ink-Rendering-Pipeline mit mehreren Inhaltsmodi (Kanji, Japanisch, Arabisch, Chinesisch, Emoji).
- 🗃️ Lokale Wortdatenbank (`words_phonetics.db`) mit CSV-basierten Wortlisten in `data/`.
- 🤖 OpenAI-basierte Wortauswahl und phonetische Anreicherung mit strukturierten JSON-Ausgaben.
- 🌐 HTTP-Service für externe Trigger und Bildabruf.
- ⚡ Caching-Schicht (`cache/`) zur Reduzierung wiederholter OpenAI-Aufrufe.

## Schnellstart
| Ziel | Befehl |
|---|---|
| HTTP-Server starten (Port `8082`) | `python app.py` |
| Standalone-Renderer ausführen (CSV) | `python words_gpt.py --use_csv` |
| Mit OpenAI + CSV ausführen | `python words_gpt.py --enable_openai --use_csv` |
| Emoji + vereinfachter CJK-Modus | `python words_gpt.py --make_emoji --simplify` |
| Raspberry Pi Auto-Setup | `bash scripts/setup_pi_wordscard.sh` |

## Demos
<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabische Wortkarte" width="48%" />
</p>

## Funktionen
- Hardware- und virtuelles Rendering (`EPaperHardware`, `EPaperDisplay`) aus `words_gpt.py`.
- Mehrsprachige Anreicherungs-Pipeline in `words_data.py` (IPA, japanische Varianten, arabische, französische und chinesische Felder).
- SQLite-basierte Persistenz mit dynamischen Feld-Update-Helfern in `words_database.py`.
- OpenAI-Helper für strukturierte JSON-Anfragen mit Dateicache in `openai_request_json.py`.
- Optionale PWA-Assets in `pwa/` für schlanke Frontend-Konfigurations-/Vorschau-Workflows.

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
- `words_data.py`: Kern-Orchestrierung für Wortabruf und Anreicherung.
- `words_database.py`: SQLite-Store-Helfer.
- `scripts/*.sh`: Raspberry-Pi-Setup, Service-Installation und tmux-Lifecycle-Skripte.

## Voraussetzungen
- Python `3.9+` (empfohlen).
- Raspberry-Pi-Zielgerät (für Hardware-Modus).
- Unterstütztes Waveshare-E-Paper-Panel.
- Aktiviertes SPI auf dem Pi (`raspi-config`) plus panel-spezifische Verdrahtung.

In diesem Projekt verwendete Python-Pakete:
- `openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.
- Das Setup-Skript installiert zusätzlich: `json5`, `pandas`, `spidev`, `RPi.GPIO`, `gpiozero`, `lgpio`.

## Installation

### Option A: Minimale/manuelle Installation
Waveshare-Treiberpaket installieren:
```bash
python setup.py install
```

Wenn du die NLTK-Wortliste verwendest, einmalig herunterladen:
```bash
python -m nltk.downloader words
```

### Option B: Automatisches Raspberry-Pi-Setup (auf dem Gerät empfohlen)
Aus dem Repository-Root:
```bash
bash scripts/setup_pi_wordscard.sh
```

Dieses Skript:
- Installiert apt-Abhängigkeiten.
- Stellt sicher, dass SPI aktiviert ist.
- Erstellt und aktiviert die virtuelle Umgebung `wordscard`.
- Installiert Python-Laufzeitabhängigkeiten.
- Installiert das Waveshare-Paket.
- Startet `app.py` in einer tmux-Session.

## Konfiguration

### Verhalten von `.env`
Dieses Repository lädt Umgebungsvariablen beim Import aus `.env` und **überschreibt** vorhandene Shell-Werte. Dadurch bleiben lokale Overrides deterministisch, auch wenn Werte bereits in Shell-Profilen exportiert wurden.

`.env` erstellen oder aktualisieren:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### Weitergabe von App-Argumenten
Die systemd/tmux-Skripte unterstützen:
```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### CLI-Flags (Server und Renderer)
Sowohl `app.py` als auch `words_gpt.py` unterstützen:
- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

## Nutzung

### HTTP-Server ausführen
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

Kompatibilitätshinweis: In älteren Dokumenten wurde `GET /current_word` genannt; die aktuelle Route in `app.py` ist `GET /get_current_word`.

### Standalone-Renderer ausführen
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

### Nächstes zufälliges Wort auslösen
```bash
curl "http://127.0.0.1:8082/next_random_word"
```

### Aktuellen Wort-Payload lesen
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
Display-spezifisches Skript verwenden:
```bash
python epd_7in3f_test.py
```

Oder:
```bash
python epd_13in3k_test.py
```

Weitere Beispiele finden sich in `waveshare/examples/`.

## Daten, Cache und Logs
| Bereich | Pfad(e) | Hinweise |
|---|---|---|
| Wortlisten | `data/` | Enthält `data/words_list.csv` und thematische CSV-Dateien |
| Persistente DB | `words_phonetics.db` | Lokaler Store für Phonetik/Anreicherung |
| OpenAI/Cache-Artefakte | `cache/` | Reduziert wiederholte Anfragen |
| Logs | `logs/`, `logs-word-phonetics/` | Laufzeit- und Update-Logs |
| Generierte Karten | `words_card_temp/` | Bildausgaben und Quelle für statisches Serving |

## Entwicklungshinweise
- Abhängigkeitsverwaltung ist skriptzentriert (`scripts/setup_pi_wordscard.sh`) plus `setup.py`; es gibt noch keine `requirements.txt` oder `pyproject.toml`.
- Mehrere Backup-/Legacy-Dateien existieren (`words_data_*`, `words_gpt_old.py`); der aktive Laufzeitpfad ist primär `app.py` + `words_gpt.py` + `words_data.py` + `words_database.py`.
- `env_loader.py` überschreibt Umgebungsvariablen aus `.env` immer, wenn Schlüssel vorhanden sind.
- Der Server-Modus führt einen periodischen Aktualisierungsfluss (etwa alle 5 Minuten) aus, der den Update-Endpunkt intern aufrufen kann.

## Fehlerbehebung
- `ModuleNotFoundError` oder Import-Probleme:
  - Sicherstellen, dass die virtuelle Umgebung aktiv ist und Abhängigkeiten installiert sind.
  - Auf dem Pi `bash scripts/setup_pi_wordscard.sh` erneut ausführen.
- OpenAI-Fehler (`401`, fehlendes Modell/Key):
  - `OPENAI_API_KEY` und optional `OPENAI_MODEL` in `.env` prüfen.
  - Netzwerkverbindung des Geräts bestätigen.
- Display wird nicht aktualisiert:
  - Panel-Modell/Verdrahtung prüfen und passendes Testskript ausführen (`epd_7in3f_test.py` oder `epd_13in3k_test.py`).
  - Bestätigen, dass SPI aktiviert ist (`sudo raspi-config nonint do_spi 0`).
  - Auf Pi 5 sicherstellen, dass ein Kompatibilitäts-Symlink für `/dev/spidev0.0` existiert, wenn das Gerät `/dev/spidev10.0` bereitstellt.
- OpenCC-Installationsprobleme:
  - Distributionskompatibles Paket verwenden (`libopencc1` oder `libopencc2`), wie im Setup-Skript.
- API-Routen-Mismatch:
  - Für den aktuellen Payload `/get_current_word` statt `/current_word` verwenden.

## Hinweise zur OpenAI-Nutzung
Der OpenAI-Zugriff ist optional, aber für frische Wortgenerierung und phonetische Anreicherung empfehlenswert. Der strukturierte JSON-Helper in `openai_request_json.py` cached Ergebnisse unter `cache/`, um wiederholte Aufrufe zu reduzieren.

## Roadmap
- Formales Abhängigkeits-Manifest (`requirements.txt` oder `pyproject.toml`) für reproduzierbare Installationen hinzufügen.
- `i18n/` um gepflegte übersetzte README-Varianten erweitern.
- Legacy-/Backup-Skriptvarianten konsolidieren, sobald der kanonische Flow finalisiert ist.
- PWA-Workflow (`pwa/`) mit Endpunkt-Beispielen und Screenshots dokumentieren.
- Wiederholbare automatisierte Tests für Daten und Routenverhalten ergänzen.

## Support

### Was deine Unterstützung ermöglicht
- <b>Tools offen halten</b>: Hosting, Inferenz, Datenspeicherung und Community-Betrieb.  
- <b>Schneller liefern</b>: fokussierte Open-Source-Zeit für WordsCardEink und verwandte Lerntools.  
- <b>Geräte prototypisieren</b>: E-Ink-Hardware-Iterationen und Forschung zu Display-Layouts.  
- <b>Zugang für alle</b>: subventionierte Deployments für Lernende, Kreative und Community-Gruppen.

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

**支援 / Spenden**

- ご支援は研究・開発と運用の継続に役立ち、より多くのオープンなプロジェクトを皆さんに届ける力になります。  
- 你的支持将用于研发与运维，帮助我持续公开分享更多项目与改进。  
- Deine Unterstützung trägt Forschung, Entwicklung und Betrieb, damit ich weiterhin mehr offene Projekte und Verbesserungen teilen kann.

## Mitwirken
Siehe `AGENTS.md` für Mitwirkungsrichtlinien, Coding-Style und PR-Erwartungen.

Empfohlene Mitwirkungs-Checkliste:
- Panel-Modell und Hardware-Hinweise bei Display-Änderungen angeben.
- Exakte Validierungsbefehle auflisten.
- Screenshots/Fotos für UI- oder E-Ink-Ausgabeänderungen beifügen.
- Datensatzänderungen beschreiben (Datei + Zeilen-/Spaltenauswirkung).

## Lizenz
Im Repository-Root ist derzeit keine `LICENSE`-Datei vorhanden (in diesem Draft-Durchlauf beobachtet). Bis eine Lizenzdatei ergänzt wird, sind Wiederverwendungsrechte nicht ausdrücklich gewährt.

Annahme: Die Maintainer ergänzen möglicherweise in einem Folge-Update eine explizite Open-Source-Lizenz.
