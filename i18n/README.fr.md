[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Eink Words GPT

**Options de langue :** Français (cette version)

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=flat-square&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=flat-square)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=flat-square)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=flat-square&logo=openai&logoColor=white)

Un projet Raspberry Pi + e-ink Waveshare qui affiche du vocabulaire sélectionné dynamiquement, avec phonétique et synonymes multilingues. Le système peut récupérer des mots depuis des jeux de données locaux ou OpenAI, les rendre dans une mise en page, puis envoyer le résultat vers des panneaux e-paper compatibles. Il expose aussi un petit service HTTP pour déclencher des mises à jour de mots et récupérer les images générées.

| 🔎 En bref | Détails |
|---|---|
| Runtime principal | `app.py` (service HTTP) + `words_gpt.py` (boucle de rendu) |
| Chemin des données | Jeux de données CSV dans `data/` + base SQLite `words_phonetics.db` |
| Cibles de sortie | Panneaux e-paper Waveshare et sorties d’images virtuelles |
| Dépendance IA | Optionnelle (`--enable_openai`) avec cache dans `cache/` |

## 📚 Table des matières
- [Vue d’ensemble](#vue-densemble)
- [Points forts](#points-forts)
- [Démarrage rapide](#démarrage-rapide)
- [Démos](#démos)
- [Fonctionnalités](#fonctionnalités)
- [Structure du projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Exemples](#exemples)
- [Données, cache et logs](#données-cache-et-logs)
- [Notes de développement](#notes-de-développement)
- [Dépannage](#dépannage)
- [Notes sur l’utilisation d’OpenAI](#notes-sur-lutilisation-dopenai)
- [Feuille de route](#feuille-de-route)
- [Support](#-support)
- [Contribution](#contribution)
- [Licence](#licence)

## Vue d’ensemble
`words_gpt` est un système Python de génération et d’affichage de cartes de vocabulaire pour appareils e-ink.

Il combine :
- L’approvisionnement en mots depuis des jeux de données CSV/locaux et, en option, la génération via OpenAI.
- L’enrichissement (phonétique IPA + champs de synonymes multilingues).
- Des pipelines de rendu pour sorties matérielles et virtuelles.
- Un service HTTP Tornado pour les déclenchements à distance et la récupération d’images.

Le code actuel s’articule autour de `app.py`, `words_gpt.py`, `words_data.py`, `words_database.py` et `openai_request_json.py`.

## Points forts
- 🖼️ Pipeline de rendu e-ink avec plusieurs modes de contenu (kanji, japonais, arabe, chinois, emoji).
- 🗃️ Base de données locale de mots (`words_phonetics.db`) avec listes de mots basées sur CSV dans `data/`.
- 🤖 Sélection de mots via OpenAI et enrichissement phonétique avec sorties JSON structurées.
- 🌐 Service HTTP pour déclencheurs externes et récupération d’images.
- ⚡ Couche de cache (`cache/`) pour réduire les appels OpenAI répétés.

## Démarrage rapide
| Objectif | Commande |
|---|---|
| Démarrer le serveur HTTP (port `8082`) | `python app.py` |
| Lancer le moteur autonome (CSV) | `python words_gpt.py --use_csv` |
| Lancer avec OpenAI + CSV | `python words_gpt.py --enable_openai --use_csv` |
| Mode emoji + CJK simplifié | `python words_gpt.py --make_emoji --simplify` |
| Configuration automatique Raspberry Pi | `bash scripts/setup_pi_wordscard.sh` |

## Démos
<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## Fonctionnalités
- Flux de rendu matériel + virtuel (`EPaperHardware`, `EPaperDisplay`) depuis `words_gpt.py`.
- Pipeline d’enrichissement multilingue dans `words_data.py` (IPA, variantes japonaises, arabe, français, champs chinois).
- Persistance SQLite avec helpers de mise à jour dynamique des champs dans `words_database.py`.
- Helper de requêtes JSON structurées OpenAI avec cache fichier dans `openai_request_json.py`.
- Ressources PWA optionnelles dans `pwa/` pour des workflows légers de configuration/aperçu frontend.

## Structure du projet
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

Fichiers d’exécution importants :
- `app.py` : serveur web Tornado (port par défaut `8082`) et boucle de mise à jour périodique.
- `words_gpt.py` : boucle de rendu autonome et classes d’affichage.
- `words_data.py` : orchestration principale de récupération/enrichissement des mots.
- `words_database.py` : helpers de stockage SQLite.
- `scripts/*.sh` : configuration Raspberry Pi, installation de service et scripts de cycle de vie tmux.

## Prérequis
- Python `3.9+` (recommandé).
- Cible Raspberry Pi (pour le mode matériel).
- Panneau e-paper Waveshare compatible.
- SPI activé sur le Pi (`raspi-config`), avec câblage spécifique au panneau.

Les paquets Python utilisés dans ce projet incluent :
- `openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.
- Le script de setup installe aussi : `json5`, `pandas`, `spidev`, `RPi.GPIO`, `gpiozero`, `lgpio`.

## Installation

### Option A : Installation minimale/manuelle
Installer le paquet de driver Waveshare :
```bash
python setup.py install
```

Si vous utilisez la liste de mots NLTK, téléchargez-la une fois :
```bash
python -m nltk.downloader words
```

### Option B : Configuration automatique Raspberry Pi (recommandée sur l’appareil)
Depuis la racine du dépôt :
```bash
bash scripts/setup_pi_wordscard.sh
```

Ce script :
- Installe les dépendances apt.
- Vérifie que SPI est activé.
- Crée et active l’environnement virtuel `wordscard`.
- Installe les dépendances Python d’exécution.
- Installe le paquet Waveshare.
- Démarre `app.py` dans une session tmux.

## Configuration

### Comportement de `.env`
Ce dépôt charge les variables d’environnement depuis `.env` au moment de l’import et **écrase** toute valeur shell existante. Cela rend les surcharges locales déterministes, même si des valeurs sont déjà exportées dans vos profils shell.

Créez ou mettez à jour `.env` :
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### Transmission d’arguments d’app
Les scripts systemd/tmux prennent en charge :
```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### Flags CLI (serveur et moteur de rendu)
`app.py` et `words_gpt.py` prennent tous deux en charge :
- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

## Utilisation

### Exécuter le serveur HTTP
Démarrer le service (port par défaut `8082`) :
```bash
python app.py
```

Routes observées dans le code :
- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (depuis `words_card_temp/`)

Note de compatibilité : les documents plus anciens faisaient référence à `GET /current_word`; la route actuelle dans `app.py` est `GET /get_current_word`.

### Exécuter le moteur de rendu autonome
Liste basée sur CSV :
```bash
python words_gpt.py --use_csv
```

Activer OpenAI :
```bash
python words_gpt.py --enable_openai --use_csv
```

Rendu emoji + CJK simplifié :
```bash
python words_gpt.py --make_emoji --simplify
```

### Mode service sur Raspberry Pi
Installer l’unité de service :
```bash
bash scripts/install_wordscard_service.sh
```

Puis :
```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## Exemples

### Déclencher le prochain mot aléatoire
```bash
curl "http://127.0.0.1:8082/next_random_word"
```

### Lire la charge utile du mot courant
```bash
curl "http://127.0.0.1:8082/get_current_word"
```

### Soumettre un mot explicite
```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

### Tests rapides matériel
Utilisez le script spécifique à l’écran :
```bash
python epd_7in3f_test.py
```

Ou :
```bash
python epd_13in3k_test.py
```

D’autres exemples se trouvent dans `waveshare/examples/`.

## Données, cache et logs
| Zone | Chemin(s) | Notes |
|---|---|---|
| Listes de mots | `data/` | Inclut `data/words_list.csv` et des fichiers CSV thématiques |
| Base persistante | `words_phonetics.db` | Stockage local de phonétique/enrichissement |
| Artefacts OpenAI/cache | `cache/` | Réduit les requêtes répétées |
| Logs | `logs/`, `logs-word-phonetics/` | Logs d’exécution et de mise à jour |
| Cartes générées | `words_card_temp/` | Sorties image et source de service statique |

## Notes de développement
- La gestion des dépendances est orientée scripts (`scripts/setup_pi_wordscard.sh`) + `setup.py` ; il n’y a pas encore de `requirements.txt` ni de `pyproject.toml`.
- Plusieurs fichiers de sauvegarde/legacy existent (`words_data_*`, `words_gpt_old.py`) ; le chemin d’exécution actif est principalement `app.py` + `words_gpt.py` + `words_data.py` + `words_database.py`.
- `env_loader.py` écrase toujours les variables d’environnement depuis `.env` lorsque les clés sont présentes.
- Le mode serveur exécute un flux de rafraîchissement périodique (toutes les ~5 minutes) qui peut appeler le endpoint de mise à jour en interne.

## Dépannage
- `ModuleNotFoundError` ou problèmes d’import :
  - Vérifiez que l’environnement virtuel est actif et que les dépendances sont installées.
  - Relancez `bash scripts/setup_pi_wordscard.sh` sur le Pi.
- Erreurs OpenAI (`401`, modèle/clé manquant) :
  - Vérifiez `OPENAI_API_KEY` et éventuellement `OPENAI_MODEL` dans `.env`.
  - Confirmez la connectivité réseau depuis l’appareil.
- L’affichage ne se met pas à jour :
  - Vérifiez le modèle/câblage du panneau et exécutez le script de test correspondant (`epd_7in3f_test.py` ou `epd_13in3k_test.py`).
  - Confirmez que SPI est activé (`sudo raspi-config nonint do_spi 0`).
  - Sur Pi 5, assurez la compatibilité du lien symbolique `/dev/spidev0.0` si l’appareil expose `/dev/spidev10.0`.
- Problèmes d’installation OpenCC :
  - Utilisez un paquet compatible avec la distribution (`libopencc1` ou `libopencc2`) comme dans le script de setup.
- Incohérence de route API :
  - Utilisez `/get_current_word` pour la charge utile courante, pas `/current_word`.

## Notes sur l’utilisation d’OpenAI
L’accès OpenAI est optionnel mais recommandé pour la génération de nouveaux mots et l’enrichissement phonétique. Le helper JSON structuré dans `openai_request_json.py` met en cache les résultats dans `cache/` pour réduire les appels répétés.

## Feuille de route
- Ajouter un manifeste formel de dépendances (`requirements.txt` ou `pyproject.toml`) pour des installations reproductibles.
- Étendre `i18n/` avec des variantes traduites du README maintenues.
- Consolider les variantes de scripts legacy/sauvegarde après finalisation du flux canonique.
- Documenter le workflow PWA (`pwa/`) avec exemples d’endpoints et captures d’écran.
- Ajouter des tests automatisés reproductibles pour les données et le comportement au niveau des routes.

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

### Ce que votre soutien rend possible
- <b>Garder les outils ouverts</b> : hébergement, inférence, stockage des données et opérations communautaires.  
- <b>Livrer plus vite</b> : temps open source concentré sur WordsCardEink et des outils d’apprentissage associés.  
- <b>Prototyper des appareils</b> : itérations matérielles e-ink et recherche sur la mise en page d’affichage.  
- <b>Accès pour toutes et tous</b> : déploiements subventionnés pour étudiantes/étudiants, créateurs et groupes communautaires.

### Faire un don

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

## Contribution
Voir `AGENTS.md` pour les directives de contribution, le style de code et les attentes PR.

Checklist de contribution suggérée :
- Inclure le modèle de panneau + des notes matérielles pour les changements d’affichage.
- Lister les commandes exactes exécutées pour la validation.
- Joindre des captures/photos pour les changements d’UI ou de sortie e-ink.
- Décrire les modifications de jeux de données (fichier + impact ligne/colonne).

## Licence
Aucun fichier `LICENSE` n’est actuellement présent à la racine du dépôt (observé dans ce passage de brouillon). Tant qu’un fichier de licence n’est pas ajouté, les droits de réutilisation ne sont pas explicitement accordés.

Hypothèse : les mainteneurs pourront ajouter une licence open source explicite lors d’une mise à jour ultérieure.
