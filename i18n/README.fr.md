[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 🖨️ Eink Words GPT

**Langue de ce draft:** Français

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

> Un projet Raspberry Pi + Waveshare e-ink qui rend des cartes de vocabulaire dynamiques avec la phonétique IPA et des indices multilingues. Il prend en charge les flux CSV locaux, l’enrichissement IA optionnel, le rendu sur e-paper et le contrôle HTTP à distance.

Vue d’ensemble des modes d’exécution :
`app.py` (service) et `words_gpt.py` (moteur autonome) peuvent être exécutés séparément ou ensemble.

| 🔎 En bref | Détails |
|---|---|
| **Runtime principal** | `app.py` (service HTTP) + `words_gpt.py` (boucle de rendu) |
| **Chemin de données** | jeux de données CSV dans `data/` + base SQLite `words_phonetics.db` |
| **Cibles de sortie** | panneaux e-paper Waveshare et sorties d’image virtuelles |
| **Dépendance IA** | optionnelle (`--enable_openai`) avec cache des requêtes dans `cache/` |
| **Fréquence par défaut** | serveur sur `8082`, rafraîchissement périodique d’environ 5 minutes |

## 📚 Table des matières
- [Aperçu](#aperçu)
- [Points forts](#points-forts)
- [Démonstrations](#démonstrations)
- [Structure du projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Exemples](#exemples)
- [Données, cache, et journaux](#données-cache-et-journaux)
- [Notes de développement](#notes-de-développement)
- [Dépannage](#dépannage)
- [Feuille de route](#feuille-de-route)
- [Support](#support)
- [Contribution](#contribution)
- [Licence](#licence)

---

## Aperçu

`words_gpt` est une pile de génération de cartes de vocabulaire Python pour écrans e-ink. Elle combine l’orchestration des données, l’enrichissement phonétique et le rendu via deux modes d’exécution :

- Un service Tornado de longue durée (`app.py`) pour le contrôle à distance et la diffusion d’images.
- Un moteur autonome (`words_gpt.py`) pouvant fonctionner en mode polling, boucle continue ou rendu direct.

Modules principaux :

- `words_data.py` / `words_data_utils.py` pour la gestion des mots et des enrichissements.
- `words_data_with_legacy.py`, `words_data_without_legacy.py`, `words_data_workable*.py` pour les flux alternatifs et la compatibilité legacy.
- `words_database.py` pour l’interaction SQLite.
- `openai_request_json.py` pour les requêtes structurées OpenAI avec cache disque et logique de retry.
- `env_loader.py` pour le chargement déterministe de la configuration d’environnement.
- `words_update.py` pour la maintenance BD et les flux de vérification.
- `app.py` et `words_gpt.py` pour le cycle de vie service/rendu.
- `pwa/` pour des outils légers de prévisualisation navigateur/configuration.

## Points forts

- Pipeline de rendu e-ink avec modes multilingues : variantes japonaises, mode kanji, arabe, chinois, et mode emoji.
- Source de mots locale et enrichissement via OpenAI dans un même flux.
- Rendu chinois simplifié optionnel (`--simplify`).
- Endpoints HTTP pour contrôle distant (`/next_random_word`, `/display_word`, `/get_current_word`, `/get_current_word_page`, `/get_words_card`).
- Mise en cache et persistance pour limiter les appels IA répétés.
- Packaging et exemples matériels via l’arborescence `waveshare/` incluse.

## Démonstrations

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## Structure du projet

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

Fichiers critiques en runtime :

- `app.py` : service Tornado sur le port `8082` + déclenchement périodique `next_random_word`.
- `words_gpt.py` : moteur autonome de rendu et abstractions d’affichage (`EPaperHardware`, `EPaperDisplay`).
- `words_data.py` : flux de récupération/enrichissement et utilitaires de sélection.
- `words_database.py` : utilitaires SQLite pour métadonnées stockées et opérations cache des mots.
- `scripts/*.sh` : installation/service lifecycle et scripts de bootstrap Raspberry Pi.
- `words_update.py` : helper de maintenance par lots pour la qualité de la base.

## Prérequis

- Python `3.9+` (recommandé)
- Raspberry Pi (obligatoire pour le mode matériel)
- Panneau e-paper Waveshare supporté (par ex. la famille 7.3F ou 13K)
- SPI activé (`raspi-config`), câblage correct, et alimentation stable
- Corpus NLTK lors de l’usage des sources de mots NLTK

Dépendances courantes présentes dans le dépôt et utilisées par les chemins d’exécution :
`openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.

## Installation

### Option 1 — Installation minimale/manuelle (desktop ou Pi)

Depuis la racine du dépôt :

```bash
python setup.py install
```

Si nécessaire :

```bash
python -m nltk.downloader words
```

### Option 2 — Installation automatisée Raspberry Pi (recommandée sur dispositif)

Depuis la racine du dépôt :

```bash
bash scripts/setup_pi_wordscard.sh
```

Ce script effectue :

- installation des dépendances spécifiques au Pi
- vérifications d’activation SPI
- configuration de l’environnement virtuel `wordscard`
- installation des paquets Python/runtime
- installation du package Waveshare
- lancement de l’application via `tmux`

### Option 3 — Installation du service systemd

Pour enregistrer le cycle de vie de l’app sous `systemd` :

```bash
bash scripts/install_wordscard_service.sh
```

Puis :

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## Configuration

### Variables d’environnement (`.env`)

`env_loader` lit les clés d’environnement et les applique au démarrage du processus. La documentation et l’usage runtime utilisent généralement :

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

Hypothèse : conservez les secrets dans la configuration locale de l’environnement et ne les commitez jamais dans le dépôt.

### Drapeaux runtime (utilisés par `app.py` et `words_gpt.py`)

| Drapeau CLI | Objectif |
| --- | --- |
| `--enable_openai` | Active le mode d’enrichissement OpenAI optionnel |
| `--make_emoji` | Rend des cartes centrées sur les emoji |
| `--ignore_list` | Ignore les mots des listes d’exclusion configurées |
| `--simplify` | Produit une sortie CJK simplifiée |
| `--use_csv` | Lit les mots depuis les datasets CSV |
| `--complete_csv` | Utilise le mode source CSV complet |
| `--filename <csv_file>` | Pointe vers un fichier CSV d’entrée spécifique |

`APP_ARGS` peut être transmis via les scripts de démarrage. Exemple :

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### Comportement des routes en mode app

Routes observées :

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (servi depuis `words_card_temp/`)

Note de compatibilité : les anciennes docs peuvent mentionner `GET /current_word`; la route courante est `GET /get_current_word`.

### Notes d’usage OpenAI

Les fonctionnalités OpenAI sont optionnelles et contrôlées par des drapeaux CLI/env. Les requêtes mises en cache dans `cache/` aident à la reproductibilité et à la maîtrise des limites de taux. Pour des exécutions offline-first déterministes, commencez par le mode CSV (`--use_csv`) et activez OpenAI de manière sélective.

## Utilisation

### Lancer le serveur HTTP

```bash
python app.py
```

Le processus conserve la dernière image dans `words_card_temp/` et expose des endpoints utilisés par des outils front-end ou scripts.

### Lancer le moteur directement

Mode CSV :

```bash
python words_gpt.py --use_csv
```

Mode OpenAI :

```bash
python words_gpt.py --enable_openai --use_csv
```

Mode emoji + CJK simplifié :

```bash
python words_gpt.py --make_emoji --simplify
```

### Exécuter sur matériel Pi

- Démarrer avec le script tmux :

```bash
bash scripts/start_wordscard.sh
```

- Arrêter avec le script tmux :

```bash
bash scripts/stop_wordscard.sh
```

## Exemples

Obtenir les métadonnées de la prochaine carte aléatoire :

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

Récupérer le mot actuellement stocké :

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

Demander la charge utile de l’image de la page rendue :

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

Soumettre un mot explicite :

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

Déclencher le rendu de carte via un endpoint de type formulaire :

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

## Données, cache, et journaux

Artefacts typiques utilisés par l’application :

- `data/` : jeux CSV curés
- `words_phonetics.db` : cache/source SQLite
- `cache/` : cache de requêtes/résultats OpenAI
- `word_phonetics_processed.csv` : dataset dérivé/traité
- `logs/`, `logs-word-phonetics/` : journaux d’exécution
- `words_card_temp/` : cartes générées et sorties temporaires
- `pic/` et `figs/` : images de référence et bannières

## Notes de développement

- Des modules/artefacts legacy existent (par exemple `words_gpt_old.py`, `lib.old`), donc considérez-les comme références sauf migration ou compatibilité intentionnelle.
- `words_update.py` contient des helpers de revalidation/batch pour la maintenance qualité de la base.
- La validation matériel est gérée par `epd_*_test.py` et les scripts de démonstration `waveshare/examples/*`.
- Il n’existe pas de `requirements.txt` ou de lockfile à la racine du dépôt ; les dépendances sont pilotées par les scripts d’installation et des flux d’installation directs.
- Aucun framework de tests automatisés n’est configuré dans ce dépôt.

## Dépannage

- `ImportError` depuis les modules GPIO/SPI du Raspberry Pi :
  - Utilisez le flux d’installation Pi (`scripts/setup_pi_wordscard.sh`) ou installez explicitement les dépendances sur une cible compatible.
- Erreurs `403/404` depuis les endpoints image/statique :
  - Vérifiez la convention de route (`/get_current_word*`) et que `words_card_temp/` est accessible en écriture.
- Réponse OpenAI vide/incorrecte :
  - Vérifiez `OPENAI_API_KEY` (et les valeurs org/modèle) puis inspectez `cache/` et les logs.
- Mauvais rendu ou texte rogné :
  - Vérifiez les chemins de polices, les constantes de résolution de l’écran, et les paramètres de mode sélectionnés dans `words_gpt.py`.
- L’API renvoie des données obsolètes :
  - Appelez manuellement `POST /next_random_word` et consultez l’intervalle de rappel dans `app.py`.
- Rendu matériel figé :
  - Vérifiez la session `tmux` et les logs système `journalctl -u wordscard`.
- Entrées ou dictionnaires manquants dans le dataset :
  - Validez les CSV dans `data/` et exécutez les tâches de maintenance `words_update.py`.

## Feuille de route

- Ajouter un `requirements.txt` / manifeste d’installation reproductible minimal.
- Clarifier les modes runtime et documenter explicitement le `--help` CLI.
- Étendre la documentation des modes de rendu (`japanese_synonym`, `arabic_synonym`, `film`, et autres flux).
- Standardiser la gestion d’erreurs et les schémas de réponses API côté utilisateur.
- Ajouter des tests smoke légers pour validation CI sans matériel.

## Contribution

Les contributions sont bienvenues. Flux suggéré :

1. Limitez chaque changement à un périmètre comportemental (rendu, données, API, scripts).
2. Mettez à jour l’usage des commandes/docs pour les évolutions visibles par l’utilisateur.
3. Préservez la compatibilité des drapeaux CLI et des endpoints quand c’est possible.
4. Si des scripts matériels changent, documentez le modèle de matériel et les commandes exactes exécutées.

## Licence

Aucun fichier `LICENSE` n’est présent à la racine actuelle du dépôt. La licence effective reste donc non définie en interne dans cette version. Merci d’en ajouter une si vous souhaitez des conditions explicites de redistribution et de réutilisation.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
