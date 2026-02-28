[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 🖨️ Eink Words GPT

**Langue de ce brouillon :** Français

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=for-the-badge&logo=githubactions&logoColor=white)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=for-the-badge&logo=python&logoColor=white)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=for-the-badge&logo=openai&logoColor=white)

Projet Raspberry Pi + Waveshare e-ink qui génère dynamiquement des cartes de vocabulaire avec phonétique IPA et indices multilingues. Il prend en charge des flux CSV locaux, un enrichissement IA optionnel, le rendu e-paper et le contrôle HTTP à distance.

| 🔎 Vue d'ensemble | Détails |
|---|---|
| Runtime principal | `app.py` (service HTTP) + `words_gpt.py` (boucle de rendu) |
| Chemin des données | jeux CSV dans `data/` + base SQLite `words_phonetics.db` |
| Cibles de sortie | panneaux e-paper Waveshare et sorties d'images virtuelles |
| Dépendance IA | Optionnelle (`--enable_openai`) avec cache dans `cache/` |
| Valeurs par défaut de la boucle principale | serveur sur `8082`, rafraîchissement périodique d'environ 5 minutes |

## 📚 Table des matières
- [Vue d'ensemble](#vue-densemble)
- [Points forts](#points-forts)
- [Démonstrations](#démonstrations)
- [Structure du projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Exemples](#exemples)
- [Données, cache et journaux](#données-cache-et-journaux)
- [Notes de développement](#notes-de-développement)
- [Dépannage](#dépannage)
- [Feuille de route](#feuille-de-route)
- [Support](#support)
- [Contribuer](#contribuer)
- [Licence](#licence)

---

## Vue d'ensemble

`words_gpt` est une pile de génération de cartes de vocabulaire Python pour écrans e-ink. Elle combine orchestration des données, enrichissement phonétique et orchestration du rendu derrière deux modes d'exécution :

- Un service Tornado en long-running (`app.py`) pour le contrôle distant et le service d’images
- Un moteur autonome (`words_gpt.py`) qui peut tourner en mode polling, boucle ou rendu direct

Modules principaux :

- `words_data.py` / `words_data_utils.py` pour les flux de mots et d’enrichissement
- `words_database.py` pour l’interaction SQLite
- `openai_request_json.py` pour les requêtes OpenAI structurées avec cache disque
- `env_loader.py` pour le chargement déterministe des variables d’environnement
- `words_update.py` pour la maintenance BD et les flux de vérification
- `app.py` et `words_gpt.py` pour le cycle de vie service/rendu

## Points forts

- Pipeline de rendu e-ink avec plusieurs modes de langue/contenu :
  - variantes japonaises, mode kanji, arabe, chinois, mode emoji
- Approvisionnement de mots en local et via OpenAI dans un même flux
- Sortie chinoise simplifiée optionnelle dans le chemin de rendu
- Endpoints serveur pour interactions directes (`/next_random_word`, `/display_word`, etc.)
- Cache et persistance qui réduisent les appels réseau répétés
- Ressources PWA optionnelles dans `pwa/` pour des flux de prévisualisation/config légers

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

Fichiers d'exécution importants :

- `app.py` : Tornado sur le port `8082` + déclencheur `next_random_word` périodique.
- `words_gpt.py` : moteur de rendu autonome et abstractions d’affichage (`EPaperHardware`, `EPaperDisplay`).
- `words_data.py` : workflow avancé de récupération/enrichissement et utilitaires associés.
- `words_database.py` : helpers SQLite pour métadonnées stockées et opérations de cache de mots.
- `scripts/*.sh` : scripts de cycle de vie install/service et bootstrap Raspberry Pi.

## Prérequis

- Python `3.9+` (recommandé)
- Raspberry Pi (requis pour le mode matériel)
- Panneau e-paper Waveshare compatible (ex. familles 7.3F / 13K)
- SPI activé (`raspi-config`), câblage correct et alimentation stable
- Corpus NLTK disponible quand vous utilisez les sources de mots `nltk`

Dépendances couramment observées dans le dépôt :
`openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.

## Installation

### Option 1 — Installation minimale/manuelle (bureau ou Pi)

Depuis la racine du dépôt :

```bash
python setup.py install
```

Si nécessaire :

```bash
python -m nltk.downloader words
```

### Option 2 — Installation automatisée Raspberry Pi (recommandée sur l'appareil)

Depuis la racine du dépôt :

```bash
bash scripts/setup_pi_wordscard.sh
```

Ceci effectue :

- dépendances spécifiques au Pi
- activation SPI
- configuration de l'environnement virtuel `wordscard`
- installation des paquets Python/runtime
- installation du package Waveshare
- lancement de l'application via `tmux`

### Option 3 — Installation du service

Pour enregistrer le cycle de vie de l'application avec `systemd` :

```bash
bash scripts/install_wordscard_service.sh
```

Puis :

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordcard -n 100 --no-pager
```

## Configuration

### Variables d’environnement (`.env`)

Le dépôt utilise un chargement `.env` qui écrase actuellement les variables de shell existantes. Utilisez ce comportement volontairement :

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### Flags runtime (utilisés par `app.py` et `words_gpt.py`)

- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

Les scripts Pi supportent la transmission d’arguments via `APP_ARGS` (exemple) :

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### Comportement de routage en mode app

Routes observées dans le code courant :

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (servi depuis `words_card_temp/`)

Note de compatibilité : la documentation antérieure mentionnait `GET /current_word` ; la route courante est `GET /get_current_word`.

### Remarques sur l’usage d’OpenAI

Les fonctionnalités OpenAI sont optionnelles et contrôlées par les flags CLI/environnement. Les charges utiles API mises en cache sont utiles pour la reproductibilité et le contrôle de débit. En environnement contraint, exécutez d’abord en mode CSV (`--use_csv`) et activez OpenAI sélectivement (`--enable_openai`) quand l’enrichissement est désiré.

## Utilisation

### Lancer le serveur HTTP

```bash
python app.py
```

Le processus maintient une image dans `words_card_temp/` et expose des endpoints HTTP utilisés par des outils front-end ou de simples scripts.

### Lancer le moteur de rendu directement

Mode CSV :

```bash
python words_gpt.py --use_csv
```

Mode OpenAI :

```bash
python words_gpt.py --enable_openai --use_csv
```

Emoji + CJK simplifié :

```bash
python words_gpt.py --make_emoji --simplify
```

### Exécuter sur un Pi matériel

- Démarrage via le script `tmux` :

```bash
bash scripts/start_wordscard.sh
```

- Arrêt via le script `tmux` :

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

Demander le payload de l’image de page rendue :

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

Soumettre un mot explicite :

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

Déclencher le rendu via un endpoint de formulaire :

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

## Données, cache et journaux

Artefacts typiques utilisés par l'application :

- `data/` : jeux de données CSV curés
- `words_phonetics.db` : cache/source SQLite
- `cache/` : cache des requêtes/résultats OpenAI
- `word_phonetics_processed.csv` : jeu de données traité/dérivé
- `logs/`, `logs-word-phonetics/` : logs d'exécution
- `words_card_temp/` : cartes générées et sorties temporaires

## Notes de développement

- Des fichiers legacy/sauvegarde existent (par exemple `words_gpt_old.py`, `lib.old`), traitez-les comme références sauf si vous migrez spécifiquement ou maintenez la compatibilité.
- `words_update.py` contient des helpers de rafraîchissement/revérification en lot utiles pour un passage qualité des données BD.
- La validation matérielle est gérée par `epd_*_test.py` et les démos `waveshare/examples/*`.
- Il n’y a pas de `requirements.txt` ni de lockfile à la racine du dépôt ; l’installation des dépendances se fait via le script de setup ou une installation directe.
- Aucun jeu de tests automatisé n’est configuré dans ce repo.

## Dépannage

- `ImportError` provenant des modules GPIO/SPI Raspberry Pi :
  - Installez via le chemin Pi (`setup_pi_wordscard.sh`), ou vérifiez `python setup.py install` sur une cible compatible.
- `403/404` sur les endpoints image/static :
  - Vérifiez l’usage des endpoints `/get_current_word*` et que `words_card_temp/` est accessible en écriture.
- Payload OpenAI vide/ invalide :
  - Vérifiez que `OPENAI_API_KEY` et les valeurs org/modèle optionnelles sont chargées ; consultez `cache/` et les logs.
- Mauvais rendu/clipping de texte :
  - Vérifiez le chemin des polices et les réglages de résolution du panneau dans la chaîne de rendu de `words_gpt.py`.
- L'API renvoie des données obsolètes :
  - Appelez manuellement `POST /next_random_word` et revoyez l’intervalle de rappel périodique dans `app.py`.
- La mise à jour matérielle semble gelée :
  - Vérifiez la session tmux et les logs systemd (`journalctl -u wordscard`).
- Jeu de données ou entrées de dictionnaire manquants :
  - Validez les fichiers CSV dans `data/` et exécutez les workflows `words_update.py` pour rafraîchir/nettoyer.

## Feuille de route

- Ajouter un `requirements.txt` / manifeste d'installation reproductible minimal.
- Ajouter des modes d'exécution plus clairs et une documentation CLI `--help` explicite.
- Étoffer la documentation du schéma de rendu pour chaque mode de contenu (`japanese_synonym`, `arabic_synonym`, `film`, etc.).
- Standardiser la gestion d’erreurs et les schémas de réponses API côté utilisateur.
- Ajouter des scripts de smoke-test pour validation CI non-hardware.

## Support

| Option de support | Lien | Objectif |
|---|---|---|
| GitHub Sponsors | https://github.com/sponsors/lachlanchen | Soutien de projet récurrent et ponctuel |
| Lazying Art | https://lazying.art | Marque et ressources associées |
| Chat | https://chat.lazying.art | Discussion et support |
| Only Ideas | https://onlyideas.art | Recherche créative et projets annexes |

## Contribuer

Les contributions sont bienvenues. Flux suggéré :

1. Concentrez les changements sur un seul domaine fonctionnel (rendu, données, API, scripts).
2. Mettez à jour l’usage des commandes/docs pour les changements visibles par l’utilisateur.
3. Préservez la compatibilité des flags CLI et des endpoints tant que possible.
4. Si les scripts matériels changent, documentez le périphérique/modèle testé et les commandes exactes exécutées.

## Licence

Aucun fichier `LICENSE` n’est présent dans le dépôt courant. La licence effective est donc, dans cet état, non définie dans l’arbre. Ajoutez-en une si vous voulez des conditions explicites de redistribution/reprise.
