[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 🖨️ Eink Words GPT

**Idioma de este borrador:** Español

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

> Un proyecto de Raspberry Pi + Waveshare e-ink que renderiza tarjetas de vocabulario dinámicas con fonética IPA y pistas multilingües. Admite flujos locales con CSV, enriquecimiento de IA opcional, renderizado en e-paper y control HTTP remoto.

Vista rápida de modos:
`app.py` (servicio) y `words_gpt.py` (renderizador independiente) pueden ejecutarse de forma independiente o juntas.

| 🔎 A simple vista | Detalles |
|---|---|
| **Núcleo de ejecución** | `app.py` (servicio HTTP) + `words_gpt.py` (bucle de renderizado) |
| **Ruta de datos** | Conjuntos CSV en `data/` + base de datos SQLite `words_phonetics.db` |
| **Objetivos de salida** | Paneles Waveshare e-paper y salidas de imagen virtual |
| **Dependencia de IA** | Opcional (`--enable_openai`) con caché de peticiones en `cache/` |
| **Ritmo por defecto** | Servidor en `8082`, refresco periódico aproximado cada 5 minutos |

## 📚 Tabla de contenidos
- [Resumen general](#resumen-general)
- [Características destacadas](#características-destacadas)
- [Demos](#demos)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos previos](#requisitos-previos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Uso](#uso)
- [Ejemplos](#ejemplos)
- [Datos, caché y registros](#datos-caché-y-registros)
- [Notas de desarrollo](#notas-de-desarrollo)
- [Solución de problemas](#solución-de-problemas)
- [Hoja de ruta](#hoja-de-ruta)
- [Support](#support)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

---

## Resumen general

`words_gpt` es una pila en Python para generar tarjetas de vocabulario para pantallas de tinta electrónica. Combina orquestación de datos, enriquecimiento fonético y renderizado detrás de dos modos de ejecución:

- Un servicio Tornado de larga duración (`app.py`) para control remoto y servicio de imágenes.
- Un renderizador independiente (`words_gpt.py`) que puede ejecutarse en modo sondeo, en bucle o de render directo.

Módulos principales:

- `words_data.py` / `words_data_utils.py` para flujos de palabras y enriquecimiento.
- `words_data_with_legacy.py`, `words_data_without_legacy.py`, `words_data_workable*.py` para variantes de flujo y compatibilidad con legado.
- `words_database.py` para interacción con SQLite.
- `openai_request_json.py` para solicitudes estructuradas a OpenAI con caché en disco y reintento.
- `env_loader.py` para carga determinista de entorno.
- `words_update.py` para mantenimiento de BD y flujos de revalidación.
- `app.py` y `words_gpt.py` para el ciclo de vida de servicio/renderizado.
- `pwa/` para herramientas ligeras de previsualización/configuración en navegador.

## Características destacadas

- Pipeline de renderizado e-ink con modos multilingües: variantes japonesas, modo kanji, árabe, chino y emoji.
- Selección de palabras local y con OpenAI en el mismo flujo.
- Renderizado de chino simplificado opcional (`--simplify`).
- Endpoints HTTP para control remoto (`/next_random_word`, `/display_word`, `/get_current_word`, `/get_current_word_page`, `/get_words_card`).
- Caché y persistencia para reducir llamadas repetidas a IA.
- Empaquetado y ejemplos de hardware mediante el árbol `waveshare/`.

## Demos

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## Estructura del proyecto

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

Archivos de runtime importantes:

- `app.py`: aplicación Tornado en el puerto `8082` + trigger periódico `next_random_word`.
- `words_gpt.py`: renderizador independiente y abstracciones de pantalla (`EPaperHardware`, `EPaperDisplay`).
- `words_data.py`: flujo de obtención/enriquecimiento y utilidades de selección.
- `words_database.py`: helpers de SQLite para metadatos almacenados y operaciones de caché de palabras.
- `scripts/*.sh`: instalación y ciclo de vida del servicio (inicialización de Pi).
- `words_update.py`: helper de refresco por lotes/rechequeo para calidad de datos.

## Requisitos previos

- Python `3.9+` (recomendado)
- Raspberry Pi (obligatorio para modo hardware)
- Panel Waveshare e-paper compatible (por ejemplo familia 7.3F o 13K)
- SPI habilitado (`raspi-config`), cableado correcto y alimentación estable
- NLTK corpus al usar fuentes de palabras NLTK

Dependencias comunes en el árbol y usadas por rutas de ejecución:
`openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.

## Instalación

### Opción 1 — Instalación manual/mínima (escritorio o Raspberry Pi)

Desde la raíz del repositorio:

```bash
python setup.py install
```

Si es necesario:

```bash
python -m nltk.downloader words
```

### Opción 2 — Configuración automática en Raspberry Pi (recomendada en dispositivo)

Desde la raíz del repositorio:

```bash
bash scripts/setup_pi_wordscard.sh
```

Esto realiza:

- Instalación específica para Pi
- Comprobaciones de habilitación de SPI
- Configuración del entorno virtual `wordscard`
- Instalación de paquetes de Python/runtime
- Instalación del paquete Waveshare
- Lanzamiento de la app por `tmux`

### Opción 3 — Instalación de servicio systemd

Para registrar el ciclo de vida de la app bajo `systemd`:

```bash
bash scripts/install_wordscard_service.sh
```

Luego:

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## Configuración

### Variables de entorno (`.env`)

`env_loader` carga claves de entorno y las aplica en el contexto de arranque del proceso. La documentación y uso actuales suelen incluir:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

Asunción: mantén secretos en configuración local del entorno y no los subas al control de versiones.

### Flags de ejecución (usados por `app.py` y `words_gpt.py`)

| Flag CLI | Propósito |
| --- | --- |
| `--enable_openai` | Habilita el modo opcional de enriquecimiento con OpenAI |
| `--make_emoji` | Renderiza tarjetas enfocadas en emoji |
| `--ignore_list` | Omite palabras de listas de exclusión configuradas |
| `--simplify` | Genera salida CJK simplificada |
| `--use_csv` | Lee palabras desde datasets CSV |
| `--complete_csv` | Usa el modo CSV completo |
| `--filename <csv_file>` | Apunta a un archivo CSV concreto |

`APP_ARGS` se puede pasar desde scripts de arranque. Ejemplo:

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### Comportamiento de rutas en modo app

Rutas observadas:

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (se sirve desde `words_card_temp/`)

Nota de compatibilidad: documentación antigua puede mencionar `GET /current_word`; la ruta actual es `GET /get_current_word`.

### Notas de uso de OpenAI

Las funciones de OpenAI son opcionales y se controlan por flags CLI/variables de entorno. Las solicitudes cacheadas en `cache/` ayudan con reproducibilidad y límites de tasa. Para ejecuciones offline-first deterministas, empieza en modo CSV (`--use_csv`) y activa OpenAI de forma selectiva.

## Uso

### Ejecutar servidor HTTP

```bash
python app.py
```

El proceso mantiene la imagen más reciente en `words_card_temp/` y expone endpoints usados por herramientas front-end o scripts.

### Ejecutar renderizador directamente

Modo CSV:

```bash
python words_gpt.py --use_csv
```

Modo OpenAI:

```bash
python words_gpt.py --enable_openai --use_csv
```

Modo emoji + CJK simplificado:

```bash
python words_gpt.py --make_emoji --simplify
```

### Ejecutar en hardware Raspberry Pi

- Inicia con script de tmux:

```bash
bash scripts/start_wordscard.sh
```

- Detén con script de tmux:

```bash
bash scripts/stop_wordscard.sh
```

## Ejemplos

Obtener metadatos de la siguiente palabra aleatoria:

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

Obtener palabra almacenada actualmente:

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

Solicitar el payload de imagen de la página renderizada:

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

Enviar una palabra explícita:

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

Disparar renderizado de tarjeta mediante endpoint tipo formulario:

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

## Datos, caché y registros

Artefactos típicos usados por la app:

- `data/`: datasets CSV curados
- `words_phonetics.db`: caché/base de datos fuente SQLite
- `cache/`: caché de solicitudes/resultados de OpenAI
- `word_phonetics_processed.csv`: dataset procesado/derivado
- `logs/`, `logs-word-phonetics/`: registros de ejecución
- `words_card_temp/`: tarjetas generadas y salida temporal
- `pic/` y `figs/`: imágenes de referencia y banners

## Notas de desarrollo

- Existen módulos y artefactos legacy/backup (`words_gpt_old.py`, `lib.old`, etc.), así que trátalos como referencia a menos que la migración o compatibilidad sea explícita.
- `words_update.py` incluye helpers de refresco/rechequeo por lotes útiles para mantenimiento de calidad de la base de datos.
- La validación de hardware la cubren `epd_*_test.py` y los scripts demo de `waveshare/examples/*`.
- No hay `requirements.txt` ni lockfile en la raíz del repositorio; las dependencias se gestionan mediante scripts de instalación y flujo directo.
- No hay suite de tests automatizada configurada en este repositorio.

## Solución de problemas

- `ImportError` desde módulos Raspberry Pi GPIO/SPI:
  - Usa el flujo de configuración de Pi (`scripts/setup_pi_wordscard.sh`) o instala explícitamente dependencias en un objetivo compatible.
- `403/404` desde endpoints de imagen/estáticos:
  - Confirma el uso de rutas (`/get_current_word*`) y que `words_card_temp/` tenga permisos de escritura.
- Payload de palabra vacío o inválido desde modo OpenAI:
  - Verifica que `OPENAI_API_KEY` (y los valores opcionales de org/modelo) estén cargados, luego inspecciona `cache/` y los logs.
- Renderizado defectuoso o texto recortado:
  - Verifica rutas de fuentes, constantes de resolución del display y configuraciones de modo dentro de `words_gpt.py`.
- API devuelve datos obsoletos:
  - Llama manualmente `POST /next_random_word` y revisa el intervalo de callback periódico en `app.py`.
- El renderizado de hardware parece congelado:
  - Revisa la sesión tmux y logs de systemd con `journalctl -u wordscard`.
- Faltan entradas de dataset/diccionario:
  - Valida los CSV en `data/` y ejecuta tareas de mantenimiento en `words_update.py`.

## Hoja de ruta

- Añadir un `requirements.txt` mínimo / manifiesto reproducible de instalación.
- Añadir modos de ejecución más claros y documentación explícita de `--help` en CLI.
- Ampliar documentación de modos de renderizado (`japanese_synonym`, `arabic_synonym`, `film`, y otros flujos).
- Estandarizar manejo de errores y esquemas de respuestas de API orientados al usuario final.
- Añadir stubs de smoke test ligeros para validación CI sin hardware.

## Contribuciones

Las contribuciones son bienvenidas. Flujo sugerido:

1. Mantén los cambios acotados a una sola área de comportamiento (renderizado, datos, API, scripts).
2. Actualiza el uso de comandos y documentación para cambios de comportamiento visible al usuario.
3. Conserva la compatibilidad de `flags` CLI y endpoints existentes cuando sea posible.
4. Si cambias scripts de hardware, documenta dispositivo/modelo probado y comandos exactos ejecutados.

## Licencia

No existe archivo `LICENSE` en la raíz actual del repositorio. La licencia efectiva, por tanto, no está definida en el árbol actual de este borrador. Añádela si deseas términos explícitos de redistribución y reutilización.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
