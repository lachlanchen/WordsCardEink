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

Un proyecto de Raspberry Pi + Waveshare e-ink que renderiza tarjetas de vocabulario seleccionadas dinámicamente con fonética IPA y pistas multilingües. Admite flujos locales de CSV, enriquecimiento opcional con IA, renderizado en e-paper y control remoto por HTTP.

| 🔎 Resumen rápido | Detalles |
|---|---|
| Núcleo de ejecución | `app.py` (servicio HTTP) + `words_gpt.py` (bucle de renderizado) |
| Ruta de datos | Conjuntos de datos CSV en `data/` + almacén SQLite `words_phonetics.db` |
| Destinos de salida | Pantallas e-paper de Waveshare y salidas de imagen virtuales |
| Dependencia de IA | Opcional (`--enable_openai`) con caché en `cache/` |
| Valores por defecto del bucle principal | Servidor en `8082`, actualización periódica aproximada de 5 minutos |

## 📚 Tabla de contenidos
- [Resumen general](#resumen-general)
- [Características destacadas](#características-destacadas)
- [Demostraciones](#demostraciones)
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
- [Soporte](#soporte)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

---

## Resumen general

`words_gpt` es una pila de generación de tarjetas de vocabulario en Python para pantallas e-ink. Combina orquestación de datos, enriquecimiento fonético y orquestación de renderizado en dos modos de ejecución:

- Un servicio Tornado de larga duración (`app.py`) para control remoto y servir imágenes
- Un renderizador autónomo (`words_gpt.py`) que puede ejecutarse en modos de sondeo, bucle o renderizado directo

Módulos principales:

- `words_data.py` / `words_data_utils.py` para flujos de palabras y enriquecimiento
- `words_database.py` para interacción con SQLite
- `openai_request_json.py` para solicitudes estructuradas de OpenAI con caché en disco
- `env_loader.py` para carga determinística de entorno
- `words_update.py` para mantenimiento de BD y flujos de revisión
- `app.py` y `words_gpt.py` para el ciclo de vida del servicio/renderizado

## Características destacadas

- Canal de renderizado e-ink con múltiples idiomas/modos de contenido:
  - Variantes japonesas, modo kanji, árabe, chino, modo emoji
- Origen de palabras local y con OpenAI en un único flujo
- Salida china simplificada opcional en la ruta de renderizado
- Endpoints del servidor para interacción directa (`/next_random_word`, `/display_word`, etc.)
- Caché y persistencia que reducen llamadas repetidas a la red
- Activos opcionales de PWA en `pwa/` para flujos ligeros de vista previa/configuración

## Demostraciones

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

Archivos de runtime importantes:

- `app.py`: app Tornado en el puerto `8082` + disparador periódico `next_random_word`.
- `words_gpt.py`: renderizador autónomo y abstracciones de pantalla (`EPaperHardware`, `EPaperDisplay`).
- `words_data.py`: flujo avanzado de obtención/enriquecimiento y utilidades auxiliares.
- `words_database.py`: helpers de SQLite para metadatos almacenados y operaciones de caché de palabras.
- `scripts/*.sh`: instalación y ciclo de vida de servicio/arranque en Raspberry Pi.

## Requisitos previos

- Python `3.9+` (recomendado)
- Raspberry Pi (requerido para modo hardware)
- Panel e-paper compatible de Waveshare (por ejemplo familia 7.3F / 13K)
- SPI habilitado (`raspi-config`), cableado correcto y alimentación estable
- NLTK disponible cuando se usen fuentes de palabras con `nltk`

Dependencias comunes observadas en el código:
`openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.

## Instalación

### Opción 1 — Instalación mínima/manual (escritorio o Pi)

Desde la raíz del repositorio:

```bash
python setup.py install
```

Si es necesario:

```bash
python -m nltk.downloader words
```

### Opción 2 — Configuración automática en Raspberry Pi (recomendada en el dispositivo)

Desde la raíz del repositorio:

```bash
bash scripts/setup_pi_wordscard.sh
```

Esto ejecuta:

- dependencias específicas de Pi
- habilitación de SPI
- configuración del entorno virtual `wordscard`
- instalación de paquetes Python/runtime
- instalación del paquete de Waveshare
- arranque de `app` en `tmux`

### Opción 3 — Instalación del servicio

Para registrar el ciclo de vida de la app con `systemd`:

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

El repositorio usa carga `.env` que actualmente sobrescribe variables de entorno preexistentes. Úsalo de forma intencionada:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### Flags de runtime (usados por `app.py` y `words_gpt.py`)

- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

Los scripts de inicio en Pi soportan pasaje de argumentos mediante `APP_ARGS` (ejemplo):

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### Comportamiento de rutas en modo app

Rutas observadas en el código actual:

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (sirve desde `words_card_temp/`)

Nota de compatibilidad: la documentación antigua hacía referencia a `GET /current_word`; la ruta actual es `GET /get_current_word`.

### Notas sobre uso de OpenAI

Las funciones de OpenAI son opcionales y se controlan con flags de CLI/entorno. Los payloads cacheados de la API son útiles para reproducibilidad y control de límites de tasa. En entornos con restricciones, ejecuta primero en modo CSV (`--use_csv`) y habilita OpenAI de forma selectiva (`--enable_openai`) cuando se necesite enriquecimiento.

## Uso

### Ejecutar servidor HTTP

```bash
python app.py
```

El proceso mantiene una imagen en `words_card_temp/` y expone endpoints HTTP usados por herramientas front-end o scripts simples.

### Ejecutar renderizador directamente

Modo CSV:

```bash
python words_gpt.py --use_csv
```

Modo OpenAI:

```bash
python words_gpt.py --enable_openai --use_csv
```

Emoji + CJK simplificado:

```bash
python words_gpt.py --make_emoji --simplify
```

### Ejecutar en hardware Raspberry Pi

- Iniciar vía script de `tmux`:

```bash
bash scripts/start_wordscard.sh
```

- Detener vía script de `tmux`:

```bash
bash scripts/stop_wordscard.sh
```

## Ejemplos

Obtener metadatos de la siguiente palabra aleatoria:

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

Obtener la palabra almacenada actualmente:

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

Solicitar payload de imagen de página renderizada:

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

- `data/`: conjuntos de datos CSV curados
- `words_phonetics.db`: caché/base de datos de origen SQLite
- `cache/`: caché de solicitudes/resultados de OpenAI
- `word_phonetics_processed.csv`: conjunto de datos procesado/derivado
- `logs/`, `logs-word-phonetics/`: registros de ejecución
- `words_card_temp/`: tarjetas generadas y salida temporal

## Notas de desarrollo

- Existen archivos heredados/respaldo (por ejemplo `words_gpt_old.py`, `lib.old`), así que trátalos como referencias salvo que estés migrando o manteniendo compatibilidad explícitamente.
- `words_update.py` contiene helpers de refresco/revisión por lotes útiles para un pase de calidad de datos.
- La validación de hardware la manejan `epd_*_test.py` y demos de `waveshare/examples/*`.
- No existe `requirements.txt` ni archivo de lock en la raíz; la configuración de dependencias se realiza mediante el script de instalación o instalación directa.
- No hay una suite de tests automatizada configurada en este repositorio.

## Solución de problemas

- `ImportError` desde módulos Raspberry Pi GPIO/SPI:
  - Instálalo con la ruta de Pi (`setup_pi_wordscard.sh`), o verifica `python setup.py install` en un objetivo compatible.
- `403/404` desde endpoints de imagen/estáticos:
  - Confirma el uso del endpoint `/get_current_word*` y que `words_card_temp/` tenga permisos de escritura.
- Carga de payload de palabra vacía/inválida desde modo OpenAI:
  - Confirma que `OPENAI_API_KEY` y los valores opcionales de org/model se cargan correctamente; revisa `cache/` y los logs.
- Mal renderizado/corte de texto:
  - Verifica la ruta de fuentes y la configuración de resolución del panel dentro de `words_gpt.py`.
- La API devuelve datos obsoletos:
  - Llama `POST /next_random_word` manualmente y revisa el intervalo de callback periódico en `app.py`.
- La actualización de hardware parece congelada:
  - Revisa la sesión tmux y los logs de systemd (`journalctl -u wordscard`).
- Falta un conjunto de datos o entradas de diccionario:
  - Valida los archivos CSV en `data/` y ejecuta flujos de `words_update.py` para refresco/limpieza.

## Hoja de ruta

- Añadir un `requirements.txt` mínimo / manifiesto reproducible de instalación.
- Añadir modos de ejecución más claros y documentación explícita de CLI con `--help`.
- Ampliar la documentación del esquema de renderizado para cada modo de contenido (`japanese_synonym`, `arabic_synonym`, `film`, etc.).
- Estandarizar el manejo de errores y los esquemas de respuesta de API orientados a usuario final.
- Añadir stubs de scripts de smoke test simples para validación CI sin hardware.

## Soporte

| Opción de soporte | Enlace | Propósito |
|---|---|---|
| GitHub Sponsors | https://github.com/sponsors/lachlanchen | Soporte continuo y donaciones únicas |
| Lazying Art | https://lazying.art | Marca y recursos relacionados |
| Chat | https://chat.lazying.art | Discusión y soporte |
| Only Ideas | https://onlyideas.art | Investigación creativa y proyectos paralelos |

## Contribuciones

Las contribuciones son bienvenidas. Flujo sugerido:

1. Mantén los cambios acotados a un área de comportamiento (render, datos, API, scripts).
2. Actualiza el uso de comandos y documentación para cambios en comportamiento visible al usuario.
3. Conserva la compatibilidad de flags CLI y endpoints existentes cuando sea posible.
4. Si cambian scripts de hardware, documenta el dispositivo/modelo probado y los comandos exactos ejecutados.

## Licencia

No hay archivo `LICENSE` en la raíz actual del repositorio. Por tanto, la licencia efectiva está indefinida en este borrador. Añade una si quieres términos explícitos de redistribución y reutilización.
