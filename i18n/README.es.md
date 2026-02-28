[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Eink Words GPT

**Opciones de idioma:** Español (este borrador)

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=flat-square&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=flat-square)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=flat-square)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=flat-square&logo=openai&logoColor=white)

Un proyecto de Raspberry Pi + e-ink de Waveshare que muestra vocabulario seleccionado dinámicamente con fonética y sinónimos multilingües. El sistema puede obtener palabras desde datasets locales o OpenAI, renderizarlas en un diseño y enviar el resultado a paneles e-paper compatibles. También expone un pequeño servicio HTTP para activar actualizaciones de palabras y recuperar imágenes renderizadas.

| 🔎 Resumen Rápido | Detalles |
|---|---|
| Entorno principal | `app.py` (servicio HTTP) + `words_gpt.py` (bucle de renderizado) |
| Ruta de datos | Datasets CSV en `data/` + base SQLite `words_phonetics.db` |
| Destinos de salida | Paneles e-paper Waveshare y salidas de imagen virtuales |
| Dependencia de IA | Opcional (`--enable_openai`) con caché en `cache/` |

## 📚 Tabla de Contenidos
- [Resumen](#resumen)
- [Aspectos destacados](#aspectos-destacados)
- [Inicio rápido](#inicio-rápido)
- [Demos](#demos)
- [Funciones](#funciones)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos previos](#requisitos-previos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Uso](#uso)
- [Ejemplos](#ejemplos)
- [Datos, caché y logs](#datos-caché-y-logs)
- [Notas de desarrollo](#notas-de-desarrollo)
- [Solución de problemas](#solución-de-problemas)
- [Notas sobre el uso de OpenAI](#notas-sobre-el-uso-de-openai)
- [Hoja de ruta](#hoja-de-ruta)
- [Support](#-support)
- [Contribuir](#contribuir)
- [Licencia](#licencia)

## Resumen
`words_gpt` es un sistema en Python para generar y mostrar tarjetas de vocabulario en dispositivos e-ink.

Combina:
- Obtención de palabras desde CSV/datasets locales y generación opcional con OpenAI.
- Enriquecimiento (fonética IPA + campos de sinónimos multilingües).
- Pipelines de renderizado para hardware y salidas virtuales.
- Un servicio HTTP con Tornado para activación remota y recuperación de imágenes.

La base de código actual se centra en `app.py`, `words_gpt.py`, `words_data.py`, `words_database.py` y `openai_request_json.py`.

## Aspectos destacados
- 🖼️ Pipeline de renderizado e-ink con múltiples modos de contenido (kanji, japonés, árabe, chino, emoji).
- 🗃️ Base de datos local de palabras (`words_phonetics.db`) con listas CSV respaldadas en `data/`.
- 🤖 Selección de palabras y enriquecimiento fonético con OpenAI usando salidas JSON estructuradas.
- 🌐 Servicio HTTP para disparadores externos y recuperación de imágenes.
- ⚡ Capa de caché (`cache/`) para reducir llamadas repetidas a OpenAI.

## Inicio rápido
| Objetivo | Comando |
|---|---|
| Iniciar servidor HTTP (puerto `8082`) | `python app.py` |
| Ejecutar renderizador independiente (CSV) | `python words_gpt.py --use_csv` |
| Ejecutar con OpenAI + CSV | `python words_gpt.py --enable_openai --use_csv` |
| Modo emoji + CJK simplificado | `python words_gpt.py --make_emoji --simplify` |
| Configuración automática de Raspberry Pi | `bash scripts/setup_pi_wordscard.sh` |

## Demos
<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Tarjeta de palabra en árabe" width="48%" />
</p>

## Funciones
- Flujo de renderizado para hardware + virtual (`EPaperHardware`, `EPaperDisplay`) desde `words_gpt.py`.
- Pipeline de enriquecimiento multilingüe en `words_data.py` (IPA, variantes japonesas, árabe, francés y campos chinos).
- Persistencia con SQLite y utilidades de actualización dinámica de campos en `words_database.py`.
- Helper de solicitudes JSON estructuradas a OpenAI con caché de archivos en `openai_request_json.py`.
- Recursos PWA opcionales en `pwa/` para flujos ligeros de configuración/preview frontend.

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

Archivos importantes en ejecución:
- `app.py`: servidor web Tornado (puerto por defecto `8082`) y bucle de actualización periódica.
- `words_gpt.py`: bucle de renderizado independiente y clases de pantalla.
- `words_data.py`: orquestación central para obtención/enriquecimiento de palabras.
- `words_database.py`: utilidades para el almacenamiento SQLite.
- `scripts/*.sh`: scripts de setup de Raspberry Pi, instalación del servicio y ciclo de vida con tmux.

## Requisitos previos
- Python `3.9+` (recomendado).
- Raspberry Pi objetivo (para modo hardware).
- Panel e-paper Waveshare compatible.
- SPI habilitado en la Pi (`raspi-config`), además del cableado específico del panel.

Los paquetes de Python usados en este proyecto incluyen:
- `openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.
- El script de setup también instala: `json5`, `pandas`, `spidev`, `RPi.GPIO`, `gpiozero`, `lgpio`.

## Instalación

### Opción A: instalación mínima/manual
Instala el paquete de drivers Waveshare:
```bash
python setup.py install
```

Si usas la lista de palabras de NLTK, descárgala una vez:
```bash
python -m nltk.downloader words
```

### Opción B: setup automatizado en Raspberry Pi (recomendado en el dispositivo)
Desde la raíz del repo:
```bash
bash scripts/setup_pi_wordscard.sh
```

Este script:
- Instala dependencias de apt.
- Garantiza que SPI esté habilitado.
- Crea y activa el entorno virtual `wordscard`.
- Instala dependencias de ejecución de Python.
- Instala el paquete Waveshare.
- Inicia `app.py` dentro de una sesión tmux.

## Configuración

### Comportamiento de `.env`
Este repositorio carga variables de entorno desde `.env` durante la importación y **sobrescribe** cualquier valor existente del shell. Esto hace que las anulaciones locales sean deterministas incluso si ya exportaste valores en perfiles de shell.

Crea o actualiza `.env`:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### Paso de argumentos a la app
Los scripts de systemd/tmux soportan:
```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### Flags de CLI (servidor y renderizador)
Tanto `app.py` como `words_gpt.py` soportan:
- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

## Uso

### Ejecutar el servidor HTTP
Inicia el servicio (puerto por defecto `8082`):
```bash
python app.py
```

Rutas observadas en el código:
- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (desde `words_card_temp/`)

Nota de compatibilidad: la documentación anterior hacía referencia a `GET /current_word`; la ruta actual en `app.py` es `GET /get_current_word`.

### Ejecutar el renderizador independiente
Lista basada en CSV:
```bash
python words_gpt.py --use_csv
```

Habilitar OpenAI:
```bash
python words_gpt.py --enable_openai --use_csv
```

Renderizado de emoji + CJK simplificado:
```bash
python words_gpt.py --make_emoji --simplify
```

### Modo servicio en Raspberry Pi
Instalar la unidad de servicio:
```bash
bash scripts/install_wordscard_service.sh
```

Luego:
```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## Ejemplos

### Activar la siguiente palabra aleatoria
```bash
curl "http://127.0.0.1:8082/next_random_word"
```

### Leer el payload de la palabra actual
```bash
curl "http://127.0.0.1:8082/get_current_word"
```

### Enviar una palabra explícita
```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

### Pruebas rápidas de hardware
Usa el script específico de pantalla:
```bash
python epd_7in3f_test.py
```

O:
```bash
python epd_13in3k_test.py
```

Hay más ejemplos en `waveshare/examples/`.

## Datos, caché y logs
| Área | Ruta(s) | Notas |
|---|---|---|
| Listas de palabras | `data/` | Incluye `data/words_list.csv` y archivos CSV temáticos |
| BD persistente | `words_phonetics.db` | Almacenamiento local de fonética/enriquecimiento |
| Artefactos OpenAI/caché | `cache/` | Reduce solicitudes repetidas |
| Logs | `logs/`, `logs-word-phonetics/` | Logs de ejecución y actualización |
| Tarjetas generadas | `words_card_temp/` | Salidas de imagen y origen para servido estático |

## Notas de desarrollo
- La gestión de dependencias está basada en scripts (`scripts/setup_pi_wordscard.sh`) + `setup.py`; todavía no existe `requirements.txt` ni `pyproject.toml`.
- Hay múltiples archivos legacy/de respaldo (`words_data_*`, `words_gpt_old.py`); la ruta activa de ejecución es principalmente `app.py` + `words_gpt.py` + `words_data.py` + `words_database.py`.
- `env_loader.py` siempre sobrescribe variables de entorno desde `.env` cuando las claves existen.
- El modo servidor ejecuta un flujo de refresco periódico (cada ~5 minutos) que puede llamar internamente al endpoint de actualización.

## Solución de problemas
- `ModuleNotFoundError` o problemas de importación:
  - Asegúrate de que el entorno virtual esté activo y las dependencias instaladas.
  - Vuelve a ejecutar `bash scripts/setup_pi_wordscard.sh` en la Pi.
- Errores de OpenAI (`401`, modelo/clave faltante):
  - Verifica `OPENAI_API_KEY` y opcionalmente `OPENAI_MODEL` en `.env`.
  - Confirma la conectividad de red desde el dispositivo.
- La pantalla no se actualiza:
  - Verifica el modelo/cableado del panel y ejecuta el script de prueba correspondiente (`epd_7in3f_test.py` o `epd_13in3k_test.py`).
  - Confirma que SPI esté habilitado (`sudo raspi-config nonint do_spi 0`).
  - En Pi 5, asegúrate del symlink de compatibilidad `/dev/spidev0.0` si el dispositivo expone `/dev/spidev10.0`.
- Problemas al instalar OpenCC:
  - Usa un paquete compatible con la distro (`libopencc1` o `libopencc2`) como en el script de setup.
- Desajuste en rutas de API:
  - Usa `/get_current_word` para el payload actual, no `/current_word`.

## Notas sobre el uso de OpenAI
El acceso a OpenAI es opcional, pero recomendable para generar palabras nuevas y enriquecer la fonética. El helper de JSON estructurado en `openai_request_json.py` guarda caché de resultados en `cache/` para reducir llamadas repetidas.

## Hoja de ruta
- Añadir un manifiesto formal de dependencias (`requirements.txt` o `pyproject.toml`) para instalaciones reproducibles.
- Ampliar `i18n/` con variantes mantenidas del README traducido.
- Consolidar variantes de scripts legacy/de respaldo una vez se finalice el flujo canónico.
- Documentar el flujo PWA (`pwa/`) con ejemplos de endpoints y capturas.
- Añadir pruebas automatizadas repetibles para datos y comportamiento a nivel de rutas.

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

### Lo que hace posible tu apoyo
- <b>Mantener herramientas abiertas</b>: hosting, inferencia, almacenamiento de datos y operaciones de comunidad.  
- <b>Publicar más rápido</b>: tiempo de código abierto enfocado en WordsCardEink y herramientas de aprendizaje relacionadas.  
- <b>Prototipar dispositivos</b>: iteraciones de hardware e-ink e investigación de layouts de pantalla.  
- <b>Acceso para todos</b>: despliegues subvencionados para estudiantes, creadores y grupos comunitarios.

### Donar

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

## Contribuir
Consulta `AGENTS.md` para las directrices de contribución, estilo de código y expectativas de PR.

Checklist sugerida para contribuir:
- Incluye modelo de panel + notas de hardware para cambios de pantalla.
- Enumera los comandos exactos ejecutados para validar.
- Adjunta capturas/fotos para cambios de UI o salida e-ink.
- Describe cambios en datasets (archivo + impacto en filas/columnas).

## Licencia
Actualmente no hay un archivo `LICENSE` en la raíz del repositorio (observado en este borrador). Hasta que se añada un archivo de licencia, los derechos de reutilización no están explícitamente concedidos.

Suposición: los maintainers pueden añadir una licencia open-source explícita en una actualización posterior.
