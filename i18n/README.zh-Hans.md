[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 🖨️ Eink Words GPT

**本草稿语言：** 简体中文

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

> 一个基于 Raspberry Pi + Waveshare e-ink 的项目，渲染支持 IPA 音标和多语言提示的动态词汇卡片。它支持本地 CSV 流程、可选的 AI 增强、e-paper 渲染以及远程 HTTP 控制。

运行模式一眼看懂：
`app.py`（服务端）和 `words_gpt.py`（独立渲染器）可以单独运行，也可以配合工作。

| 🔎 一览 | 详情 |
|---|---|
| **核心运行时** | `app.py`（HTTP 服务） + `words_gpt.py`（渲染循环） |
| **数据路径** | `data/` 中的 CSV 数据集 + SQLite 存储 `words_phonetics.db` |
| **输出目标** | Waveshare 电子纸面板与虚拟图片输出 |
| **AI 依赖** | 可选（`--enable_openai`），`cache/` 中有请求缓存 |
| **默认节奏** | 服务器端口 `8082`，约每 5 分钟自动刷新一次 |

## 📚 目录
- 概览
- 亮点
- 演示
- 项目结构
- 前置条件
- 安装
- 配置
- 使用方法
- 示例
- 数据、缓存与日志
- 开发说明
- 故障排查
- 计划路线
- 支持
- 贡献
- 许可

---

## 概览

`words_gpt` 是一个用于 e-ink 显示屏的 Python 词汇卡片生成系统。它在两个运行模式下整合了数据编排、音标增强与渲染：

- 长期运行的 Tornado 服务（`app.py`）用于远程控制和图片服务。
- 独立渲染器（`words_gpt.py`）可在轮询、循环或直接渲染模式下运行。

核心模块：

- `words_data.py` / `words_data_utils.py`：词汇与增强数据流程。
- `words_data_with_legacy.py`、`words_data_without_legacy.py`、`words_data_workable*.py`：变体流程与旧版兼容。
- `words_database.py`：SQLite 交互。
- `openai_request_json.py`：带磁盘缓存与重试行为的结构化 OpenAI 请求。
- `env_loader.py`：确定性的环境变量加载。
- `words_update.py`：数据库维护与复查工作流。
- `app.py` 与 `words_gpt.py`：服务与渲染生命周期。
- `pwa/`：轻量浏览器预览和配置工具。

## 亮点

- 支持多语言模式的电子纸渲染流水线：日语变体、汉字模式、阿拉伯语、中文与 emoji 模式。
- 单一工作流下支持本地和 OpenAI 两种词汇来源。
- 可选的简体中文渲染（`--simplify`）。
- 提供远程控制 HTTP 接口（`/next_random_word`、`/display_word`、`/get_current_word`、`/get_current_word_page`、`/get_words_card`）。
- 通过缓存与持久化减少重复 AI 请求。
- 通过内置的 `waveshare/` 驱动目录提供封装与硬件示例。

## 演示

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## 项目结构

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

关键运行文件：

- `app.py`：端口 `8082` 的 Tornado 应用 + 定期 `next_random_word` 触发。
- `words_gpt.py`：独立渲染器与显示抽象（`EPaperHardware`、`EPaperDisplay`）。
- `words_data.py`：抓取/增强流程和词条选择器。
- `words_database.py`：用于存储元数据和词条缓存操作的 SQLite 辅助函数。
- `scripts/*.sh`：安装与服务生命周期，以及 Raspberry Pi 启动脚本。
- `words_update.py`：用于数据库质量维护的批量刷新/复查工具。

## 前置条件

- Python `3.9+`（推荐）
- Raspberry Pi（硬件模式必需）
- 支持的 Waveshare e-paper 面板（例如 7.3F 或 13K 系列）
- 已启用 SPI（`raspi-config`）、布线正确且供电稳定
- 使用 NLTK 词源时需要准备 NLTK 语料库

运行时常见依赖项（项目内引用）：
`openai`、`tornado`、`Pillow`、`numpy`、`nltk`、`opencc`、`pykakasi`、`arabic_reshaper`、`python-bidi`、`pytz`。

## 安装

### 方式一：最小/手动安装（桌面或 Pi）

在仓库根目录执行：

```bash
python setup.py install
```

如需要：

```bash
python -m nltk.downloader words
```

### 方式二：Raspberry Pi 自动化安装（推荐在设备上运行）

在仓库根目录执行：

```bash
bash scripts/setup_pi_wordscard.sh
```

该流程会完成：

- Raspberry Pi 专用依赖安装
- SPI 启用与检查
- `wordscard` 虚拟环境设置
- Python /运行时包安装
- Waveshare 包安装
- 使用 `tmux` 启动 app 进程

### 方式三：systemd 服务安装

要在 `systemd` 下注册应用生命周期：

```bash
bash scripts/install_wordscard_service.sh
```

然后执行：

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## 配置

### 环境变量（`.env`）

`env_loader` 在进程启动上下文读取环境变量并应用配置。目前文档和运行实践通常包含：

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

建议：将密钥保留在本地环境配置中，不要提交到版本库。

### 运行参数（`app.py` 与 `words_gpt.py`）

| CLI 参数 | 用途 |
| --- | --- |
| `--enable_openai` | 启用可选的 OpenAI 增强模式 |
| `--make_emoji` | 渲染以 emoji 为重点的卡片 |
| `--ignore_list` | 跳过在忽略列表中的词 |
| `--use_csv` | 从 CSV 数据集中读取词条 |
| `--complete_csv` | 使用完整 CSV 源模式 |
| `--filename <csv_file>` | 指定特定 CSV 输入文件 |

`APP_ARGS` 可通过启动脚本透传，例如：

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### app 模式路由行为

已观测到的路由：

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)`（来自 `words_card_temp/`）

兼容性说明：旧文档可能提到 `GET /current_word`；当前路由为 `GET /get_current_word`。

### OpenAI 使用说明

OpenAI 功能为可选，并由 CLI/环境变量控制。`cache/` 中的请求缓存有助于提高可复现性并控制速率。若要优先离线优先运行，可先用 CSV 模式（`--use_csv`），再按需启用 OpenAI。

## 使用方法

### 启动 HTTP 服务器

```bash
python app.py
```

进程会在 `words_card_temp/` 中保留最新图片，并通过端点供前端工具或脚本使用。

### 直接运行渲染器

CSV 模式：

```bash
python words_gpt.py --use_csv
```

OpenAI 模式：

```bash
python words_gpt.py --enable_openai --use_csv
```

emoji + 简化中文模式：

```bash
python words_gpt.py --make_emoji --simplify
```

### 在 Pi 上运行硬件

- 通过 tmux 脚本启动：

```bash
bash scripts/start_wordscard.sh
```

- 通过 tmux 脚本停止：

```bash
bash scripts/stop_wordscard.sh
```

## 示例

获取下一张随机卡片元数据：

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

获取当前存储词条：

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

获取当前页面图片负载：

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

提交指定词条：

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

通过表单风格端点触发卡片渲染：

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

## 数据、缓存与日志

应用常见文件：

- `data/`：筛选后的 CSV 数据集
- `words_phonetics.db`：SQLite 缓存/源数据库
- `cache/`：OpenAI 请求/响应缓存
- `word_phonetics_processed.csv`：已处理/衍生数据集
- `logs/`、`logs-word-phonetics/`：运行日志
- `words_card_temp/`：生成的卡片和临时输出
- `pic/` 与 `figs/`：参考图像和横幅

## 开发说明

- 项目内存在旧版/备用模块与产物（如 `words_gpt_old.py`、`lib.old`），除非你有意迁移或兼容性需求，请将其视为参考。
- `words_update.py` 包含用于数据库质量维护的批量刷新/复查工具。
- 硬件验证由 `epd_*_test.py` 与 `waveshare/examples/*` 示例脚本完成。
- 仓库根目录没有 `requirements.txt` 或 lockfile；依赖由安装脚本和直接安装流程决定。
- 仓库中未配置自动化测试套件。

## 故障排查

- Raspberry Pi GPIO/SPI 模块抛出 `ImportError`：
  - 使用 Pi 安装流程（`scripts/setup_pi_wordscard.sh`）或在兼容目标上显式安装依赖。
- 图片/静态端点返回 `403/404`：
  - 确认路由用法（`/get_current_word*`）并确认 `words_card_temp/` 可写。
- OpenAI 模式返回空或无效词条：
  - 检查 `OPENAI_API_KEY`（以及 org/model）是否加载，再检查 `cache/` 与日志。
- 渲染异常或文字截断：
  - 检查 `words_gpt.py` 中字体路径、显示分辨率常量和所选模式设置。
- API 返回过期数据：
  - 手动调用 `POST /next_random_word`，并检查 `app.py` 中的回调间隔。
- 硬件渲染看似冻结：
  - 检查 tmux 会话和 `journalctl -u wordscard` 系统日志。
- 数据集/词典条目缺失：
  - 校验 `data/` 中的 CSV 文件，并执行 `words_update.py` 维护任务。

## 计划路线

- 增加精简的 `requirements.txt` / 可复现安装清单。
- 提供更清晰的运行模式说明和完整的 CLI `--help` 文档。
- 扩展渲染模式说明（`japanese_synonym`、`arabic_synonym`、`film` 等工作流）。
- 标准化错误处理和面向用户的 API 返回格式。
- 新增轻量级 smoke test 模板，支持无硬件的 CI 验证。

## 贡献

欢迎贡献，建议流程：

1. 将改动聚焦到单一行为域（渲染、数据、API、脚本）。
2. 对面向用户的行为变化更新命令与文档。
3. 在可行时保留现有 CLI 参数和端点兼容性。
4. 若修改硬件脚本，请记录测试设备/型号及执行的精确命令。

## 许可

当前仓库根目录未提供 `LICENSE` 文件，因此本仓库树内的授权定义尚未明确。若你需要明确的再发布/复用条款，请补充授权文件。


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
