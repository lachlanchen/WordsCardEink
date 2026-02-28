[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 🖨️ Eink Words GPT

**语言版本：** 中文（简体）

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=for-the-badge&logo=githubactions&logoColor=white)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=for-the-badge&logo=python&logoColor=white)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=for-the-badge&logo=openai&logoColor=white)

Raspberry Pi + Waveshare 电子墨水屏项目，可动态渲染带 IPA 音标和多语言提示的词汇卡片。它支持本地 CSV 工作流、可选 AI 增强、电子纸渲染以及远程 HTTP 控制。

| 🔎 一览 | 详情 |
|---|---|
| 核心运行时 | `app.py`（HTTP 服务） + `words_gpt.py`（渲染循环） |
| 数据路径 | `data/` 中的 CSV 数据集 + SQLite 存储 `words_phonetics.db` |
| 输出目标 | Waveshare 电子纸面板与虚拟图像输出 |
| AI 依赖 | 可选（`--enable_openai`），缓存位于 `cache/` |
| 主循环默认 | 服务器端口 `8082`，约每 5 分钟刷新一次 |

## 📚 目录
- [概览](#overview)
- [亮点](#highlights)
- [演示](#demos)
- [项目结构](#project-structure)
- [先决条件](#prerequisites)
- [安装](#installation)
- [配置](#configuration)
- [使用方法](#usage)
- [示例](#examples)
- [数据、缓存与日志](#data-cache-and-logs)
- [开发说明](#development-notes)
- [故障排查](#troubleshooting)
- [路线图](#roadmap)
- [支持](#support)
- [贡献](#contributing)
- [许可](#license)

---

<a id="overview"></a>
## 概览

`words_gpt` 是一个面向电子墨水显示器的 Python 词汇卡片生成系统。它将数据编排、音标增强和渲染调度整合在两个运行模式下：

- 长驻运行的 Tornado 服务（`app.py`），用于远程控制与图片服务
- 独立渲染器（`words_gpt.py`），支持轮询、循环或直接渲染模式

主要模块：

- `words_data.py` / `words_data_utils.py`：词条与增强工作流
- `words_database.py`：SQLite 交互
- `openai_request_json.py`：带磁盘缓存的结构化 OpenAI 请求
- `env_loader.py`：确定性的环境变量加载
- `words_update.py`：数据库维护与复检工作流
- `app.py` 与 `words_gpt.py`：服务与渲染生命周期

<a id="highlights"></a>
## 亮点

- 多语言/多内容模式的电子纸渲染流水线：
  - 日语变体、汉字模式、阿拉伯语、中文、emoji 模式
- 本地词库与 OpenAI 词源并行的单词来源
- 直接交互接口（`/next_random_word`、`/display_word` 等）
- 缓存与持久化，减少重复网络请求
- `pwa/` 下提供可选的 PWA 资源，用于轻量级预览/配置流程

<a id="demos"></a>
## 演示

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

<a id="project-structure"></a>
## 项目结构

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

关键运行文件：

- `app.py`：监听端口 `8082` 的 Tornado 应用，并周期触发 `next_random_word`
- `words_gpt.py`：独立渲染器与显示抽象（`EPaperHardware`、`EPaperDisplay`）
- `words_data.py`：高级取词/增强工作流和辅助工具
- `words_database.py`：用于已存元数据与词条缓存的 SQLite 工具
- `scripts/*.sh`：安装与服务生命周期、Raspberry Pi 引导脚本

<a id="prerequisites"></a>
## 先决条件

- Python `3.9+`（推荐）
- Raspberry Pi（硬件模式需要）
- 支持的 Waveshare 电子纸型号（例如 7.3F / 13K 系列）
- 已开启 SPI（`raspi-config`）、接线正确，并保持运行电源稳定
- 使用 `nltk` 词源时需准备 NLTK 语料

仓库中常见依赖（示例）：
`openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`。

<a id="installation"></a>
## 安装

### 方案 1 — 最小/手动安装（桌面或树莓派）

在仓库根目录执行：

```bash
python setup.py install
```

如需要：

```bash
python -m nltk.downloader words
```

### 方案 2 — Raspberry Pi 自动化安装（设备内推荐）

在仓库根目录执行：

```bash
bash scripts/setup_pi_wordscard.sh
```

该脚本会完成：

- 树莓派专用依赖安装
- SPI 启用
- `wordscard` 虚拟环境搭建
- Python 与运行时包安装
- Waveshare 包安装
- 使用 `tmux` 启动应用进程

### 方案 3 — 安装服务

将应用生命周期接入 `systemd`：

```bash
bash scripts/install_wordscard_service.sh
```

然后执行：

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

<a id="configuration"></a>
## 配置

### 环境变量（`.env`）

仓库使用 `.env` 加载，并会覆盖已有 shell 变量。请按需配置：

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### 运行参数（`app.py` 与 `words_gpt.py` 共用）

- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

树莓派启动脚本支持通过 `APP_ARGS` 透传参数（示例）：

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### 应用模式下路由行为

当前代码中的路由：

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)`（来源于 `words_card_temp/`）

兼容性说明：早期文档中出现过 `GET /current_word`，当前路由是 `GET /get_current_word`。

### OpenAI 使用说明

OpenAI 功能可选，并由 CLI/环境变量控制。缓存的 API 请求体有助于复现和限流控制。在受限环境下，建议先使用 CSV 模式（`--use_csv`），按需再开启 `--enable_openai` 进行增强。

<a id="usage"></a>
## 使用方法

### 启动 HTTP 服务

```bash
python app.py
```

进程会在 `words_card_temp/` 中持续保留一张图片，并为前端工具或脚本提供 HTTP 接口。

### 直接运行渲染器

CSV 模式：

```bash
python words_gpt.py --use_csv
```

OpenAI 模式：

```bash
python words_gpt.py --enable_openai --use_csv
```

Emoji + 简化 CJK：

```bash
python words_gpt.py --make_emoji --simplify
```

### 在 Raspberry Pi 上运行

- 通过 tmux 启动脚本启动：

```bash
bash scripts/start_wordscard.sh
```

- 通过 tmux 停止脚本关闭：

```bash
bash scripts/stop_wordscard.sh
```

<a id="examples"></a>
## 示例

获取下一条随机卡片元数据：

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

获取当前存储词条：

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

请求当前页面图片负载：

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

提交明确词条：

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

<a id="data-cache-and-logs"></a>
## 数据、缓存与日志

应用常用产物：

- `data/`：精选 CSV 数据集
- `words_phonetics.db`：SQLite 缓存/源数据库
- `cache/`：OpenAI 请求与结果缓存
- `word_phonetics_processed.csv`：处理后的衍生数据集
- `logs/`, `logs-word-phonetics/`：运行日志
- `words_card_temp/`：生成的卡片和临时输出

<a id="development-notes"></a>
## 开发说明

- 存在历史/备份文件（如 `words_gpt_old.py`、`lib.old`），除非你正在迁移或维护兼容性，否则可将其当作参考。
- `words_update.py` 提供批量刷新/复检辅助，适用于数据库数据质量处理。
- 硬件验证通过 `epd_*_test.py` 和 `waveshare/examples/*` 示例。
- 仓库根目录未配置 `requirements.txt` 或锁文件；依赖由安装脚本或手动安装完成。
- 仓库未内置自动化测试框架。

<a id="troubleshooting"></a>
## 故障排查

- 来自 Raspberry Pi GPIO/SPI 模块的 `ImportError`：
  - 使用 Pi 安装流程 (`setup_pi_wordscard.sh`)，或在兼容目标上执行 `python setup.py install`
- 图片/静态端点返回 `403/404`：
  - 确认 `/get_current_word*` 路由用法，并检查 `words_card_temp/` 是否可写
- OpenAI 模式下返回空或无效词条：
  - 确认 `OPENAI_API_KEY` 及可选组织/模型参数已加载；检查 `cache/` 与日志
- 渲染/文字截断异常：
  - 检查 `words_gpt.py` 渲染路径中的字体路径与面板分辨率配置
- 接口返回旧数据：
  - 手动调用 `POST /next_random_word`，并检查 `app.py` 中周期回调间隔
- 硬件更新似乎冻结：
  - 查看 tmux 会话和 systemd 日志（`journalctl -u wordscard`）
- 数据集或词典条目缺失：
  - 校验 `data/` 中的 CSV 文件，并运行 `words_update.py` 刷新/清理流程

<a id="roadmap"></a>
## 路线图

- 增加最小化 `requirements.txt` / 可复现安装清单。
- 明确运行模式并补充 `--help` 文档。
- 扩展各内容模式的渲染模式说明（`japanese_synonym`、`arabic_synonym`、`film` 等）。
- 统一错误处理与用户端 API 响应结构。
- 为非硬件 CI 增加轻量化烟雾测试脚本框架。

<a id="support"></a>
## 支持

| 支持方式 | 链接 | 作用 |
|---|---|---|
| GitHub Sponsors | https://github.com/sponsors/lachlanchen | 持续支持与一次性项目资助 |
| Lazying Art | https://lazying.art | 品牌与相关资源 |
| Chat | https://chat.lazying.art | 讨论与支持 |
| Only Ideas | https://onlyideas.art | 创意研究与副项目 |

<a id="contributing"></a>
## 贡献

欢迎参与贡献。建议流程：

1. 保持改动聚焦到单一行为域（渲染、数据、API、脚本）。
2. 对面向用户的行为变更同步更新命令使用说明。
3. 尽量保持现有 CLI 参数与端点兼容。
4. 硬件脚本变更时，记录测试设备/型号和执行命令。

<a id="license"></a>
## 许可证

当前仓库根目录未包含 `LICENSE` 文件，因此本草稿内的仓库许可证在树内未明确声明。若需明确的再发布/复用条款，请补充许可证文件。
