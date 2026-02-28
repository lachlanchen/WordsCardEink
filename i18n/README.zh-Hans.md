[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Eink Words GPT

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi-green)
![Display](https://img.shields.io/badge/display-Waveshare%20e--Paper-black)
![Status](https://img.shields.io/badge/status-active%20prototype-orange)
![Server](https://img.shields.io/badge/http-Tornado-0A7EA4)
![Storage](https://img.shields.io/badge/storage-SQLite-003B57)
![AI](https://img.shields.io/badge/OpenAI-optional-412991)

这是一个基于 Raspberry Pi + Waveshare 电子墨水屏的项目，可动态显示带音标与多语言同义词的词汇。系统可从本地数据集或 OpenAI 获取单词，将其渲染为版式，并推送到受支持的电子纸面板。同时还提供一个轻量 HTTP 服务，用于触发单词更新和获取渲染图像。

## 概览
`words_gpt` 是一个面向电子墨水设备的 Python 词汇卡片生成与显示系统。

它结合了：
- 从 CSV/本地数据集取词，以及可选的 OpenAI 生成。
- 内容增强（IPA 音标 + 多语言同义词字段）。
- 面向硬件与虚拟输出的渲染流水线。
- 用于远程触发与图像获取的 Tornado HTTP 服务。

当前代码库核心包括 `app.py`、`words_gpt.py`、`words_data.py`、`words_database.py` 与 `openai_request_json.py`。

## 亮点
- 🖼️ 电子墨水渲染流水线，支持多种内容模式（汉字、日语、阿拉伯语、中文、emoji）。
- 🗃️ 本地词库数据库（`words_phonetics.db`），词表来自 `data/` 中的 CSV。
- 🤖 基于 OpenAI 的选词与音标增强，采用结构化 JSON 输出。
- 🌐 供外部触发和图像获取的 HTTP 服务。
- ⚡ 缓存层（`cache/`）减少重复 OpenAI 调用。

## 快速开始
| 目标 | 命令 |
|---|---|
| 启动 HTTP 服务（端口 `8082`） | `python app.py` |
| 运行独立渲染器（CSV） | `python words_gpt.py --use_csv` |
| OpenAI + CSV 模式运行 | `python words_gpt.py --enable_openai --use_csv` |
| Emoji + 简体 CJK 模式 | `python words_gpt.py --make_emoji --simplify` |
| Raspberry Pi 自动化安装 | `bash scripts/setup_pi_wordscard.sh` |

## 演示
<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## 功能
- 来自 `words_gpt.py` 的硬件 + 虚拟渲染流程（`EPaperHardware`、`EPaperDisplay`）。
- `words_data.py` 中的多语言增强流水线（IPA、日语变体、阿拉伯语、法语、中文字段）。
- `words_database.py` 中基于 SQLite 的持久化与动态字段更新辅助函数。
- `openai_request_json.py` 中带文件缓存的 OpenAI 结构化 JSON 请求辅助工具。
- `pwa/` 下可选的 PWA 资源，用于轻量前端配置/预览流程。

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

重要运行时文件：
- `app.py`：Tornado Web 服务器（默认端口 `8082`）与周期更新循环。
- `words_gpt.py`：独立渲染循环与显示类。
- `words_data.py`：核心取词/增强编排。
- `words_database.py`：SQLite 存储辅助函数。
- `scripts/*.sh`：Raspberry Pi 安装、服务安装与 tmux 生命周期脚本。

## 前置要求
- Python `3.9+`（推荐）。
- Raspberry Pi 目标设备（硬件模式）。
- 受支持的 Waveshare 电子纸面板。
- Pi 上已启用 SPI（`raspi-config`），并完成对应面板接线。

本项目使用的 Python 包包括：
- `openai`、`tornado`、`Pillow`、`numpy`、`nltk`、`opencc`、`pykakasi`、`arabic_reshaper`、`python-bidi`、`pytz`。
- 安装脚本还会安装：`json5`、`pandas`、`spidev`、`RPi.GPIO`、`gpiozero`、`lgpio`。

## 安装

### 方案 A：最小/手动安装
安装 Waveshare 驱动包：
```bash
python setup.py install
```

如果你使用 NLTK 词表，需下载一次：
```bash
python -m nltk.downloader words
```

### 方案 B：Raspberry Pi 自动化安装（设备上推荐）
在仓库根目录执行：
```bash
bash scripts/setup_pi_wordscard.sh
```

该脚本会：
- 安装 apt 依赖。
- 确保 SPI 已启用。
- 创建并激活 `wordscard` 虚拟环境。
- 安装 Python 运行时依赖。
- 安装 Waveshare 包。
- 在 tmux 会话中启动 `app.py`。

## 配置

### `.env` 行为
此仓库在导入阶段会从 `.env` 加载环境变量，并**覆盖** shell 中已有值。即使你已经在 shell 配置中导出变量，也能保证本地覆盖行为可预测。

创建或更新 `.env`：
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### App 参数透传
systemd/tmux 脚本支持：
```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### CLI 标志（服务端与渲染器）
`app.py` 和 `words_gpt.py` 均支持：
- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

## 使用

### 运行 HTTP 服务
启动服务（默认端口 `8082`）：
```bash
python app.py
```

代码中可见的路由：
- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)`（来自 `words_card_temp/`）

兼容性说明：早期文档写的是 `GET /current_word`；当前 `app.py` 路由为 `GET /get_current_word`。

### 运行独立渲染器
基于 CSV 词表：
```bash
python words_gpt.py --use_csv
```

启用 OpenAI：
```bash
python words_gpt.py --enable_openai --use_csv
```

Emoji 渲染 + 简体 CJK：
```bash
python words_gpt.py --make_emoji --simplify
```

### Raspberry Pi 上的服务模式
安装服务单元：
```bash
bash scripts/install_wordscard_service.sh
```

然后：
```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## 示例

### 触发下一条随机单词
```bash
curl "http://127.0.0.1:8082/next_random_word"
```

### 读取当前单词载荷
```bash
curl "http://127.0.0.1:8082/get_current_word"
```

### 提交指定单词
```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

### 硬件冒烟测试
使用对应显示型号脚本：
```bash
python epd_7in3f_test.py
```

或：
```bash
python epd_13in3k_test.py
```

更多示例见 `waveshare/examples/`。

## 数据、缓存与日志
| 区域 | 路径 | 说明 |
|---|---|---|
| 词表 | `data/` | 包含 `data/words_list.csv` 及主题 CSV 文件 |
| 持久化数据库 | `words_phonetics.db` | 本地音标/增强存储 |
| OpenAI/缓存产物 | `cache/` | 减少重复请求 |
| 日志 | `logs/`、`logs-word-phonetics/` | 运行与更新日志 |
| 生成卡片 | `words_card_temp/` | 图像输出与静态资源服务来源 |

## 开发说明
- 依赖管理目前以脚本优先（`scripts/setup_pi_wordscard.sh`）+ `setup.py`；尚无 `requirements.txt` 或 `pyproject.toml`。
- 仓库中存在多份备份/历史文件（`words_data_*`、`words_gpt_old.py`）；当前主要运行路径是 `app.py` + `words_gpt.py` + `words_data.py` + `words_database.py`。
- `env_loader.py` 在键存在时总是用 `.env` 覆盖环境变量。
- 服务模式会运行周期性刷新流程（约每 5 分钟），并可能在内部调用更新端点。

## 故障排查
- `ModuleNotFoundError` 或导入问题：
  - 确认虚拟环境已激活且依赖已安装。
  - 在 Pi 上重新运行 `bash scripts/setup_pi_wordscard.sh`。
- OpenAI 错误（`401`、模型/密钥缺失）：
  - 在 `.env` 中检查 `OPENAI_API_KEY` 和可选的 `OPENAI_MODEL`。
  - 确认设备网络连通性。
- 显示未更新：
  - 确认面板型号/接线，并运行匹配测试脚本（`epd_7in3f_test.py` 或 `epd_13in3k_test.py`）。
  - 确认 SPI 已启用（`sudo raspi-config nonint do_spi 0`）。
  - 在 Pi 5 上，若设备暴露为 `/dev/spidev10.0`，请确保存在 `/dev/spidev0.0` 兼容性符号链接。
- OpenCC 安装问题：
  - 使用发行版兼容包（`libopencc1` 或 `libopencc2`），与安装脚本一致。
- API 路由不匹配：
  - 当前载荷请使用 `/get_current_word`，而不是 `/current_word`。

## OpenAI 使用说明
OpenAI 接入是可选的，但建议用于生成新词和音标增强。`openai_request_json.py` 中的结构化 JSON 辅助器会将结果缓存到 `cache/` 以减少重复调用。

## 路线图
- 增加正式依赖清单（`requirements.txt` 或 `pyproject.toml`），支持可复现安装。
- 扩展 `i18n/` 并维护多语言 README 版本。
- 在规范流程确定后，整合历史/备份脚本变体。
- 为 `pwa/` 流程补充端点示例与截图文档。
- 添加可重复的自动化测试，覆盖数据与路由级行为。

## 支持

### 你的支持可以实现
- <b>保持工具开放</b>：覆盖托管、推理、数据存储与社区运营。  
- <b>更快交付</b>：将更多开源时间投入 WordsCardEink 及相关学习工具。  
- <b>设备原型迭代</b>：支持电子墨水硬件迭代与版式研究。  
- <b>让更多人可用</b>：为学生、创作者和社区团体提供补贴部署。

### 捐赠

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

## 贡献
贡献指南、代码风格与 PR 预期请见 `AGENTS.md`。

建议的贡献检查清单：
- 涉及显示改动时附上面板型号与硬件说明。
- 列出用于验证的精确命令。
- 对 UI 或电子墨水输出改动附上截图/照片。
- 描述数据集修改（文件 + 行/列影响）。

## 许可证
当前仓库根目录中尚未发现 `LICENSE` 文件（以本次草稿扫描为准）。在补充许可证文件之前，项目复用权限尚未被明确授予。

假设：维护者可能会在后续更新中补充明确的开源许可证。
