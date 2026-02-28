[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 🖨️ Eink Words GPT

**本草稿語言：** 中文（繁體）

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=for-the-badge&logo=githubactions&logoColor=white)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=for-the-badge&logo=python&logoColor=white)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=for-the-badge&logo=openai&logoColor=white)

這是一個基於 Raspberry Pi + Waveshare 電子紙的專案，會動態產生並渲染含 IPA 音標與多語提示的詞彙卡。它支援本地 CSV 流程、可選的 AI 強化、e-paper 渲染，以及遠端 HTTP 控制。

| 🔎 一覽 | 說明 |
|---|---|
| 核心執行元件 | `app.py`（HTTP 服務）＋`words_gpt.py`（渲染循環） |
| 資料路徑 | `data/` 中的 CSV 資料集 + SQLite 存放檔 `words_phonetics.db` |
| 輸出目標 | Waveshare 電子紙面板與虛擬圖片輸出 |
| AI 依賴 | 可選（`--enable_openai`），快取位於 `cache/` |
| 主循環預設 | 伺服器在 `8082`，每約 5 分鐘定期刷新一次 |

## 目錄
- 概覽
- 重點整理
- 示範
- 專案架構
- 先決條件
- 安裝
- 設定
- 使用方式
- 範例
- 資料、快取與日誌
- 開發備註
- 疑難排解
- 路線圖
- 支援
- 貢獻
- 授權

---

## 概覽

`words_gpt` 是一個為 e-ink 顯示器打造的 Python 詞彙卡片產生架構，整合了資料流程、音標補強與渲染編排，透過兩種主要模式運行：

- 長時間運行的 Tornado 服務（`app.py`），負責遠端控制與圖片提供
- 獨立渲染器（`words_gpt.py`），可在輪詢、循環或直接渲染模式下執行

核心模組：

- `words_data.py` / `words_data_utils.py`：詞彙與增強資料流程
- `words_database.py`：SQLite 互動
- `openai_request_json.py`：含磁碟快取的 OpenAI 結構化請求
- `env_loader.py`：穩定載入環境變數
- `words_update.py`：資料庫維護與重新檢查流程
- `app.py` 與 `words_gpt.py`：服務與渲染生命週期

## 重點整理

- E-ink 渲染管線，支援多語系與內容模式：
  - 日語變體、漢字模式、阿拉伯文、中文、emoji 模式
- 本地與 OpenAI 詞彙來源可在同一流程並用
- 提供直接互動端點（`/next_random_word`、`/display_word` 等）
- 快取與持久化降低重複網路呼叫
- 可選的 PWA 資源放在 `pwa/`，適合輕量預覽與設定流程

## 示範

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## 專案架構

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

關鍵運行檔：

- `app.py`：Tornado 應用，預設 `8082` 埠 + 定期 `next_random_word` 觸發
- `words_gpt.py`：獨立渲染器，並提供顯示抽象層（`EPaperHardware`、`EPaperDisplay`）
- `words_data.py`：進階抓取與增強流程
- `words_database.py`：存放詞彙中繼資料與快取的 SQLite 輔助
- `scripts/*.sh`：安裝與服務流程，以及 Raspberry Pi 啟動腳本

## 先決條件

- Python `3.9+`（建議）
- Raspberry Pi（硬體模式必備）
- 支援的 Waveshare 電子紙面板（例如 7.3F / 13K 家族）
- 已啟用 SPI（`raspi-config`）、接線正確，並有穩定電源
- 使用 `nltk` 詞表時需先有 NLTK 語料庫

目前程式中使用到的常見套件：
`openai`、`tornado`、`Pillow`、`numpy`、`nltk`、`opencc`、`pykakasi`、`arabic_reshaper`、`python-bidi`、`pytz`。

## 安裝

### 方式一：最小/手動安裝（桌機或 Pi）

在專案根目錄：

```bash
python setup.py install
```

如有需要：

```bash
python -m nltk.downloader words
```

### 方式二：Raspberry Pi 自動安裝（建議裝置內操作）

在專案根目錄：

```bash
bash scripts/setup_pi_wordscard.sh
```

會完成以下工作：

- Pi 專用相依套件安裝
- 啟用 SPI
- 建立 `wordscard` 虛擬環境
- 安裝 Python/執行環境套件
- 安裝 Waveshare 套件
- 以 `tmux` 啟動 app 行程

### 方式三：服務安裝

註冊 app 的 systemd 生命週期：

```bash
bash scripts/install_wordscard_service.sh
```

接著：

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordcard -n 100 --no-pager
```

## 設定

### 環境變數（`.env`）

此專案會載入 `.env` 並覆蓋既有同名 shell 變數，請依需求使用：

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### 執行參數（`app.py` 與 `words_gpt.py` 共用）

- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

Pi 啟動腳本可透過 `APP_ARGS` 傳遞參數（範例）：

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### app 模式的路由行為

目前觀察到的路由：

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)`（從 `words_card_temp/` 提供）

相容性說明：舊文件曾提及 `GET /current_word`，目前路由為 `GET /get_current_word`。

### OpenAI 使用注意

OpenAI 功能為可選，受 CLI 與環境變數控制。快取的 API 請求有助於重現性與速率控制。在受限環境下可先以 CSV 模式啟動（`--use_csv`），再依需要開啟 `--enable_openai`。

## 使用方式

### 啟動 HTTP 伺服器

```bash
python app.py
```

程式會在 `words_card_temp/` 保留最新圖片，並曝光 HTTP 端點給前端工具或簡易腳本使用。

### 直接啟動渲染器

CSV 模式：

```bash
python words_gpt.py --use_csv
```

OpenAI 模式：

```bash
python words_gpt.py --enable_openai --use_csv
```

### 在 Raspberry Pi 上直接運行

- 透過 `tmux` 啟動：

```bash
bash scripts/start_wordscard.sh
```

- 透過 `tmux` 停止：

```bash
bash scripts/stop_wordscard.sh
```

## 範例

取得下一張隨機卡片的中繼資料：

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

取得目前快取詞彙：

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

取得目前渲染頁面圖片 payload：

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

提交指定詞彙：

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

透過表單式端點觸發卡片渲染：

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

## 資料、快取與日誌

應用常見使用檔案：

- `data/`：精選 CSV 資料集
- `words_phonetics.db`：SQLite 快取/來源資料庫
- `cache/`：OpenAI 請求與回應快取
- `word_phonetics_processed.csv`：已處理/衍生資料集
- `logs/`、`logs-word-phonetics/`：執行日誌
- `words_card_temp/`：產生卡片與暫存輸出

## 開發備註

- 專案存在舊版/備份檔（例如 `words_gpt_old.py`、`lib.old`），除非你正在維護相容性，否則優先參考主線實作。
- `words_update.py` 提供可批次更新與複查的流程，適合進行資料品質整理。
- 硬體驗證集中在 `epd_*_test.py` 與 `waveshare/examples/*` 示範。
- 倉庫根目錄目前沒有 `requirements.txt` 或 lockfile；依賴安裝主要透過安裝腳本或直接安裝。
- 本倉庫未配置自動化測試。

## 疑難排解

- `ImportError` 來自 Raspberry Pi GPIO/SPI 模組：
  - 請改以 Pi 專用路徑 (`setup_pi_wordscard.sh`) 安裝，或在對應目標確認 `python setup.py install`。
- 取得圖片/靜態檔 403/404：
  - 確認 `/get_current_word*` 的使用方式，並確認 `words_card_temp/` 可寫。
- OpenAI 模式下 payload 為空或無效：
  - 確認 `OPENAI_API_KEY`、可選的 org/model 已載入；檢查 `cache/` 與日誌。
- 顯示異常/文字截斷：
  - 檢查 `words_gpt.py` 中的字型路徑與面板解析度設定。
- API 回傳舊資料：
  - 手動呼叫 `POST /next_random_word`，並檢查 `app.py` 中的週期回呼間隔。
- 更新卡片卡住：
  - 檢查 tmux 會話與 systemd 日誌（`journalctl -u wordcard`）。
- 缺少詞彙或字典資料：
  - 檢查 `data/` 內 CSV，並執行 `words_update.py` 流程做重整理。

## 路線圖

- 新增最小 `requirements.txt` / 可重現安裝清單。
- 提供更明確的執行模式文件與完整 `--help` 說明。
- 擴充各內容模式的渲染格式文件（例如 `japanese_synonym`、`arabic_synonym`、`film` 等）。
- 統一錯誤處理與使用者可見 API 回應格式。
- 新增小型 smoke test 雛形，支援非硬體環境驗證。

## 支援

| 支援方式 | 連結 | 用途 |
|---|---|---|
| GitHub Sponsors | https://github.com/sponsors/lachlanchen | 持續與一次性專案支援 |
| Lazying Art | https://lazying.art | 品牌與相關資源 |
| Chat | https://chat.lazying.art | 討論與支援 |
| Only Ideas | https://onlyideas.art | 創意研究與副專案 |

## 貢獻

歡迎提交修改。建議流程：

1. 將改動限制在單一行為範圍（渲染、資料、API、腳本）。
2. 對使用者可見行為變動更新指令與文件。
3. 儘可能保留既有 CLI 參數與端點相容性。
4. 若硬體腳本有修改，請註明測試設備/型號與實際執行指令。

## 授權

目前倉庫根目錄未提供 `LICENSE` 檔案，依目前版本來看，授權條款尚未在專案內明確定義。如需明確的重用/散佈規範，請補上授權檔案。
