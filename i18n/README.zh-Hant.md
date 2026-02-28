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
![Stars](https://img.shields.io/github/stars/lachlanchen/words_gpt?style=for-the-badge&logo=github&logoColor=white&label=Stars)
![Issues](https://img.shields.io/github/issues/lachlanchen/words_gpt?style=for-the-badge&logo=github&logoColor=white&label=Open%20Issues)
![Issues](https://img.shields.io/github/license/lachlanchen/words_gpt?style=for-the-badge&logo=github&logoColor=white&label=License)

> 這是一個基於 Raspberry Pi + Waveshare e-ink 的專案，用於渲染具備 IPA 音標與多語提示的動態詞彙卡片。它支援本機 CSV 工作流程、可選擇的 AI 強化、e-paper 渲染，以及遠端 HTTP 控制。

執行模式一覽：
`app.py`（服務）與 `words_gpt.py`（獨立渲染器）可單獨運作，也可同時搭配使用。

| 🔎 一覽 | 詳情 |
|---|---|
| **核心執行時** | `app.py`（HTTP 服務）+ `words_gpt.py`（渲染循環） |
| **資料路徑** | `data/` 中的 CSV 資料集 + SQLite 存放庫 `words_phonetics.db` |
| **輸出目標** | Waveshare e-paper 螢幕與虛擬圖片輸出 |
| **AI 依賴** | 可選（`--enable_openai`），並於 `cache/` 保留請求快取 |
| **預設週期** | 伺服器於 `8082`，約每 5 分鐘自動刷新一次 |

## 📚 目錄
- 概覽
- 重點
- 展示
- 專案結構
- 先決條件
- 安裝
- 設定
- 使用方式
- 範例
- 資料、快取與日誌
- 開發備註
- 故障排除
- 開發藍圖
- Support
- 貢獻
- 授權

---

## 概覽

`words_gpt` 是一個用於 e-ink 顯示器的 Python 詞彙卡片產生系統。它將資料流程、音標強化與渲染整合於兩個執行模式：

- 長時間運行的 Tornado 服務（`app.py`），用於遠端控制與圖片提供。
- 獨立渲染器（`words_gpt.py`），可在輪詢、迴圈或直接渲染模式下運作。

主要模組：

- `words_data.py` / `words_data_utils.py`：詞彙抓取與增強流程。
- `words_data_with_legacy.py`、`words_data_without_legacy.py`、`words_data_workable*.py`：變體流程與舊版相容。
- `words_database.py`：SQLite 互動。
- `openai_request_json.py`：具磁碟快取與重試行為的結構化 OpenAI 請求。
- `env_loader.py`：決定性環境載入。
- `words_update.py`：資料庫維護與重檢流程。
- `app.py` 與 `words_gpt.py`：服務與渲染生命週期。
- `pwa/`：輕量瀏覽器預覽與設定工具。

## 重點

- 單一工作流程同時支援本機與 OpenAI 詞彙來源。
- 可選簡體中文輸出（`--simplify`）。
- 提供遠端控制 HTTP 端點（`/next_random_word`、`/display_word`、`/get_current_word`、`/get_current_word_page`、`/get_words_card`）。
- 透過快取與持久化減少重複 AI 請求。
- 透過內建 `waveshare/` 驅動程式庫與硬體範例進行封裝與整合。

## 展示

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## 專案結構

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

重要的執行檔：

- `app.py`：Tornado 應用，預設 `8082` 連接埠 + 定時 `next_random_word` 觸發。
- `words_gpt.py`：獨立渲染器與顯示抽象層（`EPaperHardware`、`EPaperDisplay`）。
- `words_data.py`：擷取/強化流程與詞彙挑選工具。
- `words_database.py`：用於存放中繼資料與詞彙快取操作的 SQLite 輔助函式。
- `scripts/*.sh`：安裝與服務生命週期腳本，以及 Raspberry Pi 啟動協助工具。
- `words_update.py`：用於資料品質維護的批次刷新/重檢流程。

## 先決條件

- Python `3.9+`（建議）
- Raspberry Pi（硬體模式必備）
- 支援的 Waveshare e-paper 面板（例如 7.3F 或 13K 系列）
- 啟用 SPI（`raspi-config`）、接線正確且供電穩定
- 使用 NLTK 詞源時需安裝 NLTK 語料庫

常見相依套件（執行路徑使用）：
`openai`、`tornado`、`Pillow`、`numpy`、`nltk`、`opencc`、`pykakasi`、`arabic_reshaper`、`python-bidi`、`pytz`。

## 安裝

### 方式一：最小／手動安裝（桌機或 Pi）

在專案根目錄執行：

```bash
python setup.py install
```

如有需要：

```bash
python -m nltk.downloader words
```

### 方式二：Raspberry Pi 自動化安裝（建議在設備上）

在專案根目錄執行：

```bash
bash scripts/setup_pi_wordscard.sh
```

這個流程會完成：

- Pi 專用相依套件安裝
- SPI 啟用與檢查
- `wordscard` 虛擬環境設定
- Python 與執行環境套件安裝
- Waveshare 套件安裝
- 透過 `tmux` 啟動 app 行程

### 方式三：systemd 服務安裝

將 app 生命週期註冊到 `systemd`：

```bash
bash scripts/install_wordscard_service.sh
```

接著：

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## 設定

### 環境變數（`.env`）

`env_loader` 會在程序啟動時讀取環境變數並套用組態。目前文件與實際使用常見如下：

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

建議：將密鑰保留在本機環境設定中，不要提交至版本控制。

### 執行旗標（`app.py` 與 `words_gpt.py`）

| CLI 旗標 | 用途 |
| --- | --- |
| `--enable_openai` | 啟用可選的 OpenAI 強化模式 |
| `--make_emoji` | 渲染以 emoji 為主的卡片 |
| `--ignore_list` | 略過設定中的忽略詞 |
| `--use_csv` | 從 CSV 資料集讀取詞彙 |
| `--complete_csv` | 使用完整 CSV 來源模式 |
| `--filename <csv_file>` | 指定特定的 CSV 輸入檔 |

`APP_ARGS` 可透過啟動腳本傳遞，例如：

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### app 模式路由行為

觀察到的路由：

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)`（由 `words_card_temp/` 提供）

相容性說明：舊文件可能提到 `GET /current_word`；目前路由為 `GET /get_current_word`。

### OpenAI 使用說明

OpenAI 功能為可選，透過 CLI / 環境變數控制。`cache/` 的快取請求有助於提高重複性並控管速率限制。若要優先離線執行，建議從 CSV 模式（`--use_csv`）起步，再按需啟用 OpenAI。

## 使用方式

### 啟動 HTTP 伺服器

```bash
python app.py
```

程序會保留最新圖片在 `words_card_temp/`，並對前端工具或腳本提供對應端點。

### 直接執行渲染器

CSV 模式：

```bash
python words_gpt.py --use_csv
```

OpenAI 模式：

```bash
python words_gpt.py --enable_openai --use_csv
```

Emoji + 簡化中文模式：

```bash
python words_gpt.py --make_emoji --simplify
```

### 在 Raspberry Pi 上執行硬體

- 透過 tmux 腳本啟動：

```bash
bash scripts/start_wordscard.sh
```

- 透過 tmux 腳本停止：

```bash
bash scripts/stop_wordscard.sh
```

## 範例

取得下一張隨機卡片中繼資料：

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

取得目前存放的詞彙：

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

取得目前頁面圖片內容：

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

送出指定詞彙：

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

透過 form 風格端點觸發卡片渲染：

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

## 資料、快取與日誌

應用常見使用檔案：

- `data/`：精選 CSV 資料集
- `words_phonetics.db`：SQLite 快取/來源資料庫
- `cache/`：OpenAI 請求/回應快取
- `word_phonetics_processed.csv`：已處理/衍生資料集
- `logs/`、`logs-word-phonetics/`：執行日誌
- `words_card_temp/`：產生卡片與暫存輸出
- `pic/` 與 `figs/`：參考影像與橫幅

## 開發備註

- 專案內包含舊版或備援模組與成果（例如 `words_gpt_old.py`、`lib.old`），除非你刻意進行相容性或遷移，否則請將其視為參考。
- `words_update.py` 含有可用於 DB 品質維護的批次刷新/複查工具。
- 硬體驗證由 `epd_*_test.py` 與 `waveshare/examples/*` 示例腳本處理。
- 倉庫根目錄沒有 `requirements.txt` 或鎖定檔；相依套件由安裝腳本與直接安裝流程驅動。
- 本專案未配置自動化測試套件。

## 故障排除

- 來自 Raspberry Pi GPIO/SPI 模組的 `ImportError`：
  - 使用 Pi 安裝流程（`scripts/setup_pi_wordscard.sh`），或在相容目標上明確安裝相依套件。
- 圖片/靜態端點回傳 `403/404`：
  - 確認路由用法（`/get_current_word*`）並確認 `words_card_temp/` 可寫。
- OpenAI 模式回傳空值或無效詞彙：
  - 檢查 `OPENAI_API_KEY`（及 org/model）是否載入，接著檢查 `cache/` 與日誌。
- 顯示異常或文字被截斷：
  - 檢查 `words_gpt.py` 中字型路徑、面板解析度常數與選取的模式設定。
- API 回傳過期資料：
  - 手動呼叫 `POST /next_random_word`，並檢查 `app.py` 的回呼間隔。
- 硬體渲染似乎卡住：
  - 檢查 tmux 工作階段與 `journalctl -u wordcard` 系統日誌。
- 缺少資料集／字典條目：
  - 驗證 `data/` 中的 CSV 檔並執行 `words_update.py` 維護任務。

## 開發藍圖

- 新增精簡版 `requirements.txt` / 可重現安裝清單。
- 提供更清晰的執行模式與明確的 CLI `--help` 文件。
- 擴充渲染模式文件（`japanese_synonym`、`arabic_synonym`、`film` 與其他流程）。
- 統一錯誤處理與對外 API 回應格式。
- 新增輕量級 smoke 測試雛形，用於無硬體 CI 驗證。

## 貢獻

歡迎提交貢獻，建議流程如下：

1. 將修改聚焦在單一行為區塊（渲染、資料、API、腳本）。
2. 對使用者可見行為變更同步更新命令與文件。
3. 在可能的情況下保留既有 CLI 參數與端點相容性。
4. 若修改硬體腳本，請註明已測試的設備型號與實際執行命令。

## 授權

目前 `README` 目錄下未提供 `LICENSE` 檔案，專案內仍未明確定義授權條款。若你需要明確的再發布／再授權規範，請補上授權檔案。


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
