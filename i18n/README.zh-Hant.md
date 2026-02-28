[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Eink Words GPT

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi-green)
![Display](https://img.shields.io/badge/display-Waveshare%20e--Paper-black)
![Status](https://img.shields.io/badge/status-active%20prototype-orange)
![Server](https://img.shields.io/badge/http-Tornado-0A7EA4)
![Storage](https://img.shields.io/badge/storage-SQLite-003B57)
![AI](https://img.shields.io/badge/OpenAI-optional-412991)

這是一個基於 Raspberry Pi + Waveshare 電子紙的專案，可顯示動態選取的詞彙，並附上音標與多語同義詞。系統可從本地資料集或 OpenAI 取得單字，將內容排版渲染後推送到支援的電子紙面板，也提供小型 HTTP 服務以觸發單字更新並取得渲染圖片。

## 概覽
`words_gpt` 是一套以 Python 開發、面向電子紙裝置的詞彙卡片產生與顯示系統。

它整合了：
- 從 CSV／本地資料集取詞，並可選擇啟用 OpenAI 產生內容。
- 資料增強（IPA 音標 + 多語同義詞欄位）。
- 針對硬體與虛擬輸出的渲染流程。
- 用於遠端觸發與圖片取得的 Tornado HTTP 服務。

目前程式碼主軸為 `app.py`、`words_gpt.py`、`words_data.py`、`words_database.py`、`openai_request_json.py`。

## 亮點
- 🖼️ 電子紙渲染流程，支援多種內容模式（漢字、日文、阿拉伯文、中文、emoji）。
- 🗃️ 本地單字資料庫（`words_phonetics.db`），搭配 `data/` 中以 CSV 維護的詞表。
- 🤖 以 OpenAI 支援的單字選取與音標增強，輸出結構化 JSON。
- 🌐 提供 HTTP 服務，供外部觸發與圖片取得。
- ⚡ 快取層（`cache/`）可減少重複 OpenAI 呼叫。

## 快速開始
| 目標 | 指令 |
|---|---|
| 啟動 HTTP 伺服器（埠號 `8082`） | `python app.py` |
| 執行獨立渲染器（CSV） | `python words_gpt.py --use_csv` |
| 使用 OpenAI + CSV 執行 | `python words_gpt.py --enable_openai --use_csv` |
| Emoji + 簡體 CJK 模式 | `python words_gpt.py --make_emoji --simplify` |
| Raspberry Pi 自動化安裝 | `bash scripts/setup_pi_wordscard.sh` |

## 展示
<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## 功能
- 由 `words_gpt.py` 提供的硬體 + 虛擬渲染流程（`EPaperHardware`、`EPaperDisplay`）。
- `words_database.py` 中以 SQLite 為基礎，並具備動態欄位更新輔助。
- `openai_request_json.py` 提供 OpenAI 結構化 JSON 請求輔助與檔案快取。
- `pwa/` 提供可選的 PWA 資源，用於輕量前端設定／預覽流程。

## 專案結構
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

重要執行檔案：
- `app.py`：Tornado Web 伺服器（預設埠號 `8082`）與定期更新循環。
- `words_gpt.py`：獨立渲染循環與顯示類別。
- `words_data.py`：核心單字擷取／增強流程協調。
- `words_database.py`：SQLite 儲存輔助。
- `scripts/*.sh`：Raspberry Pi 安裝、服務安裝與 tmux 生命週期腳本。

## 先決條件
- Python `3.9+`（建議）。
- Raspberry Pi 目標裝置（硬體模式）。
- 支援的 Waveshare 電子紙面板。
- Pi 已啟用 SPI（`raspi-config`），並完成面板對應接線。

本專案使用的 Python 套件包含：
- `openai`、`tornado`、`Pillow`、`numpy`、`nltk`、`opencc`、`pykakasi`、`arabic_reshaper`、`python-bidi`、`pytz`。
- 安裝腳本另外會安裝：`json5`、`pandas`、`spidev`、`RPi.GPIO`、`gpiozero`、`lgpio`。

## 安裝

### 選項 A：最小化／手動安裝
安裝 Waveshare 驅動套件：
```bash
python setup.py install
```

若使用 NLTK 字詞列表，請先下載一次：
```bash
python -m nltk.downloader words
```

### 選項 B：Raspberry Pi 自動化安裝（建議在裝置上使用）
在 repo 根目錄執行：
```bash
bash scripts/setup_pi_wordscard.sh
```

此腳本會：
- 安裝 apt 相依套件。
- 確保 SPI 已啟用。
- 建立並啟用 `wordscard` 虛擬環境。
- 安裝 Python 執行時相依套件。
- 安裝 Waveshare 套件。
- 在 tmux 工作階段中啟動 `app.py`。

## 設定

### `.env` 行為
此 repo 會在 import 階段從 `.env` 載入環境變數，並**覆寫** shell 中既有值。即使你已在 shell profile 匯出變數，也能確保本機覆寫具可預期性。

建立或更新 `.env`：
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### App 參數透傳
systemd/tmux 腳本支援：
```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### CLI 旗標（伺服器與渲染器）
`app.py` 與 `words_gpt.py` 皆支援：
- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

## 使用方式

### 執行 HTTP 伺服器
啟動服務（預設埠號 `8082`）：
```bash
python app.py
```

程式碼中觀察到的路由：
- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)`（來源 `words_card_temp/`）

相容性說明：較早文件曾寫為 `GET /current_word`；目前 `app.py` 路由為 `GET /get_current_word`。

### 執行獨立渲染器
基於 CSV 的詞表：
```bash
python words_gpt.py --use_csv
```

啟用 OpenAI：
```bash
python words_gpt.py --enable_openai --use_csv
```

Emoji 渲染 + 簡體 CJK：
```bash
python words_gpt.py --make_emoji --simplify
```

### Raspberry Pi 的服務模式
安裝服務單元：
```bash
bash scripts/install_wordscard_service.sh
```

接著：
```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## 範例

### 觸發下一個隨機單字
```bash
curl "http://127.0.0.1:8082/next_random_word"
```

### 讀取目前單字 payload
```bash
curl "http://127.0.0.1:8082/get_current_word"
```

### 提交指定單字
```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

### 硬體冒煙測試
使用面板對應腳本：
```bash
python epd_7in3f_test.py
```

或：
```bash
python epd_13in3k_test.py
```

更多範例在 `waveshare/examples/`。

## 資料、快取與日誌
| 區域 | 路徑 | 說明 |
|---|---|---|
| 詞表 | `data/` | 包含 `data/words_list.csv` 與主題化 CSV 檔 |
| 持久化資料庫 | `words_phonetics.db` | 本地音標／增強資料儲存 |
| OpenAI／快取產物 | `cache/` | 降低重複請求 |
| 日誌 | `logs/`、`logs-word-phonetics/` | 執行與更新日誌 |
| 產生的卡片 | `words_card_temp/` | 圖像輸出與靜態服務來源 |

## 開發備註
- 相依管理以腳本優先（`scripts/setup_pi_wordscard.sh`）+ `setup.py`；目前尚無 `requirements.txt` 或 `pyproject.toml`。
- 專案中有多個備份／舊版檔案（`words_data_*`、`words_gpt_old.py`）；目前主要執行路徑是 `app.py` + `words_gpt.py` + `words_data.py` + `words_database.py`。
- `env_loader.py` 在 key 存在時，會一律以 `.env` 值覆寫環境變數。
- 伺服器模式會執行定期刷新流程（約每 5 分鐘），可能在內部呼叫更新端點。

## 疑難排解
- `ModuleNotFoundError` 或匯入問題：
  - 確認虛擬環境已啟用且相依套件已安裝。
  - 在 Pi 上重新執行 `bash scripts/setup_pi_wordscard.sh`。
- OpenAI 錯誤（`401`、模型／金鑰缺失）：
  - 確認 `.env` 中的 `OPENAI_API_KEY` 與可選的 `OPENAI_MODEL`。
  - 確認裝置網路連線正常。
- 顯示未更新：
  - 確認面板型號／接線，並執行對應測試腳本（`epd_7in3f_test.py` 或 `epd_13in3k_test.py`）。
  - 確認 SPI 已啟用（`sudo raspi-config nonint do_spi 0`）。
  - 若為 Pi 5，當裝置提供 `/dev/spidev10.0` 時，請確認 `/dev/spidev0.0` 相容性符號連結。
- OpenCC 安裝問題：
  - 使用相容於發行版的套件（`libopencc1` 或 `libopencc2`），與安裝腳本一致。
- API 路由不一致：
  - 目前 payload 路由請使用 `/get_current_word`，不是 `/current_word`。

## OpenAI 使用說明
OpenAI 存取為可選，但建議啟用以取得更即時的單字生成與音標增強。`openai_request_json.py` 中的結構化 JSON 輔助會將結果快取到 `cache/`，以減少重複呼叫。

## 路線圖
- 新增正式相依清單（`requirements.txt` 或 `pyproject.toml`），提升安裝可重現性。
- 擴充 `i18n/`，並維護各語言 README 版本。
- 在標準流程定稿後，整併舊版／備份腳本變體。
- 補上 `pwa/` 工作流程文件（含端點範例與截圖）。
- 新增可重複執行的自動化測試，涵蓋資料與路由層行為。

## 支援

### 你的支持能帶來什麼
- <b>保持工具開放</b>：支撐主機、推論、資料儲存與社群營運。  
- <b>更快交付</b>：投入更多專注的開源時間到 WordsCardEink 與相關學習工具。  
- <b>裝置原型</b>：支持電子紙硬體迭代與顯示版面研究。  
- <b>普及使用</b>：為學生、創作者與社群團體提供補助部署。

### 捐款

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

## 貢獻
請參閱 `AGENTS.md`，了解貢獻指引、程式碼風格與 PR 預期內容。

建議的貢獻檢查清單：
- 若有顯示相關變更，請附上面板型號與硬體備註。
- 列出驗證時實際執行的完整指令。
- 針對 UI 或電子紙輸出變更附上截圖／照片。
- 說明資料集修改內容（檔案 + 列／欄影響）。

## 授權
目前在倉庫根目錄尚未發現 `LICENSE` 檔案（以本次草稿檢視為準）。在補上授權檔前，重用權利尚未被明確授予。

假設：維護者可能會在後續更新中加入明確的開源授權。
