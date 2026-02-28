[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Eink Words GPT

**語言選項：** 中文（繁體）

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=flat-square&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=flat-square)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=flat-square)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=flat-square&logo=openai&logoColor=white)

這是一個結合 Raspberry Pi 與 Waveshare 電子紙的專案，可顯示動態挑選的詞彙、音標與多語同義資訊。系統可從本地資料集或 OpenAI 取得詞彙，將內容排版渲染後推送到支援的電子紙面板，並提供小型 HTTP 服務以觸發詞彙更新與取得渲染後圖片。

| 🔎 快速總覽 | 詳細內容 |
|---|---|
| 核心執行元件 | `app.py`（HTTP 服務）+ `words_gpt.py`（渲染迴圈） |
| 資料路徑 | `data/` 內 CSV 資料集 + SQLite 儲存 `words_phonetics.db` |
| 輸出目標 | Waveshare 電子紙面板與虛擬圖片輸出 |
| AI 相依 | 可選（`--enable_openai`），快取於 `cache/` |

## 📚 目錄
- [概覽](#概覽)
- [亮點](#亮點)
- [快速開始](#快速開始)
- [展示](#展示)
- [功能](#功能)
- [專案結構](#專案結構)
- [先決條件](#先決條件)
- [安裝](#安裝)
- [設定](#設定)
- [使用方式](#使用方式)
- [範例](#範例)
- [資料、快取與日誌](#資料快取與日誌)
- [開發備註](#開發備註)
- [疑難排解](#疑難排解)
- [OpenAI 使用說明](#openai-使用說明)
- [路線圖](#路線圖)
- [Support](#-support)
- [貢獻](#貢獻)
- [授權](#授權)

## 概覽
`words_gpt` 是一套以 Python 開發、面向電子紙裝置的詞彙卡片生成與顯示系統。

它整合了：
- 來自 CSV/本地資料集與可選 OpenAI 生成的詞彙來源。
- 增強處理（IPA 音標 + 多語同義欄位）。
- 面向硬體與虛擬輸出的渲染流程。
- 用於遠端觸發與圖片取得的 Tornado HTTP 服務。

目前程式碼主要集中在 `app.py`、`words_gpt.py`、`words_data.py`、`words_database.py` 與 `openai_request_json.py`。

## 亮點
- 🖼️ 電子紙渲染流程，支援多種內容模式（漢字、日文、阿拉伯文、中文、emoji）。
- 🗃️ 本地詞庫資料庫（`words_phonetics.db`）與 `data/` 下 CSV 詞表。
- 🤖 以 OpenAI 為基礎的詞彙選取與音標增強，輸出結構化 JSON。
- 🌐 供外部觸發與圖片取得的 HTTP 服務。
- ⚡ 快取層（`cache/`）可降低重複 OpenAI 呼叫。

## 快速開始
| 目標 | 指令 |
|---|---|
| 啟動 HTTP 伺服器（連接埠 `8082`） | `python app.py` |
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
- 來自 `words_gpt.py` 的硬體 + 虛擬渲染流程（`EPaperHardware`、`EPaperDisplay`）。
- `words_database.py` 提供 SQLite 持久化與動態欄位更新輔助。
- `openai_request_json.py` 提供 OpenAI 結構化 JSON 請求輔助與檔案快取。
- `pwa/` 內可選 PWA 資源，可用於輕量前端設定/預覽流程。

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
- `app.py`：Tornado Web 伺服器（預設連接埠 `8082`）與週期更新迴圈。
- `words_gpt.py`：獨立渲染迴圈與顯示類別。
- `words_data.py`：核心詞彙擷取/增強流程協調。
- `words_database.py`：SQLite 儲存輔助。
- `scripts/*.sh`：Raspberry Pi 安裝、服務安裝與 tmux 生命週期腳本。

## 先決條件
- Python `3.9+`（建議）。
- Raspberry Pi 目標裝置（硬體模式）。
- 支援的 Waveshare 電子紙面板。
- Pi 已啟用 SPI（`raspi-config`），並完成對應面板接線。

本專案使用的 Python 套件包含：
- `openai`、`tornado`、`Pillow`、`numpy`、`nltk`、`opencc`、`pykakasi`、`arabic_reshaper`、`python-bidi`、`pytz`。
- 安裝腳本另外安裝：`json5`、`pandas`、`spidev`、`RPi.GPIO`、`gpiozero`、`lgpio`。

## 安裝

### 選項 A：最小/手動安裝
安裝 Waveshare 驅動套件：
```bash
python setup.py install
```

若要使用 NLTK 詞表，請先下載一次：
```bash
python -m nltk.downloader words
```

### 選項 B：Raspberry Pi 自動化安裝（建議於裝置上執行）
在 repo 根目錄執行：
```bash
bash scripts/setup_pi_wordscard.sh
```

此腳本會：
- 安裝 apt 相依套件。
- 確保 SPI 已啟用。
- 建立並啟用 `wordscard` 虛擬環境。
- 安裝 Python 執行期相依。
- 安裝 Waveshare 套件。
- 在 tmux session 中啟動 `app.py`。

## 設定

### `.env` 行為
此 repo 在 import 階段會從 `.env` 載入環境變數，且會**覆寫**任何既有 shell 值。即使你已在 shell profile 輸出變數，也可確保本地覆寫具可預測性。

建立或更新 `.env`：
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### App 參數傳遞
systemd/tmux 腳本支援：
```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### CLI 旗標（伺服器與渲染器）
`app.py` 與 `words_gpt.py` 都支援：
- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

## 使用方式

### 執行 HTTP 伺服器
啟動服務（預設連接埠 `8082`）：
```bash
python app.py
```

程式中可觀察到的路由：
- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)`（來自 `words_card_temp/`）

相容性說明：較早文件提到 `GET /current_word`；目前 `app.py` 路由為 `GET /get_current_word`。

### 執行獨立渲染器
以 CSV 清單為基礎：
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

### Raspberry Pi 服務模式
安裝服務單元：
```bash
bash scripts/install_wordscard_service.sh
```

接著執行：
```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## 範例

### 觸發下一個隨機詞
```bash
curl "http://127.0.0.1:8082/next_random_word"
```

### 讀取目前詞彙 payload
```bash
curl "http://127.0.0.1:8082/get_current_word"
```

### 提交指定詞彙
```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

### 硬體煙霧測試
使用對應顯示器腳本：
```bash
python epd_7in3f_test.py
```

或：
```bash
python epd_13in3k_test.py
```

更多範例位於 `waveshare/examples/`。

## 資料、快取與日誌
| 區域 | 路徑 | 備註 |
|---|---|---|
| 詞表 | `data/` | 包含 `data/words_list.csv` 與主題 CSV 檔 |
| 持久化資料庫 | `words_phonetics.db` | 本地音標/增強儲存 |
| OpenAI/快取產物 | `cache/` | 降低重複請求 |
| 日誌 | `logs/`、`logs-word-phonetics/` | 執行期與更新日誌 |
| 產生的卡片 | `words_card_temp/` | 圖片輸出與靜態服務來源 |

## 開發備註
- 相依管理目前以腳本為主（`scripts/setup_pi_wordscard.sh`）+ `setup.py`；尚無 `requirements.txt` 或 `pyproject.toml`。
- 專案內有多個備份/舊版檔案（`words_data_*`、`words_gpt_old.py`）；目前主要執行路徑為 `app.py` + `words_gpt.py` + `words_data.py` + `words_database.py`。
- `env_loader.py` 在鍵存在時會永遠以 `.env` 覆寫環境變數。
- 伺服器模式會執行週期性刷新流程（約每 5 分鐘），並可在內部呼叫更新端點。

## 疑難排解
- `ModuleNotFoundError` 或匯入問題：
  - 確認虛擬環境已啟用且相依已安裝。
  - 在 Pi 上重新執行 `bash scripts/setup_pi_wordscard.sh`。
- OpenAI 錯誤（`401`、缺少 model/key）：
  - 檢查 `.env` 中 `OPENAI_API_KEY` 與可選 `OPENAI_MODEL`。
  - 確認裝置網路連線正常。
- 顯示未更新：
  - 確認面板型號/接線，並執行對應測試腳本（`epd_7in3f_test.py` 或 `epd_13in3k_test.py`）。
  - 確認 SPI 已啟用（`sudo raspi-config nonint do_spi 0`）。
  - 在 Pi 5 上，若裝置暴露 `/dev/spidev10.0`，請確保 `/dev/spidev0.0` 相容性符號連結存在。
- OpenCC 安裝問題：
  - 使用與發行版相容套件（`libopencc1` 或 `libopencc2`），如安裝腳本所示。
- API 路由不一致：
  - 目前詞彙 payload 請使用 `/get_current_word`，不是 `/current_word`。

## OpenAI 使用說明
OpenAI 存取為可選，但建議用於取得新詞彙與音標增強。`openai_request_json.py` 的結構化 JSON 輔助會將結果快取於 `cache/`，以降低重複呼叫。

## 路線圖
- 新增正式相依清單（`requirements.txt` 或 `pyproject.toml`），讓安裝可重現。
- 擴充 `i18n/` 並維護多語 README 版本。
- 在標準流程穩定後整合舊版/備份腳本變體。
- 補充 PWA 流程（`pwa/`）文件，含端點範例與截圖。
- 增加可重複執行的自動化測試，覆蓋資料與路由層行為。

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

### 你的支持能帶來什麼
- <b>維持工具開放</b>：主機、推理、資料儲存與社群營運。  
- <b>更快交付</b>：投入更多開源時間在 WordsCardEink 與相關學習工具。  
- <b>裝置原型開發</b>：電子紙硬體迭代與版面研究。  
- <b>讓更多人可用</b>：補助學生、創作者與社群團體部署。

### Donate

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
- 你的支持將用於研發與運維，幫助我持續公開分享更多專案與改進。  
- Your support sustains my research, development, and ops so I can keep sharing more open projects and improvements.

## 貢獻
請參考 `AGENTS.md`，其中包含貢獻規範、程式風格與 PR 期待。

建議的貢獻檢查清單：
- 針對顯示變更附上面板型號與硬體備註。
- 列出驗證時實際執行的指令。
- 針對 UI 或電子紙輸出變更附上截圖/照片。
- 描述資料集調整（檔案 + 列/欄影響）。

## 授權
目前在 repository 根目錄中尚未看到 `LICENSE` 檔案（以此草稿版本觀察）。在加入授權檔案前，重用權利尚未被明確授予。

假設：維護者可能在後續更新中加入明確的開源授權。
