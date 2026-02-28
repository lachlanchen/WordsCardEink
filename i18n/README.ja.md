[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Eink Words GPT


![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=flat-square&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=flat-square)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=flat-square)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=flat-square&logo=openai&logoColor=white)

Raspberry Pi + Waveshare e-ink を使って、動的に選ばれた語彙を発音情報と多言語の類義語付きで表示するプロジェクトです。ローカルデータセットまたは OpenAI から単語を取得し、レイアウトにレンダリングして対応する e-paper パネルへ出力できます。さらに、単語更新のトリガーやレンダリング画像取得のための小規模 HTTP サービスも提供します。

| 🔎 At a Glance | Details |
|---|---|
| Core runtime | `app.py` (HTTP service) + `words_gpt.py` (renderer loop) |
| Data path | CSV datasets in `data/` + SQLite store `words_phonetics.db` |
| Output targets | Waveshare e-paper panels and virtual image outputs |
| AI dependency | Optional (`--enable_openai`) with cache in `cache/` |

## 📚 Table of Contents
- [Overview](#overview)
- [Highlights](#highlights)
- [Quick Start](#quick-start)
- [Demos](#demos)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Examples](#examples)
- [Data, Cache, and Logs](#data-cache-and-logs)
- [Development Notes](#development-notes)
- [Troubleshooting](#troubleshooting)
- [Notes on OpenAI Usage](#notes-on-openai-usage)
- [Roadmap](#roadmap)
- [Support](#-support)
- [Contributing](#contributing)
- [License](#license)

## Overview
`words_gpt` は、e-ink デバイス向けの Python ベース語彙カード生成・表示システムです。

以下を組み合わせています。
- CSV / ローカルデータセットからの単語取得と、任意の OpenAI 生成。
- 拡張処理（IPA 発音記号 + 多言語類義語フィールド）。
- ハードウェア出力と仮想出力のレンダリングパイプライン。
- リモートトリガーと画像取得のための Tornado HTTP サービス。

現在のコードベースは `app.py`、`words_gpt.py`、`words_data.py`、`words_database.py`、`openai_request_json.py` が中心です。

## Highlights
- 🖼️ 複数コンテンツモード（漢字、日本語、アラビア語、中国語、絵文字）対応の e-ink レンダリングパイプライン。
- 🗃️ ローカル単語データベース（`words_phonetics.db`）と `data/` 配下の CSV 単語リスト。
- 🤖 構造化 JSON 出力による OpenAI ベースの単語選定と発音拡張。
- 🌐 外部トリガーと画像取得のための HTTP サービス。
- ⚡ OpenAI の重複呼び出しを減らすキャッシュ層（`cache/`）。

## Quick Start
| Goal | Command |
|---|---|
| Start HTTP server (port `8082`) | `python app.py` |
| Run standalone renderer (CSV) | `python words_gpt.py --use_csv` |
| Run with OpenAI + CSV | `python words_gpt.py --enable_openai --use_csv` |
| Emoji + simplified CJK mode | `python words_gpt.py --make_emoji --simplify` |
| Raspberry Pi auto setup | `bash scripts/setup_pi_wordscard.sh` |

## Demos
<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## Features
- `words_gpt.py` の `EPaperHardware` と `EPaperDisplay` によるハードウェア + 仮想レンダリングフロー。
- `words_data.py` の多言語拡張パイプライン（IPA、日本語バリアント、アラビア語、フランス語、中国語フィールド）。
- `words_database.py` の動的フィールド更新ヘルパーを備えた SQLite 永続化。
- `openai_request_json.py` のファイルキャッシュ付き OpenAI 構造化 JSON リクエストヘルパー。
- 軽量なフロントエンド設定/プレビュー向けの任意 PWA アセット（`pwa/`）。

## Project Structure
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

重要な実行時ファイル:
- `app.py`: Tornado Web サーバー（デフォルトポート `8082`）と定期更新ループ。
- `words_gpt.py`: スタンドアロンのレンダリングループと表示クラス。
- `words_data.py`: 単語取得/拡張オーケストレーションの中核。
- `words_database.py`: SQLite ストアヘルパー。
- `scripts/*.sh`: Raspberry Pi セットアップ、サービス導入、tmux ライフサイクルスクリプト。

## Prerequisites
- Python `3.9+`（推奨）。
- Raspberry Pi ターゲット（ハードウェアモード用）。
- 対応する Waveshare e-paper パネル。
- Pi で SPI を有効化（`raspi-config`）し、パネル別の配線を設定。

このプロジェクトで使用する Python パッケージには以下が含まれます。
- `openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`。
- セットアップスクリプトで追加インストール: `json5`, `pandas`, `spidev`, `RPi.GPIO`, `gpiozero`, `lgpio`。

## Installation

### Option A: Minimal/manual install
Waveshare ドライバーパッケージをインストール:
```bash
python setup.py install
```

NLTK 単語リストを使う場合は一度だけダウンロード:
```bash
python -m nltk.downloader words
```

### Option B: Raspberry Pi automated setup (recommended on device)
リポジトリルートから実行:
```bash
bash scripts/setup_pi_wordscard.sh
```

このスクリプトで実行される内容:
- apt 依存関係をインストール。
- SPI が有効であることを確認。
- `wordscard` 仮想環境を作成して有効化。
- Python 実行時依存関係をインストール。
- Waveshare パッケージをインストール。
- `app.py` を tmux セッション内で起動。

## Configuration

### `.env` behavior
このリポジトリは import 時に `.env` から環境変数を読み込み、既存のシェル値を**上書き**します。シェルプロファイルですでに export されていても、ローカル上書きが常に同じ結果になる設計です。

`.env` を作成または更新:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### App argument passthrough
systemd / tmux スクリプトでは以下をサポート:
```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### CLI flags (server and renderer)
`app.py` と `words_gpt.py` の両方でサポート:
- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

## Usage

### Running the HTTP server
サービスを起動（デフォルトポート `8082`）:
```bash
python app.py
```

コード上で確認できるルート:
- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)`（`words_card_temp/` から配信）

互換性メモ: 以前のドキュメントでは `GET /current_word` と記載されていましたが、現在の `app.py` のルートは `GET /get_current_word` です。

### Running standalone renderer
CSV ベースのリスト:
```bash
python words_gpt.py --use_csv
```

OpenAI を有効化:
```bash
python words_gpt.py --enable_openai --use_csv
```

絵文字レンダリング + 簡体 CJK:
```bash
python words_gpt.py --make_emoji --simplify
```

### Service mode on Raspberry Pi
サービスユニットをインストール:
```bash
bash scripts/install_wordscard_service.sh
```

次に実行:
```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## Examples

### Trigger next random word
```bash
curl "http://127.0.0.1:8082/next_random_word"
```

### Read current word payload
```bash
curl "http://127.0.0.1:8082/get_current_word"
```

### Submit explicit word
```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

### Hardware smoke tests
ディスプレイに対応したスクリプトを使用:
```bash
python epd_7in3f_test.py
```

または:
```bash
python epd_13in3k_test.py
```

追加の例は `waveshare/examples/` にあります。

## Data, Cache, and Logs
| Area | Path(s) | Notes |
|---|---|---|
| Word lists | `data/` | `data/words_list.csv` とテーマ別 CSV ファイルを含む |
| Persistent DB | `words_phonetics.db` | ローカル発音/拡張ストア |
| OpenAI/cache artifacts | `cache/` | 重複リクエストを削減 |
| Logs | `logs/`, `logs-word-phonetics/` | 実行時ログと更新ログ |
| Generated cards | `words_card_temp/` | 画像出力と静的配信のソース |

## Development Notes
- 依存関係管理はスクリプト優先（`scripts/setup_pi_wordscard.sh`）+ `setup.py`。現時点では `requirements.txt` / `pyproject.toml` は未整備です。
- 複数のバックアップ/レガシーファイル（`words_data_*`、`words_gpt_old.py`）が存在し、主要な実行経路は `app.py` + `words_gpt.py` + `words_data.py` + `words_database.py` です。
- `env_loader.py` はキーが存在する場合、`.env` の環境変数で常に上書きします。
- サーバーモードでは定期更新フロー（約 5 分ごと）が動作し、内部的に更新エンドポイントを呼ぶことがあります。

## Troubleshooting
- `ModuleNotFoundError` や import 問題:
  - 仮想環境が有効で、依存関係がインストール済みか確認してください。
  - Pi 上で `bash scripts/setup_pi_wordscard.sh` を再実行してください。
- OpenAI エラー（`401`、モデル/キー未設定）:
  - `.env` の `OPENAI_API_KEY` と任意の `OPENAI_MODEL` を確認してください。
  - デバイスのネットワーク接続を確認してください。
- ディスプレイが更新されない:
  - パネル型番/配線を確認し、対応テストスクリプト（`epd_7in3f_test.py` または `epd_13in3k_test.py`）を実行してください。
  - SPI が有効化されているか確認してください（`sudo raspi-config nonint do_spi 0`）。
  - Pi 5 では、デバイスが `/dev/spidev10.0` を公開する場合に `/dev/spidev0.0` 互換シンボリックリンクが必要です。
- OpenCC インストール問題:
  - セットアップスクリプトと同様に、ディストリ互換パッケージ（`libopencc1` または `libopencc2`）を使用してください。
- API ルート不一致:
  - 現在のペイロード取得は `/current_word` ではなく `/get_current_word` を使ってください。

## Notes on OpenAI Usage
OpenAI アクセスは任意ですが、新しい単語生成と発音拡張には推奨です。`openai_request_json.py` の構造化 JSON ヘルパーは、重複呼び出しを減らすため `cache/` 配下に結果をキャッシュします。

## Roadmap
- 再現可能なインストールのための正式な依存関係マニフェスト（`requirements.txt` または `pyproject.toml`）を追加。
- `i18n/` の翻訳 README バリアントを拡充し、継続保守。
- 正式フロー確定後、レガシー/バックアップ系スクリプトを統合。
- エンドポイント例とスクリーンショット付きで PWA ワークフロー（`pwa/`）を文書化。
- データ処理とルート挙動の再現可能な自動テストを追加。

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

このプロジェクトが役に立った場合、以下の支援リンクは継続的な保守とハードウェア改善に直接つながります。

### What your support makes possible
- <b>Keep tools open</b>: ホスティング、推論、データ保管、コミュニティ運営。  
- <b>Ship faster</b>: WordsCardEink と関連学習ツールへ集中して OSS 開発。  
- <b>Prototype devices</b>: e-ink ハードウェアの反復開発と表示レイアウト研究。  
- <b>Access for all</b>: 学生、クリエイター、コミュニティ向け導入支援。  

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

- ご支援は研究・開発・運用の継続を支え、より多くのオープンプロジェクト公開につながります。  
- 你的支持将用于研发与运维，帮助我持续公开分享更多项目与改进。  
- Your support sustains my research, development, and ops so I can keep sharing more open projects and improvements.

## Contributing
コントリビューションガイド、コーディングスタイル、PR の期待事項は `AGENTS.md` を参照してください。

推奨コントリビューションチェックリスト:
- ディスプレイ変更時はパネル型番とハードウェアメモを含める。
- 検証時に実行したコマンドを正確に記載する。
- UI または e-ink 出力変更にはスクリーンショット/写真を添付する。
- データセット編集内容（ファイル + 行/列への影響）を説明する。

## License
リポジトリルートには現在 `LICENSE` ファイルがありません（このドラフト時点の観測）。ライセンスファイルが追加されるまでは、再利用権は明示的に付与されていません。

前提: メンテナーは後続アップデートで明示的なオープンソースライセンスを追加する可能性があります。
