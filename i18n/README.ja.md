[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Eink Words GPT

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi-green)
![Display](https://img.shields.io/badge/display-Waveshare%20e--Paper-black)
![Status](https://img.shields.io/badge/status-active%20prototype-orange)
![Server](https://img.shields.io/badge/http-Tornado-0A7EA4)
![Storage](https://img.shields.io/badge/storage-SQLite-003B57)
![AI](https://img.shields.io/badge/OpenAI-optional-412991)

Raspberry Pi + Waveshare e-ink を使い、動的に選ばれた語彙を発音情報と多言語類義語付きで表示するプロジェクトです。ローカルデータセットまたは OpenAI から単語を取得し、レイアウトにレンダリングして対応 e-paper パネルへ出力できます。さらに、単語更新のトリガーとレンダリング画像取得のための小規模 HTTP サービスも提供します。

## 概要
`words_gpt` は、e-ink デバイス向けの Python 製ボキャブラリーカード生成・表示システムです。

以下を組み合わせています:
- CSV/ローカルデータセットからの単語取得と、任意で OpenAI 生成。
- 拡張処理（IPA 発音記号 + 多言語類義語フィールド）。
- ハードウェア出力と仮想出力のレンダリングパイプライン。
- リモートトリガーと画像取得のための Tornado HTTP サービス。

現在のコードベースは `app.py`、`words_gpt.py`、`words_data.py`、`words_database.py`、`openai_request_json.py` を中心に構成されています。

## ハイライト
- 🖼️ 複数コンテンツモード（漢字、日本語、アラビア語、中国語、絵文字）に対応した e-ink レンダリングパイプライン。
- 🗃️ `data/` 内の CSV 単語リストと連携するローカル単語データベース（`words_phonetics.db`）。
- 🤖 構造化 JSON 出力による OpenAI ベースの単語選定と発音拡張。
- 🌐 外部トリガーと画像取得のための HTTP サービス。
- ⚡ OpenAI の重複呼び出しを減らすキャッシュ層（`cache/`）。

## クイックスタート
| 目的 | コマンド |
|---|---|
| HTTP サーバーを起動（ポート `8082`） | `python app.py` |
| スタンドアロンレンダラーを実行（CSV） | `python words_gpt.py --use_csv` |
| OpenAI + CSV で実行 | `python words_gpt.py --enable_openai --use_csv` |
| 絵文字 + 簡体 CJK モード | `python words_gpt.py --make_emoji --simplify` |
| Raspberry Pi 自動セットアップ | `bash scripts/setup_pi_wordscard.sh` |

## デモ
<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## 機能
- `words_gpt.py` の `EPaperHardware`、`EPaperDisplay` によるハードウェア + 仮想レンダリングフロー。
- `words_data.py` の多言語拡張パイプライン（IPA、日本語バリアント、アラビア語、フランス語、中国語フィールド）。
- `words_database.py` の動的フィールド更新ヘルパーを備えた SQLite 永続化。
- `openai_request_json.py` のファイルキャッシュ付き OpenAI 構造化 JSON リクエストヘルパー。
- 軽量フロントエンド設定/プレビュー向けの任意 PWA アセット（`pwa/`）。

## プロジェクト構成
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

## 前提条件
- Python `3.9+`（推奨）。
- Raspberry Pi ターゲット（ハードウェアモード時）。
- 対応 Waveshare e-paper パネル。
- Pi で SPI を有効化（`raspi-config`）、加えてパネル別の配線設定。

このプロジェクトで使用される主な Python パッケージ:
- `openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`。
- セットアップスクリプトで追加導入: `json5`, `pandas`, `spidev`, `RPi.GPIO`, `gpiozero`, `lgpio`。

## インストール

### Option A: 最小/手動インストール
Waveshare ドライバーパッケージをインストール:
```bash
python setup.py install
```

NLTK の単語リストを使う場合は一度だけダウンロード:
```bash
python -m nltk.downloader words
```

### Option B: Raspberry Pi 自動セットアップ（デバイス上で推奨）
リポジトリルートで実行:
```bash
bash scripts/setup_pi_wordscard.sh
```

このスクリプトは次を実行します:
- apt 依存関係をインストール。
- SPI が有効か確認。
- `wordscard` 仮想環境を作成して有効化。
- Python 実行時依存関係をインストール。
- Waveshare パッケージをインストール。
- tmux セッション内で `app.py` を起動。

## 設定

### `.env` の挙動
このリポジトリは import 時点で `.env` から環境変数を読み込み、既存のシェル値を**上書き**します。これにより、シェルプロファイルで値を export 済みでも、ローカル上書きが決定的になります。

`.env` を作成または更新:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### App 引数の受け渡し
systemd/tmux スクリプトでは以下をサポート:
```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### CLI フラグ（サーバーおよびレンダラー）
`app.py` と `words_gpt.py` の両方でサポート:
- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

## 使い方

### HTTP サーバーを実行
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

### スタンドアロンレンダラーを実行
CSV ベースリスト:
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

### Raspberry Pi のサービスモード
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

## 例

### 次のランダム単語をトリガー
```bash
curl "http://127.0.0.1:8082/next_random_word"
```

### 現在の単語ペイロードを取得
```bash
curl "http://127.0.0.1:8082/get_current_word"
```

### 単語を明示指定して送信
```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

### ハードウェア スモークテスト
ディスプレイに対応するスクリプトを使用:
```bash
python epd_7in3f_test.py
```

または:
```bash
python epd_13in3k_test.py
```

追加例は `waveshare/examples/` にあります。

## データ・キャッシュ・ログ
| 領域 | パス | メモ |
|---|---|---|
| 単語リスト | `data/` | `data/words_list.csv` とテーマ別 CSV を含む |
| 永続 DB | `words_phonetics.db` | ローカル発音/拡張ストア |
| OpenAI/キャッシュ成果物 | `cache/` | 重複リクエストを削減 |
| ログ | `logs/`, `logs-word-phonetics/` | 実行時ログと更新ログ |
| 生成カード | `words_card_temp/` | 画像出力および静的配信元 |

## 開発メモ
- 依存関係管理はスクリプト中心（`scripts/setup_pi_wordscard.sh`）+ `setup.py`。`requirements.txt` と `pyproject.toml` はまだありません。
- 複数のバックアップ/レガシーファイル（`words_data_*`、`words_gpt_old.py`）が存在し、主要な実行経路は `app.py` + `words_gpt.py` + `words_data.py` + `words_database.py` です。
- `env_loader.py` はキーが存在する場合、`.env` から環境変数を常に上書きします。
- サーバーモードでは定期更新フロー（約 5 分ごと）が動作し、内部的に更新エンドポイントを呼び出す場合があります。

## トラブルシューティング
- `ModuleNotFoundError` や import 問題:
  - 仮想環境が有効で、依存関係がインストール済みであることを確認してください。
  - Pi 上で `bash scripts/setup_pi_wordscard.sh` を再実行してください。
- OpenAI エラー（`401`、モデル/キー未設定）:
  - `.env` の `OPENAI_API_KEY` と任意の `OPENAI_MODEL` を確認してください。
  - デバイスのネットワーク接続を確認してください。
- ディスプレイが更新されない:
  - パネル型番/配線を確認し、対応テストスクリプト（`epd_7in3f_test.py` または `epd_13in3k_test.py`）を実行してください。
  - SPI が有効化されていることを確認してください（`sudo raspi-config nonint do_spi 0`）。
  - Pi 5 では、デバイスが `/dev/spidev10.0` を公開している場合に `/dev/spidev0.0` 互換シンボリックリンクが必要です。
- OpenCC インストール問題:
  - セットアップスクリプトと同様に、ディストリ互換パッケージ（`libopencc1` または `libopencc2`）を使用してください。
- API ルート不一致:
  - 現在のペイロード取得は `/current_word` ではなく `/get_current_word` を使用してください。

## OpenAI 利用に関するメモ
OpenAI アクセスは任意ですが、新しい単語生成と発音拡張には推奨されます。`openai_request_json.py` の構造化 JSON ヘルパーは、重複呼び出しを減らすため `cache/` 配下に結果をキャッシュします。

## ロードマップ
- 再現可能なインストールのため、正式な依存関係マニフェスト（`requirements.txt` または `pyproject.toml`）を追加。
- `i18n/` の翻訳 README バリアントを拡充し、継続保守。
- 正式フロー確定後、レガシー/バックアップスクリプトのバリアントを統合。
- エンドポイント例とスクリーンショット付きで PWA ワークフロー（`pwa/`）を文書化。
- データ処理とルート挙動向けに再現可能な自動テストを追加。

## サポート

### ご支援で実現できること
- <b>ツールをオープンに維持</b>: ホスティング、推論、データ保管、コミュニティ運営。  
- <b>開発を加速</b>: WordsCardEink と関連学習ツールへの集中した OSS 開発時間。  
- <b>デバイス試作</b>: e-ink ハードウェアの反復開発と表示レイアウト研究。  
- <b>誰でもアクセス可能に</b>: 学生、クリエイター、コミュニティ向けの導入支援。

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
- 你的支持将用于研发与运维，帮助我持续公开分享更多项目与改进。  
- Your support sustains my research, development, and ops so I can keep sharing more open projects and improvements.

## コントリビュート
コントリビューションガイド、コーディングスタイル、PR 要件は `AGENTS.md` を参照してください。

推奨コントリビュートチェックリスト:
- ディスプレイ変更時はパネル型番 + ハードウェアメモを含める。
- 検証で実行した正確なコマンドを列挙する。
- UI または e-ink 出力変更にはスクリーンショット/写真を添付する。
- データセット編集は内容（ファイル + 行/列への影響）を記述する。

## ライセンス
このドラフト時点では、リポジトリルートに `LICENSE` ファイルが存在しません。ライセンスファイルが追加されるまでは、再利用権は明示的に許諾されていません。

前提: メンテナーが後続アップデートで明示的なオープンソースライセンスを追加する可能性があります。
