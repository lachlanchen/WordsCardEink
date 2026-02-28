[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 🖨️ Eink Words GPT

**このREADMEの言語:** 日本語

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

> Raspberry Pi と Waveshare e-ink を使って、IPA の音声記号と多言語ヒント付きで語彙カードを動的に描画するプロジェクトです。ローカルの CSV 処理、必要に応じた AI 強化、e-paper 描画、リモート HTTP 制御に対応しています。

起動モードをひと目で確認:
`app.py`（サービス）と `words_gpt.py`（単体レンダラー）は独立実行または同時実行ができます。

| 🔎 一覧 | 詳細 |
|---|---|
| **主な実行系** | `app.py`（HTTP サービス）+ `words_gpt.py`（レンダーループ） |
| **データ経路** | `data/` 配下の CSV データセット + SQLite ストア `words_phonetics.db` |
| **出力先** | Waveshare e-paper パネルと仮想画像出力 |
| **AI 依存** | `--enable_openai` で任意有効化。`cache/` にリクエストキャッシュあり |
| **既定の更新間隔** | サーバーは `8082`、おおよそ 5 分おきに自動更新 |

## 📚 目次
- [概要](#overview)
- [ハイライト](#highlights)
- [デモ](#demos)
- [プロジェクト構成](#project-structure)
- [前提条件](#prerequisites)
- [インストール](#installation)
- [設定](#configuration)
- [使用方法](#usage)
- [サンプル](#examples)
- [データ・キャッシュ・ログ](#data-cache-and-logs)
- [開発メモ](#development-notes)
- [トラブルシューティング](#troubleshooting)
- [ロードマップ](#roadmap)
- [サポート](#support)
- [貢献](#contributing)
- [ライセンス](#license)

---

<a id="overview"></a>
## 概要

`words_gpt` は e-ink 表示向けの語彙カード生成スタックです。データの統合、音声記号の補完、レンダリングを 2 つの実行モードで統一しています。

- 長時間稼働する Tornado サービス (`app.py`)：リモート制御と画像配信を担当。
- 単体レンダラー (`words_gpt.py`)：ポーリング、ループ、または直接レンダリングモードで実行可能。

主要モジュール:

- `words_data.py` / `words_data_utils.py`：単語処理と拡張ワークフロー。
- `words_data_with_legacy.py`、`words_data_without_legacy.py`、`words_data_workable*.py`：互換性維持を含む複数フロー。
- `words_database.py`：SQLite 操作。
- `openai_request_json.py`：キャッシュ付き構造化 OpenAI リクエストとリトライ制御。
- `env_loader.py`：環境変数の決定的読み込み。
- `words_update.py`：DB メンテナンスと再チェック。
- `app.py` と `words_gpt.py`：サービス/レンダリングのライフサイクル管理。
- `pwa/`：軽量ブラウザプレビュー/設定ツール。

<a id="highlights"></a>
## ハイライト

- 多言語モードによる e-paper 描画パイプライン（日本語（漢字を含む）/アラビア語/中国語/絵文字モード）。
- ローカルのみ、または OpenAI 補助の単語取得を同一ワークフローで切り替え可能。
- オプションとして簡体字レンダリング (`--simplify`)。
- リモート制御用の HTTP エンドポイント（`/next_random_word`、`/display_word`、`/get_current_word`、`/get_current_word_page`、`/get_words_card`）。
- キャッシュと永続化で OpenAI 呼び出し回数を抑制。
- `waveshare/` 付属ドライバーを通じたハードウェア例とパッケージ。

<a id="demos"></a>
## デモ

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

<a id="project-structure"></a>
## プロジェクト構成

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

実行時の中核ファイル:

- `app.py`：8082 ポートの Tornado アプリ + 定期 `next_random_word` トリガ。
- `words_gpt.py`：単体レンダラーとディスプレイ抽象化（`EPaperHardware`、`EPaperDisplay`）。
- `words_data.py`：取得・拡張フローと単語選定ユーティリティ。
- `words_database.py`：保存メタデータと単語キャッシュ操作向け SQLite ヘルパー。
- `scripts/*.sh`：インストールとサービス起動、Raspberry Pi ブートストラップ支援。
- `words_update.py`：データ品質維持のための一括 DB 更新・再チェックヘルパー。

<a id="prerequisites"></a>
## 前提条件

- Python `3.9+`（推奨）
- Raspberry Pi（ハードウェア利用時は必須）
- 対応 Waveshare e-paper パネル（例: 7.3F、13K 系）
- SPI 有効化（`raspi-config`）、配線確認、安定電源
- NLTK 単語ソースを利用する場合の NLTK コーパス

実行パスで利用される主な依存関係: `openai`、`tornado`、`Pillow`、`numpy`、`nltk`、`opencc`、`pykakasi`、`arabic_reshaper`、`python-bidi`、`pytz`。

<a id="installation"></a>
## インストール

### オプション 1 — 最小/手動インストール（デスクトップまたは Pi）

リポジトリルートで実行:

```bash
python setup.py install
```

必要に応じて:

```bash
python -m nltk.downloader words
```

### オプション 2 — Raspberry Pi 自動セットアップ（デバイス上で推奨）

リポジトリルートで実行:

```bash
bash scripts/setup_pi_wordscard.sh
```

実行内容:

- Pi 向け依存関係のインストール
- SPI 有効化チェック
- `wordscard` 仮想環境のセットアップ
- Python/ランタイムパッケージのインストール
- Waveshare パッケージのインストール
- `tmux` によるアプリ起動

### オプション 3 — systemd サービス登録

`systemd` 配下でアプリライフサイクルを登録:

```bash
bash scripts/install_wordscard_service.sh
```

続けて:

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

<a id="configuration"></a>
## 設定

### 環境変数（`.env`）

`env_loader` は環境変数を読み取り、プロセス起動時コンテキストへ反映します。代表的な利用例:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

前提: シークレットはローカル環境設定に保持し、バージョン管理にはコミットしないこと。

### ランタイム引数（`app.py` / `words_gpt.py` 共通）

| CLI フラグ | 目的 |
| --- | --- |
| `--enable_openai` | OpenAI 補助モードを有効化 |
| `--make_emoji` | 絵文字特化カードを描画 |
| `--ignore_list` | 指定無視リストの単語をスキップ |
| `--simplify` | 簡体字系 CJK 出力を生成 |
| `--use_csv` | CSV データセットから単語を読み込み |
| `--complete_csv` | 完全 CSV ソースモードを使用 |
| `--filename <csv_file>` | 対象 CSV ファイルを指定 |

`APP_ARGS` は起動スクリプト経由で渡すことができます。例:

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### app モード時のルーティング

確認済みの API ルート:

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)`（配信元: `words_card_temp/`）

互換性メモ: 旧ドキュメントでは `GET /current_word` に言及される場合があります。現在のルートは `GET /get_current_word` です。

### OpenAI 利用時の注意

OpenAI の機能は任意で、CLI/環境変数フラグで制御します。`cache/` に保存されるキャッシュは再現性とレート制御に役立ちます。オフライン寄りの安定運用は、まず CSV モード (`--use_csv`) で始め、必要時にのみ OpenAI を有効化するのが安全です。

<a id="usage"></a>
## 使い方

### HTTP サーバーを起動

```bash
python app.py
```

このプロセスは `words_card_temp/` に最新画像を保持し、フロントエンド/スクリプトから参照できるエンドポイントを公開します。

### レンダラーを直接実行

CSV モード:

```bash
python words_gpt.py --use_csv
```

OpenAI モード:

```bash
python words_gpt.py --enable_openai --use_csv
```

絵文字 + 簡体字モード:

```bash
python words_gpt.py --make_emoji --simplify
```

### Pi ハードウェア上で実行

- tmux 起動スクリプトを使用:

```bash
bash scripts/start_wordscard.sh
```

- tmux 停止スクリプトを使用:

```bash
bash scripts/stop_wordscard.sh
```

<a id="examples"></a>
## 使用例

次のランダムカード情報を取得:

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

現在保存されている単語を取得:

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

描画済みページ画像のペイロードを要求:

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

明示的な単語を送信:

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

フォーム形式エンドポイントでカード描画を開始:

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

<a id="data-cache-and-logs"></a>
## データ・キャッシュ・ログ

アプリで使用する一般的な成果物:

- `data/`: 精選済み CSV データセット
- `words_phonetics.db`: SQLite キャッシュ/ソース DB
- `cache/`: OpenAI リクエスト・結果キャッシュ
- `word_phonetics_processed.csv`: 処理済み/派生データセット
- `logs/`, `logs-word-phonetics/`: 実行ログ
- `words_card_temp/`: 生成カードと一時出力
- `pic/` と `figs/`: 参考画像とバナー

<a id="development-notes"></a>
## 開発メモ

- レガシー/バックアップ用モジュールや成果物が存在します（例: `words_gpt_old.py`、`lib.old`）。移植や互換性が意図されていない限り、参照用途として扱うべきです。
- `words_update.py` には DB 品質維持のための一括更新・再チェックヘルパーが含まれます。
- ハードウェア検証は `epd_*_test.py` と `waveshare/examples/*` のデモスクリプトで行います。
- リポジトリ直下に `requirements.txt` やロックファイルはありません。依存関係はセットアップスクリプトと直接インストールフローで管理されます。
- このリポジトリには自動テストスイートは設定されていません。

<a id="troubleshooting"></a>
## トラブルシューティング

- Raspberry Pi の GPIO/SPI モジュールからの `ImportError`:
  - `scripts/setup_pi_wordscard.sh` を使った Pi セットアップを実行するか、互換デバイスで依存関係を明示インストールしてください。
- 画像・静的ファイル系で `403/404` が出る:
  - ルート利用法（`/get_current_word*`）を確認し、`words_card_temp/` が書き込み可能か確認してください。
- OpenAI モードで単語ペイロードが空/不正:
  - `OPENAI_API_KEY`（および org/model 値）が読み込まれているか確認し、`cache/` とログを調査してください。
- 描画不良・文字欠け:
  - `words_gpt.py` 内のフォントパス、表示解像度定数、選択モードを確認してください。
- API が古いデータを返す:
  - `POST /next_random_word` を手動で呼び出し、`app.py` のコールバック間隔を確認してください。
- ハードウェア描画が停止しているように見える:
  - tmux セッションと systemd ログを `journalctl -u wordscard` で確認してください。
- データセット/辞書のエントリ不足:
  - `data/` 内の CSV を検証し、`words_update.py` のメンテナンスタスクを実行してください。

<a id="roadmap"></a>
## ロードマップ

- 最小構成の `requirements.txt` / 再現可能な導入マニフェストを追加。
- より明確な実行モードと `--help` の CLI ドキュメントを追加。
- レンダリングモードのドキュメント拡充（`japanese_synonym`、`arabic_synonym`、`film`、その他のワークフロー）。
- エラーハンドリングと利用者向け API レスポンススキーマの標準化。
- ハードウェア不要の CI 検証向け軽量スモークテストスタブを追加。

<a id="support"></a>
## 貢献

コントリビューションは歓迎です。推奨手順:

1. 変更範囲を1つの振る舞い領域（レンダリング、データ、API、スクリプト）に限定する。
2. ユーザー向け挙動が変わる場合は、コマンドの使用方法と説明を更新する。
3. 可能な限り既存の CLI フラグとエンドポイント互換性を維持する。
4. ハードウェア関連スクリプトを変更する場合は、テスト済みデバイス/モデルと実行コマンドを明記する。

<a id="license"></a>
## ライセンス

現在のリポジトリルートには `LICENSE` ファイルが存在しません。そのため、本リポジトリ内で配布条件が明示されていない状態です。再配布・再利用条件を明確にしたい場合は追加してください。


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
