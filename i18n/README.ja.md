[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 🖨️ Eink Words GPT

**この版の言語:** 日本語

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=for-the-badge&logo=githubactions&logoColor=white)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=for-the-badge&logo=python&logoColor=white)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=for-the-badge&logo=openai&logoColor=white)

Raspberry Pi + Waveshare e-ink 向けの語彙カード生成プロジェクトで、IPA 表記と多言語ヒントを付与した単語カードを動的にレンダリングします。ローカル CSV ワークフロー、AI 補完（任意）、e-paper レンダリング、リモート HTTP 制御をサポートします。

| 🔎 At a Glance | Details |
|---|---|
| Core runtime | `app.py` (HTTP サービス) + `words_gpt.py` (レンダーループ) |
| Data path | `data/` の CSV データセット + SQLite ストア `words_phonetics.db` |
| Output targets | Waveshare e-paper パネルと仮想画像出力 |
| AI dependency | オプション (`--enable_openai`) と `cache/` のキャッシュ |
| Main loop defaults | サーバーは `8082`、更新周期は約 5 分 |

## 📚 目次
- [概要](#概要)
- [ハイライト](#ハイライト)
- [デモ](#デモ)
- [プロジェクト構成](#プロジェクト構成)
- [前提条件](#前提条件)
- [インストール](#インストール)
- [設定](#設定)
- [使い方](#使い方)
- [使用例](#使用例)
- [データ・キャッシュ・ログ](#データ・キャッシュ・ログ)
- [開発ノート](#開発ノート)
- [トラブルシューティング](#トラブルシューティング)
- [ロードマップ](#ロードマップ)
- [サポート](#サポート)
- [コントリビューション](#コントリビューション)
- [ライセンス](#ライセンス)

---

## 概要

`words_gpt` は、e-ink 表示向けに語彙カードを生成する Python スタックです。データ連携、音声表記の補完、レンダリング制御を 2 つの実行モードでまとめて提供します。

- 長時間稼働の Tornado サービス (`app.py`)（リモート制御と画像提供）
- 単独実行レンダラー (`words_gpt.py`)（ポーリング、ループ、直接描画モード）

主要モジュール:

- `words_data.py` / `words_data_utils.py`（単語取得と補完ワークフロー）
- `words_database.py`（SQLite 連携）
- `openai_request_json.py`（ディスクキャッシュ付き OpenAI 構造化リクエスト）
- `env_loader.py`（決定論的な環境読み込み）
- `words_update.py`（DB メンテナンスと再確認ワークフロー）
- `app.py` と `words_gpt.py`（サービス/レンダリング寿命）

## ハイライト

- 多言語・多コンテンツモードを持つ e-ink レンダーパイプライン
  - 日本語変種、漢字モード、アラビア語、中国語、絵文字モード
- ローカル単語ソースと OpenAI ソースを単一フローで併用可能
- レンダーパスでの簡体字出力モード（`--simplify`）がオプション
- 直接操作用 API (`/next_random_word`、`/display_word` など)
- キャッシュと永続化で API 呼び出しを抑制
- `pwa/` 以下の軽量プレビュー・設定フロー向け PWA アセット（任意）

## デモ

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

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

重要な実行ファイル:

- `app.py`: ポート `8082` で Tornado アプリを起動し、定期的に `next_random_word` をトリガーします。
- `words_gpt.py`: 単体レンダラーと表示抽象化（`EPaperHardware`、`EPaperDisplay`）。
- `words_data.py`: 進化した単語取得/補完ワークフローとヘルパー。
- `words_database.py`: メタデータ保管と単語キャッシュ操作のための SQLite ヘルパー。
- `scripts/*.sh`: サービス運用と Raspberry Pi 初期化ヘルパー。

## 前提条件

- Python `3.9+`（推奨）
- Raspberry Pi（ハードウェアモードでは必須）
- 対応する Waveshare e-paper パネル（例: 7.3F / 13K ファミリー）
- SPI 有効化 (`raspi-config`)、配線、電源が安定した実行環境
- `nltk` 単語ソースを使う場合は NLTK corpus の準備

コードベースで使用されている主な依存パッケージ:
`openai`、`tornado`、`Pillow`、`numpy`、`nltk`、`opencc`、`pykakasi`、`arabic_reshaper`、`python-bidi`、`pytz`。

## インストール

### オプション 1 — 最小構成 / 手動インストール（PC または Pi）

リポジトリルートから:

```bash
python setup.py install
```

必要な場合:

```bash
python -m nltk.downloader words
```

### オプション 2 — Raspberry Pi 自動セットアップ（本体推奨）

リポジトリルートから:

```bash
bash scripts/setup_pi_wordscard.sh
```

このスクリプトは次を実行します:

- Pi 固有の依存関係
- SPI 有効化
- `wordscard` 仮想環境のセットアップ
- Python / runtime パッケージのインストール
- Waveshare パッケージのインストール
- `tmux` によるアプリ起動

### オプション 3 — systemd サービス化

`systemd` でアプリのライフサイクルを登録します:

```bash
bash scripts/install_wordscard_service.sh
```

続けて:

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## 設定

### 環境変数（`.env`）

本リポジトリは、既存のシェル変数を上書きする `.env` ローダーを利用します。意図して使用してください:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### 実行フラグ（`app.py` と `words_gpt.py` の両方で使用）

- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

Pi 起動スクリプトは `APP_ARGS` で引数を受け渡せます（例）:

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### app モードのルーティング

現在の実装で確認されるルート:

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)`（`words_card_temp/` から配信）

互換性メモ: 旧仕様で参照されていた `GET /current_word` は、現在は `GET /get_current_word` に統一されています。

### OpenAI 利用に関する注意

OpenAI 機能は任意で、CLI/env フラグで制御します。API 応答はキャッシュされるため、再現性とレート制御に有効です。リソースの限られた環境では、まず CSV モード（`--use_csv`）で起動し、必要時のみ `--enable_openai` を有効にする運用が推奨です。

## 使い方

### HTTP サーバーを起動

```bash
python app.py
```

プロセスは `words_card_temp/` に画像を保持し、フロントエンドや簡易スクリプト向けに HTTP エンドポイントを提供します。

### レンダラーを直接実行

CSV モード:

```bash
python words_gpt.py --use_csv
```

OpenAI モード:

```bash
python words_gpt.py --enable_openai --use_csv
```

絵文字 + 簡体字対応:

```bash
python words_gpt.py --make_emoji --simplify
```

### Pi ハードウェアで実行

- `tmux` 起動スクリプトで開始:

```bash
bash scripts/start_wordscard.sh
```

- `tmux` 停止スクリプトで終了:

```bash
bash scripts/stop_wordscard.sh
```

## 使用例

次のランダムカードのメタデータを取得:

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

保存済みの現在語を取得:

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

レンダリング済みページ画像のペイロードを要求:

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

明示的な単語を送信:

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

フォーム形式エンドポイントでレンダリングをトリガー:

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

## データ・キャッシュ・ログ

アプリで使用される主な成果物:

- `data/`: 監修済み CSV データセット
- `words_phonetics.db`: SQLite のキャッシュ/ソース DB
- `cache/`: OpenAI リクエストと結果のキャッシュ
- `word_phonetics_processed.csv`: 加工済み/派生データセット
- `logs/`, `logs-word-phonetics/`: 実行ログ
- `words_card_temp/`: 生成済みカードと一時出力

## 開発ノート

- 過去互換・バックアップ用ファイルが存在します（例: `words_gpt_old.py`、`lib.old`）。互換性維持や移行作業を行う場合を除き参照用として扱ってください。
- `words_update.py` には、DB の品質向上に使えるバッチ更新・再確認ヘルパーがあります。
- ハードウェア検証は `epd_*_test.py` と `waveshare/examples/*` のデモで実施します。
- リポジトリルートには `requirements.txt` やロックファイルはありません。依存関係はセットアップスクリプトか直接インストールで管理します。
- 本リポジトリには自動化されたテストスイートはありません。

## トラブルシューティング

- Raspberry Pi GPIO/SPI モジュール由来の `ImportError`
  - Pi 向け手順 (`setup_pi_wordscard.sh`) でインストール、またはターゲット互換環境で `python setup.py install` を実行
- 画像/静的配信で `403/404` が返る
  - `/get_current_word*` の使い方を確認し、`words_card_temp/` の書き込み権限を確認
- OpenAI モードで空/不正なペイロード
  - `OPENAI_API_KEY` と任意の org/model が正しく読み込まれているか確認。`cache/` とログを確認
- レンダリング不具合（文字欠け・切れ）
  - `words_gpt.py` の描画フロー内でフォントパスとパネル解像度の設定を確認
- API が古いデータを返す
  - `POST /next_random_word` を手動実行し、`app.py` の定期コールバック間隔を確認
- ハードウェア更新が停止したように見える
  - tmux セッションと systemd ログ（`journalctl -u wordscard`）を確認
- データセットや辞書項目が不足する
  - `data/` の CSV を検証し、`words_update.py` で更新/クリーンアップを実施

## ロードマップ

- `requirements.txt` / 再現可能なインストールマニフェストを追加
- 実行モードを明確化し、CLI `--help` ドキュメントを充実
- 各コンテンツモード（`japanese_synonym`、`arabic_synonym`、`film` など）のレンダリング仕様を拡張
- エラー処理とユーザー向け API 応答スキーマを標準化
- 非ハードウェア環境での簡易 CI 確認向けスモークテストスクリプトを追加

## サポート

| Support option | Link | Purpose |
|---|---|---|
| GitHub Sponsors | https://github.com/sponsors/lachlanchen | 継続的・一回限りのプロジェクト支援 |
| Lazying Art | https://lazying.art | ブランド情報と関連リソース |
| Chat | https://chat.lazying.art | 質問・サポート |
| Only Ideas | https://onlyideas.art | クリエイティブ研究とサイドプロジェクト |

## コントリビューション

コントリビューションは歓迎します。推奨フロー:

1. 変更範囲は 1 つの動作領域（レンダリング、データ、API、スクリプト）に絞る
2. ユーザー向け挙動変更はコマンド利用法とドキュメントを更新
3. 可能な限り既存 CLI フラグとエンドポイント互換性を維持
4. ハードウェア関連スクリプトを変更する場合、デバイス/モデル名と実行コマンドを明示

## ライセンス

現在のリポジトリルートには `LICENSE` ファイルがありません。したがってこの版では、このリポジトリ内で有効なライセンスが未定義です。明示的な再配布・再利用条件を設ける場合は追加してください。
