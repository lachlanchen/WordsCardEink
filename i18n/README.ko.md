[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 🖨️ Eink Words GPT

**이 문서의 언어:** 한국어

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

> Raspberry Pi + Waveshare e-ink 프로젝트로 IPA 음성기호와 다국어 힌트가 포함된 동적 단어 카드를 렌더링합니다. 로컬 CSV 워크플로우, 선택적 AI 확장, e-paper 렌더링, 원격 HTTP 제어를 지원합니다.

실행 모드 한눈에 보기:
`app.py`(서비스)와 `words_gpt.py`(독립 실행형 렌더러)를 각각 또는 함께 실행할 수 있습니다.

| 🔎 한눈에 보기 | 상세 |
|---|---|
| **핵심 실행 방식** | `app.py` (HTTP 서비스) + `words_gpt.py` (렌더러 루프) |
| **데이터 경로** | `data/`의 CSV 데이터셋 + SQLite 저장소 `words_phonetics.db` |
| **출력 대상** | Waveshare e-paper 패널 및 가상 이미지 출력 |
| **AI 의존성** | 선택적(`--enable_openai`), `cache/`에 요청 캐시 존재 |
| **기본 갱신 주기** | 서버 포트 `8082`, 약 5분 간격의 주기적 갱신 |

## 📚 목차
- [개요](#개요)
- [주요 기능](#주요-기능)
- [데모](#데모)
- [프로젝트 구조](#프로젝트-구조)
- [사전 준비](#사전-준비)
- [설치](#설치)
- [설정](#설정)
- [사용법](#사용법)
- [예시](#예시)
- [데이터, 캐시, 로그](#데이터-cache-및-로그)
- [개발 노트](#개발-노트)
- [문제 해결](#문제-해결)
- [로드맵](#로드맵)
- [❤️ Support](#support)
- [기여](#기여)
- [라이선스](#라이선스)

---

## 개요

`words_gpt`는 e-ink 디스플레이용 파이썬 단어 카드 생성 스택입니다. 두 가지 실행 모드 뒤에서 데이터 오케스트레이션, 음성기호 확장, 렌더링을 결합합니다.

- 원격 제어와 이미지 서빙을 위한 장시간 실행 Tornado 서비스 (`app.py`).
- 폴링, 루프, 또는 직접 렌더링 모드로 실행 가능한 독립 렌더러 (`words_gpt.py`).

주요 모듈:

- 단어 처리 및 확장 워크플로우는 `words_data.py` / `words_data_utils.py`.
- 레거시 호환 및 변형 워크플로우는 `words_data_with_legacy.py`, `words_data_without_legacy.py`, `words_data_workable*.py`.
- SQLite 상호작용은 `words_database.py`.
- 디스크 캐시/재시도 동작이 포함된 구조화된 OpenAI 요청은 `openai_request_json.py`.
- 결정적 환경 변수를 위한 로드러는 `env_loader.py`.
- DB 유지보수와 재확인 워크플로우는 `words_update.py`.
- 서비스/렌더 생명주기는 `app.py` 및 `words_gpt.py`.
- 가벼운 브라우저 미리보기/설정 도구는 `pwa/`.

## 주요 기능

- 다국어 모드 렌더링 파이프라인: 일본어 변형, 한자 모드, 아랍어, 중국어, 이모지 모드.
- 로컬 전용 및 OpenAI 기반 단어 소싱을 하나의 워크플로우에서 지원.
- 선택적 중국어 간체 렌더링 (`--simplify`).
- 원격 제어용 HTTP 엔드포인트 (`/next_random_word`, `/display_word`, `/get_current_word`, `/get_current_word_page`, `/get_words_card`).
- 캐싱 및 영속 저장으로 반복 AI 호출 감소.
- 벤더링된 `waveshare/` 드라이버 트리를 통한 패키징 및 하드웨어 예제 제공.

## 데모

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## 프로젝트 구조

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

중요 런타임 파일:

- `app.py`: 포트 `8082`에서 동작하는 Tornado 앱 + 주기적 `next_random_word` 트리거.
- `words_gpt.py`: 독립 렌더러 및 디스플레이 추상화 (`EPaperHardware`, `EPaperDisplay`).
- `words_data.py`: 단어 조회/확장 워크플로우와 선택 유틸리티.
- `words_database.py`: 저장된 메타데이터와 단어 캐시 작업을 위한 SQLite 헬퍼.
- `scripts/*.sh`: 설치/서비스 생명주기 및 Raspberry Pi 부트스트랩 도구.
- `words_update.py`: DB 품질 유지보수를 위한 일괄 DB 갱신/재확인 헬퍼.

## 사전 준비

- Python `3.9+` (권장)
- Raspberry Pi (하드웨어 모드에서 필수)
- 지원되는 Waveshare e-paper 패널(예: 7.3F 또는 13K 계열)
- SPI 활성화(`raspi-config`), 정확한 배선, 안정적인 전원
- NLTK 단어 소스를 사용할 때 필요한 NLTK 코퍼스

실행 경로에서 트리의 공통 의존성:
`openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.

## 설치

### 옵션 1 — 최소/수동 설치 (데스크톱 또는 Pi)

저장소 루트에서:

```bash
python setup.py install
```

필요한 경우:

```bash
python -m nltk.downloader words
```

### 옵션 2 — Raspberry Pi 자동 설치 (장치에서 권장)

저장소 루트에서:

```bash
bash scripts/setup_pi_wordscard.sh
```

다음 작업을 수행합니다:

- Pi 전용 의존성 설치
- SPI 활성화 도우미 점검
- `wordscard` 가상환경 설정
- Python/런타임 패키지 설치
- Waveshare 패키지 설치
- `tmux`로 앱 프로세스 실행

### 옵션 3 — systemd 서비스 설치

`systemd`에 앱 생명주기를 등록하려면:

```bash
bash scripts/install_wordscard_service.sh
```

그런 다음:

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## 설정

### 환경 변수 (`.env`)

`env_loader`는 환경 키를 읽어 프로세스 시작 컨텍스트에 적용합니다. 현재 문서와 런타임에서 일반적으로 사용되는 값:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

가정: 비밀 값은 로컬 환경 설정에 보관하고 버전 관리에 커밋하지 마세요.

### 런타임 플래그 (`app.py` 및 `words_gpt.py`에서 사용)

| CLI 플래그 | 목적 |
| --- | --- |
| `--enable_openai` | 선택적 OpenAI 확장 모드 활성화 |
| `--make_emoji` | 이모지 중심 카드 렌더링 |
| `--ignore_list` | 설정된 무시 목록의 단어 건너뛰기 |
| `--simplify` | CJK 간체 결과 생성 |
| `--use_csv` | CSV 데이터셋에서 단어 읽기 |
| `--complete_csv` | 전체 CSV 소스 모드 사용 |
| `--filename <csv_file>` | 특정 CSV 입력 파일 지정 |

`APP_ARGS`는 시작 스크립트를 통해 전달할 수 있습니다. 예:

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### app 모드의 라우팅 동작

사용되는 API 경로:

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (`words_card_temp/`에서 제공)

호환성 참고: 이전 문서에서는 `GET /current_word`를 언급할 수 있으나, 현재 라우트는 `GET /get_current_word`입니다.

### OpenAI 사용 안내

OpenAI 기능은 선택적이며 CLI/환경 변수 플래그로 제어됩니다. `cache/`의 캐시된 요청은 재현성 및 rate-limit 제어에 도움을 줍니다. 오프라인 우선 실행의 경우 CSV 모드(`--use_csv`)로 시작하고 OpenAI는 필요한 경우에만 활성화하세요.

## 사용법

### HTTP 서버 실행

```bash
python app.py
```

이 프로세스는 최신 이미지를 `words_card_temp/`에 유지하고 프런트엔드 도구나 스크립트에서 사용하는 엔드포인트를 노출합니다.

### 렌더러 직접 실행

CSV 모드:

```bash
python words_gpt.py --use_csv
```

OpenAI 모드:

```bash
python words_gpt.py --enable_openai --use_csv
```

이모지 + 간체 CJK 모드:

```bash
python words_gpt.py --make_emoji --simplify
```

### Pi 하드웨어에서 실행

- tmux 스크립트로 시작:

```bash
bash scripts/start_wordscard.sh
```

- tmux 스크립트로 중지:

```bash
bash scripts/stop_wordscard.sh
```

## 예시

다음 무작위 카드 메타데이터 요청:

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

현재 저장된 단어 조회:

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

렌더링된 페이지 이미지 페이로드 요청:

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

특정 단어 제출:

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

폼 스타일 엔드포인트로 카드 렌더링 트리거:

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

## 데이터, 캐시, 로그

앱에서 사용하는 일반적인 산출물:

- `data/`: 큐레이션된 CSV 데이터셋
- `words_phonetics.db`: SQLite 캐시/소스 데이터베이스
- `cache/`: OpenAI 요청/응답 캐시
- `word_phonetics_processed.csv`: 처리/파생 데이터셋
- `logs/`, `logs-word-phonetics/`: 런타임 로그
- `words_card_temp/`: 생성된 카드와 임시 출력물
- `pic/` 및 `figs/`: 참고 이미지와 배너

## 개발 노트

- 레거시/백업 모듈과 산출물이 존재합니다(예: `words_gpt_old.py`, `lib.old`), 마이그레이션 또는 호환성 목적이 아니라면 참조 용도로만 다루세요.
- `words_update.py`에는 DB 품질 유지를 위한 일괄 갱신/재확인 헬퍼가 포함됩니다.
- 하드웨어 검증은 `epd_*_test.py`와 `waveshare/examples/*` 데모 스크립트에서 수행합니다.
- 저장소 루트에는 `requirements.txt` 또는 락 파일이 없으며, 의존성은 설치 스크립트 및 직접 설치 흐름으로 관리됩니다.
- 이 저장소에는 자동화된 테스트 스위트가 구성되어 있지 않습니다.

## 문제 해결

- Raspberry Pi GPIO/SPI 모듈의 `ImportError`:
  - Pi 설정 흐름(`scripts/setup_pi_wordscard.sh`)을 사용하거나 호환되는 대상에서 의존성을 명시적으로 설치하세요.
- 이미지/정적 엔드포인트의 `403/404`:
  - 라우트 사용(` /get_current_word*`)을 확인하고 `words_card_temp/`에 쓰기 권한이 있는지 확인하세요.
- OpenAI 모드에서 빈/잘못된 단어 페이로드:
  - `OPENAI_API_KEY`(및 org/model 값)가 로드되었는지 확인한 뒤 `cache/`와 로그를 점검하세요.
- 렌더링 불량 또는 텍스트 잘림:
  - `words_gpt.py`의 폰트 경로, 디스플레이 해상도 상수, 선택한 모드 설정을 확인하세요.
- API가 오래된 데이터를 반환:
  - `POST /next_random_word`를 수동으로 호출하고 `app.py`의 콜백 간격을 확인하세요.
- 하드웨어 렌더링이 멈춘 것처럼 보임:
  - `tmux` 세션과 `journalctl -u wordscard`로 systemd 로그를 확인하세요.
- 데이터셋/사전 항목 누락:
  - `data/`의 CSV 파일을 검증하고 `words_update.py` 유지보수 작업을 실행하세요.

## 로드맵

- 최소 `requirements.txt` / 재현 가능한 설치 매니페스트 추가.
- 더 명확한 런타임 모드와 CLI `--help` 문서 보완.
- 렌더링 모드 문서 확장 (`japanese_synonym`, `arabic_synonym`, `film` 및 기타 워크플로우).
- 오류 처리 및 사용자 대상 API 응답 스키마 표준화.
- 비하드웨어 CI 검증용 경량 smoke-test 스텁 추가.

## 기여

기여를 환영합니다. 권장 작업 흐름:

1. 변경 범위를 하나의 동작 영역(렌더링, 데이터, API, 스크립트)으로 유지합니다.
2. 사용자에 영향을 주는 동작 변경 시 명령어 사용법/문서 업데이트를 수행합니다.
3. 가능하면 기존 CLI 플래그 및 엔드포인트 호환성을 유지하세요.
4. 하드웨어 스크립트를 변경할 경우, 테스트한 장치/모델과 실행한 정확한 명령어를 문서화하세요.

## 라이선스

현재 저장소 루트에는 `LICENSE` 파일이 없습니다. 이 초안 기준 유효한 라이선스는 저장소 내에서 정의되지 않습니다. 재배포/재사용 조항이 필요하면 명시적으로 추가하세요.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
