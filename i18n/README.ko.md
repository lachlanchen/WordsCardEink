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

IPA 음성기호와 다국어 힌트가 포함된 동적 단어 카드를 렌더링하는 Raspberry Pi + Waveshare e-ink 프로젝트입니다. 로컬 CSV 워크플로우, 선택적 AI 보강, e-paper 렌더링, 원격 HTTP 제어를 모두 지원합니다.

| 🔎 한눈에 보기 | 상세 내역 |
|---|---|
| 핵심 런타임 | `app.py` (HTTP 서비스) + `words_gpt.py` (렌더링 루프) |
| 데이터 경로 | `data/`의 CSV 데이터셋 + SQLite 저장소 `words_phonetics.db` |
| 출력 대상 | Waveshare e-paper 패널 및 가상 이미지 출력 |
| AI 의존성 | 선택 사항 (`--enable_openai`) 및 `cache/` 캐시 |
| 기본 메인 루프 | 서버는 `8082`, 갱신 주기는 약 5분 |

## 📚 목차
- [개요](#개요)
- [하이라이트](#하이라이트)
- [데모](#데모)
- [프로젝트 구조](#프로젝트-구조)
- [요구 사항](#요구-사항)
- [설치](#설치)
- [설정](#설정)
- [사용법](#사용법)
- [예시](#예시)
- [데이터, 캐시 및 로그](#데이터-캐시-및-로그)
- [개발 노트](#개발-노트)
- [문제 해결](#문제-해결)
- [로드맵](#로드맵)
- [지원](#지원)
- [기여](#기여)
- [라이선스](#라이선스)

---

## 개요

`words_gpt`는 e-ink 디스플레이용 단어 카드 생성 Python 스택입니다. 데이터 오케스트레이션, 음성 기호 보강, 렌더링 오케스트레이션을 두 가지 실행 모드로 묶어 제공합니다.

- 원격 제어 및 이미지 제공을 위한 장기 실행 Tornado 서비스 (`app.py`)
- 폴링/루프/직접 렌더링 모드로 실행 가능한 독립형 렌더러 (`words_gpt.py`)

주요 모듈:

- 단어 및 보강 워크플로우를 위한 `words_data.py` / `words_data_utils.py`
- SQLite 상호작용을 위한 `words_database.py`
- 디스크 캐시가 포함된 구조화 OpenAI 요청을 위한 `openai_request_json.py`
- 결정론적 환경 로딩을 위한 `env_loader.py`
- DB 유지관리 및 재확인 워크플로우를 위한 `words_update.py`
- 서비스/렌더링 라이프사이클을 담당하는 `app.py`, `words_gpt.py`

## 하이라이트

- 다국어/다콘텐츠 모드가 있는 e-ink 렌더링 파이프라인:
  - 일본어 변형, 한자 모드, 아랍어, 중국어, 이모지 모드
- 로컬 단어 소스와 OpenAI 단어 소스를 하나의 워크플로우에서 사용
- 렌더러 경로에서 선택 가능한 간체 중국어 출력 (`--simplify`)
- 직접 상호작용 가능한 서버 엔드포인트 (`/next_random_word`, `/display_word` 등)
- 캐싱과 영속성으로 반복 네트워크 호출 감축
- 경량 미리보기/설정 흐름을 위한 `pwa/`의 선택적 PWA 에셋

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

중요한 실행 파일:

- `app.py`: 포트 `8082`에서 Tornado 앱을 실행하고, 주기적으로 `next_random_word` 트리거를 수행합니다.
- `words_gpt.py`: 독립형 렌더러와 디스플레이 추상화(`EPaperHardware`, `EPaperDisplay`).
- `words_data.py`: 고급 단어 조회/보강 워크플로우 및 유틸리티.
- `words_database.py`: 저장 메타데이터 및 단어 캐시 작업을 위한 SQLite 헬퍼.
- `scripts/*.sh`: 서비스 라이프사이클과 라즈베리파이 부트스트랩 도구.

## 요구 사항

- Python `3.9+` (권장)
- Raspberry Pi (하드웨어 모드 필수)
- 지원되는 Waveshare e-paper 패널(예: 7.3F / 13K 계열)
- SPI 활성화 (`raspi-config`), 올바른 배선, 안정적인 전원 환경
- `nltk` 단어 소스를 사용할 경우 NLTK 코퍼스 필요

코드베이스에서 사용되는 주요 의존성:
`openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.

## 설치

### 옵션 1 — 최소 설치 / 수동 설치(데스크톱 또는 Pi)

저장소 루트에서 실행:

```bash
python setup.py install
```

필요 시:

```bash
python -m nltk.downloader words
```

### 옵션 2 — Raspberry Pi 자동 설치(기기에서 권장)

저장소 루트에서 실행:

```bash
bash scripts/setup_pi_wordscard.sh
```

이 스크립트는 다음 작업을 수행합니다:

- Pi 전용 의존성 처리
- SPI 활성화
- `wordscard` 가상환경 설정
- Python/런타임 패키지 설치
- Waveshare 패키지 설치
- `tmux`를 통한 앱 프로세스 실행

### 옵션 3 — 서비스 등록

`systemd`로 앱 라이프사이클을 등록하려면:

```bash
bash scripts/install_wordscard_service.sh
```

그다음:

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## 설정

### 환경 변수 (`.env`)

이 저장소는 기존 셸 변수보다 우선 적용되는 `.env` 로더를 사용합니다. 의도적으로 사용할 것:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### 런타임 플래그 (`app.py`, `words_gpt.py` 모두 사용)

- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

Pi 시작 스크립트는 `APP_ARGS`를 통해 인자 전달을 지원합니다(예시):

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### 앱 모드 라우팅 동작

현재 코드에서 확인되는 라우트:

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (`words_card_temp/`에서 제공)

호환성 참고: 이전 문서에서는 `GET /current_word`가 언급되어 있었으나 현재 라우트는 `GET /get_current_word`입니다.

### OpenAI 사용 참고사항

OpenAI 기능은 선택 사항이며 CLI/환경 변수 플래그로 제어합니다. 캐시된 API 페이로드는 재현성과 요청 제한 제어에 유용합니다. 제약이 있는 환경에서는 먼저 CSV 모드(`--use_csv`)로 실행한 뒤, 필요 시에만 `--enable_openai`를 켜고 AI 보강을 수행하세요.

## 사용법

### HTTP 서버 실행

```bash
python app.py
```

이 프로세스는 `words_card_temp/`에 이미지를 보관하고, 프런트엔드 도구나 간단한 스크립트가 사용하는 HTTP 엔드포인트를 제공합니다.

### 렌더러 직접 실행

CSV 모드:

```bash
python words_gpt.py --use_csv
```

OpenAI 모드:

```bash
python words_gpt.py --enable_openai --use_csv
```

이모지 + CJK 간화:

```bash
python words_gpt.py --make_emoji --simplify
```

### Pi 하드웨어에서 실행

- `tmux` 스크립트로 시작:

```bash
bash scripts/start_wordscard.sh
```

- `tmux` 스크립트로 중지:

```bash
bash scripts/stop_wordscard.sh
```

## 예시

다음 랜덤 카드 메타데이터 가져오기:

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

명시적 단어 제출:

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

폼 스타일 엔드포인트로 렌더링 트리거:

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

## 데이터, 캐시 및 로그

앱에서 사용하는 전형적인 산출물:

- `data/`: 선별된 CSV 데이터셋
- `words_phonetics.db`: SQLite 캐시/원본 데이터베이스
- `cache/`: OpenAI 요청/응답 캐시
- `word_phonetics_processed.csv`: 처리/파생 데이터셋
- `logs/`, `logs-word-phonetics/`: 런타임 로그
- `words_card_temp/`: 생성된 카드 및 임시 출력물

## 개발 노트

- 레거시/백업 파일이 존재합니다(예: `words_gpt_old.py`, `lib.old`). 호환성 유지 또는 마이그레이션을 수행할 때를 제외하고 참조용으로만 취급하세요.
- `words_update.py`에는 DB 데이터 품질 점검에 유용한 일괄 갱신/재점검 헬퍼가 포함되어 있습니다.
- 하드웨어 검증은 `epd_*_test.py`와 `waveshare/examples/*` 데모에서 수행합니다.
- 저장소 루트에는 `requirements.txt` 또는 lockfile이 없습니다. 의존성 설치는 setup 스크립트 또는 직접 설치로 처리됩니다.
- 이 저장소에는 자동 테스트 스위트가 구성되어 있지 않습니다.

## 문제 해결

- Raspberry Pi GPIO/SPI 모듈에서 `ImportError` 발생:
  - Pi 경로(`setup_pi_wordscard.sh`)로 설치하거나 호환 타깃에서 `python setup.py install`을 확인하세요.
- 이미지/정적 엔드포인트에서 `403/404` 반환:
  - `/get_current_word*` 엔드포인트 사용법을 확인하고 `words_card_temp/` 쓰기 가능 여부를 점검하세요.
- OpenAI 모드에서 빈/잘못된 페이로드 발생:
  - `OPENAI_API_KEY`와 선택적 org/model 값이 로드되었는지 확인하고, `cache/`와 로그를 확인하세요.
- 렌더링 불량/텍스트 잘림:
  - `words_gpt.py` 내부 렌더 경로에서 글꼴 경로와 패널 해상도 설정을 확인하세요.
- API가 오래된 데이터 반환:
  - `POST /next_random_word`를 수동 호출하고, `app.py`의 주기적 콜백 간격을 점검하세요.
- 하드웨어 업데이트가 멈춘 것처럼 보임:
  - tmux 세션과 systemd 로그(`journalctl -u wordscard`)를 확인하세요.
- 데이터셋 또는 사전 항목이 없음:
  - `data/`의 CSV를 검증하고 `words_update.py` 워크플로우로 갱신/정리하세요.

## 로드맵

- 최소 `requirements.txt` / 재현 가능한 설치 매니페스트 추가
- 실행 모드 표시를 더 명확히 하고 CLI `--help` 문서 보강
- 각 콘텐츠 모드에 대한 렌더링 스키마 문서 확대 (`japanese_synonym`, `arabic_synonym`, `film` 등)
- 오류 처리와 사용자 대상 API 응답 스키마를 표준화
- 비하드웨어 CI 검증을 위한 경량 smoke-test 스크립트 추가

## 지원

| 지원 항목 | 링크 | 목적 |
|---|---|---|
| GitHub Sponsors | https://github.com/sponsors/lachlanchen | 지속적/일회성 프로젝트 후원 |
| Lazying Art | https://lazying.art | 브랜드 및 관련 자료 |
| Chat | https://chat.lazying.art | 질의 및 지원 |
| Only Ideas | https://onlyideas.art | 창의적 리서치 및 사이드 프로젝트 |

## 기여

기여를 환영합니다. 권장하는 흐름:

1. 변경 범위를 하나의 동작 영역(렌더링, 데이터, API, 스크립트)으로 제한하세요.
2. 사용자에게 보이는 동작 변경 시, 명령 사용법 및 문서를 업데이트하세요.
3. 가능하면 기존 CLI 플래그와 엔드포인트 호환성을 유지하세요.
4. 하드웨어 스크립트를 변경할 경우, 테스트한 기기 모델과 실행한 정확한 명령을 문서화하세요.

## 라이선스

현재 저장소 루트에는 `LICENSE` 파일이 없습니다. 따라서 이 문서 기준으로는 저장소 내에 유효한 라이선스가 정의되어 있지 않습니다. 재배포/재사용 조건을 명확히 하려면 `LICENSE`를 추가해 주세요.
