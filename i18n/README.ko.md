[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Eink Words GPT

**언어 옵션:** 한국어 (이 문서)

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=flat-square&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=flat-square)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=flat-square)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=flat-square&logo=openai&logoColor=white)

Raspberry Pi + Waveshare e-ink 기반 프로젝트로, 음성(phonetics)과 다국어 동의어가 포함된 단어 카드를 동적으로 표시합니다. 로컬 데이터셋 또는 OpenAI에서 단어를 가져와 레이아웃으로 렌더링한 뒤 지원되는 e-paper 패널로 출력할 수 있습니다. 또한 단어 업데이트 트리거와 렌더링 이미지 조회를 위한 경량 HTTP 서비스도 제공합니다.

| 🔎 한눈에 보기 | 상세 |
|---|---|
| 핵심 런타임 | `app.py` (HTTP 서비스) + `words_gpt.py` (렌더링 루프) |
| 데이터 경로 | `data/`의 CSV 데이터셋 + SQLite 저장소 `words_phonetics.db` |
| 출력 대상 | Waveshare e-paper 패널 및 가상 이미지 출력 |
| AI 의존성 | 선택 사항 (`--enable_openai`), 캐시 위치: `cache/` |

<a id="table-of-contents"></a>
## 📚 목차
- [개요](#overview)
- [주요 특징](#highlights)
- [빠른 시작](#quick-start)
- [데모](#demos)
- [기능](#features)
- [프로젝트 구조](#project-structure)
- [사전 요구사항](#prerequisites)
- [설치](#installation)
- [설정](#configuration)
- [사용법](#usage)
- [예제](#examples)
- [데이터, 캐시, 로그](#data-cache-and-logs)
- [개발 노트](#development-notes)
- [문제 해결](#troubleshooting)
- [OpenAI 사용 메모](#notes-on-openai-usage)
- [로드맵](#roadmap)
- [지원](#-support)
- [기여](#contributing)
- [라이선스](#license)

<a id="overview"></a>
## 개요
`words_gpt`는 e-ink 디바이스용 어휘 카드 생성 및 표시 시스템입니다.

다음 요소를 결합합니다:
- CSV/로컬 데이터셋 기반 단어 소싱 및 선택적 OpenAI 생성.
- 보강 처리(IPA 음성 표기 + 다국어 동의어 필드).
- 하드웨어 및 가상 출력 렌더링 파이프라인.
- 원격 트리거 및 이미지 조회를 위한 Tornado HTTP 서비스.

현재 코드베이스의 핵심 파일은 `app.py`, `words_gpt.py`, `words_data.py`, `words_database.py`, `openai_request_json.py`입니다.

<a id="highlights"></a>
## 주요 특징
- 🖼️ 다양한 콘텐츠 모드(한자, 일본어, 아랍어, 중국어, 이모지)를 지원하는 e-ink 렌더링 파이프라인.
- 🗃️ 로컬 단어 데이터베이스(`words_phonetics.db`)와 `data/`의 CSV 기반 단어 목록.
- 🤖 구조화된 JSON 출력 기반 OpenAI 단어 선택 및 음성 정보 보강.
- 🌐 외부 트리거 및 이미지 조회용 HTTP 서비스.
- ⚡ 반복 OpenAI 호출을 줄이는 캐시 계층(`cache/`).

<a id="quick-start"></a>
## 빠른 시작
| 목표 | 명령어 |
|---|---|
| HTTP 서버 시작 (포트 `8082`) | `python app.py` |
| 독립 렌더러 실행 (CSV) | `python words_gpt.py --use_csv` |
| OpenAI + CSV 모드 실행 | `python words_gpt.py --enable_openai --use_csv` |
| 이모지 + 간체 CJK 모드 | `python words_gpt.py --make_emoji --simplify` |
| Raspberry Pi 자동 설정 | `bash scripts/setup_pi_wordscard.sh` |

<a id="demos"></a>
## 데모
<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

<a id="features"></a>
## 기능
- `words_gpt.py`의 하드웨어 + 가상 렌더링 흐름(`EPaperHardware`, `EPaperDisplay`).
- `words_data.py`의 다국어 보강 파이프라인(IPA, 일본어 변형, 아랍어, 프랑스어, 중국어 필드).
- `words_database.py`의 SQLite 기반 영속성 및 동적 필드 업데이트 헬퍼.
- `openai_request_json.py`의 OpenAI 구조화 JSON 요청 헬퍼와 파일 캐시.
- 경량 프런트엔드 설정/미리보기 워크플로를 위한 선택적 PWA 자산(`pwa/`).

<a id="project-structure"></a>
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

중요 런타임 파일:
- `app.py`: Tornado 웹 서버(기본 포트 `8082`) 및 주기적 업데이트 루프.
- `words_gpt.py`: 독립 렌더링 루프 및 디스플레이 클래스.
- `words_data.py`: 핵심 단어 수집/보강 오케스트레이션.
- `words_database.py`: SQLite 저장소 헬퍼.
- `scripts/*.sh`: Raspberry Pi 설정, 서비스 설치, tmux 라이프사이클 스크립트.

<a id="prerequisites"></a>
## 사전 요구사항
- Python `3.9+` (권장).
- Raspberry Pi 대상 환경(하드웨어 모드용).
- 지원되는 Waveshare e-paper 패널.
- Pi에서 SPI 활성화(`raspi-config`) 및 패널별 배선.

이 프로젝트에서 사용하는 주요 Python 패키지:
- `openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.
- 설정 스크립트 추가 설치: `json5`, `pandas`, `spidev`, `RPi.GPIO`, `gpiozero`, `lgpio`.

<a id="installation"></a>
## 설치

### 옵션 A: 최소/수동 설치
Waveshare 드라이버 패키지 설치:
```bash
python setup.py install
```

NLTK 단어 목록을 사용하는 경우 1회 다운로드:
```bash
python -m nltk.downloader words
```

### 옵션 B: Raspberry Pi 자동 설정 (디바이스 권장)
저장소 루트에서 실행:
```bash
bash scripts/setup_pi_wordscard.sh
```

이 스크립트는 다음을 수행합니다:
- apt 의존성 설치.
- SPI 활성화 보장.
- `wordscard` 가상환경 생성 및 활성화.
- Python 런타임 의존성 설치.
- Waveshare 패키지 설치.
- tmux 세션 안에서 `app.py` 시작.

<a id="configuration"></a>
## 설정

### `.env` 동작
이 저장소는 import 시점에 `.env` 환경 변수를 로드하며, **기존 셸 값도 덮어씁니다**. 이미 셸 프로필에 값을 export 했더라도 로컬 오버라이드는 일관되게 적용됩니다.

`.env` 생성 또는 수정:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### App 인자 전달
systemd/tmux 스크립트에서 다음을 지원합니다:
```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### CLI 플래그 (서버/렌더러 공통)
`app.py`와 `words_gpt.py` 모두 다음 옵션을 지원합니다:
- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

<a id="usage"></a>
## 사용법

### HTTP 서버 실행
서비스 시작 (기본 포트 `8082`):
```bash
python app.py
```

코드에서 확인되는 라우트:
- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (`words_card_temp/`에서 제공)

호환성 참고: 이전 문서에는 `GET /current_word`가 언급되었지만, 현재 `app.py` 라우트는 `GET /get_current_word`입니다.

### 독립 렌더러 실행
CSV 기반 목록:
```bash
python words_gpt.py --use_csv
```

OpenAI 활성화:
```bash
python words_gpt.py --enable_openai --use_csv
```

이모지 렌더링 + 간체 CJK:
```bash
python words_gpt.py --make_emoji --simplify
```

### Raspberry Pi 서비스 모드
서비스 유닛 설치:
```bash
bash scripts/install_wordscard_service.sh
```

그다음:
```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

<a id="examples"></a>
## 예제

### 다음 랜덤 단어 트리거
```bash
curl "http://127.0.0.1:8082/next_random_word"
```

### 현재 단어 페이로드 조회
```bash
curl "http://127.0.0.1:8082/get_current_word"
```

### 특정 단어 제출
```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

### 하드웨어 스모크 테스트
디스플레이별 스크립트 실행:
```bash
python epd_7in3f_test.py
```

또는:
```bash
python epd_13in3k_test.py
```

추가 예제는 `waveshare/examples/`에 있습니다.

<a id="data-cache-and-logs"></a>
## 데이터, 캐시, 로그
| 영역 | 경로 | 비고 |
|---|---|---|
| 단어 목록 | `data/` | `data/words_list.csv` 및 테마별 CSV 포함 |
| 영구 DB | `words_phonetics.db` | 로컬 음성/보강 저장소 |
| OpenAI/캐시 산출물 | `cache/` | 반복 요청 감소 |
| 로그 | `logs/`, `logs-word-phonetics/` | 런타임 및 업데이트 로그 |
| 생성 카드 | `words_card_temp/` | 이미지 출력 및 정적 서빙 소스 |

<a id="development-notes"></a>
## 개발 노트
- 의존성 관리는 스크립트 중심(`scripts/setup_pi_wordscard.sh`) + `setup.py`이며, 아직 `requirements.txt` 또는 `pyproject.toml`은 없습니다.
- 백업/레거시 파일(`words_data_*`, `words_gpt_old.py`)이 다수 존재하며, 활성 런타임 경로는 주로 `app.py` + `words_gpt.py` + `words_data.py` + `words_database.py`입니다.
- `env_loader.py`는 키가 존재하면 `.env` 값으로 환경 변수를 항상 덮어씁니다.
- 서버 모드는 주기적 새로고침 흐름(약 5분마다)을 실행하며 내부적으로 업데이트 엔드포인트를 호출할 수 있습니다.

<a id="troubleshooting"></a>
## 문제 해결
- `ModuleNotFoundError` 또는 import 문제:
  - 가상환경 활성화 및 의존성 설치 여부를 확인하세요.
  - Pi에서 `bash scripts/setup_pi_wordscard.sh`를 다시 실행하세요.
- OpenAI 오류 (`401`, 모델/키 누락):
  - `.env`의 `OPENAI_API_KEY` 및 선택적 `OPENAI_MODEL`을 확인하세요.
  - 디바이스 네트워크 연결 상태를 확인하세요.
- 디스플레이가 갱신되지 않음:
  - 패널 모델/배선을 확인하고 일치하는 테스트 스크립트(`epd_7in3f_test.py` 또는 `epd_13in3k_test.py`)를 실행하세요.
  - SPI 활성화 여부를 확인하세요 (`sudo raspi-config nonint do_spi 0`).
  - Pi 5에서는 `/dev/spidev10.0`만 노출될 경우 `/dev/spidev0.0` 호환 심볼릭 링크가 필요할 수 있습니다.
- OpenCC 설치 문제:
  - 설정 스크립트와 동일하게 배포판 호환 패키지(`libopencc1` 또는 `libopencc2`)를 사용하세요.
- API 라우트 불일치:
  - 현재 페이로드는 `/current_word`가 아니라 `/get_current_word`를 사용하세요.

<a id="notes-on-openai-usage"></a>
## OpenAI 사용 메모
OpenAI 연동은 선택 사항이지만, 새로운 단어 생성과 음성 정보 보강에는 권장됩니다. `openai_request_json.py`의 구조화 JSON 헬퍼는 `cache/`에 결과를 캐시하여 중복 호출을 줄입니다.

<a id="roadmap"></a>
## 로드맵
- 재현 가능한 설치를 위한 공식 의존성 매니페스트(`requirements.txt` 또는 `pyproject.toml`) 추가.
- 유지 관리되는 번역 README 확장을 위해 `i18n/` 보강.
- 정식 플로우 확정 후 레거시/백업 스크립트 변형 정리.
- 엔드포인트 예시와 스크린샷을 포함한 PWA 워크플로(`pwa/`) 문서화.
- 데이터 및 라우트 레벨 동작을 위한 반복 가능한 자동 테스트 추가.

## ❤️ Support

이 프로젝트가 도움이 되었다면, 아래 링크를 통해 유지보수와 하드웨어 개선 작업을 직접 지원할 수 있습니다.

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

### 후원이 만드는 변화
- <b>오픈 도구 유지</b>: 호스팅, 추론, 데이터 저장, 커뮤니티 운영 유지.  
- <b>더 빠른 개발</b>: WordsCardEink 및 관련 학습 도구에 오픈소스 개발 시간을 집중.  
- <b>디바이스 프로토타이핑</b>: e-ink 하드웨어 반복 개발 및 디스플레이 레이아웃 연구.  
- <b>접근성 확대</b>: 학생, 크리에이터, 커뮤니티 그룹 대상 지원 배포.

### Donate

<div align="center">
<table style="margin:0 auto; text-align:center; border-collapse:collapse;">
  <tr>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://chat.lazying.art/donate">https://chat.lazying.art/donate</a>
    </td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;">
      <a href="https://chat.lazying.art/donate"><img src="../figs/donate_button.svg" alt="Donate" height="44"></a>
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
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;"><img alt="WeChat QR" src="../figs/donate_wechat.png" width="240"/></td>
    <td style="text-align:center; vertical-align:middle; padding:6px 12px;"><img alt="Alipay QR" src="../figs/donate_alipay.png" width="240"/></td>
  </tr>
</table>
</div>

**支援 / Donate**

- ご支援は研究・開発と運用の継続に役立ち、より多くのオープンなプロジェクトを皆さんに届ける力になります。  
- 你的支持将用于研发与运维，帮助我持续公开分享更多项目与改进。  
- Your support sustains my research, development, and ops so I can keep sharing more open projects and improvements.

<a id="contributing"></a>
## 기여
기여 가이드, 코딩 스타일, PR 기대사항은 `AGENTS.md`를 참고하세요.

권장 체크리스트:
- 디스플레이 변경 시 패널 모델 + 하드웨어 노트를 포함하세요.
- 검증에 사용한 정확한 명령어를 기재하세요.
- UI 또는 e-ink 출력 변경 시 스크린샷/사진을 첨부하세요.
- 데이터셋 수정 사항(파일 + 행/열 영향)을 설명하세요.

<a id="license"></a>
## 라이선스
현재 저장소 루트에는 `LICENSE` 파일이 없습니다(이 초안 시점 기준). 라이선스 파일이 추가되기 전까지는 재사용 권한이 명시적으로 부여되지 않습니다.

가정: 유지관리자가 후속 업데이트에서 명시적 오픈소스 라이선스를 추가할 수 있습니다.
