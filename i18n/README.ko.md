[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Eink Words GPT

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi-green)
![Display](https://img.shields.io/badge/display-Waveshare%20e--Paper-black)
![Status](https://img.shields.io/badge/status-active%20prototype-orange)
![Server](https://img.shields.io/badge/http-Tornado-0A7EA4)
![Storage](https://img.shields.io/badge/storage-SQLite-003B57)
![AI](https://img.shields.io/badge/OpenAI-optional-412991)

Raspberry Pi + Waveshare e-ink 기반 프로젝트로, 동적으로 선택된 어휘를 발음 정보와 다국어 동의어와 함께 표시합니다. 시스템은 로컬 데이터셋 또는 OpenAI에서 단어를 가져와 레이아웃으로 렌더링한 뒤 지원되는 e-paper 패널에 출력할 수 있습니다. 또한 단어 갱신 트리거와 렌더링 이미지 조회를 위한 소형 HTTP 서비스도 제공합니다.

## 개요
`words_gpt`는 e-ink 기기를 위한 Python 기반 어휘 카드 생성 및 표시 시스템입니다.

다음을 결합합니다:
- CSV/로컬 데이터셋 기반 단어 소싱과 선택적 OpenAI 생성.
- 보강 처리(IPA 발음 기호 + 다국어 동의어 필드).
- 하드웨어 출력 및 가상 출력용 렌더링 파이프라인.
- 원격 트리거 및 이미지 조회를 위한 Tornado HTTP 서비스.

현재 코드베이스의 중심은 `app.py`, `words_gpt.py`, `words_data.py`, `words_database.py`, `openai_request_json.py`입니다.

## 주요 특징
- 🖼️ 다양한 콘텐츠 모드(kanji, Japanese, Arabic, Chinese, emoji)를 지원하는 e-ink 렌더링 파이프라인.
- 🗃️ `data/`의 CSV 단어 목록과 연동되는 로컬 단어 DB(`words_phonetics.db`).
- 🤖 구조화된 JSON 출력 기반 OpenAI 단어 선택 및 발음 보강.
- 🌐 외부 트리거와 이미지 조회를 위한 HTTP 서비스.
- ⚡ 중복 OpenAI 호출을 줄이는 캐시 계층(`cache/`).

## 빠른 시작
| 목표 | 명령어 |
|---|---|
| HTTP 서버 시작 (포트 `8082`) | `python app.py` |
| 단독 렌더러 실행 (CSV) | `python words_gpt.py --use_csv` |
| OpenAI + CSV로 실행 | `python words_gpt.py --enable_openai --use_csv` |
| 이모지 + 간체 CJK 모드 | `python words_gpt.py --make_emoji --simplify` |
| Raspberry Pi 자동 설정 | `bash scripts/setup_pi_wordscard.sh` |

## 데모
<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## 기능
- `words_gpt.py`의 `EPaperHardware`, `EPaperDisplay`를 통한 하드웨어 + 가상 렌더링 흐름.
- `words_data.py`의 다국어 보강 파이프라인(IPA, 일본어 변형, 아랍어, 프랑스어, 중국어 필드).
- `words_database.py`의 동적 필드 업데이트 헬퍼를 포함한 SQLite 영속화.
- `openai_request_json.py`의 파일 캐시 기반 OpenAI 구조화 JSON 요청 헬퍼.
- 경량 프론트엔드 설정/미리보기 워크플로를 위한 선택적 PWA 에셋(`pwa/`).

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
- `words_gpt.py`: 단독 렌더링 루프 및 디스플레이 클래스.
- `words_data.py`: 단어 수집/보강 오케스트레이션의 핵심.
- `words_database.py`: SQLite 저장소 헬퍼.
- `scripts/*.sh`: Raspberry Pi 설정, 서비스 설치, tmux 라이프사이클 스크립트.

## 사전 요구사항
- Python `3.9+` (권장).
- Raspberry Pi 대상 장치(하드웨어 모드).
- 지원되는 Waveshare e-paper 패널.
- Pi에서 SPI 활성화(`raspi-config`) 및 패널별 배선.

이 프로젝트에서 사용하는 Python 패키지:
- `openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.
- 설정 스크립트가 추가로 설치: `json5`, `pandas`, `spidev`, `RPi.GPIO`, `gpiozero`, `lgpio`.

## 설치

### Option A: 최소/수동 설치
Waveshare 드라이버 패키지 설치:
```bash
python setup.py install
```

NLTK 단어 목록을 사용할 경우 1회 다운로드:
```bash
python -m nltk.downloader words
```

### Option B: Raspberry Pi 자동 설정 (장치에서 권장)
리포지토리 루트에서 실행:
```bash
bash scripts/setup_pi_wordscard.sh
```

이 스크립트는 다음을 수행합니다:
- apt 의존성 설치.
- SPI 활성화 여부 확인.
- `wordscard` 가상환경 생성 및 활성화.
- Python 런타임 의존성 설치.
- Waveshare 패키지 설치.
- tmux 세션에서 `app.py` 시작.

## 구성

### `.env` 동작
이 저장소는 import 시점에 `.env`에서 환경 변수를 로드하고, 기존 셸 값을 **덮어씁니다**. 따라서 셸 프로필에 이미 값을 export해도 로컬 오버라이드가 결정적으로 적용됩니다.

`.env` 생성 또는 업데이트:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### App 인자 전달
systemd/tmux 스크립트는 다음을 지원합니다:
```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### CLI 플래그 (서버 및 렌더러)
`app.py`와 `words_gpt.py` 모두 지원:
- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

## 사용법

### HTTP 서버 실행
서비스 시작(기본 포트 `8082`):
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

호환성 참고: 이전 문서에는 `GET /current_word`가 있었지만, 현재 `app.py` 라우트는 `GET /get_current_word`입니다.

### 단독 렌더러 실행
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

그다음 실행:
```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## 예시

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
디스플레이별 스크립트 사용:
```bash
python epd_7in3f_test.py
```

또는:
```bash
python epd_13in3k_test.py
```

더 많은 예시는 `waveshare/examples/`에 있습니다.

## 데이터, 캐시, 로그
| 영역 | 경로 | 비고 |
|---|---|---|
| 단어 목록 | `data/` | `data/words_list.csv` 및 주제별 CSV 파일 포함 |
| 영구 DB | `words_phonetics.db` | 로컬 발음/보강 저장소 |
| OpenAI/캐시 아티팩트 | `cache/` | 반복 요청 감소 |
| 로그 | `logs/`, `logs-word-phonetics/` | 런타임 및 업데이트 로그 |
| 생성 카드 | `words_card_temp/` | 이미지 출력 및 정적 제공 소스 |

## 개발 노트
- 의존성 관리는 스크립트 우선(`scripts/setup_pi_wordscard.sh`) + `setup.py` 방식이며, 아직 `requirements.txt`나 `pyproject.toml`은 없습니다.
- 백업/레거시 파일(`words_data_*`, `words_gpt_old.py`)이 여러 개 존재하며, 활성 런타임 경로는 주로 `app.py` + `words_gpt.py` + `words_data.py` + `words_database.py`입니다.
- `env_loader.py`는 키가 존재할 때 `.env`의 환경 변수를 항상 덮어씁니다.
- 서버 모드는 내부적으로 업데이트 엔드포인트를 호출할 수 있는 주기적 새로고침 흐름(약 5분마다)을 실행합니다.

## 문제 해결
- `ModuleNotFoundError` 또는 import 문제:
  - 가상환경이 활성화되어 있고 의존성이 설치되었는지 확인하세요.
  - Pi에서 `bash scripts/setup_pi_wordscard.sh`를 다시 실행하세요.
- OpenAI 오류(`401`, 모델/키 누락):
  - `.env`의 `OPENAI_API_KEY`와 선택적 `OPENAI_MODEL`을 확인하세요.
  - 장치의 네트워크 연결 상태를 확인하세요.
- 디스플레이가 갱신되지 않음:
  - 패널 모델/배선을 확인하고 일치하는 테스트 스크립트(`epd_7in3f_test.py` 또는 `epd_13in3k_test.py`)를 실행하세요.
  - SPI 활성화 여부를 확인하세요(`sudo raspi-config nonint do_spi 0`).
  - Pi 5에서는 장치가 `/dev/spidev10.0`으로 노출될 경우 `/dev/spidev0.0` 호환 심볼릭 링크를 확인하세요.
- OpenCC 설치 문제:
  - 설정 스크립트처럼 배포판 호환 패키지(`libopencc1` 또는 `libopencc2`)를 사용하세요.
- API 라우트 불일치:
  - 현재 페이로드는 `/current_word`가 아니라 `/get_current_word`를 사용하세요.

## OpenAI 사용 관련 참고
OpenAI 접근은 선택 사항이지만 새로운 단어 생성과 발음 보강에는 권장됩니다. `openai_request_json.py`의 구조화 JSON 헬퍼는 `cache/` 아래에 결과를 캐시해 반복 호출을 줄입니다.

## 로드맵
- 재현 가능한 설치를 위해 정식 의존성 매니페스트(`requirements.txt` 또는 `pyproject.toml`) 추가.
- 유지보수되는 번역 README 변형을 `i18n/`에 확장.
- 정식 흐름이 확정된 뒤 레거시/백업 스크립트 변형 통합.
- 엔드포인트 예시와 스크린샷을 포함해 PWA 워크플로(`pwa/`) 문서화.
- 데이터 및 라우트 단위 동작에 대한 반복 가능한 자동 테스트 추가.

## 지원

### 여러분의 지원으로 가능한 일
- <b>도구를 계속 공개</b>: 호스팅, 추론, 데이터 저장, 커뮤니티 운영.  
- <b>더 빠른 개발</b>: WordsCardEink 및 관련 학습 도구에 집중된 오픈소스 개발 시간.  
- <b>디바이스 프로토타이핑</b>: e-ink 하드웨어 반복 실험과 디스플레이 레이아웃 연구.  
- <b>모두를 위한 접근성</b>: 학생, 크리에이터, 커뮤니티 그룹을 위한 배포 지원.

### 후원

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

## 기여
기여 가이드, 코딩 스타일, PR 기대사항은 `AGENTS.md`를 참고하세요.

권장 기여 체크리스트:
- 디스플레이 변경 시 패널 모델 + 하드웨어 노트를 포함하세요.
- 검증에 사용한 정확한 명령어를 나열하세요.
- UI 또는 e-ink 출력 변경 시 스크린샷/사진을 첨부하세요.
- 데이터셋 수정 내용(파일 + 행/열 영향)을 설명하세요.

## 라이선스
저장소 루트에 `LICENSE` 파일이 현재 없습니다(이 초안 작성 시점 기준 관찰). 라이선스 파일이 추가되기 전까지는 재사용 권한이 명시적으로 부여되지 않습니다.

가정: 유지관리자가 후속 업데이트에서 명시적인 오픈소스 라이선스를 추가할 수 있습니다.
