[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 🖨️ Eink Words GPT

**Ngôn ngữ của bản nháp này:** Tiếng Việt

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

> Dự án Raspberry Pi + Waveshare e-ink tạo thẻ từ vựng động với phiên âm IPA và gợi ý đa ngôn ngữ. Nó hỗ trợ luồng làm việc CSV cục bộ, làm giàu bằng AI khi cần, hiển thị e-paper và điều khiển HTTP từ xa.

Nhìn nhanh chế độ chạy:
`app.py` (dịch vụ) và `words_gpt.py` (renderer độc lập) có thể hoạt động riêng rẽ hoặc cùng lúc.

| 🔎 Tóm tắt nhanh | Chi tiết |
|---|---|
| **Thời gian chạy chính** | `app.py` (dịch vụ HTTP) + `words_gpt.py` (vòng lặp render) |
| **Đường dẫn dữ liệu** | Bộ dữ liệu CSV trong `data/` + kho SQLite `words_phonetics.db` |
| **Đích đầu ra** | Màn hình e-paper Waveshare và ảnh đầu ra ảo |
| **Phụ thuộc AI** | Tùy chọn (`--enable_openai`) với bộ nhớ đệm trong `cache/` |
| **Chu kỳ mặc định** | Máy chủ tại cổng `8082`, làm mới định kỳ khoảng 5 phút |

## 📚 Mục lục
- [Tổng quan](#overview)
- [Điểm nổi bật](#highlights)
- [Demos](#demos)
- [Cấu trúc dự án](#project-structure)
- [Yêu cầu hệ thống](#prerequisites)
- [Cài đặt](#installation)
- [Cấu hình](#configuration)
- [Cách sử dụng](#usage)
- [Ví dụ](#examples)
- [Dữ liệu, cache, và nhật ký](#data-cache-and-logs)
- [Ghi chú phát triển](#development-notes)
- [Khắc phục sự cố](#troubleshooting)
- [Lộ trình](#roadmap)
- [Support](#support)
- [Đóng góp](#contributing)
- [Giấy phép](#license)

---

<a id="overview"></a>
## Tổng quan

`words_gpt` là một stack tạo thẻ từ vựng Python cho màn hình e-ink. Dự án kết hợp điều phối dữ liệu, làm giàu phiên âm và render trong hai chế độ chạy:

- Dịch vụ Tornado chạy liên tục (`app.py`) cho điều khiển từ xa và phục vụ ảnh.
- Renderer độc lập (`words_gpt.py`) có thể chạy ở chế độ polling, loop hoặc render trực tiếp.

Các module chính:

- `words_data.py` / `words_data_utils.py` cho quy trình lấy từ và làm giàu dữ liệu.
- `words_data_with_legacy.py`, `words_data_without_legacy.py`, `words_data_workable*.py` cho các luồng thay thế và tương thích legacy.
- `words_database.py` cho tương tác SQLite.
- `openai_request_json.py` cho các yêu cầu OpenAI có cấu trúc, có cache/retry ghi trên đĩa.
- `env_loader.py` cho việc tải biến môi trường có tính xác định.
- `words_update.py` cho bảo trì DB và quy trình re-check.
- `app.py` và `words_gpt.py` cho vòng đời service/render.
- `pwa/` cho công cụ preview/config trình duyệt nhẹ.

<a id="highlights"></a>
## Điểm nổi bật

- Pipeline render e-ink với các chế độ đa ngôn ngữ: các biến thể tiếng Nhật, chế độ kanji, Arabic, Chinese và emoji.
- Nguồn từ cục bộ và nguồn từ OpenAI trong cùng một luồng làm việc.
- Tùy chọn render tiếng Trung giản thể (`--simplify`).
- Các endpoint HTTP cho điều khiển từ xa (`/next_random_word`, `/display_word`, `/get_current_word`, `/get_current_word_page`, `/get_words_card`).
- Cache và persistence giúp giảm số lần gọi lại API AI.
- Bao gồm driver `waveshare/` giúp đóng gói/đi kèm ví dụ phần cứng.

<a id="demos"></a>
## Demos

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

<a id="project-structure"></a>
## Cấu trúc dự án

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

Các tệp runtime quan trọng:

- `app.py`: ứng dụng Tornado trên cổng `8082` + trigger `next_random_word` định kỳ.
- `words_gpt.py`: renderer độc lập và lớp trừu tượng hiển thị (`EPaperHardware`, `EPaperDisplay`).
- `words_data.py`: quy trình lấy/enrichment và tiện ích chọn từ.
- `words_database.py`: helper SQLite cho metadata đã lưu và thao tác cache từ vựng.
- `scripts/*.sh`: cài đặt, vòng đời service và các script bootstrap Raspberry Pi.
- `words_update.py`: helper cập nhật/bảo trì hàng loạt cho chất lượng DB.

<a id="prerequisites"></a>
## Yêu cầu hệ thống

- Python `3.9+` (khuyến nghị)
- Raspberry Pi (bắt buộc khi chạy phần cứng)
- Màn hình e-paper Waveshare tương thích (ví dụ họ 7.3F hoặc 13K)
- SPI đã bật (`raspi-config`), dây nối đúng, nguồn điện ổn định
- Corpus NLTK khi dùng nguồn từ vựng NLTK

Các phụ thuộc phổ biến trong source và thường được dùng ở đường đi chạy chính:
`openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.

<a id="installation"></a>
## Cài đặt

### Tùy chọn 1 — Cài đặt tối thiểu/thủ công (desktop hoặc Pi)

Từ root repo:

```bash
python setup.py install
```

Nếu cần:

```bash
python -m nltk.downloader words
```

### Tùy chọn 2 — Thiết lập tự động trên Raspberry Pi (khuyến nghị khi chạy trực tiếp trên thiết bị)

Từ root repo:

```bash
bash scripts/setup_pi_wordscard.sh
```

Hành động này:

- Cài phụ thuộc riêng cho Pi
- Kiểm tra bật SPI
- Tạo môi trường ảo `wordscard`
- Cài đặt package Python/runtime
- Cài đặt package Waveshare
- Khởi động app bằng `tmux`

### Tùy chọn 3 — Cài đặt bằng systemd

Đăng ký chu kỳ sống ứng dụng dưới `systemd`:

```bash
bash scripts/install_wordscard_service.sh
```

Rồi chạy:

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

<a id="configuration"></a>
## Cấu hình

### Biến môi trường (`.env`)

`env_loader` đọc khóa môi trường và áp dụng khi khởi tạo process. Tài liệu/runtime hiện thường dùng:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

Giả định: giữ secrets trong cấu hình môi trường local và không commit lên version control.

### Cờ runtime (dùng bởi `app.py` và `words_gpt.py`)

| Cờ CLI | Mục đích |
| --- | --- |
| `--enable_openai` | Bật chế độ làm giàu OpenAI tùy chọn |
| `--make_emoji` | Render thẻ ưu tiên emoji |
| `--ignore_list` | Bỏ qua các từ trong danh sách ignore đã cấu hình |
| `--simplify` | Tạo đầu ra CJK giản thể |
| `--use_csv` | Đọc từ vựng từ bộ dữ liệu CSV |
| `--complete_csv` | Sử dụng chế độ CSV đầy đủ |
| `--filename <csv_file>` | Trỏ tới file CSV đầu vào cụ thể |

`APP_ARGS` có thể truyền qua các startup script. Ví dụ:

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### Hành vi routing ở app mode

Các route API đã quan sát:

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (phục vụ từ `words_card_temp/`)

Ghi chú tương thích: tài liệu cũ có thể nhắc `GET /current_word`; route hiện tại là `GET /get_current_word`.

### Ghi chú sử dụng OpenAI

Tính năng OpenAI là tùy chọn và được kiểm soát bằng cờ CLI/env. Cache trong `cache/` giúp tái lập và kiểm soát rate limit tốt hơn. Để chạy ổn định ngoài mạng, bắt đầu với chế độ CSV (`--use_csv`) và bật OpenAI có chọn lọc.

<a id="usage"></a>
## Cách sử dụng

### Chạy HTTP server

```bash
python app.py
```

Quá trình chạy giữ ảnh mới nhất trong `words_card_temp/` và cung cấp endpoint cho công cụ frontend hoặc script.

### Chạy renderer trực tiếp

Chế độ CSV:

```bash
python words_gpt.py --use_csv
```

Chế độ OpenAI:

```bash
python words_gpt.py --enable_openai --use_csv
```

Emoji + CJK giản thể:

```bash
python words_gpt.py --make_emoji --simplify
```

### Chạy trên phần cứng Pi

- Khởi động bằng tmux script:

```bash
bash scripts/start_wordscard.sh
```

- Dừng bằng tmux script:

```bash
bash scripts/stop_wordscard.sh
```

<a id="examples"></a>
## Ví dụ

Lấy metadata thẻ ngẫu nhiên tiếp theo:

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

Lấy từ hiện tại đang lưu:

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

Yêu cầu payload ảnh trang đã render:

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

Gửi một từ cụ thể:

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

Kích hoạt render thẻ qua endpoint kiểu form:

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

<a id="data-cache-and-logs"></a>
## Dữ liệu, cache, và nhật ký

Các artifact phổ biến được app sử dụng:

- `data/`: dataset CSV đã tuyển chọn
- `words_phonetics.db`: cache/source SQLite
- `cache/`: cache request/kết quả OpenAI
- `word_phonetics_processed.csv`: dataset đã xử lý/sinh ra
- `logs/`, `logs-word-phonetics/`: nhật ký runtime
- `words_card_temp/`: thẻ đã tạo và output tạm
- `pic/` và `figs/`: ảnh tham chiếu và banner

<a id="development-notes"></a>
## Ghi chú phát triển

- Các module và artifacts legacy/backup vẫn tồn tại (ví dụ `words_gpt_old.py`, `lib.old`), nên coi đó là tài liệu tham khảo trừ khi có kế hoạch migrate hoặc cần tương thích.
- `words_update.py` chứa helper refresh/recheck hàng loạt hữu ích cho kiểm soát chất lượng DB.
- Kiểm thử phần cứng được xử lý bởi `epd_*_test.py` và các demo script trong `waveshare/examples/*`.
- Repository không có `requirements.txt` hay lockfile ở root; phụ thuộc được quản lý bởi setup scripts và cài đặt trực tiếp.
- Chưa có bộ test tự động trong repo.

<a id="troubleshooting"></a>
## Khắc phục sự cố

- `ImportError` từ module GPIO/SPI Raspberry Pi:
  - Dùng luồng setup Pi (`scripts/setup_pi_wordscard.sh`) hoặc cài rõ dependency trên thiết bị tương thích.
- `403/404` từ endpoint image/static:
  - Kiểm tra cú pháp route (`/get_current_word*`) và đảm bảo `words_card_temp/` có quyền ghi.
- Payload từ OpenAI rỗng/không hợp lệ:
  - Xác nhận `OPENAI_API_KEY` (và org/model) đã được load, sau đó kiểm tra `cache/` và logs.
- Render kém chất lượng hoặc cắt chữ:
  - Kiểm tra font path, hằng số độ phân giải màn hình và cấu hình mode trong `words_gpt.py`.
- API trả dữ liệu cũ:
  - Gọi thủ công `POST /next_random_word` rồi xem lại interval callback trong `app.py`.
- Hiển thị phần cứng có vẻ bị treo:
  - Kiểm tra session tmux và logs systemd qua `journalctl -u wordscard`.
- Thiếu dữ liệu/thiếu mục từ điển:
  - Kiểm tra file CSV trong `data/` và chạy tác vụ bảo trì của `words_update.py`.

<a id="roadmap"></a>
## Lộ trình

- Thêm `requirements.txt` tối thiểu / manifest cài đặt tái tạo được.
- Bổ sung tài liệu rõ ràng hơn cho các runtime mode và `--help` CLI.
- Mở rộng tài liệu cho các chế độ render (`japanese_synonym`, `arabic_synonym`, `film`, và các workflow khác).
- Chuẩn hóa xử lý lỗi và schema phản hồi API hướng người dùng.
- Thêm smoke test stub nhẹ cho xác thực CI không cần phần cứng.

<a id="support"></a>
## Đóng góp

Đóng góp luôn được chào đón. Quy trình đề xuất:

1. Giữ thay đổi trong một phạm vi hành vi (render, data, API, scripts).
2. Cập nhật lệnh/sách hướng dẫn khi hành vi hiển thị cho người dùng thay đổi.
3. Giữ khả năng tương thích CLI flags và endpoint càng nhiều càng tốt.
4. Nếu sửa scripts phần cứng, ghi rõ thiết bị/model đã test và lệnh chạy.

<a id="license"></a>
## Giấy phép

Không có tệp `LICENSE` tại thư mục root repo hiện tại. Vì vậy giấy phép hiệu lực trong tree hiện tại chưa được xác định rõ. Vui lòng bổ sung nếu bạn muốn quy định rõ điều khoản phân phối/ tái sử dụng.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
