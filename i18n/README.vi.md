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

Một dự án Raspberry Pi + e-ink Waveshare hiển thị các thẻ từ vựng được chọn động, có phiên âm IPA và gợi ý đa ngôn ngữ. Hỗ trợ quy trình làm việc CSV cục bộ, làm giàu bằng AI theo nhu cầu, render cho e-paper, và điều khiển từ xa qua HTTP.

| 🔎 Tổng quan nhanh | Chi tiết |
|---|---|
| Runtime cốt lõi | `app.py` (dịch vụ HTTP) + `words_gpt.py` (vòng lặp render) |
| Đường dẫn dữ liệu | Các bộ dữ liệu CSV trong `data/` + kho SQLite `words_phonetics.db` |
| Mục tiêu đầu ra | Panel e-paper Waveshare và ảnh đầu ra ảo |
| Phụ thuộc AI | Tùy chọn (`--enable_openai`) với cache trong `cache/` |
| Mặc định vòng lặp chính | Server trên cổng `8082`, làm mới định kỳ khoảng 5 phút |

## 📚 Mục lục
- [Tổng quan](#tổng-quan)
- [Điểm nổi bật](#điểm-nổi-bật)
- [Demo](#demo)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Điều kiện tiên quyết](#điều-kiện-tiên-quyết)
- [Cài đặt](#cài-đặt)
- [Cấu hình](#cấu-hình)
- [Cách sử dụng](#cách-sử-dụng)
- [Ví dụ](#ví-dụ)
- [Dữ liệu, cache và logs](#dữ-liệu-cache-và-logs)
- [Ghi chú phát triển](#ghi-chú-phát-triển)
- [Khắc phục sự cố](#khắc-phục-sự-cố)
- [Lộ trình](#lộ-trình)
- [Hỗ trợ](#hỗ-trợ)
- [Đóng góp](#đóng-góp)
- [Giấy phép](#giấy-phép)

---

## Tổng quan

`words_gpt` là một stack tạo thẻ từ vựng bằng Python cho màn hình e-ink. Dự án kết hợp điều phối dữ liệu, làm giàu phiên âm và orchestrate render trong hai chế độ chạy:

- Dịch vụ Tornado chạy liên tục (`app.py`) để điều khiển từ xa và phục vụ ảnh
- Trình render độc lập (`words_gpt.py`) có thể chạy theo chế độ polling, loop, hoặc render trực tiếp

Các module chính:

- `words_data.py` / `words_data_utils.py` cho quy trình lấy từ và làm giàu dữ liệu
- `words_database.py` cho tương tác SQLite
- `openai_request_json.py` cho request OpenAI có cấu trúc JSON kèm cache trên đĩa
- `env_loader.py` cho tải biến môi trường có tính xác định
- `words_update.py` cho duy trì DB và quy trình recheck
- `app.py` và `words_gpt.py` cho vòng đời service/render

## Điểm nổi bật

- Pipeline render e-ink với nhiều chế độ ngôn ngữ/nội dung:
  - Biến thể tiếng Nhật, chế độ kanji, Arabic, Chinese, emoji
- Nguồn từ cục bộ và OpenAI trong cùng một quy trình
- Tùy chọn kết quả tiếng Trung giản thể trong luồng render
- API endpoint để tương tác trực tiếp (`/next_random_word`, `/display_word`, ...)
- Cache và persistence giúp giảm số lần gọi mạng lặp lại
- Tùy chọn assets PWA trong `pwa/` cho luồng preview/config nhẹ

## Demo

<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## Cấu trúc dự án

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

Các tệp runtime quan trọng:

- `app.py`: Tornado app trên cổng `8082` + trigger `next_random_word` theo chu kỳ.
- `words_gpt.py`: renderer độc lập và abstractions hiển thị (`EPaperHardware`, `EPaperDisplay`).
- `words_data.py`: quy trình fetch/làm giàu nâng cao và utilities.
- `words_database.py`: helpers SQLite cho metadata đã lưu và thao tác cache từ vựng.
- `scripts/*.sh`: cài đặt/service lifecycle và helper bootstrap Raspberry Pi.

## Điều kiện tiên quyết

- Python `3.9+` (khuyến nghị)
- Raspberry Pi (bắt buộc cho chế độ phần cứng)
- Panel e-paper Waveshare được hỗ trợ (ví dụ: họ 7.3F / 13K)
- SPI đã bật (`raspi-config`), dây kết nối đúng, và runtime ổn định nguồn điện
- NLTK corpus sẵn sàng khi dùng nguồn từ `nltk`

Dependencies phổ biến trong codebase:
`openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.

## Cài đặt

### Lựa chọn 1 — Cài đặt tối thiểu/thủ công (desktop hoặc Pi)

Từ root repo:

```bash
python setup.py install
```

Nếu cần:

```bash
python -m nltk.downloader words
```

### Lựa chọn 2 — Thiết lập tự động cho Raspberry Pi (khuyến nghị trên thiết bị)

Từ root repo:

```bash
bash scripts/setup_pi_wordscard.sh
```

Lệnh này thực hiện:

- Phụ thuộc riêng cho Pi
- Bật SPI
- Thiết lập `wordscard` virtual environment
- Cài đặt package Python/runtime
- Cài đặt package Waveshare
- Khởi chạy app trong `tmux`

### Lựa chọn 3 — Cài đặt service

Đăng ký vòng đời ứng dụng với `systemd`:

```bash
bash scripts/install_wordscard_service.sh
```

Sau đó:

```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## Cấu hình

### Biến môi trường (`.env`)

Repo sử dụng cơ chế tải `.env` và hiện đang ghi đè các biến shell đã có trước đó. Hãy dùng có chủ đích:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### Tham số runtime (dùng cho cả `app.py` và `words_gpt.py`)

- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

Scripts Pi hỗ trợ truyền tham số qua `APP_ARGS` (ví dụ):

```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### Hành vi định tuyến trong app mode

Các route quan sát được trong code hiện tại:

- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (phục vụ từ `words_card_temp/`)

Lưu ý tương thích: tài liệu cũ từng tham chiếu `GET /current_word`; route hiện tại là `GET /get_current_word`.

### Ghi chú về sử dụng OpenAI

Tính năng OpenAI là tùy chọn và được điều khiển bởi CLI/env flags. Payload API được cache giúp tái lập và kiểm soát giới hạn tần suất. Trong môi trường hạn chế, chạy CSV trước (`--use_csv`) và chỉ bật OpenAI có chọn lọc (`--enable_openai`) khi cần làm giàu dữ liệu.

## Cách sử dụng

### Chạy HTTP server

```bash
python app.py
```

Process duy trì một ảnh trong `words_card_temp/` và expose các HTTP endpoint cho công cụ front-end hoặc script đơn giản.

### Chạy renderer trực tiếp

Chế độ CSV:

```bash
python words_gpt.py --use_csv
```

Chế độ OpenAI:

```bash
python words_gpt.py --enable_openai --use_csv
```

Emoji + CJK simplified:

```bash
python words_gpt.py --make_emoji --simplify
```

### Chạy trên phần cứng Pi

- Khởi động qua script `tmux`:

```bash
bash scripts/start_wordscard.sh
```

- Dừng qua script `tmux`:

```bash
bash scripts/stop_wordscard.sh
```

## Ví dụ

Lấy metadata thẻ ngẫu nhiên tiếp theo:

```bash
curl "http://127.0.0.1:8082/next_random_word"
```

Lấy từ đang lưu hiện tại:

```bash
curl "http://127.0.0.1:8082/get_current_word"
```

Yêu cầu payload ảnh trang đã render:

```bash
curl "http://127.0.0.1:8082/get_current_word_page"
```

Gửi một từ rõ ràng:

```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

Gọi endpoint render thẻ qua form style:

```bash
curl -X POST "http://127.0.0.1:8082/get_words_card" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "word=bonjour&phonetic=%CB%88b%C9%94n%C9%93r%E2%80%AD"
```

## Dữ liệu, cache, và logs

Các artifact điển hình mà app sử dụng:

- `data/`: bộ dữ liệu CSV đã được chọn lọc
- `words_phonetics.db`: cache/source SQLite
- `cache/`: OpenAI request/result cache
- `word_phonetics_processed.csv`: dataset đã xử lý và sinh ra
- `logs/`, `logs-word-phonetics/`: log runtime
- `words_card_temp/`: thẻ đã sinh và output tạm thời

## Ghi chú phát triển

- File legacy/backup vẫn tồn tại (ví dụ `words_gpt_old.py`, `lib.old`), xem các file này như tài liệu tham chiếu trừ khi đang di chuyển hoặc giữ tương thích.
- `words_update.py` chứa helper làm mới/kiểm tra batch hữu ích cho bước làm sạch chất lượng dữ liệu DB.
- Kiểm thử phần cứng được xử lý bởi `epd_*_test.py` và demo `waveshare/examples/*`.
- Repo chưa có `requirements.txt` hay lockfile ở root; setup phụ thuộc qua setup script hoặc cài trực tiếp.
- Repo chưa có suite test tự động.

## Khắc phục sự cố

- `ImportError` từ module Raspberry Pi GPIO/SPI:
  - Cài qua đường dẫn Pi (`setup_pi_wordscard.sh`), hoặc kiểm tra `python setup.py install` trên target tương thích.
- `403/404` từ các endpoint hình/static:
  - Kiểm tra việc dùng endpoint `/get_current_word*` và xác nhận `words_card_temp/` có quyền ghi.
- Payload OpenAI rỗng/không hợp lệ:
  - Kiểm tra `OPENAI_API_KEY` và các giá trị org/model tùy chọn đã được load; xem `cache/` và logs.
- Render kém/chữ bị cắt:
  - Kiểm tra `font path` và cấu hình độ phân giải panel trong luồng render của `words_gpt.py`.
- API trả về dữ liệu cũ:
  - Gọi `POST /next_random_word` thủ công và kiểm tra `interval` callback định kỳ trong `app.py`.
- Cập nhật phần cứng dường như bị treo:
  - Kiểm tra phiên tmux và log systemd (`journalctl -u wordscard`).
- Thiếu dữ liệu hoặc thiếu mục từ điển:
  - Kiểm tra file CSV trong `data/` và chạy `words_update.py` để refresh/cleanup.

## Lộ trình

- Thêm `requirements.txt` tối thiểu / manifest cài đặt tái lập.
- Bổ sung mô tả chế độ runtime rõ hơn và tài liệu CLI `--help` đầy đủ.
- Mở rộng docs schema render cho từng content mode (`japanese_synonym`, `arabic_synonym`, `film`, ...).
- Chuẩn hóa xử lý lỗi và schema phản hồi API hướng người dùng.
- Thêm stub smoke-test nhỏ cho validation CI không cần phần cứng.

## Hỗ trợ

| Tùy chọn hỗ trợ | Liên kết | Mục đích |
|---|---|---|
| GitHub Sponsors | https://github.com/sponsors/lachlanchen | Hỗ trợ dự án thường xuyên và một lần |
| Lazying Art | https://lazying.art | Nhãn hiệu và tài nguyên liên quan |
| Chat | https://chat.lazying.art | Thảo luận và hỗ trợ |
| Only Ideas | https://onlyideas.art | Nghiên cứu sáng tạo và dự án phụ |

## Đóng góp

Những đóng góp đều được chào đón. Gợi ý quy trình:

1. Giữ thay đổi gọn trong một vùng hành vi (render, data, API, scripts).
2. Cập nhật hướng dẫn lệnh/docs khi có thay đổi ảnh hưởng người dùng.
3. Giữ khả năng tương thích cờ CLI và endpoint hiện có khi có thể.
4. Nếu thay đổi script phần cứng, ghi rõ thiết bị/model đã test và lệnh chạy chính xác.

## Giấy phép

Hiện chưa có tệp `LICENSE` trong thư mục gốc repo hiện tại. Do đó license hiệu lực chưa được định nghĩa trong repo tại bản nháp này. Vui lòng bổ sung nếu bạn muốn điều khoản phân phối/tái sử dụng rõ ràng.
