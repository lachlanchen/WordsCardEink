[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Eink Words GPT

**Tùy chọn ngôn ngữ:** Tiếng Việt (bản dịch)

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-C51A4A?style=flat-square&logo=raspberrypi&logoColor=white)
![Display](https://img.shields.io/badge/Display-Waveshare%20e--Paper-111111?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Prototype-F59E0B?style=flat-square)
![Server](https://img.shields.io/badge/HTTP-Tornado-0A7EA4?style=flat-square)
![Storage](https://img.shields.io/badge/Storage-SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![AI](https://img.shields.io/badge/OpenAI-Optional-412991?style=flat-square&logo=openai&logoColor=white)

Dự án Raspberry Pi + e-ink Waveshare hiển thị từ vựng được chọn động kèm phiên âm và từ đồng nghĩa đa ngôn ngữ. Hệ thống có thể lấy từ từ bộ dữ liệu cục bộ hoặc OpenAI, dựng thành bố cục hiển thị và đẩy kết quả lên các màn hình e-paper được hỗ trợ. Dự án cũng cung cấp một dịch vụ HTTP nhỏ để kích hoạt cập nhật từ và truy xuất ảnh đã dựng.

| 🔎 Tổng quan nhanh | Chi tiết |
|---|---|
| Runtime cốt lõi | `app.py` (dịch vụ HTTP) + `words_gpt.py` (vòng lặp dựng) |
| Luồng dữ liệu | Bộ dữ liệu CSV trong `data/` + kho SQLite `words_phonetics.db` |
| Mục tiêu đầu ra | Màn hình e-paper Waveshare và đầu ra ảnh ảo |
| Phụ thuộc AI | Tùy chọn (`--enable_openai`) với cache trong `cache/` |

## 📚 Mục lục
- [Tổng quan](#tổng-quan)
- [Điểm nổi bật](#điểm-nổi-bật)
- [Bắt đầu nhanh](#bắt-đầu-nhanh)
- [Demo](#demo)
- [Tính năng](#tính-năng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Điều kiện tiên quyết](#điều-kiện-tiên-quyết)
- [Cài đặt](#cài-đặt)
- [Cấu hình](#cấu-hình)
- [Cách dùng](#cách-dùng)
- [Ví dụ](#ví-dụ)
- [Dữ liệu, Cache và Log](#dữ-liệu-cache-và-log)
- [Ghi chú phát triển](#ghi-chú-phát-triển)
- [Khắc phục sự cố](#khắc-phục-sự-cố)
- [Ghi chú về việc dùng OpenAI](#ghi-chú-về-việc-dùng-openai)
- [Lộ trình](#lộ-trình)
- [Support](#-support)
- [Đóng góp](#đóng-góp)
- [Giấy phép](#giấy-phép)

## Tổng quan
`words_gpt` là hệ thống tạo và hiển thị thẻ từ vựng trên thiết bị e-ink, được xây dựng bằng Python.

Hệ thống kết hợp:
- Nguồn từ từ CSV/bộ dữ liệu cục bộ và tùy chọn sinh từ bằng OpenAI.
- Làm giàu dữ liệu (phiên âm IPA + các trường từ đồng nghĩa đa ngôn ngữ).
- Pipeline dựng hình cho cả phần cứng và đầu ra ảo.
- Dịch vụ HTTP Tornado để kích hoạt từ xa và truy xuất ảnh.

Codebase hiện tập trung quanh `app.py`, `words_gpt.py`, `words_data.py`, `words_database.py`, và `openai_request_json.py`.

## Điểm nổi bật
- 🖼️ Pipeline dựng hình e-ink với nhiều chế độ nội dung (kanji, tiếng Nhật, tiếng Ả Rập, tiếng Trung, emoji).
- 🗃️ Cơ sở dữ liệu từ cục bộ (`words_phonetics.db`) với danh sách từ dựa trên CSV trong `data/`.
- 🤖 Chọn từ và làm giàu phiên âm bằng OpenAI với đầu ra JSON có cấu trúc.
- 🌐 Dịch vụ HTTP cho kích hoạt bên ngoài và truy xuất ảnh.
- ⚡ Tầng cache (`cache/`) giúp giảm các lần gọi OpenAI lặp lại.

## Bắt đầu nhanh
| Mục tiêu | Lệnh |
|---|---|
| Khởi động HTTP server (cổng `8082`) | `python app.py` |
| Chạy trình dựng độc lập (CSV) | `python words_gpt.py --use_csv` |
| Chạy với OpenAI + CSV | `python words_gpt.py --enable_openai --use_csv` |
| Chế độ Emoji + CJK giản thể | `python words_gpt.py --make_emoji --simplify` |
| Thiết lập tự động cho Raspberry Pi | `bash scripts/setup_pi_wordscard.sh` |

## Demo
<p align="center">
  <img src="demos/demo.jpg" alt="Demo" width="48%" />
  <img src="demos/words_card_arabic.JPG" alt="Arabic word card" width="48%" />
</p>

## Tính năng
- Luồng dựng hình phần cứng + ảo (`EPaperHardware`, `EPaperDisplay`) từ `words_gpt.py`.
- Pipeline làm giàu đa ngôn ngữ trong `words_data.py` (IPA, biến thể tiếng Nhật, tiếng Ả Rập, tiếng Pháp, các trường tiếng Trung).
- Lưu trữ SQLite với các hàm hỗ trợ cập nhật trường động trong `words_database.py`.
- Trình hỗ trợ request JSON có cấu trúc cho OpenAI kèm cache tệp trong `openai_request_json.py`.
- Tài nguyên PWA tùy chọn trong `pwa/` cho workflow cấu hình/preview frontend gọn nhẹ.

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

Tệp runtime quan trọng:
- `app.py`: web server Tornado (cổng mặc định `8082`) và vòng lặp cập nhật định kỳ.
- `words_gpt.py`: vòng lặp dựng độc lập và các lớp hiển thị.
- `words_data.py`: điều phối lõi cho lấy từ/làm giàu dữ liệu.
- `words_database.py`: các hàm hỗ trợ kho SQLite.
- `scripts/*.sh`: script thiết lập Raspberry Pi, cài service và quản lý vòng đời tmux.

## Điều kiện tiên quyết
- Python `3.9+` (khuyến nghị).
- Raspberry Pi mục tiêu (cho chế độ phần cứng).
- Màn hình e-paper Waveshare được hỗ trợ.
- Đã bật SPI trên Pi (`raspi-config`) và đấu dây theo đúng model panel.

Các gói Python dùng trong dự án bao gồm:
- `openai`, `tornado`, `Pillow`, `numpy`, `nltk`, `opencc`, `pykakasi`, `arabic_reshaper`, `python-bidi`, `pytz`.
- Script setup còn cài thêm: `json5`, `pandas`, `spidev`, `RPi.GPIO`, `gpiozero`, `lgpio`.

## Cài đặt

### Tùy chọn A: Cài tối thiểu/thủ công
Cài gói driver Waveshare:
```bash
python setup.py install
```

Nếu dùng danh sách từ của NLTK, tải một lần:
```bash
python -m nltk.downloader words
```

### Tùy chọn B: Thiết lập tự động cho Raspberry Pi (khuyến nghị trên thiết bị)
Từ thư mục gốc repo:
```bash
bash scripts/setup_pi_wordscard.sh
```

Script này sẽ:
- Cài phụ thuộc apt.
- Đảm bảo SPI đã bật.
- Tạo và kích hoạt virtual env `wordscard`.
- Cài các phụ thuộc Python runtime.
- Cài gói Waveshare.
- Khởi chạy `app.py` trong một phiên tmux.

## Cấu hình

### Hành vi của `.env`
Repo này nạp biến môi trường từ `.env` ngay khi import và **ghi đè** mọi giá trị shell đang có. Cách này giúp giá trị ghi đè cục bộ luôn xác định, kể cả khi bạn đã export biến trong profile shell.

Tạo hoặc cập nhật `.env`:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id
OPENAI_MODEL=gpt-4o-mini
```

### Truyền tham số cho app
Các script systemd/tmux hỗ trợ:
```bash
APP_ARGS="--enable_openai --use_csv" ./scripts/start_wordscard.sh
```

### Cờ CLI (server và renderer)
Cả `app.py` và `words_gpt.py` đều hỗ trợ:
- `--enable_openai`
- `--make_emoji`
- `--ignore_list`
- `--simplify`
- `--use_csv`
- `--complete_csv`
- `--filename <csv_file>`

## Cách dùng

### Chạy HTTP server
Khởi động service (cổng mặc định `8082`):
```bash
python app.py
```

Các route quan sát trong mã:
- `POST /display_word`
- `GET /get_current_word`
- `GET /get_current_word_page`
- `GET /next_random_word`
- `POST /get_words_card`
- `GET /static/(.*)` (từ `words_card_temp/`)

Ghi chú tương thích: tài liệu cũ từng tham chiếu `GET /current_word`; route hiện tại trong `app.py` là `GET /get_current_word`.

### Chạy renderer độc lập
Danh sách dựa trên CSV:
```bash
python words_gpt.py --use_csv
```

Bật OpenAI:
```bash
python words_gpt.py --enable_openai --use_csv
```

Dựng emoji + CJK giản thể:
```bash
python words_gpt.py --make_emoji --simplify
```

### Chế độ service trên Raspberry Pi
Cài service unit:
```bash
bash scripts/install_wordscard_service.sh
```

Sau đó:
```bash
sudo systemctl start wordscard
sudo systemctl status wordscard -n 50
journalctl -u wordscard -n 100 --no-pager
```

## Ví dụ

### Kích hoạt từ ngẫu nhiên tiếp theo
```bash
curl "http://127.0.0.1:8082/next_random_word"
```

### Đọc payload từ hiện tại
```bash
curl "http://127.0.0.1:8082/get_current_word"
```

### Gửi từ cụ thể
```bash
curl -X POST "http://127.0.0.1:8082/display_word" \
  -H "Content-Type: application/json" \
  -d '{"word":"serendipity"}'
```

### Hardware smoke test
Dùng script theo đúng màn hình:
```bash
python epd_7in3f_test.py
```

Hoặc:
```bash
python epd_13in3k_test.py
```

Nhiều ví dụ khác nằm trong `waveshare/examples/`.

## Dữ liệu, Cache và Log
| Khu vực | Đường dẫn | Ghi chú |
|---|---|---|
| Danh sách từ | `data/` | Bao gồm `data/words_list.csv` và các tệp CSV theo chủ đề |
| DB bền vững | `words_phonetics.db` | Kho cục bộ cho phiên âm/làm giàu |
| Hiện vật OpenAI/cache | `cache/` | Giảm request lặp lại |
| Log | `logs/`, `logs-word-phonetics/` | Log runtime và cập nhật |
| Thẻ sinh ra | `words_card_temp/` | Ảnh đầu ra và nguồn phục vụ static |

## Ghi chú phát triển
- Quản lý phụ thuộc hiện theo hướng script-first (`scripts/setup_pi_wordscard.sh`) + `setup.py`; chưa có `requirements.txt` hoặc `pyproject.toml`.
- Có nhiều tệp backup/legacy (`words_data_*`, `words_gpt_old.py`); luồng runtime đang dùng chủ yếu là `app.py` + `words_gpt.py` + `words_data.py` + `words_database.py`.
- `env_loader.py` luôn ghi đè biến môi trường từ `.env` khi khóa tồn tại.
- Chế độ server chạy luồng làm mới định kỳ (mỗi ~5 phút), có thể tự gọi endpoint cập nhật nội bộ.

## Khắc phục sự cố
- `ModuleNotFoundError` hoặc lỗi import:
  - Đảm bảo virtual environment đang hoạt động và đã cài dependencies.
  - Chạy lại `bash scripts/setup_pi_wordscard.sh` trên Pi.
- Lỗi OpenAI (`401`, thiếu model/key):
  - Kiểm tra `OPENAI_API_KEY` và tùy chọn `OPENAI_MODEL` trong `.env`.
  - Xác nhận thiết bị có kết nối mạng.
- Màn hình không cập nhật:
  - Kiểm tra model/đấu dây panel và chạy đúng script test (`epd_7in3f_test.py` hoặc `epd_13in3k_test.py`).
  - Xác nhận SPI đã bật (`sudo raspi-config nonint do_spi 0`).
  - Trên Pi 5, đảm bảo symlink tương thích `/dev/spidev0.0` nếu thiết bị hiển thị `/dev/spidev10.0`.
- Lỗi cài OpenCC:
  - Dùng gói tương thích distro (`libopencc1` hoặc `libopencc2`) như trong script setup.
- Lệch route API:
  - Dùng `/get_current_word` cho payload hiện tại, không phải `/current_word`.

## Ghi chú về việc dùng OpenAI
Truy cập OpenAI là tùy chọn, nhưng được khuyến nghị để sinh từ mới và làm giàu phiên âm. Trình hỗ trợ JSON có cấu trúc trong `openai_request_json.py` lưu cache kết quả dưới `cache/` để giảm số lần gọi lặp lại.

## Lộ trình
- Thêm manifest phụ thuộc chính thức (`requirements.txt` hoặc `pyproject.toml`) để cài đặt có thể tái lập.
- Mở rộng `i18n/` với các bản README dịch được duy trì.
- Hợp nhất các biến thể script legacy/backup sau khi luồng chuẩn được chốt.
- Tài liệu hóa quy trình PWA (`pwa/`) với ví dụ endpoint và ảnh chụp màn hình.
- Thêm test tự động lặp lại được cho dữ liệu và hành vi ở mức route.

## ❤️ Support

Nếu dự án này hữu ích với bạn, các liên kết dưới đây sẽ hỗ trợ trực tiếp cho việc bảo trì liên tục và thử nghiệm phần cứng.

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

### Sự hỗ trợ của bạn giúp thực hiện
- <b>Duy trì công cụ mở</b>: hosting, suy luận, lưu trữ dữ liệu và vận hành cộng đồng.
- <b>Phát hành nhanh hơn</b>: dành thời gian nguồn mở tập trung cho WordsCardEink và các công cụ học tập liên quan.
- <b>Tạo mẫu thiết bị</b>: lặp thiết kế phần cứng e-ink và nghiên cứu bố cục hiển thị.
- <b>Tiếp cận cho mọi người</b>: tài trợ triển khai cho học sinh/sinh viên, nhà sáng tạo và các nhóm cộng đồng.

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

## Đóng góp
Xem `AGENTS.md` để biết hướng dẫn cộng tác viên, chuẩn mã nguồn và kỳ vọng cho PR.

Checklist đóng góp được khuyến nghị:
- Bao gồm model panel + ghi chú phần cứng cho thay đổi hiển thị.
- Liệt kê chính xác các lệnh đã chạy để xác thực.
- Đính kèm screenshot/ảnh chụp cho thay đổi UI hoặc đầu ra e-ink.
- Mô tả chỉnh sửa dataset (tệp + tác động hàng/cột).

## Giấy phép
Hiện chưa có tệp `LICENSE` trong thư mục gốc repo (ghi nhận ở lần rà soát bản nháp này). Cho tới khi tệp license được thêm, quyền tái sử dụng chưa được cấp rõ ràng.

Giả định: các maintainer có thể thêm giấy phép nguồn mở rõ ràng trong bản cập nhật tiếp theo.
