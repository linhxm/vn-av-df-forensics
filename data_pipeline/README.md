# Pipeline dữ liệu real v003

Chạy ở **máy local**: từ danh sách URL YouTube ra **part** real (20 nam + 20 nữ × 15-20 clip = 600-800 clip, chia sẵn train / validation / test theo người). Part là đầu vào của bước sinh fake (`generation/`, `notebooks/generate.ipynb`).

- Chạy từng lệnh, đọc kết quả in ra, từ điển thuật ngữ: [RUNBOOK.md](RUNBOOK.md).
- Căn cứ các con số và luật: [docs/v003_data.md](../docs/v003_data.md) (tổng quan tài liệu Ch.2, số người / clip Ch.3, pipeline Ch.4: chọn và cắt §4.1-§4.2, độ lệch tiếng-hình §4.3, lưu trữ §4.8; thí nghiệm Ch.5; lỗi và thay đổi Ch.6).

Chỉ dùng video của chính mình hoặc đã được phép. Raw, clip và part là dữ liệu người thật: **không chia sẻ công khai**.

## Cài đặt

```powershell
conda activate vn-av-df
python 00_install.py                     # từ thư mục gốc repo: cài data_pipeline + package chính
python data_pipeline/steps/00_setup.py   # tải YuNet, Silero VAD, SFace, WeSpeaker, SyncNet v2
```

Cần Node.js trong PATH (yt-dlp dùng để tải YouTube).

## Các bước

| Bước | Lệnh | Làm gì |
|---|---|---|
| 01 | `01_plan.py [--cookies-from-browser firefox]` | Đọc dòng mới trong `data/inputs/urls.csv` (url, speakers = 1 / 2 / n, note), chọn video, loại video dọc / < 25 fps / < 720p / lồng tiếng / trùng. Chia **batch ≤ 20 giờ và ≤ 16 GB**. Gặp xác minh bot thì dừng ngay; chạy lại với `--cookies-from-browser firefox` |
| 02 | `02_download.py [N]` | Tải batch (720p). YouTube đòi xác minh bot thì chạy lại với `--cookies-from-browser firefox` |
| 03 | `03_cut.py [N]` | Cắt clip ứng viên 4-7 s, mép clip ở chỗ ngắt hơi (VAD). Nguồn `1`: 30-40 clip rải đều; `2`: tối đa 150 clip / video rải đều; `n`: cắt hết. Mỗi frame đúng 1 mặt, không chuyển cảnh, không jump cut. Đo độ lệch tiếng-hình (SyncNet) |
| 04 | `04_identify.py [N]` | Gán clip vào speaker bằng mặt + giọng, nhận lại người đã có, gợi ý gộp. `--calibrate`: gợi ý ngưỡng sau vài chục người |
| 05 | `05_review.py` | Trang duyệt http://127.0.0.1:8001: gộp / tách người, gán giới, **chọn 15-20 clip / người**. Dưới 15 clip: N để chờ batch sau |
| 06 | `06_parts.py` | Gom người đã duyệt xong thành part 20 nam + 20 nữ, split 12 / 3 / 5 mỗi giới. Part tạo xong bị khoá; người dư chờ part sau |
| 07 | `07_export.py N` | Xuất part N ra thư mục `exports/vn-av-df-data/vn-av-df-data-partN/` (không zip), **dịch tiếng cho khớp môi** theo độ lệch đo ở 03 |
| 08 | `08_archive.py raw N`<br>`08_archive.py part N` | Zip raw batch N / part N vào `data/archive/` |
| — | *(bạn tự up)* | Up file zip vừa tạo + cả thư mục `data/ledger/` lên Drive. Pipeline không tự up |
| 08 | `08_archive.py raw N --uploaded`<br>`08_archive.py part N --uploaded` | Ghi vào ledger: zip đã lên Drive. Chưa chạy lệnh này thì 09 không cho xoá |
| 09 | `09_clean.py raw N` / `part N` / `pool` | Xoá bản ở máy, chỉ khi đã `--uploaded`. `pool`: xoá clip không chọn của người đã duyệt xong / bị loại |
| 10 | `10_status.py` | Toàn cảnh: batch, clip, người, part, dung lượng |

`[N]` = số batch (bỏ trống: batch đầu còn việc). Mọi bước dừng được bằng Ctrl+C, chạy lại thì làm tiếp.

## Thứ tự

```
mỗi batch:  01 plan → 02 download → 03 cut → 04 identify → 05 review
đủ người:   06 parts → 07 export N → up thư mục part lên Kaggle (private, bạn tự up)
```

Raw và part cùng một nhịp ở 08-09: **zip → up Drive → `--uploaded` → xoá**. Raw làm được ngay sau 04, không cần chờ duyệt.

```powershell
# raw batch N
python steps/08_archive.py raw N              # zip → data/archive/raw_batch_00N.zip
#   up zip + data/ledger/ lên Drive (tay)
python steps/08_archive.py raw N --uploaded
python steps/09_clean.py raw N

# part N
python steps/08_archive.py part N             # zip → data/archive/vn-av-df-data-partN.zip
#   up zip + data/ledger/ lên Drive (tay)
python steps/08_archive.py part N --uploaded
python steps/09_clean.py part N
python steps/09_clean.py pool                 # sau cùng: clip thừa của người đã duyệt xong
```

## Thư mục `data/` (một thư mục cho mọi batch và part)

| Đường dẫn | Nội dung | Giữ / xoá |
|---|---|---|
| `inputs/urls.csv` | danh sách URL | giữ |
| `ledger/ledger.sqlite` | **sổ cái**: URL, video, lý do loại, batch, clip, độ lệch, speaker, quyết định duyệt, part, nơi lưu trên Drive. Kèm `backups/` (mọi bản, không tự xoá) và `csv/` (xem bằng Excel) | **giữ, up Drive mỗi lần archive** |
| `raw/batch_XXX/` | video gốc + `videos.csv` | xoá sau khi lên Drive (`09_clean raw`) |
| `pool/clips/`, `pool/faces/` | clip ứng viên, ảnh dải mặt | clip xoá theo part / pool; ảnh mặt giữ |
| `registry/` | ảnh mặt đại diện mỗi người | giữ |
| `archive/` | zip raw / part chờ up Drive (bước 08) | xoá cùng raw / part |

Part export nằm ngoài `data/`: `exports/vn-av-df-data/vn-av-df-data-partN/`, xoá bằng `09_clean part N`.

Cấu hình: [configs/data.yaml](configs/data.yaml). Đổi luật cắt sau khi đã có clip sẽ bị chặn (luật cắt phải thống nhất); thử luật mới trên workspace riêng: `$env:VNAV_DATA="data_trial"`.

## Kiểm thử

```powershell
python -m pytest data_pipeline/tests
```
