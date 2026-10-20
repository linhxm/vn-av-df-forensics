# Sổ tay chạy pipeline real v003

Từ URL YouTube đến part real đã lên Drive / Kaggle, từng lệnh một: chạy gì, in ra gì, khi nào chạy lệnh tiếp. Bảng tóm tắt ở [README.md](README.md); căn cứ các con số ở [docs/v003_data.md](../docs/v003_data.md).

Mọi lệnh chạy trong `data_pipeline/` với env `vn-av-df`:

```powershell
conda activate vn-av-df
cd D:\COMP\RESEARCH\Deepfake_VN\vn-av-df-forensics\data_pipeline
```

## 1. Từ điển

| Từ | Nghĩa |
|---|---|
| **input** | Một dòng trong `data/inputs/urls.csv`: một URL (video, kênh, playlist) + `speakers` = `1` (một người nói), `2` (hai người: phỏng vấn 1-1, podcast) hoặc `n` (nhiều người: talkshow). Dòng đã plan thì không đổi `speakers` được |
| **raw** | Video gốc tải từ YouTube, nằm trong `data/raw/batch_XXX/` |
| **batch** | Một nhóm video raw tải cùng lúc, ≤ 20 giờ và ≤ 16 GB (vừa ổ đĩa). Bước 01 tự chia. Mọi việc tải / cắt / dọn đều làm theo batch |
| **workspace** | Thư mục `data/` chứa mọi thứ của pipeline: ledger, raw, pool, registry, archive |
| **ledger** | Sổ cái `data/ledger/ledger.sqlite`: ghi mọi URL, video, lý do loại, clip, speaker, quyết định duyệt, part, nơi lưu. Mỗi lệnh tự sao lưu (`backups/`, giữ mọi bản, không tự xoá) và xuất CSV (`csv/`, mở bằng Excel). Mất ledger là mất toàn bộ quyết định, nên **luôn up ledger lên Drive kèm zip** |
| **clip ứng viên** | Đoạn 4-7 s cắt từ raw (25 fps, ≤ 720p), đã qua kiểm mặt, chưa ai duyệt |
| **pool** | `data/pool/`: kho clip ứng viên (`clips/`) và ảnh dải mặt (`faces/`) của mọi batch |
| **VAD** | Voice Activity Detection (Silero VAD): đoán từng khúc 32 ms có tiếng nói không. Dùng để cắt clip ở chỗ người nói ngắt hơi, không cắt ngang từ |
| **jump cut** | Chỗ video bị cắt ghép (nhảy hình) ngay trong một cảnh. Clip có jump cut bị loại |
| **SyncNet / độ lệch** | Mô hình đo tiếng sớm hay muộn so với hình bao nhiêu frame. Đo mỗi video ở bước 03, sửa ở bước 07. `ok` = đo chắc chắn; `unreliable` = không đo được, clip của video đó tự loại |
| **fps** | Số hình / giây. Pipeline cần ≥ 25 fps; video 24 fps bị loại vì nhân frame lên 25 sẽ sinh giật hình chỉ có ở real (mô hình học lỏm) |
| **speaker** | Một người, mã `spk_0001`… Bước 04 tự gán clip vào speaker bằng mặt và giọng; bước 05 bạn sửa và xác nhận |
| **registry** | `data/registry/`: ảnh mặt đại diện mỗi speaker, dùng để nhận lại người đó ở batch sau |
| **identify** | Bước 04: gán clip → speaker (mới hoặc đã có) |
| **review / duyệt** | Bước 05: bạn chọn clip tốt, gán giới, gộp / tách người trên trang web |
| **keep** | Clip bạn chọn (S). Chỉ clip keep vào part |
| **done / pending / excluded** | Trạng thái duyệt của speaker: xong (đủ 15-20 clip, đã gán giới) / còn chờ (chưa duyệt hoặc chưa đủ 15) / bị loại (E) |
| **part** | Gói dữ liệu cố định: 20 nam + 20 nữ, mỗi người 15-20 clip keep (600-800 clip). Tạo xong là **khoá**, không đổi thành viên |
| **split** | Chia người trong part thành train / validation / test = 12 / 3 / 5 người mỗi giới. Một người chỉ nằm ở một split |
| **dư speaker** | Người đã duyệt xong nhưng giới đó đã đủ 20 trong part: nằm chờ part sau, không mất gì |
| **export** | Bước 07: chép clip keep của part ra thư mục riêng, sửa độ lệch tiếng, ghi manifest. Không zip |
| **manifest** | `manifest.jsonl`: mỗi dòng mô tả một clip (người, giới, split, nguồn, độ lệch…). Generation và training đọc file này |
| **archive** | Bước 08: zip raw batch hoặc part vào `data/archive/` để up Drive. Raw và part zip cùng một kiểu lệnh |
| **`--uploaded`** | Bạn báo cho ledger "zip này đã lên Drive". Trước khi báo, lệnh xoá (09) từ chối chạy |
| **location** | Nơi lưu ghi trong ledger: `local` (chỉ ở máy) → `packed` (đã có zip) → `drive:<tên zip>` (đã lên Drive) → `deleted` (đã xoá ở máy, ledger vẫn nhớ) |
| **cut signature** | Dấu vân tay của cách làm ra một clip (độ dài, mặt, chuyển cảnh, jump cut, VAD, SyncNet: config + code + mô hình). Đổi những luật này sau khi đã có clip thì bước 03 chặn, để mọi clip cắt cùng một luật. Luật số lượng (`single`, `pair`, `quota.py`) đổi được, không bị chặn. Thử luật mới trên workspace khác: `$env:VNAV_DATA="data_trial"` |

## 2. Bức tranh

```
                     ┌────────────── mỗi batch ──────────────┐
urls.csv ─ 01 plan ─▶ 02 download ─▶ 03 cut ─▶ 04 identify ─▶ 05 review
                                       │                        │
                                       ▼                        ▼
                         08 archive raw N → up Drive    đủ 20 nam + 20 nữ?
                         → --uploaded → 09 clean raw N          │
                                                                ▼
                         06 parts ─▶ 07 export N ─▶ up thư mục lên Kaggle (private)
                         ─▶ 08 archive part N → up Drive → --uploaded ─▶ 09 clean part N ─▶ 09 clean pool
```

Raw và part đi cùng một nhịp ở bước 08-09: **zip → up Drive → `--uploaded` → xoá**.

`10_status.py` xem toàn cảnh, chạy lúc nào cũng được (chỉ đọc).

## 3. Từng lệnh

### 00 — Cài mô hình (một lần)

```powershell
python steps/00_setup.py
```

Tải YuNet (dò mặt), Silero VAD (tiếng nói), SFace (nhận mặt), WeSpeaker (nhận giọng), SyncNet v2 (độ lệch) vào `checkpoints/`.

### 01 — Lập kế hoạch

```powershell
python steps/01_plan.py
python steps/01_plan.py --cookies-from-browser firefox   # khi YouTube chặn
```

- Đọc **dòng mới hoặc dòng lỗi** trong `urls.csv`; dòng đã xử lý không đọc lại. Thêm URL thì chỉ cần chạy lại.
- Nguồn 1 người: chọn video trải theo ngày đăng tới khi đủ ≥ 20 phút (3-10 video). Nguồn `2` / `n`: 15 video mới nhất (link video lẻ: chính video đó).
- Loại: video dọc, < 25 fps, < 720p, lồng tiếng, trùng.
- Chia video đã chọn vào batch.
- YouTube đòi xác minh bot ("Sign in to confirm you're not a bot"): **dừng ngay**, không hỏi tiếp các dòng sau (hỏi tiếp chỉ làm bị chặn lâu hơn). Dòng bị chặn ghi `error`, dòng chưa hỏi vẫn `pending`; kết quả có mục `stopped`. Chạy lại với `--cookies-from-browser firefox` hoặc chờ vài giờ.

In ra:

| Trường | Nghĩa |
|---|---|
| `inputs_added` | Dòng mới đọc từ `urls.csv` |
| `inputs_processed` | Dòng xử lý lần này |
| `errors` | Dòng lỗi (thường do mạng): chạy lại 01 |
| `empty` | Input không còn video nào đạt (ví dụ cả kênh 24 fps). Lý do từng video: `data/ledger/csv/videos.csv`, cột `reason` |
| `batches` | Mỗi batch: số video, giờ, GB ước tính |

### 02 — Tải

```powershell
python steps/02_download.py        # batch đầu tiên còn video chưa tải
python steps/02_download.py 2      # batch 2
```

- Ctrl+C rồi chạy lại: tải tiếp từ chỗ dừng.
- Lỗi 403 tạm thời: tự thử lại. YouTube đòi xác minh bot: lệnh dừng, chạy lại với `--cookies-from-browser firefox`.
- Tải xong, video nào kiểm lại không đạt (fps, độ phân giải) thì bị loại.

In ra: `downloaded` (tải xong), `failed` (lỗi, chạy lại), `rejected` (tải về nhưng không đạt), `size_gb`.

### 03 — Cắt clip ứng viên

```powershell
python steps/03_cut.py          # batch đầu còn video chưa cắt
python steps/03_cut.py 2
```

- VAD tìm đoạn nói; đoạn dài hơn 7 s được chia ở chỗ ngắt dài nhất trong khoảng 4-7 s.
- Nguồn `1`: 30-40 clip / input, rải đều cả video. Nguồn `2`: tối đa 150 clip / video, rải đều (từ batch 3). Nguồn `n`: cắt hết. Kết quả mỗi video ghi `mode` (single / pair / multi) và `quota` trong `cut_info`.
- Mỗi frame phải có đúng 1 mặt ≥ 48 px, không chuyển cảnh, không jump cut.
- SyncNet đo độ lệch tiếng-hình mỗi video.
- Ctrl+C rồi chạy lại: video đang dở cắt lại từ đầu, video xong giữ nguyên.

Mỗi video in một dòng, ví dụ:

```
[input 12] abc123 (14.2 phút)
  38 clip / 61 cửa sổ kiểm, loại 23 | lệch 2 frame (ok) | 95s
```

| Phần | Nghĩa |
|---|---|
| `38 clip` | Clip ứng viên giữ lại |
| `61 cửa sổ kiểm` | Số đoạn 4-7 s đã thử |
| `loại 23` | Bị loại vì mặt / chuyển cảnh / jump cut |
| `lệch 2 frame (ok)` | Tiếng chậm hơn hình 2 frame (80 ms), đo chắc chắn. `unreliable` → clip video này tự loại |

Cuối lệnh: `videos` (đã cắt), `clips` (tổng clip), `skipped` (đã cắt từ trước), `failed` (lỗi, chạy lại).

### 04 — Gán speaker

```powershell
python steps/04_identify.py          # batch đầu còn clip chưa gán
python steps/04_identify.py 2
python steps/04_identify.py --calibrate   # sau khi duyệt vài chục người: gợi ý ngưỡng mới
```

In ra:

| Trường | Nghĩa |
|---|---|
| `inputs` | Input đã xử lý |
| `clips` | Clip đã gán speaker |
| `new_speakers` | Người mới tạo |
| `matched` | Nhóm clip khớp với người đã có (cùng kênh hoặc batch trước) |
| `suggestions` | Người mới nhưng giống một người cũ: trang duyệt sẽ gợi ý gộp |

Clip lạc (mặt / giọng khác hẳn phần còn lại) được gắn cờ `face_outlier` / `voice_outlier` để bạn để ý khi duyệt.

### 05 — Duyệt

```powershell
python steps/05_review.py
```

Mở http://127.0.0.1:8001. Mọi thao tác tự lưu; tắt bằng Ctrl+C.

| Phím | Việc |
|---|---|
| ← → | Clip trước / sau |
| ↑ ↓ | Video trước / sau của người này |
| S | Chọn clip (keep) |
| X | Loại clip |
| R | Clip là người khác: chuyển sang người gần nhất |
| M / F | Nam / nữ |
| G | Gộp vào người được gợi ý |
| E | Loại cả người |
| B | Người trước |
| N | Xong người này, sang người tiếp |

Luật chọn: **15-20 clip / người**. Đạt 15 là được; còn clip tốt thì chọn tới 20. Clip không chọn thì để nguyên.
- Dưới 15 clip mà bấm N: người đó **chờ** (pending). Batch sau khớp thêm clip của họ thì họ hiện lại.
- Từ 15 trở lên: N hỏi xác nhận giới rồi đánh dấu **xong** (done).

### 06 — Tạo part

```powershell
python steps/06_parts.py
```

- Lấy người **done**, đã có giới, có 15-20 clip keep, theo thứ tự duyệt xong.
- Đủ 20 nam + 20 nữ → tạo part mới, khoá lại. Còn đủ thì tạo tiếp part sau.
- Chia split 12 / 3 / 5 mỗi giới, cân tổng clip giữa các split.
- Người xuất hiện chung một video đi cùng nhau (cùng part, cùng split).

In ra:

| Trường | Nghĩa |
|---|---|
| `created` | Part vừa tạo: số người, số người và clip mỗi split |
| `ready_left_over` | **Dư speaker**: người sẵn sàng nhưng giới đó đã đủ, chờ part sau |
| `ready_but_waiting_for_group` | Người sẵn sàng nhưng còn người chung video chưa duyệt xong |
| `blocked_groups` | Nhóm dính với người đã ở part cũ: cần xử lý tay (hiếm) |

### 07 — Export part

```powershell
python steps/07_export.py 1
```

- Ghi vào `exports/vn-av-df-data/vn-av-df-data-part1.partial/`. Xong thì đổi tên bỏ `.partial` (thấy `.partial` là chưa xong hoặc bị ngắt; chạy lại sẽ làm lại).
- Dịch tiếng mỗi clip theo độ lệch đo ở bước 03, để tiếng khớp hình.
- Kiểm toàn bộ (`quality_report.json`). **Không zip**: zip ở bước 08, giống raw.
- Export lại một part đã zip: zip cũ bị xoá (không còn khớp), chạy lại 08.

Thư mục part:

| File | Nội dung |
|---|---|
| `clips/*.mp4` | 600-800 clip |
| `manifest.jsonl` | Mỗi clip một dòng |
| `dataset_info.json` | Số clip, số người, số người mỗi split × giới |
| `quality_report.json` | Kết quả kiểm |

### Up part lên Kaggle (bạn tự làm)

Kaggle → New Dataset → up thư mục `exports/vn-av-df-data/vn-av-df-data-part1/` → **Private**. Notebook `generate.ipynb` trỏ `CLEAN_PART` tới thư mục có `manifest.jsonl` trong dataset này.

### 08 — Zip và báo đã lên Drive

Raw và part cùng một kiểu:

```powershell
python steps/08_archive.py raw 1              # zip raw batch 1 → data/archive/raw_batch_001.zip
python steps/08_archive.py raw 1 --uploaded   # sau khi đã up zip + ledger/ lên Drive

python steps/08_archive.py part 1             # zip part 1 → data/archive/vn-av-df-data-part1.zip
python steps/08_archive.py part 1 --uploaded  # sau khi đã up zip + ledger/ lên Drive
```

| | Zip được khi | Trong zip |
|---|---|---|
| raw N | Mọi video của batch đã tải + cắt + identify xong (**không** cần duyệt xong: clip đã nằm trong pool) | Video gốc + `videos.csv` + `SHA256SUMS.txt` |
| part N | 07 export xong | Thư mục part (clip + manifest + …) |

- Chạy lại lệnh zip khi zip đã có: không zip lại, chỉ in tên zip.
- Mỗi lần up Drive, up **cả thư mục `data/ledger/`**.

### 09 — Dọn máy

```powershell
python steps/09_clean.py raw 1    # xoá raw batch 1 + zip; clip trong pool giữ nguyên
python steps/09_clean.py part 1   # xoá export part 1 + zip + clip trong pool của 40 người đó
python steps/09_clean.py pool     # xoá clip không chọn của người đã done / excluded
```

- Chỉ chạy được với thứ đã `--uploaded`. Mở thử zip trên Drive trước khi xoá.
- `pool` **không bao giờ** xoá clip của người còn chờ, hoặc clip chưa gán người. Clip thuộc batch chưa up raw thì giữ lại, kết quả báo `kept_until_raw_uploaded`.
- Ledger vẫn nhớ mọi thứ đã xoá (`location = deleted`); cần lại thì lấy raw trên Drive cắt lại.
- Sau `pool`, người đã done chỉ còn clip đã chọn. Muốn còn đổi clip cho ai thì chưa chạy `pool`.

### 10 — Xem toàn cảnh

```powershell
python steps/10_status.py
```

Mỗi batch: số video, giờ, GB, đã tải / cắt / identify, số clip, raw ở đâu. Số người theo trạng thái, các part và nơi lưu, ổ đĩa còn trống.

## 4. Chạy song song

- Được: `05_review` ở một terminal, `02_download` / `03_cut` / `04_identify` **batch khác** ở terminal khác. Ledger chờ khoá tới 60 s.
- Không nên: hai lệnh cùng bước, hoặc hai lệnh ghi cùng một batch cùng lúc.

## 5. Dừng giữa chừng

Mọi bước dừng bằng Ctrl+C và chạy lại cùng lệnh là làm tiếp. Ledger ghi theo từng video / clip, nên không cần tách file hay ghi chú gì thêm.

## 6. Ví dụ: batch 1 (20/10/2026)

| Bước | Kết quả |
|---|---|
| 01 | 89 URL; 79 input có video, 10 input rỗng (cả 10 đều 24 fps) |
| 05 | 41 người done (20 nam, 21 nữ), 1 người bị loại |
| 06 | Part 1: 40 người, 800 clip; train 480 / validation 120 / test 200. Dư 1 nữ (`spk_0041`) chờ part 2 |
| 07 | Export part 1 |

Việc tiếp theo:

1. Chờ 07 xong → up thư mục part 1 lên Kaggle (private).
2. `08_archive.py part 1` và `08_archive.py raw 1` → up hai zip + `ledger/` lên Drive → `08_archive.py part 1 --uploaded`, `08_archive.py raw 1 --uploaded`.
3. `09_clean.py raw 1`, `09_clean.py part 1`, rồi `09_clean.py pool`.
4. `10_status.py` kiểm lại.
5. Hai việc song song:
   - Generation part 1 trên Kaggle (`notebooks/generate.ipynb`).
   - Batch 2 cho part 2 (còn thiếu 20 nam + 19 nữ): thêm URL vào `urls.csv` → 01 → 02 → 03 → 04 → 05.
