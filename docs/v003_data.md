> Bản sao từ nhật ký nghiên cứu (`RESEARCH_WORKLOG/v003_data.md`, ngoài repo) ngày 22/10/2026. Sửa ở nhật ký rồi chép lại vào đây.

# v003 · Dữ liệu real: từ URL YouTube tới part

Cập nhật 22/10/2026. File 1/3 của v003:
- **v003_data.md** (file này): vì sao cần bộ dữ liệu, số người và số clip, pipeline dữ liệu real, độ lệch tiếng-hình, so với chuẩn hiện nay.
- [v003_generation.md](v003_generation.md): sáu ô, generator và giọng, số fake mỗi clip, cách gán, chặn lối tắt, chạy sinh trên Kaggle.
- [v003_model.md](v003_model.md): khảo sát phương pháp, baseline, kiến trúc đề xuất, huấn luyện, đánh giá, W&B, demo.

Khảo sát bộ dữ liệu ban đầu (14/10, mức S / M / L): `RESEARCH_WORKLOG/dataset_survey.md` (ngoài repo). Code: `vn-av-df-forensics/data_pipeline/` (hướng dẫn chạy ở `data_pipeline/README.md`).

**File này là nguồn đầy đủ cho báo cáo phần dữ liệu real**, tính tới lúc tạo ra một clip real hoàn chỉnh: mọi con số, luật, lý do, bằng chứng, thí nghiệm đã làm (S00-S04) và gợi ý thí nghiệm nằm ở đúng mục tương ứng. §14 chỉ cách điền vào template báo cáo capstone (AIP491). Mọi số ● dưới đây chụp từ ledger ngày 22/10/2026 trừ khi ghi khác; dữ liệu (ledger, pool, raw, part) được giữ nguyên làm bằng chứng.

Quy ước mức tin:
- **●** đo trên dữ liệu của mình (batch thử 17/10, batch 1-6 thật 18-22/10, part1 v001 / v002);
- **✔** đọc từ paper gốc hoặc trang chính thức;
- **◐** chỉ đọc tóm tắt / metadata / một phần bài: đủ để trích thông tin xuất bản và ý chính, không trích số chi tiết;
- **○** suy luận hoặc ước lượng (ghi cách tính).

Riêng tư: báo cáo không nêu tên người, tên kênh, tiêu đề, mã video, mã speaker; các bảng trong file này dùng nhãn chung (video A, B…).

"Chưa tìm thấy" nghĩa là chưa tìm thấy tính đến ngày ghi, không khẳng định là chưa ai làm.

---

## 0. Tóm tắt

| Hạng mục | Giá trị | Mục |
|---|---|---|
| Đơn vị phát hành | **part** = 40 người (20 nam + 20 nữ) × **15-20 clip** = **600-800 clip real** | §3 |
| Split | theo người, mỗi giới 12 / 3 / 5 (train 24, validation 6, test 10 người; 60 / 15 / 25 %) | §2, §7 |
| Clip | 4-7 s, 25 fps, ≤ 720p, đúng 1 mặt người đang nói, tiếng Việt gốc | §5 |
| Nguồn | YouTube; nguồn 1 người ≥ 20 phút, 3-10 video; chỉ video ngang, ≥ 720p, ≥ 25 fps | §4 |
| Chỉ tiêu cắt nguồn 1 người | 30-40 clip ứng viên **mỗi input** (tối đa), rải đều theo thời gian | §5.3 |
| Chỉ tiêu cắt nguồn 2 người | tối đa **150 clip / video**, rải đều (từ batch 3, 20/10); nguồn nhiều người (`n`) cắt hết | §5.3 |
| Độ lệch tiếng-hình | đo bằng SyncNet theo video trong bước cắt; dịch tiếng về 0 khi export; video không đo được tự loại | §6 |
| Duyệt | người chọn 15-20 clip / người (còn clip tốt thì chọn đủ 20), gán giới, góc mặt lệch ≤ ~30 % | §7 |
| Quy mô mục tiêu | ≥ 5 part = 200 người ○ | §3 |
| **Hiện có (22/10)** | **3 part = 120 người (60 nam, 60 nữ) × 20 clip = 2.400 clip real**, từ 259 video / 115,6 giờ / 92 kênh (đã tải và cắt 251 video) | §9 |
| Clip real hoàn chỉnh | định nghĩa 10 thuộc tính, bước nào bảo đảm | §8.1 |

**Một câu:** nút thắt là **số người khác nhau**, không phải số clip; mọi luật chọn, cắt, duyệt đều nhằm có nhiều người, mỗi người đủ clip từ nhiều buổi quay, và real không mang đặc điểm cấu trúc nào (độ lệch tiếng-hình, nén) khác với fake sẽ sinh từ chính nó.

## 1. Vì sao cần bộ dữ liệu này

**Khoảng trống:** chưa tìm thấy bộ deepfake **có hình, lip-sync, tiếng Việt** (tính đến 14/10/2026). Tiếng Việt mới có VietPrism (deepfake chỉ âm thanh, 1.262 người) và ViCocktail (tiếng nói thật có hình, ~269 giờ) ✔. Hai bộ này cho thấy lấy 1.000+ người Việt từ YouTube là khả thi.

**Bài học từ v001 / v002 liên quan tới dữ liệu:**

| Bài học (số đo) | Nguồn | Yêu cầu cho dữ liệu v003 |
|---|---|---|
| Test chỉ 2-3 người, không tính được CI theo người; một người (01) đủ đảo kết luận ● | v001 §8.2, v002 §17 | 40 người / part, test 10 người / part; nhiều part |
| Điểm nền lệch theo người: AUC trong từng người 0,95-0,99, gộp chỉ 0,78 ● | v002 §15.4 | số clip mỗi người cố định; đủ clip real để chuẩn hoá theo người |
| Mất cân bằng 15-158 clip mỗi người ● | v001 §9 | số clip mỗi người **cố định 20** |
| Speaker gán tay theo video, chưa kiểm mặt trong clip đúng người nói | v001 §9 | gán tự động (mặt + giọng) + duyệt; SyncNet bắt lời dẫn đè hình |
| "Lệch A/V sẵn có của real chưa đo" | v001 §9, R2 | đo và sửa (§6) |
| Nhánh artifact học dấu vết xử lý (E6) ● | v002 §15.6 | real và fake encode giống hệt (xem v003_generation §6) |

## 2. Benchmark là gì, vì sao chia split ở bước dữ liệu

Một benchmark gồm 4 thứ cố định, công bố cùng nhau: **dữ liệu có nhãn**, **split chính thức**, **protocol chấm**, **kết quả baseline** trên protocol đó. Mục đích là so được con số giữa các nhóm. FakeAVCeleb không có split chính thức nên số liệu giữa các bài không so được; FaceForensics++, LAV-DF, AV-Deepfake1M có split chính thức.

Phần dùng để **chấm** là test. Test của v003 đo tổng quát theo 3 trục: **người chưa thấy** (split theo người, ở file này), **generator chưa thấy** và **giọng chưa thấy** (v003_generation §3).

**Vì sao chia ngay ở bước dữ liệu, không "fake hết rồi tự lọc":**
1. Split phải theo người và cố định từ đầu; thông tin "ai là ai, ai xuất hiện chung video nào" chỉ có ở bước dữ liệu. Cùng người hoặc cùng video ở cả train và test thì model nhận ra người chứ không nhận ra fake.
2. Bước sinh phụ thuộc split: ② sham và ⑤ donor lấy tiếng của cùng người, cùng split; clip tham chiếu cũng trong cùng split; generator / giọng "chỉ test" chỉ chạy trên clip test.
3. Chạy mọi generator trên mọi clip tốn thêm ~45-50 % GPU ○ mà phần thêm không dùng được (generator test-only phải "chưa thấy").

"Tự lọc" vẫn làm được (manifest ghi đủ generator, giọng, ô, full / partial). Thứ duy nhất không được đổi là split.

## 3. Bao nhiêu người, bao nhiêu clip mỗi người

### 3.1. Bằng chứng (đã đọc bản gốc ✔)

| Nghiên cứu | Kết quả | Hệ quả |
|---|---|---|
| Scaling Laws for Deepfake Detection (2025; 5,8 triệu real / 51 nguồn, 8,8 triệu fake / 102 phương pháp) | 1 − AUC giảm theo luỹ thừa khi tăng **số nguồn real** (α = 0,476), chưa bão hoà. Giảm số ảnh real 10 lần chỉ làm AUC 0,886 → 0,879. Khuyên không "thu thêm ảnh từ cùng nguồn" | đa dạng người > số clip lặp |
| Implicit Identity Leakage (CVPR 2023) | detector tự học danh tính; nguyên nhân chính của kém tổng quát | ít người → dễ học danh tính |
| Bansal et al. 2017 (nhận dạng mặt) | cùng tổng số ảnh: nhiều người tốt hơn nhiều ảnh cho ít người (mạng cỡ thường) | cùng hướng (bài toán gần, không phải deepfake) |
| GBDF (2022) | lệch giới → detector kém hơn ở nữ; cân giới công bằng hơn, không giảm độ chính xác | 20 nam / 20 nữ |
| Referee (2025), ID-Reveal (ICCV 2021), POI-Forensics | phương pháp tham chiếu người cần ít real: Referee 1 clip 1-3 s (AUC 99,71 %), ID-Reveal 6 s | 3 clip tham chiếu (~16 s) là đủ |
| Layton et al. (USENIX Security 2024) | khuyên tăng lượng media thật; FPR cần nhiều real ở test | thêm real mỗi người có lợi cho đo báo nhầm |

**Thống kê** ○: đơn vị độc lập khi chấm là **người** (bootstrap theo speaker), nên độ rộng CI phụ thuộc số người ở test, không phụ thuộc số clip. Trên part1, độ lệch chuẩn ΔAUC giữa người ~0,10 ● → nửa CI 95 % ≈ 2 × 0,10 / √(số người test): 10 người ±0,06; 50 người ±0,03.

### 3.2. Các bộ dữ liệu (số kiểm trên paper gốc ✔)

| Bộ | Người | Real | Real / người | Fake / real |
|---|---|---|---|---|
| FakeAVCeleb | 500 | 500 | 1 | 39 |
| Celeb-DF | 59 | 590 | 10 | ~9,6 |
| LAV-DF | 153 | 36.431 | ~238 | ~2,7 |
| KoDF | 403 | 62.166 | ~154 | ~2,8 |
| DFDC (bản phát hành; quay 3.426 người, 14,4 video / người) | 960 (train 486) | ~19 nghìn ở train ○ | ~245 ○ | ~5 |
| AV-Deepfake1M | 2.068 | 286.721 | ~139 | 3,00 |
| AV-Deepfake1M++ | 7.109 | 627.936 | ~88 | 2,27 |

- FakeAVCeleb, LAV-DF, AV-Deepfake1M **đều lấy real từ VoxCeleb2**, người có thể trùng nhau.
- Các bộ lớn có nhiều real mỗi người vì **cắt tự động gần như miễn phí**, nhưng phần lớn từ cùng vài buổi quay (DFDC 245 clip / người nhưng ~14 video gốc; LAV-DF, AV-Deepfake1M cắt từ một số ít video phỏng vấn VoxCeleb2). Nhiều clip không đồng nghĩa đa dạng.

### 3.3. Quyết định

**Số người mỗi part: 40 (20 nam + 20 nữ).** Trước là 50 (25 + 25), split 15 / 4 / 6; user chốt 40 ngày 18/10, split 12 / 3 / 5 giữ tỉ lệ 60 / 15 / 25 %. Thêm part = thêm người; mục tiêu ≥ 5 part (200 người, nằm giữa LAV-DF 153 và KoDF 403).

**Clip mỗi người: 20, cố định.**

| Phương án đã cân nhắc | Kết luận |
|---|---|
| 10 clip / người | thiết kế ban đầu. Mỏng: model học chỉ từ real (AVH-Align) có ~3 giờ real cho 4 part; 3 clip tham chiếu chiếm 30 % số clip; FPR đo trên ít real |
| 20 clip / người, cố định | chọn 18/10; thay bằng 15-20 ngày 19/10 |
| **15-20 clip / người** | **chọn (19/10)**, kèm các biện pháp chống lệch bên dưới |
| 20-60 clip ngẫu nhiên | bỏ: người nhiều clip lấn át (identity leakage); AUC gộp bị người nhiều clip chi phối; cân giới / split theo clip lệch; cùng GPU làm được thêm part (thêm người) |
| 10 clip sinh fake + 10 clip chỉ làm real (real_only) | bỏ, vì phương án "mọi clip đều sinh fake, mỗi clip một nửa số generator" cho tỉ lệ tốt hơn mà mọi real đều có fake cặp (xem v003_generation §4) |

Vì sao 20 chứ không 10:
1. Nút thắt là **tìm người** (người Việt, đủ chất lượng, cân giới). Đã có một người thì thêm 10 clip gần như không tốn công thu thập.
2. 20 vẫn xa vùng bão hoà của các bộ lớn (88-139 real mỗi người), nên thêm clip còn lợi rõ: gấp đôi real cho model học từ real; 3 clip tham chiếu không còn chiếm 30 %; FPR đo trên 200 real / part thay vì 100; gấp đôi clip real làm nền cho fake.
3. Khi nào 10 tốt hơn: GPU là nút thắt, hoặc nhiều khách podcast không đủ 20 clip tốt. Quy tắc ưu tiên: **không đổi người lấy clip**; nếu mất nhiều người tốt, có thể hạ chuẩn part xuống 16 clip thay vì bỏ người.

**19/10: đổi sang 15-20 clip mỗi người** (user chốt; part sau cũng cần, làm đồng bộ cho mọi part). Lý do: người tốt nhưng chỉ có 15-19 clip dùng được không bị rơi; với luật cắt mới, clip ngắn hơn nên số cửa sổ không phải nút thắt, nút thắt là clip **tốt** khi duyệt. Một part còn **600-800 clip**.

**Biện pháp để số clip khác nhau không làm lệch kết quả:**

| Rủi ro | Biện pháp | Ở đâu |
|---|---|---|
| Người nhiều clip lấn át khi train (identity leakage) | lấy mẫu train theo **1 / số clip của người đó** → mỗi người đóng góp như nhau; mọi người cùng số clip thì giữ shuffle thường | `training.speaker_balance`, `train.speaker_weights` |
| AUC gộp bị người nhiều clip chi phối | báo thêm **AUC cân theo người** (trọng số 1 / số file của người đó); AUC trong từng người và CI bootstrap theo người vốn có | `evaluate.speaker_balanced_auc`, cột "ai (cân theo người)" trong `report.md` |
| Split lệch số clip (người ít clip dồn vào test) | chia split vẫn ưu tiên tuyệt đối chỉ tiêu người theo giới; trong giới hạn đó **cân tổng clip** mỗi split theo tỉ lệ số người | `parts.assign_splits` |
| Số clip mang thông tin nhãn | không thể: mọi clip của mọi người sinh **cùng số file** (9 / 12), tỉ lệ fake / real mỗi người như nhau | v003_generation §4 |
| Chênh tỉ trọng giữa người | tối đa 20 / 15 = 1,33× (v001 từng 15-158 clip, 10×); dưới 15 không vào part, trên 20 trang duyệt chặn | `min / max_clips_per_speaker` |
| Người duyệt chọn clip kém để đủ số | quy tắc duyệt: chỉ chọn clip tốt; còn clip tốt thì chọn tới 20; số clip không là tiêu chí chọn người | trang duyệt (thông báo "ĐẠT … chọn thêm tới 20") |

**15-20 clip nên đến từ 2-3 video khác nhau** (khác buổi quay, ánh sáng, áo, góc máy). 20 clip cắt liền từ một video gần như là một mẫu lặp.

---

# Pipeline dữ liệu real (chạy ở máy local)

```
mỗi batch:  01 plan → 02 download → 03 cut (+ SyncNet) → 04 identify → 05 review → 08 archive raw N → 09 clean raw N
đủ người:   06 parts → 07 export N (dịch tiếng) → up Kaggle (private) + Drive → 08 archive part N → 09 clean part N
bất cứ lúc: 10 status
```

Mọi bước ghi vào **ledger SQLite** (`data/ledger/ledger.sqlite`): URL, video, lý do loại, batch, clip, độ lệch, speaker, quyết định duyệt, part, nơi lưu trên Drive. Sau mỗi lệnh có ghi (kể cả khi Ctrl+C), ledger tự sao lưu (giữ mọi bản, không tự xoá; từ 20/10, trước đó giữ 30 bản) và xuất CSV. Mất ledger là mất thông tin quản lý, nên ledger được upload Drive mỗi lần archive. Không có gì chạy ngầm: giới hạn batch áp khi chạy `01_plan`, part chỉ tạo khi chạy `06_parts`.

## 4. Chọn và tải video (01 plan, 02 download)

| Luật | Giá trị | Lý do |
|---|---|---|
| Đầu vào | `urls.csv`: `url, speakers (1 / 2 / n), note` | `note` là ghi chú tự do, code không dùng. `2` có từ 20/10 (§5.3) |
| Nguồn 1 người | trải theo ngày đăng, thêm video tới khi **≥ 20 phút**, **3-10 video**; ưu tiên video ≤ 90 phút | ≥ 3 buổi quay cho đa dạng và clip tham chiếu; tối đa 10 chặn trường hợp toàn video rất ngắn. Trước đây đếm số video (tối đa 3): video dài phí raw, video ngắn thiếu clip |
| Nguồn 2 người / nhiều người | 15 video mới nhất (link video lẻ: chính video đó) | `2` và `n` chọn video như nhau, chỉ khác chỉ tiêu cắt |
| Chỉ tab `/videos`, chỉ video **ngang** | video dọc bị loại ở plan (metadata) và sau khi tải (file) | user chốt 17/10. Video dọc dùng được về kỹ thuật, nhưng hay có phụ đề to đè miệng, cắt cảnh 1-3 s, filter làm đẹp |
| Chất lượng | ≥ 720p (cạnh ngắn), **≥ 25 fps**, tiếng Việt gốc (không lồng tiếng AI), không trùng | **24 fps bị loại**: chuẩn hoá lên 25 fps phải lặp 1 frame mỗi giây, môi "khựng" đều đặn, nhiễu tín hiệu khớp tiếng-môi (luật từ v001) |
| Tải | bản 720p, chỉ HTTPS | |
| Batch raw | **≤ 20 giờ và ≤ 16 GB**, giới hạn nào đến trước | ổ máy không chứa hết raw; mỗi lúc chỉ có khoảng một batch. Batch đã bắt đầu tải thì khoá; URL mới vào batch mới. Batch không cần đầy |
| Lỗi tải | lỗi tạm thời (403, 429, 5xx, timeout) thử lại sau 5 / 20 / 60 s, mỗi lần xin link mới; lỗi "xác minh bot" thì **dừng cả batch**, chạy lại với `--cookies-from-browser firefox` | 403 là link tạm bị từ chối, xin link mới thì tải được ●; gọi tiếp khi bị chặn bot chỉ bị chặn lâu hơn. Firefox vì cookie Chrome / Edge trên Windows thường không đọc được |

- Thêm dữ liệu: thêm dòng vào cuối `urls.csv`, chạy lại từ `01_plan`. URL đã plan giữ nguyên; video trùng tự bỏ; không sửa `speakers` của dòng đã chạy; muốn bỏ nguồn thì để nguyên dòng.
- Khi `urls.csv` ghi **từng link video riêng**, mỗi video là một input (thiết kế ban đầu giả định một input là một kênh 3-10 video). Điều này ảnh hưởng chỉ tiêu cắt (§5.3).

## 5. Cắt clip (03 cut)

### 5.1. Luật giữ clip

| Lớp | Điều kiện | Lý do loại |
|---|---|---|
| Tiếng nói | Silero VAD (xác suất ≥ 0,5), nối đoạn cách nhau < 0,6 s; cửa sổ **4-7 s** trên lưới 25 fps, không chồng nhau; đoạn nói dài hơn 7 s được chia **ở chỗ ngắt dài nhất** trong khoảng 4-7 s (§5.4, từ 19/10); đoạn nói < 4 s bỏ | |
| Đúng 1 mặt | YuNet độ tin ≥ 0,85; 0 hoặc ≥ 2 mặt đều loại | `face_frame` |
| Mặt đủ to | cạnh hộp mặt **≥ 48 px** ở khung cạnh dài 640 (≈ 96 px ở 720p) | `face_frame` |
| Hình học mặt | hộp nằm trọn trong khung; hai khoé miệng ở nửa dưới mặt, cách nhau ≥ 12 % bề ngang mặt | `face_frame` |
| Mặt liền mạch | tâm không nhảy quá 0,6 lần cỡ mặt, cỡ không đổi quá 2 lần (so với frame tốt gần nhất) | `face_frame` |
| Hụt mặt ngắn | cho phép **≤ 3 frame hụt, không quá 2 liền nhau**; landmark nội suy qua chỗ hụt; cờ `face_gap` | |
| Không chuyển cảnh | tương quan histogram màu giữa hai frame ≥ 0,55 | `scene_change` |
| Đủ frame | đọc được ≥ 95 % frame | `coverage` |
| Jump cut | một cú giật đơn lẻ của điểm mốc mặt kèm thay đổi ảnh mặt hoặc âm thanh → loại; chỉ một dấu hiệu → giữ, cờ `jump_suspect` | `jump_cut` |
| Cùng người (nguồn nhiều người) | quét mặt sơ bộ 4 fps, ≥ 90 % điểm quét đạt; 5 frame SFace cos ≥ 0,5 với mặt trung bình | `face_scan`, `identity_change` |
| Độ lệch tiếng-hình | §6 | `sync_unreliable` |

**Hai ngưỡng được nới ngày 17/10, có số đo** ● (batch thử 5 video, lấy mẫu 25 cửa sổ mỗi video):

| Thay đổi | Trước → sau | Bằng chứng |
|---|---|---|
| Cỡ mặt tối thiểu | 64 px (≈ 128 px ở 720p) → **48 px** (≈ 96 px, cỡ mặt đầu vào của Wav2Lip) | podcast quay vừa người có mặt ~110-125 px → ra **0 clip**. Ở 48 px frame đạt tăng 12 % → 76 %. Hạ xuống 40 px chỉ cứu ô webcam nhỏ, quá nhỏ để vẽ môi tốt → không hạ |
| Frame hụt | 1 frame hụt là bỏ cả cửa sổ → **≤ 3, không quá 2 liền nhau** | 9/25 cửa sổ podcast chỉ hụt 1-2 frame (nghiêng đầu, mờ chuyển động); cửa sổ đạt tăng 7 → 16/25 |

Cắt lại batch thử ●: **296 clip** (podcast 0 → 241). Những gì vẫn bị loại là loại **đúng**: cảnh minh hoạ, lời dẫn đè hình, quay màn hình với ô webcam nhỏ, vlog đi giữa đám đông. Lời khuyên chọn nguồn: kênh **mặt nhìn camera, quay vừa người hoặc cận**.

### 5.2. Những gì bước cắt không kiểm

| Không kiểm | Xử lý |
|---|---|
| **Góc quay mặt**: luật hình học chỉ chặn mặt gần nghiêng hẳn (~60-70° trở lên); 40-50° vẫn lọt | user tự chọn khi duyệt, lệch ≤ ~30 % (chốt 18/10, không thêm code đo góc) |
| Người trong hình có phải người đang nói | bắt gián tiếp ở mức video bằng SyncNet (§6) |
| Nhạc nền to, chữ đè miệng, tay che miệng, filter làm đẹp, hình mờ | người duyệt loại |

Chủ ý từ v001: **không chọn từng clip theo điểm sync**, vì như vậy các detector dựa vào sync trông mạnh giả tạo.

### 5.3. Chỉ tiêu cắt nguồn 1 người và cách rải clip

| Thời điểm | Chỉ tiêu | Lý do |
|---|---|---|
| đến 18/10 sáng | 30-50 clip / người | thiết kế 10 clip / người |
| 18/10 | 40-60 | lên 20 clip / người |
| **18/10 tối (hiện hành)** | **30-40 mỗi input**: 30 khi ≤ 5 phút tiếng nói, tăng tuyến tính tới 40 ở ≥ 20 phút | 9 video đầu batch 1 cho thấy clip qua bộ lọc đã sạch ●; URL nhập theo từng video nên chỉ tiêu tính cho từng video: người có 3-4 video sẽ có 160-240 ứng viên cho 20 clip. 30 vẫn dư 10 cho phần loại khi duyệt |

- Chỉ tiêu là **mức tối đa** cố lấy, không bắt buộc. Thiếu thì giữ những gì cắt được. Chỗ bắt buộc là duyệt: đủ 15 clip mới vào part.
- Thời lượng nguồn **20 phút là đủ**: nguồn tốt giữ 60-100 % cửa sổ kiểm ●, nên 40-60 clip chỉ cần ~6-9 phút tiếng nói; video 20 phút có ~14-16 phút tiếng nói. (Có lúc tính nhầm cần 30 phút, đã sửa.)
- **Rải đều** (`spread_order`): trục thời gian chia N khúc (N = chỉ tiêu), mỗi khúc lấy cửa sổ gần tâm nhất. Khúc hỏng thì thử cửa sổ kế trong khúc; khúc hết cửa sổ thì bỏ; khi mọi khúc còn lại đã có clip thì lấy clip thứ 2, 3 từ khúc còn cửa sổ tới đủ chỉ tiêu, nên một khúc không có clip tốt **không làm hụt** chỉ tiêu. Trần số lần thử = 2,5 × chỉ tiêu + 5.
- Rải đều **không làm chậm**: mỗi cửa sổ được ffmpeg nhảy thẳng tới (`-ss`) và giải mã riêng; thời gian chỉ phụ thuộc số cửa sổ phải kiểm. Khi duyệt, clip khác thời điểm nên dễ chọn hơn.

**Nguồn hai người (`speakers = 2`, từ batch 3, 20/10).** Nguồn nhiều người (`n`) cắt mọi cửa sổ đạt. Đo trên 7 video `n` đầu của batch 2 ●:

| Video | Dài | Clip / cửa sổ kiểm | Thời gian cắt |
|---|---|---|---|
| Video 1 | 81 phút | 362 / 370 | 32,5 phút |
| Video 2 | 64 phút | 432 / 479 | 34,6 phút |
| Video 3 | 56 phút | 321 / 333 | 28,9 phút |
| Video 4 | 57 phút | 299 / 321 | 27,4 phút |
| Video 5 | 58 phút | 343 / 360 | 29,7 phút |
| Video 6 | 33 phút | 146 / 151 | 14,3 phút |
| Video 7 | 26 phút | 61 / 138 | 10,6 phút |

Trung bình ~280 clip / video (~5 s mỗi cửa sổ kiểm), trong khi 2 người chỉ cần ~2 × 20 clip được chọn. Luật mới:
- `2` = phỏng vấn 1-1, podcast 2 người. Kiểm giống `n` (quét mặt 4 fps + chuyển cảnh, 5 điểm cùng một mặt), nhưng chỉ lấy **tối đa 150 clip / video**, rải đều bằng `spread_order` (`cut.pair.max_clips`, trần thử 2,5 × 150 + 5). Video ngắn không đủ 150 thì ra như cắt hết.
- 150 ≈ 75 clip / người, dư ~4 lần so với 15-20 clip được chọn (nguồn 1 người dư ~2 lần với 30-40). Dư hơn vì clip 2 người dễ bị loại khi duyệt (cắt sang người kia, nói chồng). Rủi ro: camera chia không đều (khách lên hình nhiều hơn MC); MC thường lặp lại qua nhiều tập cùng kênh nên identify cộng dồn clip.
- `n` giữ cắt hết (talkshow đông người, không biết trước số người).
- Không đổi dữ liệu cuối: mỗi người vẫn 15-20 clip được chọn, train vẫn cân theo người; chỉ đổi số ứng viên khi duyệt.
- Áp từ batch 3: batch 1-2 giữ nguyên (nguồn nhiều người của batch 2 cắt hết). User đánh dấu 28 link batch 3-4 là `2` (giữ 4 link `n`); ledger đổi nhãn đúng 28 input đó, chưa cắt, có ghi sự kiện. Ước thời gian cắt batch 3-4 giảm từ ~16 giờ còn ~8-9 giờ ○.
- Mỗi video ghi luật đã dùng vào `cut_info` (`mode` = single / pair / multi, `quota`).

**Kiểm chứng giới hạn 150 (S02, 22/10)** ●. Mô phỏng trên 18 video nguồn nhiều người đã cắt hết (6.112 clip; clip / video p10 / p50 / p90 = 61 / 361 / 475): lấy các clip đã giữ làm cửa sổ, chọn N clip bằng đúng thuật toán rải đều của pipeline, rồi đếm ứng viên còn lại của từng cặp (người, video) theo speaker đã gán ở bước identify. Gần đúng ở chỗ luật thật rải trên cửa sổ trước khi kiểm (tỉ lệ giữ của nguồn nhiều người ~94 %), ở đây rải trên clip đã giữ.

| Giới hạn | Clip còn lại | Cặp (người duyệt xong, video) còn ≥ 20 ứng viên | Ứng viên / cặp p10 / p50 / p90 |
|---|---|---|---|
| không (cắt hết) | 6.112 (100 %) | - | - |
| 100 | 1.684 (27,6 %) | 32 / 36 | 18 / 37 / 80 |
| **150** | **2.480 (40,6 %)** | **34 / 36** | 32 / 56 / 114 |
| 200 | 3.230 (52,8 %) | 35 / 36 | 46 / 69 / 154 |

- 150 giữ ~41 % số clip mà 34/36 cặp vẫn đủ 20 ứng viên; hai cặp hụt vẫn có thể đủ nhờ video khác của cùng người.
- Kết quả thật của luật `2` (29 video): clip / video p10 / p50 / p90 = 85 / 150 / 150, 25 video chạm 150, ~22 phút cắt / video (trung vị); 52 cặp (người duyệt xong, video) có ứng viên 27 / 70 / 121.

### 5.4. Điểm cắt ở chỗ ngắt (từ 19/10)

**Vấn đề** ●: luật cũ nối chỗ lặng < 0,6 s rồi chặt đoạn nói dài **cứng mỗi 7 s**. Trên batch 1 (84 video, 2.392 clip): 84 % clip dài đúng 7 s; chỉ **8 %** clip có cả 2 mép rơi vào chỗ ngắt, **61 %** có cả 2 mép bên trong đoạn nói (có thể cắt ngang từ). 38 % đoạn nói sau VAD dài hơn 7 s. Không ảnh hưởng phát hiện (fake cùng mép với real) nhưng làm track A (TTS đọc lại cả câu) gặp từ cụt, và real gần như không có khoảng lặng đầu clip.

**Luật mới:** VAD vốn tính xác suất có tiếng cho từng khúc 32 ms. Lưu thêm các **chỗ ngắt** (khúc xác suất < 0,5 nằm trong đoạn nói, `pause_spans`). Khi một đoạn dài hơn 7 s, điểm cuối mỗi cửa sổ là **tâm chỗ ngắt dài nhất** trong khoảng [đầu + 4 s, đầu + 7 s] (phần còn lại phải đủ ≥ 4 s; không đủ chỗ cho 2 cửa sổ thì lấy 1, bỏ đuôi), làm tròn theo lưới 25 fps; không có chỗ ngắt thì cắt ở mức dài nhất cho phép như cũ. Cửa sổ kế bắt đầu từ điểm vừa cắt, không chồng nhau.

**Thử trước khi áp** ● (6 video batch 1, 6 kênh, ~1,8 giờ tiếng nói; mép "sạch" = mọi khúc 32 ms quanh mép có xác suất < 0,5):

| | Luật cũ | Luật mới (bản thử) | Luật mới (code thật) |
|---|---|---|---|
| Cửa sổ | 930 | 1.187 | 1.176 |
| Cả 2 mép sạch | 74 (8 %) | 745 (63 %) | 723 (61 %) |
| Cả 2 mép giữa tiếng | 533 (57 %) | 77 (6 %) | 86 (7 %) |
| Độ dài | 87 % đúng 7 s | p10 / p50 / p90 = 4,2 / 5,4 / 6,6 s | 4,2 / 5,4 / 6,7 s |
| Tổng thời lượng phủ | 6.305 s | 6.398 s | |

- Phần còn lại chưa sạch là người **nói một mạch, không ngắt suốt 4-7 s**. Nới chỗ ngắt từ ≥ 96 ms xuống ≥ 64 ms không đổi tỉ lệ sạch (63 %).
- Chi phí thời gian gần như bằng 0 (xác suất VAD đã tính sẵn). Clip ngắn hơn (trung vị 5,4 s) → bước sinh tốn ít GPU hơn một chút; vẫn đủ chỗ cho đoạn partial 0,3-3 s.
- Giới hạn: chỗ ngắt chưa chắc là hết câu (có thể giữa hai cụm); VAD có thể tụt ở phụ âm vô thanh. Cắt **tròn câu** cần ASR có mốc từng từ + dấu câu (PhoWhisper / Whisper / WhisperX), nặng hơn nhiều trên CPU → không làm.
- Tham khảo: code nhóm Capstone dùng Silero mặc định (ngắt ≥ 100 ms, đệm 30 ms), giữ nguyên đoạn 2-12 s làm một clip, đoạn dài hơn cắt cửa sổ 5 s bước 4 s (chồng 1 s).
- Áp dụng → đổi luật cắt → user xoá toàn bộ dữ liệu batch 1 (đã duyệt 38 người) và tải, cắt lại từ đầu (19/10).

**Đo lại trên toàn bộ dữ liệu (S01, 22/10)** ●. Với mỗi video đã cắt (251), lấy lại đúng đoạn nói và chỗ ngắt VAD đã lưu trong `cut_info`, chạy hàm chia cửa sổ hai lần: không truyền chỗ ngắt (= luật cũ, cắt ở mức dài nhất 7 s) và có chỗ ngắt (= luật mới); bỏ qua ranh giới chuyển cảnh cho cả hai (như nhau). Hai thước đo mép:
- **A. Sạch theo VAD** (mọi video): mép trùng đầu / cuối đoạn nói (±0,04 s; đầu / cuối đoạn đã đệm 0,15 s lặng) hoặc nằm trong một chỗ ngắt. Gần với điều luật mới tối ưu nên chỉ **mô tả**, không là bằng chứng độc lập.
- **B. Lặng theo năng lượng** (độc lập với VAD): năng lượng RMS của khúc 32 ms chứa mép và hai khúc kề (±1 khúc, cùng lưới 16 kHz) đều thấp hơn **trung vị năng lượng các khúc có tiếng của chính video đó** ít nhất 20 dB (hoặc 15 dB). Chạy trên 30 video raw còn ở máy chọn ngẫu nhiên (seed 0), 10 video mỗi loại nguồn `1` / `2` / `n`.

| | Chặt cứng 7 s (cũ) | Cắt ở chỗ ngắt (mới) |
|---|---|---|
| Cửa sổ | 55.957 | 64.145 |
| Giờ tiếng nói được phủ | 98,1 | 97,1 |
| Dài đúng 7 s | 65,7 % | 4,8 % |
| Độ dài p10 / p50 / p90 (s) | 4,04 / 7,0 / 7,0 | 4,2 / 5,4 / 6,8 |
| A. Cả 2 mép sạch theo VAD (mô tả) | 15,0 % | 87,1 % |
| A. Không mép nào sạch | 43,5 % | 0,5 % |
| **B. Cả 2 mép lặng, −20 dB (30 video)** | **10,2 %** | **41,4 %** |
| B. Cả 2 mép lặng, −15 dB (30 video) | 12,2 % | 50,3 % |

- Bằng chứng độc lập (B): tỉ lệ clip có cả 2 mép lặng tăng ~4 lần, tổng tiếng nói phủ gần như không đổi.
- Số thử 19/10 (8 % → 61 %) dùng định nghĩa chặt hơn (mọi khúc quanh mép xác suất VAD < 0,5) trên 6 video: không so trực tiếp được với bảng trên.
- **Gợi ý thí nghiệm** (chưa làm): (1) **S05 - mép rơi giữa từ**: ASR có mốc thời gian từng từ (PhoWhisper / WhisperX) trên ~20 video raw, đo % mép nằm bên trong một từ cho hai luật; là bằng chứng độc lập thứ hai, nên chạy trên GPU (Kaggle). (2) Mở rộng B lên ~90 video (≈ 30 phút CPU). Cả hai cần video raw, nên làm **trước khi** `09_clean raw` (raw đã zip lên Drive thì vẫn lấy lại được).

### 5.5. Tốc độ, chạy tuần tự, khoá luật cắt

| Số đo ● | Giá trị |
|---|---|
| Batch thử (2,6 giờ video, có SyncNet) | ~45 phút; podcast 84 phút (nguồn nhiều người, quét mặt cả video 4 fps) chiếm 37 phút |
| SyncNet | ~4,5 s mỗi clip đo, ~1-2 phút mỗi video |
| VAD trước khi cắt | batch 1: 914 phút audio trong ~7,5 phút; không in tiến độ nên trông như đứng |
| Batch 1, 9 video đầu (chỉ tiêu 40-60) | 123,5 phút video, 345 clip, 2.325 s → ~258 s / video, ~6,7 s / clip lấy được |
| RAM | một tiến trình cao nhất ~0,75-1,05 GB (lúc SyncNet), phần lớn thời gian ~0,3 GB |
| Cắt song song 3 video | 459 s so với 593 s tuần tự: chỉ nhanh **1,3×** (bản tuần tự vốn đa luồng) |

- **Giữ cắt tuần tự** (user chốt 17/10). Ước tính với chỉ tiêu 30-40: batch 20 giờ video mất ~4,5-5,5 giờ cắt ○.
- **Dừng giữa chừng:** một ledger duy nhất; mỗi video ghi một lần khi xong; video dở không ghi gì và file dở bị xoá; chạy lại thì làm tiếp từ video dở, VAD đã có không dò lại.
- **Luật cắt bị khoá:** chữ ký (config cắt + mã nguồn `cut.py`, `jumpcut.py`, `vad.py`, `face.py`, `syncnet.py` + model) ghi vào ledger. Đổi luật sau khi đã có clip thì lệnh từ chối chạy, để mọi clip theo cùng một luật. Thử luật mới dùng workspace riêng (`VNAV_DATA=data_trial`). Sửa code không đổi luật thì đóng dấu lại chữ ký.
- **Từ 20/10 chữ ký chỉ khoá cách làm ra một clip.** Luật số lượng (config `single`, `pair` và `quota.py`) nằm ngoài chữ ký: đổi chỉ tiêu không làm khác chất lượng clip đã cắt, nên không chặn batch cũ. Đóng dấu lại một lần khi tách (ledger ghi sự kiện `pair`: `b90ec5a90295 → 6ae7ba9ede0a`; luật làm clip không đổi).

## 6. Độ lệch tiếng-hình sẵn có của real

### 6.1. Vấn đề

v001 ghi "lệch A/V sẵn có của real chưa đo" nhưng chưa xử lý. Generator lip-sync luôn vẽ môi khớp tiếng ở độ lệch 0 (theo "điểm 0" của SyncNet chúng dùng khi train). Real của một kênh lệch d frame thì có độ lệch d. Hệ quả:
1. **Lối tắt "khớp quá mức":** ④ source lệch 0, real cùng clip lệch d; partial có đoạn lệch 0 nằm giữa hai đoạn lệch d. Detector có thể học "khớp hoàn hảo → fake", tức học đặc điểm của **kênh**. Báo nhầm sẽ dao động theo người.
2. **Nhãn mismatch nhiễu:** real lệch vài frame vẫn mang nhãn "không lệch".
3. **Ảnh hưởng không đều:** AVH-Align coi độ lệch 0 là cặp đúng khi train nên nhạy nhất; SpeechForensics / A1 dò độ lệch ±15 nên bền với độ lệch cố định nhưng có thể dùng "lệch 0, đỉnh nhọn" làm lối tắt; ref-norm khử phần lớn độ lệch cố định theo người.

Hai kết quả cũ **có thể** liên quan (giả thuyết, chưa kiểm): v001 Wav2Lip giữ tiếng gốc khớp tiếng hơn cả real (nhánh sync AUC 0,436 < 0,5); v002 speaker 01 có điểm real cao bất thường (0,73).

Mục tiêu: **real và fake có cùng phân bố độ lệch**. Bản thân độ lệch không có hại; nó có hại khi đi kèm nhãn hoặc đi kèm người.

| Cách | Ưu | Nhược |
|---|---|---|
| Lọc video lệch ≥ N frame | đơn giản | mất dữ liệu; lệch 1-2 frame vẫn lọt; phần giữ lại thiên về kênh "sạch" |
| **Sửa theo video** (đo độ lệch cố định của video, dịch tiếng về 0) | giữ dữ liệu; real về 0 như fake; LRS2 / LRS3 / VoxCeleb làm vậy với SyncNet | phụ thuộc độ chính xác bộ đo (±1 frame) |
| Sinh fake mang cùng độ lệch với real | không đụng real | phức tạp; real vẫn lệch, nhãn mismatch vẫn nhiễu |

### 6.2. Đo (17/10 và 18/10)

- Bộ đo: SyncNet (Chung & Zisserman 2016, code `syncnet_python`, MIT), crop mặt như `run_pipeline.py`, CPU, độ lệch −15…+15 frame.
- **Phép dịch biết trước** ●: dịch tiếng 30 clip k ∈ {−6, −3, −1, +1, +3, +6} frame → đo lại đúng k ở **180/180** lần (độ phân giải 1 frame; chỉ kiểm phần chênh, không kiểm "điểm 0" tuyệt đối).
- **Theo video, batch thử** ● (95 clip, 5 video):

| Video | Loại | Độ lệch cả video | Clip đồng ý ±1 frame | Độ nhọn đỉnh |
|---|---|---|---|---|
| Video A (podcast) | nhiều người | tiếng trễ 2 frame | 40/40 | 5,9 |
| Video B | 1 người | trễ 1 | 12/12 | 5,9 |
| Video C | 1 người | trễ 1 | 14/14 | 5,2 |
| Video D | 1 người (lời dẫn đè cảnh) | - | 8/22 | 1,6 |
| Video E | 1 người (vlog, ghép cảnh) | - | 2/7 | 1,9 |

- **Batch 1 thật, 9 video đầu** ●: 7 video `ok`, tiếng trễ 1-3 frame; 2/9 `unreliable` (98 clip tự loại).
- Kết luận: video quay mặt người nói lệch **ổn định** 1-2 frame (tiếng trễ 40-80 ms, dưới ngưỡng mắt thường). Video lời dẫn đè hình cho độ lệch lung tung → bộ đo đồng thời **phát hiện mặt trên hình không phải người nói** ở mức video (lỗ hổng v001 "chưa kiểm mặt trong clip đúng người nói").

### 6.3. Chọn bộ đo

| Model | LRS2, 5 frame | LRS2, 15 frame | Weight công khai |
|---|---|---|---|
| SyncNet (2016) | 75,8 % | 96,1 % | có |
| Perfect Match (2019) | 88,1 | 99,1 | ? |
| VocaLiST (2022) | 92,8 | 99,8 | không lấy được (theo PVSync) |
| ModEFormer (2023) | 94,5 | 99,8 | không thấy |
| IC-SyncNet (2024) | 96,5 | 99,9 | không thấy |
| StableSyncNet (LatentSync) | 94 % HDTF (tác giả); 90,6 % ở cửa sổ 16 frame theo PVSync | | có |
| PVSync (10/2026) | 95,1 % (16 frame, bộ test riêng) | | chưa phát hành |

(Số theo IC-SyncNet Bảng 1 và PVSync; độ chính xác = đoán đúng ±1 frame ✔.)

- SyncNet kém ở cửa sổ ngắn nhưng ngang các model mới khi gộp đủ dài; ta gộp 15-20 clip × 100-175 frame mỗi video → **đủ dùng**, khớp số đo 180/180.
- **Không dùng StableSyncNet làm bộ đo chính**: thuộc LatentSync (generator chỉ ở test); sửa real theo "điểm 0" của nó có thể làm real giống fake LatentSync hơn.
- **Không dùng AV-HuBERT ở local**: fairseq bản cũ (Python 3.10, biên dịch extension), script cài chỉ cho Linux, bản Large ~300 triệu tham số nặng với máy CPU 7,8 GB RAM. AV-HuBERT làm **ý kiến thứ hai** ở bước prepare trên Kaggle (v003_model §9), không chặn pipeline.
- **Không hạ Python xuống 3.10**: không giúp AV-HuBERT (fairseq cần torch 2.0, phần còn lại cần torch ≥ 2.6); yt-dlp sắp bỏ 3.10 (hết hỗ trợ 10/2026). AV-HuBERT vẫn ở worker riêng.
- **Đo trong bước cắt, không đo trước:** đo cần đoạn "có tiếng + đúng 1 mặt rõ", chính là việc bước cắt làm; đo trên raw gặp cảnh minh hoạ, quay màn hình, chuyển cảnh và cho kết quả rác.

### 6.4. Quyết định

1. **Đo trong `03_cut`**: sau mỗi video, SyncNet trên tối đa 20 clip rải đều; độ lệch video = đỉnh đường cong trung bình. Trạng thái:
   - `ok`: đỉnh ≥ 3,0, ≥ 80 % clip đồng ý ±1 frame, ≥ 3 clip;
   - `few`: đạt nhưng < 3 clip → vẫn sửa, cờ `sync_few` (thêm vì test cho thấy video ra 2 clip bị loại oan);
   - `unreliable`: còn lại → mọi clip tự `reject`, cờ `sync_unreliable`; người duyệt vẫn chọn lại được (khi đó tiếng không được sửa, manifest ghi `av_sync_status: unreliable`).
   Ngưỡng 3,0 đặt giữa nhóm tin được (5,2-5,9) và nhóm hỏng (1,6-1,9) ●.
2. **Sửa khi export:** dịch tiếng mọi clip theo độ lệch của video về 0 (copy luồng hình, encode lại tiếng, đệm lặng ≤ vài frame ở mép); manifest ghi `audio_lag_frames`, `av_sync_status`. Clip trong pool giữ bản gốc, nên đổi cách đo về sau không phải cắt lại. Thử trên 8 clip thật ●: dịch đúng số frame, sau sửa còn trong ±1 frame quanh 0.
3. **Lệch lớn nhưng ổn định thì sửa, không bỏ.** Chỉ bỏ video **không đo được**.
4. **Không trái nguyên tắc v001**: v001 tránh chọn **từng clip** theo điểm sync. Ở đây chỉ dịch một **hằng số cho cả video**; clip khớp kém tự nhiên vẫn ở lại.
5. **Augment khi train: dịch tiếng so với hình k ∈ [−1, +2] frame** (dương = tiếng trễ), đều cho real và fake, nhãn giữ nguyên (cách áp dụng: v003_model §7). Căn cứ:
   - **ITU-R BT.1359** (thử nghiệm chủ quan trên clip người đọc tin): người bắt đầu nhận ra khi tiếng **sớm ~45 ms** hoặc **trễ ~125 ms**; chấp nhận được tới khoảng +90 / −185 ms. Không đối xứng: tiếng trễ được chịu đựng gấp ~3 lần. ATSC còn chặt hơn cho phát sóng (sớm ≤ 15 ms, trễ ≤ 45 ms). −1…+2 frame (sớm 40 ms … trễ 80 ms) nằm trong vùng chưa nhận ra của BT.1359;
   - **"Điểm 0" giữa các công cụ lệch ~1 frame**: ModEFormer thấy LRS2 lệch ~1 frame so với LRS3 / VoxCeleb2 dù đều đã sửa bằng SyncNet; Wav2Lip train trên LRS2, LatentSync theo StableSyncNet → fake của các generator có thể lệch nhau ±1 frame;
   - **sai số đo**: độ phân giải 1 frame → real sau sửa còn sai ±0,5-1 frame;
   - **augment lớn có hại**: DFD-Lab augment lệch tới ±200 ms làm giảm hiệu năng ngoài miền ở cả 3 model; PVSync coi clip lệch > 3 frame so với mode của bộ dữ liệu là dữ liệu bẩn.
   - User yêu cầu con số phải có căn cứ nghiên cứu, không chọn theo ý người làm. Thí nghiệm chọn biên độ (không / ±1 / −1…+2 / ±3, đo AUC + báo nhầm khi real bị dịch k = −3…+3) **để sau**.
6. **Không nghiên cứu lệch thời gian thuần** (cùng tiếng, dịch đi, "sham_shift") trong v003 (user chốt 17/10). ② sham vẫn là thay tiếng bằng đoạn nói khác. (Phương án đã nêu nếu muốn làm: |lệch| ≤ 2 frame là vùng khớp; 3-4 frame bỏ; ≥ 5 frame (≥ 200 ms, vượt ngưỡng chấp nhận BT.1359) gán nhãn lệch.)
7. **Demo**: đo độ lệch toàn video, dịch về 0 rồi mới chấm "AI?"; lệch vượt vùng chấp nhận thì báo riêng "lệch thời gian ≈ X ms".

### 6.5. Số đo trên toàn bộ dữ liệu (S03, 22/10)

Đọc `cut_info.sync` của mọi video đã cắt (đo lúc cắt, tối đa 20 clip / video) ●:

| Độ lệch (frame 25 fps, dương = tiếng trễ) | −4 | 0 | 1 | 2 | 3 | 4 | 7 |
|---|---|---|---|---|---|---|---|
| Số video (`ok` + `few`, 246) | 1 | 5 | 50 | 86 | 73 | 30 | 1 |

- Trạng thái: `ok` 245 · `few` 1 · `unreliable` 4 trên 250 video đo được (1 video không còn clip nào để đo).
- Tiếng trễ ở **97,6 %** video; ms p10 / p50 / p90 = 40 / **80** / 160; **13,0 %** video lệch > 125 ms (ngưỡng bắt đầu nhận ra khi tiếng trễ theo BT.1359, qua nguồn thứ cấp ◐). Hai video ngoại lệ (−4 và +7 frame) vẫn được sửa vì đo tin cậy.
- `conf` p10 / p50 / p90 = 4,76 / 6,16 / 7,00 (ngưỡng 3,0); `agree` = 0,90 / 1,00 / 1,00 (ngưỡng 0,8).
- **Clip so với cả video** (4.715 lần đo trên từng clip): trùng đúng độ lệch video 76,2 %, chênh 1 frame 21,5 %, 2-3 frame 1,1 %, > 3 frame 1,2 % → ủng hộ sửa bằng một hằng số cho cả video.
- **Độ nhạy ngưỡng** (số video `unreliable`):

| `min_conf` \ `min_agree` | 0,7 | 0,8 | 0,9 |
|---|---|---|---|
| 2,5 | 1 | 4 | 17 |
| 3,0 (hiện hành) | 1 | **4** | 17 |
| 3,5 | 1 | 4 | 17 |
| 4,0 | 4 | 6 | 18 |

  `conf` nằm trong vùng ổn định (2,5-3,5 cho cùng kết quả); `agree` là thông số nhạy (0,9 → 17 video). Bốn video `unreliable` hiện hành: (conf, agree, số clip đo) = (4,56; 0,75; 20), (3,84; 0,73; 11), (4,12; 0,75; 4), (2,83; 0,50; 20) - ba video đầu chỉ thiếu ít so với 0,8.
- 4 clip `unreliable` được chọn lại nằm trong part 1 (cùng một người, video có 15/20 clip đo trễ 2-3 frame, 5 clip ±15 frame ở mép khoảng tìm) → export không dịch tiếng, nhiều khả năng còn trễ ~3 frame; user quyết định giữ (20/10).
- **Gợi ý thí nghiệm** (chưa làm): **S06 - kiểm chéo "điểm 0"**: đo lại độ lệch bằng một mô hình đồng bộ khác (ví dụ AV-HuBERT) trên cùng clip, so với SyncNet; nếu lệch hệ thống ~1 frame thì ghi thành giới hạn.

## 7. Gán người, duyệt, tạo part

### 7.1. Gán speaker (04 identify, máy làm, trước khi duyệt)

- Nguồn 1 người: mọi clip mặc định cùng người; clip lệch mặt / giọng chỉ gắn cờ `face_outlier` / `voice_outlier`.
- Nguồn nhiều người: gom cụm theo mặt (SFace), mỗi cụm một người; cụm < 3 clip không khớp ai thì để chưa gán.
- So với người đã có (ảnh trong `registry/`): trong kênh trước, rồi toàn bộ. Rất giống → gán; hơi giống → speaker mới kèm **gợi ý gộp**; giọng (WeSpeaker) mâu thuẫn thì không tự gộp.
- Ngưỡng ban đầu là ước lượng (SFace cos 0,55 / 0,35); chạy `--calibrate` sau vài chục người để chỉnh theo quyết định gộp / tách thật.
- Batch thử ●: nhận ra **cùng một người qua 2 nguồn khác nhau**; podcast tách đúng 2 người (44 và 197 clip).

### 7.2. Duyệt (05 review, người làm)

- Mỗi speaker: kiểm cụm đúng một người (gộp **G** / "Khác người" / **R** chuyển clip), **gán giới M / F**, **chọn 15-20 clip** (**S**, **X** loại; đạt từ 15, tối đa 20), **N** chốt, **E** loại người. Quyết định ghi ngay vào ledger; server chạy tới khi Ctrl+C.
- **Giới tính đánh tay**: đoán từ mặt / giọng dễ sai, mà part cần đúng 20 + 20; người duyệt đằng nào cũng xem từng người.
- Chỉ clip `keep` đi tiếp. `pending` (chưa xem) và `reject` (đã xem, hỏng) được các bước sau xử lý như nhau; chọn đủ 20 rồi N là xong.
- Video `sync_unreliable`: không cần đánh gì; speaker chỉ có clip từ video đó thì E.
- **Người chưa đủ 15 clip** (19/10): bấm N là để **chờ**, không đánh dấu xong. Batch sau `04_identify` khớp thêm clip vào người đó thì người đó hiện lại trong "chưa duyệt"; người đã duyệt xong mà còn thiếu clip cũng được mở lại như vậy (người đủ 20 thì giữ nguyên). Người không bao giờ dùng được thì E.
- **Góc mặt**: chỉ lấy clip lệch ≤ ~30 %. Mặt nghiêng tới ~45° vẫn dùng được về kỹ thuật (đúng thực tế talkshow; generator, AV-HuBERT, SyncNet làm được); 45-60° rủi ro (VideoReTalking, MuseTalk dễ lỗi); > 60° dò mặt hụt, generator lỗi.
- Rủi ro đã ghi ○: mặt càng nghiêng generator càng hay lỗi và job lỗi bị bỏ, nên real có thể nhiều mặt nghiêng hơn fake → lối tắt "nghiêng → real". Kiểm sau khi sinh: xem các job bị bỏ trong `records/`.
- Nguồn nhiều người (podcast): cả 15-20 clip có thể cùng một tập, nên để ý chọn từ nhiều tập.
- "Done" chưa phải khoá: người đã duyệt xong vẫn sửa được (chọn thêm tới 20) cho tới khi `06_parts` đưa vào part. Muốn đưa về chờ: gọi API trang duyệt `POST /api/speaker/<id>` với `{"review": "pending"}`.
- Nút "Khác người" chỉ bỏ **gợi ý gộp**; người đó vẫn là speaker riêng, vẫn có clip, vẫn pending nếu chưa duyệt.

**Cờ cảnh báo so với quyết định duyệt (S04, 22/10)** ●, trên 13.809 clip của 120 người đã duyệt xong (keep 2.400 · reject 708 · chưa xem 10.701):

| Cờ | Clip | Được chọn | Tỉ lệ chọn |
|---|---|---|---|
| tất cả | 13.809 | 2.400 | 17,4 % |
| không cờ | 10.424 | 1.812 | 17,4 % |
| `jump_suspect` | 3.046 | 567 | 18,6 % |
| `face_gap` | 341 | 24 | 7,0 % |
| `face_outlier` | 44 | 0 | 0 % |
| `voice_outlier` | 41 | 2 | 4,9 % |
| `sync_unreliable` | 33 | 4 | 12,1 % |

- Tỉ lệ thô: người duyệt dừng khi đủ 20 nên clip không chọn gồm cả clip chưa xem.
- Cờ về mặt / giọng khớp với việc người duyệt bỏ clip; `jump_suspect` có tỉ lệ chọn ngang clip không cờ và xuất hiện ở ~24 % clip được chọn (571 / 2.429) → ở mức hiện tại gần như không mang thông tin cho người duyệt (bộ dò quá nhạy, hoặc người duyệt không thấy jump cut nhỏ).
- Người duyệt không gộp / chuyển clip lần nào (0 merge, 0 move) dù có 92 gợi ý gộp; 3.574 thao tác duyệt ghi trong ledger tới 21/10 (chọn 2.515 · loại 722 · bỏ chọn 64 · cập nhật người 273).
- **Gợi ý thí nghiệm** (chưa làm): **S07** - gắn nhãn tay ~200 clip (100 có, 100 không `jump_suspect`) để đo precision / recall bộ dò jump cut (~2 giờ xem). **S08** - người thứ hai duyệt lại ~10 người, tính Cohen's kappa cho quyết định chọn clip và giới tính (đo độ tin cậy khâu duyệt một người).

### 7.3. Tạo part (06 parts)

- Part tạo khi có **đủ 20 nam + 20 nữ** đã duyệt xong, có 15-20 clip được chọn, theo thứ tự duyệt xong; đủ cho mấy part thì tạo bấy nhiêu. Chưa đủ thì không tạo gì, không báo lỗi; người đã duyệt chờ part sau.
- Split **theo người**, cân giới: 12+12 / 3+3 / 5+5; trong giới hạn đó cân tổng clip giữa các split. Người xuất hiện chung một video đi cùng nhau vào một part và một split (nhóm liên thông). Từ 22/10, người chưa duyệt xong (pending) không giữ nhóm lại: người đã duyệt xong trong nhóm vẫn vào part; nếu người pending sau này duyệt xong, nhóm của họ đã có người trong part cũ nên vào `blocked_groups` (không tự xếp, xử lý tay), tránh cảnh quay rơi vào split khác. Nhóm lớn (quanh một MC xuất hiện ở nhiều tập) phải vào nguyên một split nên có thể làm split lệch nhẹ khỏi 12 / 3 / 5 (part 3 dự kiến: validation 2 nam, test 6 nam).
- Part tạo xong bị **khoá** (`split-lock.json`).
- Nhóm liên thông có thể rất lớn: một MC xuất hiện ở 13 video nối mọi khách mời của chương trình thành một nhóm (part 3: 3 nhóm 15 người = 10 nam + 5 nữ, 10 người = 6 nam + 4 nữ, 5 người = 1 nam + 4 nữ). Nhóm phải vào nguyên một split, nên part 3 lệch nhẹ: validation 2 nam + 3 nữ (100 clip), test 6 nam + 5 nữ (220 clip). Nếu nhóm vượt 12 người một giới thì không vừa split nào → khi đó cân nhắc loại MC để tách nhóm.
- Trang duyệt đếm "sẵn cho part" theo người, **không** tính luật nhóm, nên có thể báo 20/20 trong khi `06_parts` chưa tạo được part.
- Nên nhập URL **dư 25-30 %** (~25-26 người mỗi giới) và cân nam / nữ ngay từ danh sách URL, vì người rơi rụng (không đủ 15 clip, sync hỏng, mặt nghiêng, trùng). Xem thiếu giới nào bằng `10_status`.
- Muốn thử bước sinh trước khi đủ part: làm part nhỏ trên workspace riêng, không gộp vào bộ chính.

## 8. Đầu ra part và lưu trữ

### 8.1. Một clip real hoàn chỉnh

Đầu ra cuối cùng của pipeline là một clip real **hoàn chỉnh**: đã qua mọi bước dưới đây, dùng ngay được, không cần xử lý thêm.

| Thuộc tính | Giá trị / điều kiện | Bước bảo đảm |
|---|---|---|
| Nguồn | video YouTube ngang, ≥ 720p, ≥ 25 fps, tiếng Việt gốc, không trùng | 01 plan, 02 download |
| Độ dài | 4-7 s, nằm trong một đoạn nói, mép ở chỗ ngắt khi có | 03 cut (VAD) |
| Hình | đúng một mặt người đang nói ở mọi frame (hụt ≤ 3, ≤ 2 liền nhau), mặt ≥ 48 px ở khung 640, không chuyển cảnh, không jump cut chắc chắn; nguồn `2` / `n`: 5 điểm cùng một người | 03 cut |
| Định dạng | MP4, 25 fps CFR, cạnh ngắn ≤ 720, H.264 CRF 18, AAC 48 kHz | 03 cut, 07 export |
| Đồng bộ | tiếng đã dịch về 0 theo độ lệch SyncNet của video (trừ clip `unreliable` được chọn lại, ghi rõ `av_sync_status`) | 03 cut (đo), 07 export (sửa) |
| Người | thuộc một speaker; người duyệt xác nhận đúng người, gán giới | 04 identify, 05 review |
| Chất lượng | được chọn (keep): góc mặt lệch ≤ ~30 %, không che miệng, không chữ đè miệng, không nhạc nền to | 05 review |
| Split | đúng một split theo người, khoá cùng part | 06 parts |
| Mô tả | một dòng `manifest.jsonl`: `clip_id`, người, giới, split, độ dài, nguồn (URL, kênh, đoạn giây trong video gốc), `audio_lag_frames`, `av_sync_status`, cờ, hash | 07 export |
| Truy vết | mọi thông tin trên + lý do loại các clip khác nằm trong ledger | ledger |

- Export **chép** clip từ pool sang thư mục part (độ lệch 0: chép nguyên file; khác 0: tạo file mới, chép luồng hình, encode lại tiếng); pool giữ bản gốc chưa sửa, đổi cách đo về sau chỉ cần export lại.
- Clip đã xoá ở máy vẫn còn đủ thông tin trong ledger (`clips`: `clip_id` tính cố định từ video + mốc giây, `start_s`, `end_s`, hash, người, quyết định, cờ, nơi lưu; `videos`: URL, hash raw, zip raw trên Drive, `cut_info`) → tái hiện được bằng cách cắt lại từ raw cùng mốc và cùng thông số (trùng nội dung và mốc, chưa chắc trùng hash từng byte nếu ffmpeg khác phiên bản). Pipeline chưa có lệnh cắt lại theo ledger.


### 8.2. Part export (07 export) - thứ đồng đội nhận

Thư mục `exports/vn-av-df-data/vn-av-df-data-partN/` (07 không zip), upload thư mục lên Kaggle dataset **private**; zip `vn-av-df-data-partN.zip` cho Drive do `08_archive part N` tạo:

| File | Nội dung |
|---|---|
| `clips/<clip_id>.mp4` | 600-800 clip real (15-20 / người), 4-7 s, 25 fps, cạnh ngắn ≤ 720, H.264 + AAC 48 kHz; **tiếng đã dịch cho khớp môi** |
| `manifest.jsonl` | mỗi dòng một clip: `clip_id`, `speaker_id`, `gender`, `split`, `duration_s`, nguồn (`url`, `channel`, đoạn giây trong video gốc), `audio_lag_frames`, `av_sync_status`, `flags` |
| `speakers.csv` | 40 người: giới, split, kênh, số video |
| `split-lock.json` | khoá split theo người |
| `dataset_info.json`, `quality_report.json` | thống kê part, hash manifest, kết quả kiểm định dạng |

### 8.3. Lưu trữ và dọn máy

| Lệnh | Việc | Điều kiện |
|---|---|---|
| `08_archive raw N` / `part N` | zip raw batch N (+ `SHA256SUMS.txt`) / zip thư mục part N. Raw và part cùng một kiểu (từ 20/10; trước đó 07 tự zip part) | raw: tải + cắt + identify xong; part: 07 export xong |
| `08_archive raw N --uploaded` / `part N --uploaded` | ghi ledger: đã lên Drive | user tự upload zip + thư mục `ledger/` |
| `09_clean raw N` | xoá video raw + zip (clip trong pool giữ lại để duyệt) | đã `--uploaded` |
| `09_clean part N` | xoá export + zip + mọi clip pool của 40 người trong part | đã `--uploaded` |
| `09_clean pool` | xoá clip không được chọn của speaker đã duyệt xong / bị loại | chỉ clip thuộc batch raw đã lên Drive (luôn cắt lại được) |

Ba lệnh xoá tách riêng và đều bắt buộc đã đánh dấu upload (user đề xuất 18/10). Raw không bao giờ tự xoá. Ledger, `registry/`, ảnh mặt, `urls.csv` không bao giờ xoá. Raw batch nào cắt và gán speaker xong thì dọn được ngay, không cần chờ đủ part.

## 9. Số đo thực tế

### 9.1. Lịch sử các lần chạy

| Lần chạy | Kết quả ● |
|---|---|
| Batch thử (17/10; 4 nguồn, 5 video, 2,6 giờ) | tải 5/5 (403 tự thử lại; bot → cookie); cắt 296 clip; sync 3 `ok` (trễ 1-2 frame), 2 `unreliable`; identify gộp đúng 1 người qua 2 nguồn, tách podcast 2 người; duyệt 3 speaker đủ clip; dịch tiếng export đúng trên 8 clip. Đã xoá toàn bộ để chạy bản thật |
| Batch 1 thật (18/10) | 83 URL video lẻ; 4 URL bị loại vì 24 fps; batch 1 = 79 video, 17,4 giờ, ước 5,22 GB, thực tế **4,9 GB** (≈ 0,28 GB / giờ); 9 video đầu: 345 clip, 7 `ok` / 2 `unreliable`. Chỉ tiêu hạ còn 30-40, user xoá hết và tải lại từ đầu |
| Batch 1, lần chạy với chỉ tiêu 30-40 (18-19/10) | 84 video, 2.392 clip; cửa sổ kiểm 5.165 → giữ 46 % (loại vì mặt 1.810, chuyển cảnh 572, jump cut 391); sync 78 `ok` / 6 `unreliable`; độ lệch: 0 frame 3 video · 1: 20 · **2: 36** · 3: 13 · **4: 6** (160 ms, quá ngưỡng nhận ra 125 ms → phải sửa); identify 43 speaker mới / 41 khớp / 19 gợi ý gộp; duyệt 38 xong / 4 loại. Mép clip: chỉ 8 % cả 2 mép ở chỗ ngắt (§5.4) → áp luật cắt mới, xoá hết, chạy lại |
| Batch 1 với luật cắt mới → part 1 (19-20/10) | 80 video, 18,5 giờ; duyệt 41 người xong (20 nam, 21 nữ) / 1 loại; **part 1 = 40 người, 800 clip** (train 480 / validation 120 / test 200), dư 1 nữ chờ part 2; 75 video nguồn, 796 clip `ok` + 4 clip `unreliable` (user chọn lại, export không dịch tiếng; `agree` theo clip: trung vị 1,0, p10 0,95). Zip raw 5,4 GB, zip part 0,84 GB, cả hai lên Drive rồi xoá ở máy. Raw batch 1 lỡ xoá trước khi up, tải lại (hash raw trên Drive là của lần tải thứ hai) |
| Batch 2 (20/10) | 32 video (18 nguồn 1 người, 14 nguồn nhiều người), 5.040 clip ứng viên; identify 31 người mới / 15 khớp / 17 gợi ý gộp. Nguồn nhiều người ~280 clip / video 60 phút (§5.3) |
| Batch 3-6 (20-22/10) | batch 3: 36 video · batch 4: 17 · batch 5: 22 · batch 6: 64; identify 32 / 21 / 11 / 19 người mới; luật `2` áp từ batch 3 |
| Part 2, part 3 (22/10) | mỗi part 40 người, 800 clip; part 3 lệch split nhẹ do nhóm lớn (§7.3) |

### 9.2. Toàn bộ dữ liệu (S00, ledger 22/10/2026)

**Phễu dữ liệu**

| Giai đoạn | Số |
|---|---|
| Input (`urls.csv`, đều là link video lẻ) | 288: `1` 239 · `2` 29 · `n` 20; input không còn video nào đạt: 29 |
| Video bị loại khi plan | 29: 24 fps 25 · track tiếng ngôn ngữ khác 2 · < 720p 2; trùng 0 |
| Video được chọn | **259 video, 115,6 giờ, 92 kênh**; `1` 212 video / 69,1 giờ · `2` 29 / 28,4 · `n` 18 / 18,1 |
| Độ dài video p10 / p50 / p90 | 6,3 / 16,4 / 67,4 phút |
| Năm đăng | 2026: 187 (72 %) · 2025: 40 · 2024: 13 · 2023: 10 · 2022: 3 · 2021: 4 · 2020: 2 |
| Định dạng gốc (720p) | AV1 ≥ 84 % · H.264 ≥ 14 %; fps gốc 30 / 29,97 ~66 % · 25 ~20 % · 50 / 60 ~13 % (đều chuẩn hoá về 25 fps CFR, nguồn > 25 fps bị bỏ bớt frame) |
| Đã tải và cắt (batch 1-6) | 251 video, 28,3 GB raw (≈ 0,25 GB / giờ); batch 7 (8 video, 2,1 giờ) chưa tải |
| Clip ứng viên | **16.371**, độ dài p10 / p50 / p90 = 4,16 / 5,36 / 6,72 s |
| Người tạo tự động | 156; khớp người đã có 144 lượt; gợi ý gộp 92 |
| Người (22/10) | duyệt xong 121 (120 trong part + 1 nữ dư) · loại 10 · chờ 25 |
| Clip được chọn | 2.400 trong part (đúng 20 / người ở cả 120 người) |

**Batch raw** (≤ 20 giờ và ≤ 16 GB)

| Batch | Video | Giờ | GB | Lưu trữ (22/10) |
|---|---|---|---|---|
| 1 | 80 | 18,5 | 5,4 | zip trên Drive, đã xoá ở máy |
| 2 | 32 | 19,0 | 4,7 | còn ở máy |
| 3 | 36 | 19,6 | 4,5 | còn ở máy |
| 4 | 17 | 17,7 | 3,8 | còn ở máy |
| 5 | 22 | 19,7 | 4,3 | còn ở máy |
| 6 | 64 | 18,9 | 5,6 | còn ở máy |
| 7 | 8 | 2,1 | - | chưa tải |

**Cắt theo luật số lượng**

| Luật | Video | Phút video | Cửa sổ kiểm | Clip giữ | Tỉ lệ giữ | Clip / video | Giây cắt / phút video (trung vị) |
|---|---|---|---|---|---|---|---|
| `1` (30-40 / input) | 204 | 4.020 | 13.586 | 6.227 | 46 % | 31 | 13,3 |
| `2` (≤ 150 / video) | 29 | 1.705 | 4.410 | 4.032 | 91 % | 139 | 22,6 |
| `n` (cắt hết) | 18 | 1.085 | 6.528 | 6.112 | 94 % | 340 | 29,6 |
| Tổng | 251 | 6.810 | | 16.371 | | | |

- Nguồn `2` / `n` lọc trước bằng quét mặt 4 fps (9.777 cửa sổ loại ở bước này, không tính vào "cửa sổ kiểm"), nên tỉ lệ giữ cao; nguồn `1` không quét trước nên mọi cửa sổ có tiếng đều kiểm kỹ, kể cả cảnh minh hoạ.
- Lý do loại ở bước kiểm từng frame: mặt không đạt ở một frame 5.524 · chuyển cảnh 1.387 · jump cut 1.183 · đổi người giữa clip 59.
- Cờ trên clip ứng viên: `jump_suspect` 3.810 · `face_gap` 399 · `sync_unreliable` 198 · `face_outlier` 64 · `voice_outlier` 49 · `sync_few` 2.
- Thời gian cắt ≈ 30,6 giờ CPU (single 10,9 · pair 10,3 · multi 9,4) trên máy CPU 16 luồng, RAM 7,8 GB.
- Cách gán clip ở identify: người mới 10.470 · khớp trong kênh 5.504 · khớp toàn cục 396 · chưa gán 1.

**Người đã duyệt xong (120 người trong part)**

| Số video có clip được chọn | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Người | **72 (60 %)** | 37 | 9 | 1 | 1 |

- 117 / 120 người chỉ từ 1 kênh. Clip được chọn trên một cặp (người, video): trung vị 11, tối đa 20. Ứng viên / người p10 / p50 / p90 = 32 / 77 / 236.

| Loại nguồn | Nam | Nữ | Người | Clip được chọn |
|---|---|---|---|---|
| `1` | 33 | 36 | 69 | 1.380 |
| `2` | 20 | 9 | 29 | 580 |
| `n` | 7 | 15 | 22 | 440 |

**Part**

| | Part 1 | Part 2 | Part 3 |
|---|---|---|---|
| Người | 40 (20 + 20) | 40 (20 + 20) | 40 (20 + 20) |
| Clip | 800 | 800 | 800 |
| Người theo split (nam + nữ) | 12+12 / 3+3 / 5+5 | 12+12 / 3+3 / 5+5 | 12+12 / **2+3** / **6+5** |
| Clip train / validation / test | 480 / 120 / 200 | 480 / 120 / 200 | 480 / 100 / 220 |
| Video / kênh nguồn | 75 / 39 | 45 / 18 | 44 / 10 |
| Clip `unreliable` (không dịch tiếng) | 4 | 0 | 0 |
| Zip (lưu Drive) | 0,84 GB | 0,68 GB | 0,65 GB |

`06_parts` → `09_clean` chạy thật lần đầu với part 1 (20/10).

## 10. So với chuẩn hiện nay (AV-Deepfake1M++)

Tiêu chí cộng đồng dùng để đánh giá bộ dữ liệu: quy mô và số người; đa dạng nguồn real; đa dạng phương pháp, có phương pháp chưa thấy; nhiễu thực tế; nhãn định vị; chia theo người; không lối tắt; đa ngôn ngữ; độ khó; khoảng cách tới deepfake ngoài đời.

| Tiêu chí | AV-Deepfake1M++ ✔ | v003 | |
|---|---|---|---|
| Quy mô | 2,05 triệu video, 4.656 giờ, 7.109 người | hiện có 3 part = 120 người, 2.400 clip real ●; ~5.850-7.800 file / part; 5 part = 200 người, ~29-39 nghìn file ○ | **kém xa**, giới hạn chính |
| Nguồn real | VoxCeleb2, LRS3, EngageNet | YouTube tiếng Việt, nhiều kênh | khác loại; **tiếng Việt** là điểm mới |
| Chia theo người | chỉ ở test (train / validation chia ngẫu nhiên theo mẫu) | **cả 3 tập** | v003 chặt hơn |
| Phương pháp chưa thấy | có | có (2 generator + 3 kiểu giọng) | ngang |
| Đối chứng lệch không AI (② sham) | không | **có** | v003 thêm |
| Tách tác vụ | nhãn hình / tiếng | 6 ô, 4 tác vụ | v003 chi tiết hơn |
| Sửa độ lệch tiếng-hình real | không nhắc | **có** | v003 thêm |
| Encode giống hệt real / fake | cùng nguồn | cùng nguồn + encode giống hệt | ngang / nhỉnh |
| Kiểm lối tắt | bản trước bị phát hiện có | **có** (`shortcut_audit`) | v003 thêm |
| **Nhiễu thực tế ở test** | 36 loại, nhiều codec | **chưa có** | v003 thiếu |
| Tỉ lệ real | 31 % | 31 % | ngang |

Phản biện quan trọng: **Deepfake-Eval-2024** (45 giờ video lưu hành năm 2024, 88 website, 52 ngôn ngữ) cho thấy detector mã nguồn mở giảm ~50 % AUC so với benchmark học thuật ✔.

## 11. Hạn chế và việc mở (phần dữ liệu)

**Hạn chế** (nêu trong báo cáo):
1. Quy mô nhỏ hơn hàng chục lần so với bộ chuẩn (120 người so với hàng nghìn); một loại nguồn (YouTube), chủ yếu nói chuyện trước camera / phỏng vấn.
2. Chưa có bản test nhiễu thực tế (nén lại, giảm độ phân giải, đổi fps).
3. **Đa dạng buổi quay thấp**: 60 % người chỉ có clip từ 1 video, 117 / 120 người từ 1 kênh (phần lớn URL là link video lẻ) - thấp hơn khuyến nghị "15-20 clip từ 2-3 video" (§3.3).
4. **Một người duyệt**, chưa đo độ đồng thuận; góc mặt chọn bằng mắt, không có số đo.
5. Độ lệch đo theo video, độ phân giải ±1 frame; "điểm 0" của SyncNet chưa kiểm chéo; 4 clip `unreliable` trong part 1 không được sửa.
6. `jump_suspect` có ở ~24 % clip được chọn mà không ảnh hưởng quyết định duyệt (§7.2).
7. **Giới lệch theo loại nguồn** (`2`: 20 nam / 9 nữ; `n`: 7 nam / 15 nữ) dù tổng cân 60 / 60 → nếu loại nguồn tương quan với điều kiện quay, nên báo kết quả theo giới × loại nguồn khi dùng dữ liệu.
8. Nguồn 30 / 50 / 60 fps chuyển về 25 fps bằng bỏ frame; nguồn 24 fps bị loại → bộ dữ liệu lệch về kênh quay ≥ 25 fps.
9. 72 % video đăng năm 2026.
10. Nhóm lớn quanh MC làm part 3 lệch split nhẹ (validation 2 nam, test 6 nam).
11. Raw batch 1 trên Drive là lần tải thứ hai (hash khác lần đầu; mốc thời gian và clip không đổi).
12. Dữ liệu là người thật trên YouTube: **không công bố video**; nếu phát hành thì chỉ ID + mốc thời gian + script tái tạo.

**Việc mở:**

| # | Việc | Ghi chú |
|---|---|---|
| 1 | Zip + up raw batch 2-6 (`08_archive raw N`, rồi `--uploaded`) | chưa xoá gì; mở khoá `09_clean pool` (~9,1 GB clip thừa) |
| 2 | Batch 7 (8 video): tải, cắt, gán | chưa tải |
| 3 | 25 người chờ duyệt; 3 người trong số đó chung video với người đã vào part 3 (qua nhóm MC) → nếu duyệt xong sau này sẽ vào `blocked_groups` | xử lý tay khi xảy ra (ví dụ theo split của nhóm) |
| 4 | **Bản test nhiễu thực tế** | chỉ cần ffmpeg |
| 5 | Thí nghiệm gợi ý S05 (mép giữa từ, §5.4), S06 (kiểm chéo độ lệch, §6.5), S07 (nhãn tay jump cut), S08 (đồng thuận người duyệt) (§7.2) | S05 / mở rộng S01 cần raw: làm trước `09_clean raw` |
| 6 | Lệnh cắt lại clip theo ledger (tái hiện clip đã xoá) | chỉ khi cần |
| 7 | In tiến độ cho bước VAD trong `03_cut`; sửa link docstring `features/syncnet.py` | đổi code trong chữ ký luật cắt → đóng dấu lại |

## 12. Nhật ký câu hỏi và quyết định (phần dữ liệu)

| Ngày | Câu hỏi / việc | Trả lời / quyết định |
|---|---|---|
| 14/10 | Khảo sát bộ dữ liệu, quy mô | dataset_survey: mức S / M / L; ưu tiên số người hơn số giờ |
| 17/10 | Bỏ mục tiêu "đủ 200 / 300 speaker thì dừng" | bỏ (chỉ là con số hiển thị); giữ giới hạn batch raw và part |
| 17/10 | Giới hạn raw / part áp khi nào | chỉ khi chạy lệnh |
| 17/10 | Nguồn 1 người lấy nhiều hơn 3 video? Video dọc? | **≥ 20 phút, 3-10 video**; chỉ `/videos`, chỉ ngang (user) |
| 17/10 | Folder `data/`, ledger, batch, giới tính, Drive | một `data/` cho mọi batch / part; ledger là sổ cái; giới tính đánh tay |
| 17/10 | Speaker gán khi nào | máy gán ở 04, người sửa ở 05 |
| 17/10 | Lỗi 403 / "xác minh bot" | 403 thử lại 4 lần; bot dừng batch, dùng cookie Firefox |
| 17/10 | Luật cắt quá chặt | 64 → 48 px, hụt ≤ 3 frame (số đo §5.1). User: không thêm `--reset`, tự xoá kết quả cắt để cắt lại |
| 17/10 | YouTube lệch tiếng-môi sẵn có ảnh hưởng gì | có: lối tắt "khớp quá mức", nhãn nhiễu (§6.1) |
| 17/10 | AV-HuBERT đo độ lệch được không | được nhưng phải kiểm; chọn SyncNet (§6.3) |
| 17/10 | Chỉ đo thì làm gì, lọc bỏ video? | sửa theo video, chỉ bỏ video không đo được, thêm augment |
| 17/10 | Vì sao đo sau cắt, vì sao không AV-HuBERT local | đo **trong** bước cắt; AV-HuBERT không chạy dễ trên Windows |
| 17/10 | Video lệch nhiều có bị bỏ? Lệch 1 frame có đủ? Augment tạo file mới? | lệch ổn định thì sửa; ±1 là sai số còn lại; augment không tạo file |
| 17/10 | "Augment +2 có chắc? Cần research, không phải vì tôi mà sửa số" | −1…+2 theo BT.1359 + 3 căn cứ (§6.4). User: đưa bằng chứng 45 ms vào; thí nghiệm để sau |
| 17/10 | Sửa độ lệch có bỏ nhánh lệch thời gian? Hạ Python 3.10? SyncNet đủ tốt? | không nghiên cứu sham_shift (user); không hạ Python; SyncNet đủ khi gộp |
| 17/10 | Cắt lâu; 3 video song song tốn bao RAM | đo RAM; song song chỉ 1,3× → user: quay lại tuần tự |
| 18/10 | pending vs reject; video bị loại do sync | như nhau; không cần đánh gì |
| 18/10 | 06 / 07 / 08 làm gì; dọn gì | user: tách 3 lệnh xoá (raw / part / pool), đều phải qua đánh dấu upload → `09_clean`; status thành `10_status` |
| 18/10 | Có nên chia split ở bước dữ liệu? Benchmark là gì? | có (§2) |
| 18/10 | Mặt nghiêng có dùng; code có đo góc | dùng được tới ~45°; code **không đo góc**. User: tự chọn, lệch ≤ ~30 % |
| 18/10 | Nhiều người, mỗi người 10 clip có tốt? | bằng chứng §3.1 |
| 18/10 | Số người các bộ có đúng không? | đính chính DFDC: 960 người phát hành (3.426 là số quay) |
| 18/10 | Các bộ có rất nhiều real mỗi người, 10 có ổn? | nhiều clip từ ít buổi quay; xét real phụ |
| 18/10 | "Tôi chọn 20; chỉ tiêu cắt 40-60" | chọn 20; thời lượng nguồn 20 phút là đủ (đã sửa tính nhầm 30 phút) |
| 18/10 | 20-60 clip ngẫu nhiên mỗi người? 10 hay 20? | bác 20-60; chọn **20 cố định** |
| 18/10 | Part 40 người, 20 nam 20 nữ, raw giữ nguyên | sửa, split 12 / 3 / 5 |
| 18/10 | `urls.csv` tạo ở đâu; thêm part sau; không đủ part thì sao | `01_plan` tạo mẫu; thêm URL rồi chạy lại; part chỉ tạo khi đủ 20 + 20 |
| 18/10 | `"empty": [19, 57, 67, 71]` | 4 URL video lẻ bị loại vì 24 fps |
| 18/10 | Batch 20 GB / 16 giờ? | đính chính: **20 giờ / 16 GB** |
| 18/10 | Cắt batch 1 bao lâu; sao log lâu | VAD cả batch chạy trước (~8 phút); 9 video đầu ~258 s / video |
| 18/10 | Hạ chỉ tiêu 30-40, rải khắp video; khúc không có clip thì hụt? | đồng ý; không hụt, khúc khác bù |
| 18/10 | Rải clip có làm chậm? Dừng giữa chừng ghi ledger thế nào? | không chậm; một ledger, mỗi video ghi một lần khi xong |
| 18/10 | Xoá hết, chỉ giữ `urls.csv`, tải lại từ đầu (user) | đã xoá raw, ledger, pool |
| 19/10 | Trang duyệt tràn màn hình | 20 ô "đã chọn" xếp 2 hàng × 10 |
| 19/10 | Speaker chưa đủ 20 clip thì sao, thêm batch 2 thế nào | N để chờ (không đánh dấu xong); batch 2 khớp thêm clip thì người đó hiện lại để duyệt tiếp |
| 19/10 | Cho số clip dao động 10-20 hoặc 15-20 (phỏng vấn đường phố khó đủ 20)? | giữ **20 cố định** (user). Batch 1: 40/43 người có ≥ 20 clip ứng viên ●; phỏng vấn đường phố ước chỉ 2-8 clip / người ○, nên 15-20 cũng không cứu được; nếu cần thì thử trên workspace riêng hoặc làm tập test phụ |
| 19/10 | Hiện tại cắt câu thế nào, có tròn câu không? Code nhóm Capstone cắt thế nào? | không tròn câu: chặt cứng 7 s trong đoạn nói; Capstone giữ nguyên đoạn 2-12 s (tròn cụm hơn) |
| 19/10 | Cắt semantic thì sao? | cần ASR + dấu câu → nặng; thay bằng dời điểm cắt về chỗ ngắt (VAD), thử 8 % → 63 % mép sạch |
| 19/10 | Đổi sang 15-20 clip / người cho mọi part, "phải đảm bảo không ảnh hưởng" (user) | part 600-800 clip; train lấy mẫu cân theo người; AUC cân theo người; split cân tổng clip; trang duyệt đạt từ 15, tối đa 20 (§3.3) |
| 19/10 | Áp luật cắt mới (user) | xoá hết dữ liệu, tải lại; bỏ khỏi `urls.csv` 10 link: 6 video sync `unreliable` + các video của 4 speaker bị loại; thêm 8 link mới (1 người / link) |
| 20/10 | Duyệt xong mà dư speaker thì sao? | chạy `06_parts`; người dư (đã duyệt xong) chờ part sau, không phải duyệt lại |
| 20/10 | 07 tự zip part còn raw zip ở 08: đồng bộ (user) | 07 chỉ xuất thư mục; zip cả raw lẫn part ở `08_archive` (zip → up Drive → `--uploaded` → `09_clean`) |
| 20/10 | Lỡ chạy `09_clean raw 1` trước khi up | ledger về bản sau export, tải lại raw, zip lại; `08_archive raw` từ chối khi thiếu file raw |
| 20/10 | Plan bị chặn bot | plan dừng ngay khi gặp bot check (như download); cookie Edge lỗi DPAPI (yt-dlp không giải mã cookie Chromium mới) |
| 20/10 | Clip `unreliable` được chọn lại có tính là lệch 0? | không: export **không dịch** (độ lệch gốc giữ nguyên); `few` vẫn dịch. User tự cân nhắc khi duyệt, không sửa code |
| 20/10 | Backup ledger tự xoá? | trước giữ 30 bản; từ 20/10 giữ mọi bản, chỉ xoá tay |
| 20/10 | Thêm loại `2` cắt tối đa 150 clip, từ batch 3 (user) | §5.3; chữ ký luật cắt tách luật số lượng (§5.5) |
| 20/10 | Ledger có trong zip không? | không: user tự up thư mục `data/ledger/`; Claude không xoá / ghi đè ledger khi chưa được duyệt |
| 20/10 | identify theo batch hay toàn bộ? Vừa duyệt vừa cắt được không? Dừng cắt giữa chừng? | theo batch, so khớp với mọi người đã có; duyệt song song với cắt batch khác được; Ctrl+C rồi chạy lại, video dở cắt lại |
| 20/10 | `09_clean raw` có xoá clip? | không; clip chỉ bị xoá bởi `09_clean part` / `pool` |
| 21/10 | Lỡ chốt 15 clip mà muốn thêm | "done" chưa khoá; sửa được tới khi vào part, hoặc đưa về pending qua API |
| 22/10 | Thống kê, thí nghiệm kiểm chứng, báo cáo | S00-S04 (§5.3, §5.4, §6.5, §7.2, §9); báo cáo chỉ tới clip real hoàn chỉnh (§8.1); mọi nội dung báo cáo nằm trong file này (§14) |
| 22/10 | Part 3 không tạo được dù trang duyệt báo 20/20 | người pending chung video (qua MC) giữ nhóm; "Khác người" chỉ bỏ gợi ý gộp. Sửa: người pending không giữ nhóm nữa, đến sau thì vào `blocked_groups` (§7.3) |
| 22/10 | Export lấy clip từ pool là chép hay cắt? Clip đã xoá còn tái hiện được không? | chép, pool giữ nguyên; ledger giữ mốc giây + hash → cắt lại được từ raw (§8.1) |

## 13. Nguồn

**Dữ liệu và bằng chứng số lượng (đã đọc bản gốc)**
- Scaling Laws for Deepfake Detection, 2025 - [arXiv 2510.16320](https://arxiv.org/abs/2510.16320)
- Dong et al., Implicit Identity Leakage, CVPR 2023 - [link](https://openaccess.thecvf.com/content/CVPR2023/html/Dong_Implicit_Identity_Leakage_The_Stumbling_Block_to_Improving_Deepfake_Detection_CVPR_2023_paper.html)
- Bansal et al., The Do's and Don'ts for CNN-based Face Verification, ICCVW 2017 - [arXiv 1705.07426](https://arxiv.org/abs/1705.07426)
- Nadimpalli & Rattani, GBDF, ICPRW 2022 - [arXiv 2207.10246](https://arxiv.org/abs/2207.10246)
- Referee - [arXiv 2510.27475](https://arxiv.org/abs/2510.27475) · ID-Reveal, ICCV 2021 · POI-Forensics - [arXiv 2204.03083](https://arxiv.org/abs/2204.03083)
- Layton et al., USENIX Security 2024 - [pdf](https://www.usenix.org/system/files/sec24fall-prepub-1479-layton.pdf)
- Deepfake-Eval-2024 - [arXiv 2503.02857](https://arxiv.org/abs/2503.02857)
- Bộ dữ liệu: FakeAVCeleb [2108.05080](https://arxiv.org/abs/2108.05080) · Celeb-DF [1909.12962](https://arxiv.org/abs/1909.12962) · LAV-DF [2204.06228](https://arxiv.org/abs/2204.06228) · KoDF [2103.10094](https://arxiv.org/abs/2103.10094) · DFDC [2006.07397](https://arxiv.org/abs/2006.07397) · AV-Deepfake1M [2311.15308](https://arxiv.org/abs/2311.15308) · AV-Deepfake1M++ [2507.20579](https://arxiv.org/abs/2507.20579)
- Tiếng Việt: ViCocktail [2506.04635](https://arxiv.org/abs/2506.04635) · VietPrism [2609.30005](https://arxiv.org/abs/2609.30005) · VoxVietnam [2501.00328](https://arxiv.org/abs/2501.00328)

**Độ lệch tiếng-hình**
- ITU-R BT.1359 (qua bản tóm tắt; chưa đọc bản gốc ITU, con số chấp nhận được có nguồn ghi +90 / −185, nguồn khác +90 / −190): [Fora Soft](https://www.forasoft.com/learn/audio-for-video/articles-audio/lip-sync-itu-r-bt-1359-tolerance-windows), [TV Technology](https://www.tvtechnology.com/opinions/av-synchronization-how-bad-is-bad), [arXiv 2212.01686](https://arxiv.org/abs/2212.01686)
- SyncNet: Chung & Zisserman, "Out of Time", ACCVW 2016; [joonson/syncnet_python](https://github.com/joonson/syncnet_python)
- Perfect Match [1809.08001](https://arxiv.org/abs/1809.08001) · VocaLiST [2204.02090](https://arxiv.org/abs/2204.02090) · ModEFormer [2303.11551](https://arxiv.org/abs/2303.11551) · IC-SyncNet [2409.00971](https://arxiv.org/abs/2409.00971) · PVSync [2610.09223](https://arxiv.org/abs/2610.09223) · LatentSync / StableSyncNet [2412.09262](https://arxiv.org/abs/2412.09262)
- DFD-Lab [2609.23830](https://arxiv.org/abs/2609.23830) · DeepFake Doctor [2506.05851](https://arxiv.org/abs/2506.05851)
- yt-dlp releases (Python 3.10): [GitHub](https://github.com/yt-dlp/yt-dlp/releases)

**Kiểm thêm ngày 22/10** (mức kiểm ghi sau mỗi nguồn)
- Smeu et al., Circumventing Shortcuts in Audio-visual Deepfake Detection Datasets with Unsupervised Learning, CVPR 2025 - [arXiv 2412.00175](https://arxiv.org/abs/2412.00175): fake trong FakeAVCeleb và AV-Deepfake1M bắt đầu bằng 25-30 ms lặng; bộ phân loại chỉ dùng khoảng lặng đạt AUC 98,4 / 98,2 % ✔ (abstract + Bảng 1, bản arXiv HTML).
- Nagrani et al., VoxCeleb, Interspeech 2017 - [arXiv 1706.08612](https://arxiv.org/abs/1706.08612): dò ranh giới cảnh bằng so histogram màu giữa frame liên tiếp; SyncNet xác định mặt đang nói, loại lồng tiếng / lời dẫn; ngưỡng bảo thủ (precision 1,000, recall 0,613 ở bước này) ✔ (toàn văn ar5iv).
- Chung et al., VoxCeleb2, Interspeech 2018 - [arXiv 1806.05622](https://arxiv.org/abs/1806.05622): pipeline tương tự VoxCeleb, thêm bước bỏ video trùng ◐.
- Afouras et al., LRS3-TED, 2018 - [arXiv 1809.00496](https://arxiv.org/abs/1809.00496): tiếng và hình có thể lệch tới khoảng 1 s, dùng SyncNet đồng bộ và loại lời dẫn; câu ≤ 100 ký tự hoặc 6 s (train-val, test); chia split theo video ✔ (toàn văn ar5iv).
- Gebru et al., Datasheets for Datasets, Communications of the ACM 64(12):86-92, 2021 - khung tài liệu hoá bộ dữ liệu ◐.
- Wu, Peng, Yu, YuNet: A Tiny Millisecond-level Face Detector, Machine Intelligence Research 20(5):656-665, 2023 ◐.
- Zhong et al., SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition, IEEE TIP 2021 - [arXiv 2205.12010](https://arxiv.org/abs/2205.12010) ◐.
- Wang et al., Wespeaker, ICASSP 2023 - [arXiv 2210.17016](https://arxiv.org/abs/2210.17016) ◐.
- Silero VAD - [snakers4/silero-vad](https://github.com/snakers4/silero-vad) ◐ (trích dẫn theo CITATION của repo).
- Thời gian bị YouTube chặn bot: không có số chính thức; người dùng diễn đàn kể vài giờ ◐ (không dùng làm bằng chứng trong báo cáo).

## 14. Điền báo cáo theo template AIP491 (phần dữ liệu real)

Template: `Template_AIP491_CP_StudentsGuide.docx` (gốc thư mục Deepfake_VN, không sửa bản gốc). Phạm vi hiện tại (user chốt 22/10): **chỉ tới khi tạo ra một clip real hoàn chỉnh**; thiết kế generator / kiến trúc mô hình bỏ qua cho tới khi user yêu cầu. Viết hai bản Việt / Anh cùng cấu trúc, cùng số (bản Việt dấu phẩy thập phân, bản Anh dấu chấm).

### 14.1. Mục template → nội dung trong file này

| Mục template | Lấy từ | Ghi chú |
|---|---|---|
| Bìa, thông tin nhóm, lời cảm ơn | - | user điền (chưa có thông tin) |
| Thuật ngữ và từ viết tắt | AV, AUC, VAD, SyncNet, YuNet, SFace, WeSpeaker, fps, input, batch, clip ứng viên, keep, speaker, part, split, ledger, cut signature, shortcut | định nghĩa ở §0, §4-§8 |
| I.1 Thông tin, tổng quan | §0, §8.1 | nêu rõ phạm vi: tạo clip real hoàn chỉnh; sinh fake và mô hình ngoài phạm vi |
| I.2 Bối cảnh | §1 (khoảng trống tiếng Việt, bài học v001 / v002) | |
| I.3 Mục tiêu | §0, §3.3; tình trạng §9.2 | 5 mục tiêu đo được: nhiều người, split theo người, không lối tắt cấu trúc, tái lập, mọi thông số có bằng chứng |
| I.4 Phát biểu vấn đề | §3.1 (số người), §6.1 + Smeu 2025 (lối tắt), §5.1-§5.2 (video lộn xộn), §11 mục 12 (riêng tư) | |
| I.5 Ý nghĩa | §10; số S01 (10,2 → 41,4 %) | |
| I.6 Phạm vi và giới hạn | §11 | |
| II.2.1 WBS | gói việc theo bước 01-10 + trạng thái §9.2, §11 | |
| II.2.2 Rủi ro | bảng 14.2 dưới đây | |
| II.2.3 Chất lượng | 40 test tự động cho pipeline, `quality_report.json` mỗi part, thí nghiệm S00-S04, mọi clip có người duyệt | |
| II.2.5 Quản lý thay đổi | quy trình 14.3 + nhật ký §12 | |
| II (các mục còn lại) | - | thông tin nhóm, user điền |
| III Liên quan | §3.1, §3.2, §6.3, §10, §13 | lịch sử bộ dữ liệu, nghiên cứu số người, lối tắt (Smeu), cách xây VoxCeleb / LRS3, công cụ, bảng so sánh, khoảng trống |
| IV.1 Câu hỏi nghiên cứu | RQ-D1 (đặc điểm dữ liệu, §9.2), RQ-D2 (luật cắt, §5.4), RQ-D3 (giới hạn 150, §5.3), RQ-D4 (độ lệch, §6.5), RQ-D5 (cờ duyệt, §7.2) | |
| IV.2 Thu thập và tiền xử lý | §4, §5.1-§5.3, §6.2-§6.4, §7.1-§7.3, §8.2-§8.3 | |
| IV.3 Đặc trưng | Silero VAD, YuNet, SFace, WeSpeaker, SyncNet, histogram màu - chỉ để làm sạch / tổ chức dữ liệu | |
| IV.4 (đổi thành "Chia dữ liệu và kiểm tra tính đúng") | §2, §7.3; tỉ lệ clip = tỉ lệ người; kiểm số clip, hash khi export | không huấn luyện mô hình ở phạm vi này |
| IV.5 Chỉ số | tỉ lệ giữ, mép sạch (VAD, mô tả) / mép lặng (năng lượng, độc lập), độ lệch, `conf`, `agree`, ứng viên / người, tỉ lệ chọn theo cờ | |
| IV.6 Triển khai | §5.5, §9.2 (thời gian cắt), phần mềm: Python 3.11, yt-dlp, ffmpeg / PyAV, ONNX Runtime, PyTorch CPU, SQLite, FastAPI | |
| IV.7 Đạo đức | §11 mục 12; giới tính là cảm nhận của người duyệt, không tự khai; fake không phát hành | |
| V Thiết kế hệ thống | luồng 00-10 (`data_pipeline/README.md`), mô hình tích hợp (§6.3, §7.1), triển khai (máy cá nhân, Kaggle riêng tư, Drive), mở rộng theo batch / part | |
| VI.1 Kết quả | §9.2 (S00), §5.4 (S01), §5.3 (S02), §6.5 (S03), §7.2 (S04) | |
| VI.2 Thảo luận | luật cắt (S01 + Smeu), độ lệch (S03 + LRS3 / VoxCeleb), đa dạng buổi quay, giới theo loại nguồn, cờ `jump_suspect` | |
| VI.3 Khuyến nghị | gợi ý S05-S08 (§5.4, §6.5, §7.2), thu thập 2-3 video / người, cân giới trong từng loại nguồn, test nhiễu | |
| VI.4 Nhìn lại | đo trước khi đổi luật (7 s → chỗ ngắt), cái giá của đổi luật giữa chừng (cắt lại batch 1), chốt chặn thao tác xoá (sự cố raw batch 1), số lượng không thay được đa dạng | |
| VII Kết luận | tóm tắt §0, §9.2 và các thí nghiệm; giới hạn §11 | |
| VIII Tài liệu | §13 | kiểm đủ tác giả các mục "et al." trước khi nộp; nguồn ◐ chỉ trích thông tin xuất bản |
| IX Phụ lục | tham số §5.1, §5.3, §6.4, §7.2, §7.3; thay đổi luật §5.1, §5.4, §3.3, §5.3, §12 | |

### 14.2. Rủi ro đã gặp (cho mục II.2.2)

| Rủi ro | Đã xảy ra? | Phòng / xử lý |
|---|---|---|
| YouTube chặn vì nghi bot | có, khi plan (20/10) | gặp xác minh bot thì dừng ngay; chạy lại với cookie trình duyệt đã đăng nhập hoặc chờ |
| Cookie trình duyệt bị mã hoá kiểu mới | có (Edge, lỗi DPAPI) | dùng Firefox hoặc chờ hết chặn |
| Xoá raw trước khi sao lưu | có (raw batch 1, 20/10) | khôi phục ledger từ backup, tải lại; chặn đóng gói khi thiếu file raw; lệnh xoá chỉ chạy sau `--uploaded` |
| Đổi luật giữa chừng | có (19/10, cắt lại batch 1) | cut signature khoá cách làm clip |
| Ổ đĩa đầy | nguy cơ | batch ≤ 20 giờ / 16 GB; raw lên Drive mới xoá |
| Mất ledger | nguy cơ | mỗi lệnh tự sao lưu (giữ mọi bản); ledger up Drive cùng zip |
| Part kẹt vì nhóm chung video | có (22/10) | người pending không giữ nhóm; người đến sau vào `blocked_groups` |
| Lộ dữ liệu người thật | nguy cơ | không đẩy dữ liệu lên dịch vụ công khai (Kaggle riêng tư); báo cáo chỉ có số tổng hợp |

### 14.3. Quy trình thay đổi luật (cho mục II.2.5)

1. Nêu vấn đề bằng số đo (ví dụ 84 % clip dài đúng 7 s, 8 % mép ở chỗ ngắt).
2. Đề xuất kèm bằng chứng, chờ chốt rồi mới sửa ("kế hoạch trước").
3. Thử trên mẫu (luật cắt mới thử trên 6 video).
4. Áp dụng; luật làm clip đổi → cut signature đổi → cắt lại (đã làm một lần với batch 1).
5. Ghi quyết định, ngày, lý do vào §12 và vào ledger (sự kiện).

### 14.4. Quy ước viết

- Giọng sinh viên nghiên cứu: rõ ràng, đủ chi tiết, giải thích thuật ngữ lần đầu, không lan man; mỗi luật kèm lý do một câu.
- Không bịa: số lấy từ file này (●), trích dẫn chỉ từ §13 (nguồn ◐ chỉ trích thông tin xuất bản / ý chính); thiếu thì ghi `[cần số liệu]` / `[cần trích dẫn]`.
- Phân biệt đo trên dữ liệu (●) / tài liệu (✔, ◐) / lựa chọn thiết kế / ước tính (○); thước đo chỉ mô tả (mép sạch theo VAD) phải ghi rõ.
- Riêng tư: không tên người, tên kênh, tiêu đề, mã video, mã speaker.

