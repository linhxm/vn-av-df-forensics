> Bản sao từ nhật ký nghiên cứu (`RESEARCH_WORKLOG/v003_data.md`, ngoài repo) ngày 22/10/2026. Sửa ở nhật ký rồi chép lại vào đây.

# v003 · Dữ liệu real: từ URL YouTube tới một clip real hoàn chỉnh

Báo cáo nghiên cứu phần dữ liệu real của VN-AV-DF v003 (benchmark phát hiện lip-sync deepfake nghe-nhìn tiếng Việt). Cập nhật **22/10/2026**.

File 1/3 của v003:
- **v003_data.md** (file này): toàn bộ phần dữ liệu real, từ khảo sát bộ dữ liệu, thiết kế, pipeline, thí nghiệm, tới quá trình thực hiện và lỗi đã mắc.
- [v003_generation.md](v003_generation.md): sinh fake (ngoài phạm vi file này).
- [v003_model.md](v003_model.md): mô hình phát hiện (ngoài phạm vi file này).

Code: `vn-av-df-forensics/data_pipeline/` (hướng dẫn chạy: `data_pipeline/README.md`, `data_pipeline/RUNBOOK.md`). Dữ liệu (ledger, pool, raw, part) được giữ nguyên làm bằng chứng cho mọi số đo trong file này.

---

## Cách đọc file này

**Phạm vi.** File chỉ viết tới khi **tạo ra một clip real hoàn chỉnh** nằm trong một part (định nghĩa ở §4.7). Thiết kế generator và kiến trúc mô hình nằm ở hai file còn lại và không được nhắc chi tiết ở đây.

**Đây là báo cáo đồ án, không phải paper.** Một paper chỉ giữ thông tin cốt lõi; báo cáo đồ án phải đủ để người khác hiểu, đánh giá và tái lập: mọi tham số kèm lý do, các phương án đã cân nhắc và bị bác, các lỗi đã mắc, các lần đổi luật, số đo trước và sau. Cấu trúc theo tinh thần tài liệu hoá bộ dữ liệu của *Datasheets for Datasets* [54] và *Data Cards* [55]: động cơ, thành phần, quy trình thu thập, tiền xử lý, cách dùng, phân phối, bảo trì.

**Thuật ngữ.** Thuật ngữ chuyên ngành giữ tiếng Anh (lần đầu có giải nghĩa), để khi viết báo cáo chính thức không dịch lệch nghĩa. Bảng đầy đủ ở Phụ lục B.

**Mức bằng chứng** (gắn sau mỗi số liệu / khẳng định):
- **●** đo trên dữ liệu của mình (batch thử 17/10, batch 1-6 thật 18-22/10, part1 của v001 / v002); số liệu toàn bộ chụp từ ledger ngày 22/10/2026;
- **✔** đọc từ paper gốc / trang chính thức / mã nguồn chính thức;
- **◐** chỉ đọc abstract, metadata, nguồn thứ cấp hoặc một phần bài: đủ để trích thông tin xuất bản và ý chính, **không** trích số chi tiết;
- **○** suy luận, ước lượng hoặc lựa chọn thiết kế (ghi cách tính / lý do).

"Chưa tìm thấy" nghĩa là chưa tìm thấy tính tới ngày ghi, không khẳng định là chưa ai làm. Tài liệu tham khảo đánh số [n], danh mục ở cuối file.

**Riêng tư.** Dữ liệu là video công khai của người thật. File này và báo cáo không nêu tên người, tên kênh, tiêu đề, mã video, mã speaker; các bảng dùng nhãn chung (video A, B…).

---

## Tóm tắt

Phát hiện lip-sync deepfake (video chỉ bị vẽ lại vùng miệng cho khớp một đoạn tiếng khác) cần dữ liệu có cả hình và tiếng, đúng ngôn ngữ. Tính tới 14/10/2026 chưa tìm thấy bộ lip-sync deepfake nghe-nhìn tiếng Việt nào. Phần này xây **dữ liệu real** làm nền cho benchmark: video YouTube tiếng Việt được tải, cắt thành clip 4-7 s, kiểm tự động (đúng một mặt, không chuyển cảnh, không jump cut, cùng một người), đo và sửa độ lệch tiếng-hình bằng SyncNet, gán người bằng mặt + giọng, duyệt thủ công, rồi đóng thành **part** cố định: 40 người (20 nam, 20 nữ) × 15-20 clip, chia train / validation / test theo người.

Nguyên tắc xuyên suốt: **nút thắt là số người khác nhau, không phải số clip** (scaling laws [25], identity leakage [26]); và **real không được mang đặc điểm cấu trúc nào khác với fake** sẽ sinh từ chính nó (bài học lối tắt khoảng lặng của Smeu et al. [31]).

Kết quả tới 22/10/2026 ●: 288 input → 259 video / 115,6 giờ / 92 kênh → 16.371 clip ứng viên → **3 part = 120 người × 20 clip = 2.400 clip real**, đã lưu trữ. Năm thí nghiệm kiểm chứng (S00-S04) cho thấy: cắt ở chỗ ngắt hơi tăng tỉ lệ clip có hai mép lặng (đo bằng năng lượng, độc lập với VAD) từ 10,2 % lên 41,4 %; gần như mọi video YouTube có tiếng trễ (trung vị 80 ms, 13 % vượt 125 ms) và sửa được bằng một hằng số mỗi video; giới hạn 150 clip cho nguồn hai người giữ 41 % số clip mà 34/36 cặp người-video vẫn đủ ứng viên; cờ cảnh báo về mặt / giọng khớp với người duyệt còn cờ nghi jump cut thì không. Giới hạn chính: quy mô nhỏ, một loại nguồn, đa dạng buổi quay thấp (60 % người chỉ từ một video), một người duyệt.

| Hạng mục | Giá trị | Mục |
|---|---|---|
| Đơn vị phát hành | **part** = 40 người (20 nam + 20 nữ) × **15-20 clip** = 600-800 clip real | §3 |
| Split | theo người, mỗi giới 12 / 3 / 5 (train 24, validation 6, test 10 người; 60 / 15 / 25 %) | §3.5, §4.6 |
| Clip | 4-7 s, 25 fps, cạnh ngắn ≤ 720, đúng 1 mặt người đang nói, tiếng Việt gốc, tiếng đã sửa về độ lệch 0 | §4.2, §4.7 |
| Nguồn | YouTube, chỉ video ngang, ≥ 720p, ≥ 25 fps; loại nguồn `1` / `2` / `n` | §4.1 |
| Chỉ tiêu cắt | `1`: 30-40 clip / input · `2`: ≤ 150 clip / video · `n`: cắt hết | §4.2.4 |
| Độ lệch tiếng-hình | SyncNet đo theo video khi cắt; dịch tiếng về 0 khi export; video không đo được tự loại | §4.3 |
| Duyệt | người chọn 15-20 clip / người, gán giới, góc mặt lệch ≤ ~30 % | §4.5 |
| Quy mô mục tiêu | ≥ 5 part = 200 người ○ | §3.2 |
| **Hiện có (22/10)** | **3 part = 120 người (60 nam, 60 nữ) × 20 clip = 2.400 clip real** | §5.2 |

---

# Chương 1. Giới thiệu

## 1.1. Bối cảnh

Một **deepfake** là media được tạo hoặc sửa bằng mô hình sinh. Ba dạng thường gặp trong video có người nói: **face swap** (thay mặt), **lip-sync** (vẽ lại vùng miệng cho khớp một đoạn tiếng mới) và **voice cloning** (nhân bản giọng). Lip-sync khó nhận ra vì phần lớn khung hình là thật: danh tính, bối cảnh, chuyển động đầu giữ nguyên, chỉ miệng thay đổi. Kết hợp lip-sync với giọng AI cho phép "đặt lời" vào miệng một người thật.

Bài toán **audio-visual deepfake detection** (phát hiện deepfake nghe-nhìn) cần dữ liệu có cả hình và tiếng. Ngôn ngữ quan trọng vì hình miệng (viseme) và đặc trưng giọng phụ thuộc ngôn ngữ; tiếng Việt là ngôn ngữ có thanh điệu (tone), ở giọng Hà Nội thanh điệu gắn với cả cao độ lẫn chất giọng (phonation) [56] ◐, nên kết quả trên dữ liệu tiếng Anh không chắc chuyển được sang tiếng Việt ○.

## 1.2. Khoảng trống

Tính tới 14/10/2026, **chưa tìm thấy bộ deepfake có hình, dạng lip-sync, tiếng Việt** nào. Tài nguyên tiếng Việt gần nhất (§2.4):
- VietPrism [23]: deepfake **chỉ âm thanh**, 1.262 người đã xác minh, 5 nhóm phương ngữ, gần một nửa có chêm tiếng Anh (code-switching) ✔;
- ViCocktail [22]: tiếng nói thật có hình (audio-visual speech recognition, AVSR), khoảng 269 giờ lọc từ 528,8 giờ YouTube (giữ ~51 %) ✔;
- VoxVietnam [24]: nhận dạng người nói, 2.523 người, 261,5 giờ ✔.

Hai bộ đầu cho thấy lấy dữ liệu người Việt quy mô lớn từ YouTube bằng pipeline tự động là khả thi.

## 1.3. Bài học từ v001 / v002 (của chính dự án)

| Bài học (số đo) | Nguồn | Yêu cầu cho dữ liệu v003 |
|---|---|---|
| Test chỉ 2-3 người, không tính được khoảng tin cậy (CI) theo người; một người đủ đảo kết luận ● | v001 §8.2, v002 §17 | 40 người / part, test 10 người / part; nhiều part |
| Điểm nền lệch theo người: AUC trong từng người 0,95-0,99, gộp chỉ 0,78 ● | v002 §15.4 | số clip mỗi người gần cố định; đủ real để chuẩn hoá theo người |
| Mất cân bằng 15-158 clip mỗi người ● | v001 §9 | 15-20 clip / người |
| Speaker gán tay theo video, chưa kiểm mặt trong clip có đúng người nói | v001 §9 | gán tự động (mặt + giọng) + duyệt; SyncNet bắt lời dẫn đè hình |
| "Độ lệch tiếng-hình sẵn có của real chưa đo" | v001 §9, R2 | đo và sửa (§4.3) |
| Nhánh artifact học dấu vết xử lý ● | v002 §15.6 | real và fake encode giống hệt (v003_generation) |
| Điểm video real: 0,16 với người đã thấy khi train, 0,53 với người mới ● | part1 v001 | thêm người để model không học thuộc người |

## 1.4. Mục tiêu (đo được)

1. **Nhiều người khác nhau, mỗi người đủ clip:** part 40 người (20 + 20), 15-20 clip / người; dài hạn ≥ 5 part.
2. **Split theo người, cố định trước khi sinh fake:** 12 / 3 / 5 người mỗi giới; người chung video luôn đi cùng nhau.
3. **Không lối tắt cấu trúc:** sửa độ lệch tiếng-hình về 0; loại nguồn phải nhân frame (24 fps); chuẩn hoá một định dạng.
4. **Tái lập:** mọi quyết định ghi vào sổ cái (ledger); luật làm clip bị khoá bằng dấu vân tay (cut signature).
5. **Mọi thông số có bằng chứng:** tài liệu đã đọc hoặc số đo trên chính dữ liệu (S00-S04).

## 1.5. Câu hỏi nghiên cứu (phần dữ liệu)

Câu hỏi chính: **làm thế nào xây một bộ video thật tiếng Việt, nhiều người, chia theo người, không mang lối tắt cấu trúc, đủ điều kiện làm nền cho benchmark lip-sync deepfake, trên phần cứng của một sinh viên (CPU, 7,8 GB RAM)?**

| Mã | Câu hỏi con | Trả lời ở |
|---|---|---|
| RQ-D1 | Cần bao nhiêu người, bao nhiêu clip mỗi người, chia thế nào? | §2.5, §3 |
| RQ-D2 | Dữ liệu cuối có đặc điểm gì (nguồn, người, giới, đa dạng buổi quay, part)? | §5.2 (S00) |
| RQ-D3 | Cắt ở chỗ ngắt hơi có làm mép clip rơi vào chỗ lặng nhiều hơn chặt cứng 7 s? | §5.3 (S01) |
| RQ-D4 | Với video hai người, giới hạn 150 clip / video tiết kiệm bao nhiêu, có làm thiếu ứng viên? | §5.4 (S02) |
| RQ-D5 | Video YouTube thật lệch tiếng-hình bao nhiêu; ngưỡng tin cậy có ổn định; sửa theo video có hợp lý? | §5.5 (S03) |
| RQ-D6 | Cờ cảnh báo tự động có khớp với quyết định của người duyệt? | §5.6 (S04) |

## 1.6. Đóng góp

- Bộ real tiếng Việt có hình, chia theo người ở cả ba tập, đã sửa độ lệch tiếng-hình.
- Pipeline thu thập chạy được trên CPU, dừng / chạy tiếp được, có sổ cái, khoá luật và chốt chặn an toàn.
- Mỗi tham số quan trọng gắn với bằng chứng: tài liệu, hoặc thí nghiệm trên chính dữ liệu.
- Ghi đầy đủ quá trình, các lỗi và các lần đổi luật (Chương 6).

---

# Chương 2. Tổng quan tài liệu

## 2.1. Benchmark là gì

Một **benchmark** gồm 4 thứ cố định, công bố cùng nhau: **dữ liệu có nhãn**, **split chính thức**, **protocol chấm**, **kết quả baseline** trên protocol đó, để so được con số giữa các nhóm. FakeAVCeleb [4] không có split chính thức nên số liệu giữa các bài không so được; FaceForensics++ [1], LAV-DF [6], AV-Deepfake1M [7] có split chính thức. Phần dùng để chấm là test; test của v003 đo tổng quát theo 3 trục: **người chưa thấy** (split theo người, file này), **generator chưa thấy** và **giọng chưa thấy** (v003_generation).

## 2.2. Các bộ deepfake nghe-nhìn và lip-sync (khảo sát 14/10, đính chính 18/10)

| Bộ (năm, nơi công bố) | Ngôn ngữ | Người | Real | Fake | Độ dài clip | Cách tạo fake | Nguồn real / đồng ý | Tin |
|---|---|---|---|---|---|---|---|---|
| FakeAVCeleb [4] (2021, NeurIPS D&B) | Anh | **500** (4 nhóm sắc tộc × 100, nam:nữ 50:50) | 500 (1 video / người, từ VoxCeleb2) | ~19.500 | TB **7,8 s** | face swap, lip-sync (Wav2Lip), nhân bản giọng (SV2TTS) | YouTube người nổi tiếng | ✔ |
| KoDF [5] (2021, ICCV) | **Hàn** | **403** | 62.166 | 175.776 | quay ≥ 90 s / clip | 6 cách, có Wav2Lip | **diễn viên trả tiền, có đồng ý** | ✔ |
| LAV-DF [6] (2022) | Anh | **153** (91 / 31 / 31, không trùng người) | 36.431 | 99.873 | ≤ 20 s, 69,6 % < 10 s; đoạn giả TB 0,65 s | sửa lời + TTS + lip-sync, có định vị | VoxCeleb2; CC BY-NC 4.0 | ✔ |
| AV-Deepfake1M [7] (2024, ACM MM) | Anh | **2.068** (train 1.657 / test 411, không trùng) | 286.721 | ~860 nghìn | ngắn | LLM sửa lời + TTS + lip-sync, có định vị | VoxCeleb2 | ✔ |
| AV-Deepfake1M++ [8] (2025) | Anh | **7.109** | 627.936 | | | thêm nhiễu thực tế (36 loại); challenge 2025 | VoxCeleb2, LRS3, EngageNet; EULA chỉ nghiên cứu | ✔ |
| PolyGlotFake [10] (2024) | 7 ngôn ngữ | ? | 766 | 14.472 | ? | TTS, nhân bản giọng, lip-sync | ? | ◐ |
| HAV-DF [11] (2024) | **Hindi** | ~200 | 200 | 308 | 15 s, 1080p30 | face swap, lip-sync, nhân bản giọng | YouTube | ◐ |
| MAVOS-DD [12] (2025) | 8 ngôn ngữ | ? | | 60 % là fake | > 250 giờ | 7 generator / ngôn ngữ; open-set (generator và ngôn ngữ chưa thấy) | ? | ✔ |
| TalkingHeadBench [13] (2025, WACV 2026) | Anh | ? | | | | 6 generator + **2 chỉ ở test**; chuyên gia loại > 60 % mẫu lộ lỗi | ? | ✔ |
| CHN-DF [14] (2026) | **Trung** | ? | | 434.727 mẫu | | nhánh hình và nghe; chéo miền dao động TB 19,6 % | ? | ◐ |
| InDeepFake [15] (2025) | ngôn ngữ Ấn | ? | | | | 7 cách | IEEE DataPort | ◐ |
| AVLips / LipFD [16] (2024, NeurIPS) | Anh | ? | | tới 340.000 mẫu | | MakeItTalk, Wav2Lip, TalkLip, SadTalker | LRS3, FF++, DFDC | ◐ |
| LipSyncTIMIT [17] (2025) | Anh | 43 (VidTIMIT) | | | câu ngắn | 5 generator lip-sync | tình nguyện viên | ◐ |

**Đính chính so với khảo sát 14/10:** DFDC bản phát hành có **960 người** (train 486); con số 3.426 là số người được quay ✔ [3].

### 2.2.1. Mổ xẻ FakeAVCeleb (toàn văn [4] ✔)

| Nhóm | Hình | Tiếng | Số video | Cách tạo |
|---|---|---|---|---|
| RVRA (real) | thật | thật | 500 | 1 video / người từ VoxCeleb2, TB 7,8 s |
| RVFA | thật | giả | 500 | SV2TTS; không lip-sync lại |
| FVRA | giả | thật | 9.000 | FaceSwap, FSGAN, Wav2Lip |
| FVFA | giả | giả | 10.000 | SV2TTS rồi Wav2Lip / ghép vào video đã thay mặt |

Hệ quả khi làm benchmark: (1) chỉ 500 nguồn real (~1,1 giờ) và 39 fake của cùng một real tương quan nhau, nên số mẫu độc lập ≈ 500; (2) lệch lớp 39 : 1 nên accuracy / F1 dễ gây hiểu lầm, phải đọc AUC / AP; (3) phần lớn là thay mặt, lip-sync chỉ là một phần; (4) có lối tắt khoảng lặng đầu clip [31]; (5) không có đối chứng lệch tiếng không AI (sham). Quy trình theo paper: chọn tay real từ VoxCeleb2 (một người, mặt rõ, không mũ / kính / khẩu trang), chép lời bằng IBM Watson, giọng giả SV2TTS, ghép cặp thay mặt cùng sắc tộc và giới, duyệt tay một phần mẫu; cách cắt clip và tham số Wav2Lip: paper không nêu.

## 2.3. Bộ face swap và bộ nền tham chiếu

| Bộ | Người | Real | Ghi chú | Tin |
|---|---|---|---|---|
| FaceForensics++ [1] (2019, ICCV) | ~1.000 | 1.000 video YouTube (509.914 ảnh) | chọn mặt gần chính diện, không che, ≥ 480p, ≥ 280 frame, duyệt tay; 3 mức nén (raw / c23 / c40) | ✔ |
| Celeb-DF v2 [2] (2020, CVPR) | 59 | 590 | khó trong thời gian dài dù ít người | ✔ |
| DFDC [3] (2020) | 960 người phát hành (3.426 được quay) | | diễn viên trả tiền, có đồng ý | ✔ |
| DeeperForensics-1.0 [9] (2020, CVPR) | 100 diễn viên, 1080p | | 35 loại nhiễu | ✔ / ◐ |
| LRS3-TED [19] (2018) | pretrain 5.090 / train-val 4.004 / test 451 | ~407 giờ | câu ≤ 100 ký tự hoặc 6 s; chia theo video | ✔ |
| VoxCeleb [20] (2017) | 1.251 (55 % nam) | > 100 nghìn đoạn | đoạn nói min 4,0 s, TB 8,2 s; âm thanh 16 kHz mono | ✔ |
| VoxCeleb2 [21] (2018) | 6.112 | > 1 triệu đoạn | nguồn real của FakeAVCeleb, LAV-DF, AV-Deepfake1M | ◐ |

Nhận xét: FakeAVCeleb, LAV-DF, AV-Deepfake1M **đều lấy real từ VoxCeleb2**, nên người có thể trùng giữa các bộ. Các bộ lớn có nhiều real mỗi người vì cắt tự động gần như miễn phí, nhưng phần lớn từ cùng vài buổi quay (DFDC ~245 clip / người từ ~14 video gốc; LAV-DF, AV-Deepfake1M cắt từ ít video phỏng vấn) → **nhiều clip không đồng nghĩa đa dạng** ○.

## 2.4. Tài nguyên tiếng Việt

| Bộ | Loại | Quy mô | Đáng chú ý | Tin |
|---|---|---|---|---|
| ViCocktail [22] (2025, Interspeech) | AVSR, tiếng nói thật có hình | ~269 giờ (lọc từ 528,8 giờ YouTube) | bộ AVSR tiếng Việt đầu tiên; pipeline tự động; AV-HuBERT tốt nhất | ✔ |
| VietPrism [23] (2026) | deepfake chỉ âm thanh | 1.262 người; 993,4 giờ thật + > 3.100 giờ giả | 5 nhóm phương ngữ, gần một nửa có chêm tiếng Anh | ✔ |
| VoxVietnam [24] (2025) | nhận dạng người nói | 2.523 người, 261,5 giờ | | ✔ |
| Bộ giọng AI lừa đảo trên Hugging Face | âm thanh | | | ◐ |

## 2.5. Số người quan trọng hơn số clip

| Nghiên cứu | Kết quả | Hệ quả cho thiết kế |
|---|---|---|
| Scaling Laws for Deepfake Detection [25] (2025; 5,8 triệu real / 51 nguồn, 8,8 triệu fake / 102 phương pháp) | 1 − AUC giảm theo luỹ thừa khi tăng **số nguồn real** (α = 0,476), chưa bão hoà; giảm số ảnh real 10 lần chỉ làm AUC 0,886 → 0,879; khuyên không "thu thêm ảnh từ cùng nguồn" | đa dạng người > số clip lặp |
| Implicit Identity Leakage [26] (CVPR 2023) | detector tự học danh tính; nguyên nhân chính của kém tổng quát | ít người → dễ học danh tính |
| Bansal et al. [27] (2017, nhận dạng mặt) | cùng tổng số ảnh: nhiều người tốt hơn nhiều ảnh cho ít người | cùng hướng (bài toán gần) |
| Referee [32], ID-Reveal [33], POI-Forensics [34] | phương pháp tham chiếu người cần ít real (Referee: 1 clip 1-3 s, AUC 99,71 %; ID-Reveal: 6 s) | 3 clip tham chiếu (~16 s) là đủ |
| Layton et al. [30] (USENIX Security 2024) | khuyên tăng media thật; đo tỉ lệ báo nhầm (FPR) cần nhiều real ở test | real mỗi người có lợi cho đo báo nhầm |

**Thống kê** ○: đơn vị độc lập khi chấm là **người** (bootstrap theo speaker), nên độ rộng CI phụ thuộc số người ở test, không phụ thuộc số clip. Trên part1 v001, độ lệch chuẩn ΔAUC giữa người ~0,10 ● → nửa CI 95 % ≈ 2 × 0,10 / √(số người test):

| Người ở test | 5 | 10 | 25 | 50 | 75 | 250 |
|---|---|---|---|---|---|---|
| Nửa CI (ước) | ±0,09 | ±0,06 | ±0,04 | ±0,03 | ±0,023 | ±0,013 |

## 2.6. Công bằng và đa dạng

- **Giới:** GBDF [28] cho thấy dữ liệu lệch giới làm detector kém hơn ở nữ; dữ liệu cân giới công bằng hơn mà không giảm độ chính xác ✔ → 20 nam / 20 nữ mỗi part.
- **Sắc tộc / nhóm:** Trinh & Liu [29] (IJCAI 2021) đánh giá detector trên dữ liệu cân sắc tộc và giới, thấy chênh lệch lỗi giữa các nhóm lớn và FaceForensics++ lệch về một nhóm ◐. Với bộ toàn người Việt, trục đa dạng tương ứng là **phương ngữ, tuổi, điều kiện quay** ○ (VietPrism ghi 5 nhóm phương ngữ [23]).
- **Bộ lớn ghi đa dạng có kiểm soát:** VoxCeleb ghi cân giới (55 % nam) và đa dạng sắc tộc, giọng, nghề, tuổi [20] ✔; FakeAVCeleb cân 4 nhóm sắc tộc × giới [4] ✔.
- **Đa dạng buổi quay:** 20 clip cắt liền từ một video gần như là một mẫu lặp; nên 2-3 video khác buổi (khác ánh sáng, áo, góc máy) ○, khớp khuyến nghị của [25].

## 2.7. Lối tắt (shortcut) và thiên lệch bộ dữ liệu

- **Khoảng lặng đầu clip:** Smeu et al. [31] (CVPR 2025): fake trong FakeAVCeleb và AV-Deepfake1M bắt đầu bằng 25-30 ms lặng; bộ phân loại chỉ dùng khoảng lặng đạt AUC 98,4 / 98,2 %; các model trước đó (chỉ âm thanh và nghe-nhìn) dựa vào đặc điểm này và giảm hiệu năng khi khoảng lặng bị cắt ✔.
- **Hạn chế cấu trúc:** Layton et al. [30] chỉ ra các hạn chế cấu trúc của bộ deepfake ✔.
- **Nén:** FaceForensics++ [1]: XceptionNet (crop mặt) đạt 99,26 % trên raw, 95,73 % ở H.264 c23 ("visually nearly lossless"), 81,00 % ở c40 ✔ → nén làm thay đổi tín hiệu phát hiện; real và fake phải qua **cùng** chuỗi encode.
- **Khoảng cách tới thực tế:** Deepfake-Eval-2024 [18] (45 giờ video lưu hành 2024, 88 website, 52 ngôn ngữ): detector mã nguồn mở giảm ~50 % AUC so với benchmark học thuật ✔.

Hệ quả cho dữ liệu real: mọi đặc điểm có thể tương quan với nhãn (khoảng lặng ở mép, độ lệch tiếng-hình, fps, codec, độ to) phải được kiểm soát ngay từ bước real, vì fake được sinh từ chính clip real.

## 2.8. Cách xây bộ nghe-nhìn từ video trên mạng

- **VoxCeleb** [20] ✔: tìm video YouTube theo tên + từ khoá "interview"; dò mặt mọi frame; **dò ranh giới cảnh bằng so histogram màu giữa các frame liên tiếp**; gom mặt thành track trong mỗi cảnh; **SyncNet** xác định mặt nào đang nói, loại clip lồng tiếng / lời dẫn; ngưỡng đặt rất bảo thủ (precision 1,000, recall 0,613 ở bước này); xác minh danh tính bằng CNN mặt với ngưỡng cao.
- **VoxCeleb2** [21] ◐: pipeline tương tự, đổi từng thành phần, thêm bước bỏ video trùng (YouTube hay có video đăng lại).
- **LRS3-TED** [19] ✔: tiếng và hình có thể lệch **tới khoảng một giây**; dùng SyncNet đồng bộ và loại lời dẫn; căn chỉnh từ bằng forced aligner; câu ≤ 100 ký tự hoặc 6 s (train-val, test); **chia split theo video**.
- **ViCocktail** [22] ✔: pipeline tự động cho tiếng Việt, giữ ~51 % thời lượng.

Pipeline của dự án kế thừa trực tiếp: histogram cho chuyển cảnh, SyncNet cho đồng bộ và phát hiện lời dẫn đè hình, bỏ video trùng; khác ở chỗ chia split **theo người** (chặt hơn theo video) và **sửa** độ lệch thay vì chỉ lọc.

## 2.9. Nền tảng tín hiệu (căn cứ cho các thông số)

### 2.9.1. Âm thanh

- **Dải tần tiếng nói và tần số lấy mẫu:** điện thoại kiểu cũ (narrowband, G.711) truyền 300-3400 Hz; chuẩn wideband ITU-T G.722 truyền 50-7000 Hz với tần số lấy mẫu 16 kHz [46] ◐. Theo định lý Nyquist, 16 kHz giữ được tới 8 kHz. Các công cụ nền dùng 16 kHz: VoxCeleb chuyển mọi âm thanh về mono 16-bit 16 kHz [20] ✔; Wav2Lip dùng `sample_rate=16000` [36] ✔; Silero VAD hỗ trợ 8 kHz và 16 kHz [49] ✔. → Pipeline đưa âm thanh về **16 kHz mono** cho VAD / SyncNet / embedding giọng; clip xuất giữ **AAC 48 kHz** để không mất thông tin cho các bước sau ○.
- **SyncNet** dùng 13 hệ số MFCC tính ở 100 Hz (20 bước cho 0,2 s) [35] ✔.
- **VAD** (voice activity detection, dò tiếng nói): Silero VAD [49] ✔ ra xác suất có tiếng cho từng khúc 512 mẫu ở 16 kHz (32 ms); mặc định ngưỡng 0,5, khoảng lặng tối thiểu 100 ms, đệm 30 ms, đoạn nói tối thiểu 250 ms. Pipeline dùng ngưỡng 0,5 và khúc 32 ms như mặc định, nhưng nối khoảng lặng < 0,6 s và đệm 0,15 s (§4.2.1) để không chặt câu thành mảnh quá nhỏ ○.
- **Độ to (loudness):** đo theo ITU-R BS.1770; khuyến nghị EBU R128 chuẩn hoá chương trình về −23 LUFS [47, 48] ◐. Video YouTube từ nhiều kênh có độ to khác nhau; pipeline **không chuẩn hoá độ to** (giữ nguyên tín hiệu gốc). Nếu bước sinh làm độ to của fake khác real một cách có hệ thống thì đó là một lối tắt cùng họ với khoảng lặng [31] → là một mục của kiểm lối tắt sau khi sinh (§5.8).
- **Thanh điệu tiếng Việt** [56] ◐: thanh điệu gắn với cao độ và chất giọng; các bước xử lý (resample 16 kHz chỉ để phân tích, AAC 48 kHz cho clip) không đổi cao độ ○.

### 2.9.2. Hình

- **25 fps:** SyncNet học trên cửa sổ 0,2 s = 5 frame, dữ liệu BBC News [35] ✔; Wav2Lip dùng `fps=25`, cửa sổ 5 frame, train trên LRS2 [36] ✔. → Mọi clip chuẩn hoá **25 fps CFR** (constant frame rate).
- **Vì sao loại 24 fps:** đưa 24 lên 25 fps phải lặp 1 frame mỗi giây → môi "khựng" đều đặn, một đặc điểm chỉ có ở real (fake sinh từ clip đã chuẩn hoá) → nguy cơ lối tắt; luật kế thừa từ v001 ○. Nguồn 30 / 50 / 60 fps được bỏ bớt frame khi đưa về 25 fps; hệ quả này áp như nhau cho mọi clip real và fake sinh sau ○.
- **Cỡ mặt:** Wav2Lip làm việc ở ảnh mặt 96 × 96 (`img_size=96`) [36] ✔ → ngưỡng cạnh hộp mặt ≥ 48 px ở khung 640 (≈ 96 px ở 720p).
- **Nén:** CRF (constant rate factor) của x264 có thang 0-51, mặc định 23; giá trị 17-18 thường được coi là "visually lossless hoặc gần như vậy" [53] ◐; FaceForensics++ gọi c23 là "visually nearly lossless" [1] ✔. → Clip encode **H.264 CRF 18**.
- **Chuyển cảnh:** dò bằng so histogram màu giữa frame liên tiếp như VoxCeleb [20] ✔.

### 2.9.3. Đồng bộ tiếng-hình

- **SyncNet** [35] ✔: hai nhánh (âm thanh MFCC, hình 5 frame vùng miệng); với mỗi mẫu tính khoảng cách giữa một đặc trưng hình 5 frame với mọi đặc trưng âm thanh trong khoảng ±1 s, độ lệch đúng là nơi khoảng cách nhỏ nhất; lấy trung bình trên cửa sổ 10 / 100 frame cho kết quả gần hoàn hảo; dùng được cho active speaker detection và phát hiện lồng tiếng.
- **Ngưỡng cảm nhận:** ITU-R BT.1359 (thử nghiệm chủ quan): người bắt đầu nhận ra khi tiếng **sớm ~45 ms** hoặc **trễ ~125 ms**, chấp nhận được tới khoảng +90 / −185 ms (có nguồn ghi −190) [45] ◐; Wav2Lip ghi người xem phát hiện được đoạn lệch ≈ 0,05-0,1 s [36] ✔.
- **Các bộ đo mới** (độ chính xác đoán đúng ±1 frame trên LRS2, theo IC-SyncNet và PVSync [40, 41] ✔):

| Model | Cửa sổ 5 frame | Cửa sổ 15 frame | Trọng số công khai |
|---|---|---|---|
| SyncNet [35] (2016) | 75,8 % | 96,1 % | có |
| Perfect Match [37] (2019) | 88,1 % | 99,1 % | chưa rõ |
| VocaLiST [38] (2022) | 92,8 % | 99,8 % | không lấy được (theo PVSync) |
| ModEFormer [39] (2023) | 94,5 % | 99,8 % | không thấy |
| IC-SyncNet [40] (2024) | 96,5 % | 99,9 % | không thấy |
| StableSyncNet / LatentSync [42] | 94 % HDTF (tác giả); 90,6 % ở cửa sổ 16 frame theo PVSync | | có |
| PVSync [41] (10/2026) | 95,1 % (16 frame, bộ test riêng) | | chưa phát hành |

- **"Điểm 0" giữa các công cụ lệch ~1 frame:** ModEFormer [39] thấy LRS2 lệch ~1 frame so với LRS3 / VoxCeleb2 dù đều đã sửa bằng SyncNet ✔.

## 2.10. Tài liệu hoá bộ dữ liệu

*Datasheets for Datasets* [54] ◐ đề xuất mỗi bộ dữ liệu kèm một "datasheet" về động cơ, thành phần, quy trình thu thập, tiền xử lý, cách dùng, phân phối, bảo trì. *Data Cards* [55] ◐ (Google, FAccT 2022) là bản tóm tắt có cấu trúc các sự thật cần cho người dùng bộ dữ liệu trong suốt vòng đời. File này đi theo hai khung đó: Chương 3-4 (thành phần, thu thập, tiền xử lý), Chương 6 (quá trình, lỗi), Chương 7 (cách dùng, đạo đức, giới hạn), §4.8 (phân phối, bảo trì).

## 2.11. Thế nào là benchmark tốt (tổng hợp từ các bộ trên)

| # | Nguyên tắc | Bằng chứng | Áp dụng ở v003 (phần dữ liệu) |
|---|---|---|---|
| 1 | Chia split theo người | AV-Deepfake1M [7], LAV-DF [6], LRS3 [19] (theo video) | split theo người, nhóm người chung video đi cùng (§4.6) |
| 2 | Generator / giọng chưa thấy ở test | MAVOS-DD [12], TalkingHeadBench [13] | v003_generation |
| 3 | Không lối tắt: real và fake qua cùng xử lý | Smeu [31], FF++ [1] | cắt + encode chuẩn hoá; sửa độ lệch; loại 24 fps; kiểm sau khi sinh |
| 4 | Nhiễu thực tế ở test | DeeperForensics [9], AV-Deepfake1M++ [8] | chưa có (việc mở) |
| 5 | Lọc chất lượng | TalkingHeadBench [13], FakeAVCeleb [4] (duyệt tay) | duyệt tay mọi clip real |
| 6 | Đa dạng có kiểm soát + metadata | FakeAVCeleb [4], VietPrism [23], VoxCeleb [20] | cân giới; manifest ghi nguồn, độ lệch, cờ |
| 7 | Pháp lý rõ ràng | DFDC [3], KoDF [5] (đồng ý); LAV-DF CC BY-NC; AV-Deepfake1M++ EULA | dữ liệu YouTube chỉ nghiên cứu nội bộ; nếu phát hành chỉ ID + mốc thời gian + script |
| 8 | Protocol + baseline + mã đánh giá | AV-Deepfake1M++ challenge | v003_model |

## 2.12. Khoảng trống và định vị

1. Chưa có bộ lip-sync deepfake nghe-nhìn tiếng Việt.
2. Nhiều bộ dùng chung real VoxCeleb2 → người có thể trùng; nhiều clip mỗi người nhưng ít buổi quay.
3. Split theo người chưa phổ biến ở mọi tập (AV-Deepfake1M++ chỉ ở test).
4. Độ lệch tiếng-hình sẵn có của real không được nhắc trong các bộ deepfake đã đọc, dù LRS3 ghi lệch tới ~1 s.
5. Lối tắt thường chỉ bị phát hiện sau khi phát hành.

| Tiêu chí | AV-Deepfake1M++ [8] ✔ | v003 | Nhận xét |
|---|---|---|---|
| Quy mô | 2,05 triệu video, 4.656 giờ, 7.109 người | hiện có 3 part = 120 người, 2.400 clip real ●; 5 part = 200 người ○ | **kém xa**, giới hạn chính |
| Nguồn real | VoxCeleb2, LRS3, EngageNet | YouTube tiếng Việt, 92 kênh | **tiếng Việt** là điểm mới |
| Chia theo người | chỉ ở test | **cả 3 tập** | chặt hơn |
| Sửa độ lệch tiếng-hình real | không nhắc | **có** | thêm |
| Kiểm lối tắt | bản trước bị phát hiện có | xử lý từ bước dữ liệu + kiểm sau khi sinh | thêm |
| Nhiễu thực tế ở test | 36 loại | chưa có | thiếu |

---

# Chương 3. Thiết kế bộ dữ liệu

## 3.1. Nguyên tắc

1. **Người trước, clip sau:** mọi luật chọn, cắt, duyệt nhằm có nhiều người, mỗi người đủ clip từ nhiều buổi quay (§2.5).
2. **Real không mang đặc điểm cấu trúc khác fake** (§2.7): độ lệch tiếng-hình, fps, codec, mép clip.
3. **Split cố định ở bước dữ liệu:** chỉ bước này biết ai là ai, ai xuất hiện chung video nào; chia muộn thì khó bảo đảm cùng người / cùng cảnh quay không lọt vào cả train lẫn test.
4. **Mọi quyết định có dấu vết:** sổ cái ledger, khoá luật, nhật ký.

## 3.2. Quy mô: từ khảo sát tới part

Khảo sát 14/10 đề xuất ba mức (số giờ / dung lượng ước từ part1 v001 ○):

| Hạng mục | Mức S | Mức M (đề xuất 14/10) | Mức L |
|---|---|---|---|
| Speaker | 100 | 300 | 1.000 |
| Train / val / test | 60 / 15 / 25 | 180 / 45 / 75 | 600 / 150 / 250 |
| Clip real (≈ 20 / người) | 2.000 | 6.000 | 20.000 |
| Video cần tải (ước) | ~35-40 giờ | ~100-125 giờ | ~350-400 giờ |

Lý do ưu tiên số người: clip của cùng người tương quan, số mẫu độc lập ≈ số người (và số video); FakeAVCeleb có ~1,1 giờ real nhưng 500 người; LRS3 test 451 người chỉ ~1 giờ [19]. Mức M tương đương KoDF (403) và FakeAVCeleb (500). Khảo sát cũng ghi: báo kết quả theo giới, tuổi, vùng miền cần ≥ 10-15 người mỗi nhóm ○.

Ngày 17-18/10 thiết kế chuyển sang **part** để xây dần: mỗi part 40 người (20 + 20), part tạo xong bị khoá; mục tiêu ≥ 5 part = 200 người (giữa LAV-DF 153 và KoDF 403) ○. Mục tiêu "đủ 200 / 300 speaker thì dừng" bị bỏ (17/10). Dung lượng đo thật: raw 720p ≈ 0,25 GB / giờ ● (khảo sát ước 0,68 GB / giờ cho 1080p), nên batch ≤ 20 giờ chỉ ~4-6 GB.

## 3.3. Số người mỗi part

**40 người (20 nam + 20 nữ).** Trước là 50 (25 + 25), split 15 / 4 / 6; user chốt 40 ngày 18/10, split 12 / 3 / 5 giữ tỉ lệ 60 / 15 / 25 %. Test 10 người / part → nửa CI ~±0,06; 5 part → 50 người test → ~±0,03 (§2.5) ○.

## 3.4. Số clip mỗi người

| Phương án | Kết luận |
|---|---|
| 10 clip / người | thiết kế ban đầu; mỏng: model học từ real có ~3 giờ cho 4 part; 3 clip tham chiếu chiếm 30 %; FPR đo trên ít real |
| 20 clip / người, cố định | chọn 18/10 |
| **15-20 clip / người** | **chọn 19/10** kèm biện pháp chống lệch (dưới) |
| 20-60 clip ngẫu nhiên | bác: người nhiều clip lấn át (identity leakage [26]); AUC gộp bị người nhiều clip chi phối; cân giới / split theo clip lệch |
| 10 clip sinh fake + 10 clip chỉ làm real | bác (v003_generation §4) |

Vì sao không 10: nút thắt là **tìm người**; đã có một người thì thêm 10 clip gần như không tốn công thu thập; 20 vẫn xa vùng bão hoà của các bộ lớn (88-139 real mỗi người, §2.12). Quy tắc ưu tiên: **không đổi người lấy clip**. Khảo sát 14/10 ghi thêm: clip thứ 21 trở đi của cùng người gần như không thêm thông tin (part1 v001 có người 157 clip) ●.

**19/10 đổi sang 15-20** (user chốt, đồng bộ cho mọi part): người tốt chỉ có 15-19 clip dùng được không bị rơi; với luật cắt mới, clip ngắn hơn nên số cửa sổ không phải nút thắt. Biện pháp để số clip khác nhau không làm lệch kết quả:

| Rủi ro | Biện pháp | Ở đâu |
|---|---|---|
| Người nhiều clip lấn át khi train | lấy mẫu theo 1 / số clip của người đó | `training.speaker_balance` |
| AUC gộp bị người nhiều clip chi phối | báo thêm AUC cân theo người (trọng số 1 / số file của người đó) | `evaluate.speaker_balanced_auc` |
| Split lệch số clip | ưu tiên chỉ tiêu người theo giới; trong giới hạn đó cân tổng clip mỗi split | `parts.assign_splits` |
| Số clip mang thông tin nhãn | không thể: mọi clip sinh cùng số file, tỉ lệ fake / real mỗi người như nhau | v003_generation §4 |
| Chênh tỉ trọng giữa người | tối đa 20 / 15 = 1,33× (v001 từng 10×) | `min / max_clips_per_speaker` |

Thực tế ●: **cả 120 người đều chọn đúng 20 clip**, nên các biện pháp trên tương đương lấy mẫu đều; vẫn giữ cho part sau. Khuyến nghị 15-20 clip từ 2-3 video khác nhau chỉ đạt ở 40 % người (§5.2).

## 3.5. Split và lý do chia ở bước dữ liệu

Split **theo người**, cân giới: mỗi giới 12 / 3 / 5. Người xuất hiện chung một video đi cùng nhau vào một part và một split (nhóm liên thông), để cùng cảnh quay (phông nền, ánh sáng, micro, có khi lẫn tiếng nhau) không lọt vào cả train lẫn test; LRS3 chia theo video vì cùng lý do [19]. Bước sinh cũng phụ thuộc split (tiếng thay thế lấy từ cùng người, cùng split; generator chỉ ở test chạy trên clip test), nên split phải cố định trước khi sinh. "Tự lọc" vẫn làm được nhờ manifest đủ thông tin; thứ duy nhất không được đổi là split.

## 3.6. Ba loại nguồn

| Loại | Nghĩa | Chọn video | Cắt |
|---|---|---|---|
| `1` | một người nói trước camera | kênh / playlist: trải theo ngày đăng tới ≥ 20 phút, 3-10 video; link lẻ: chính video | 30-40 clip / input, rải đều |
| `2` (từ 20/10) | hai người (phỏng vấn 1-1, podcast) | như `n` | ≤ 150 clip / video, rải đều |
| `n` | nhiều người (talkshow) | kênh: 15 video mới nhất; playlist: hỏi tối đa 100 mục rồi lấy 15 mới nhất; link lẻ: chính video | cắt hết |

Lý do và số đo cho `2`: §4.2.4 và §5.4.

---

# Chương 4. Pipeline và phương pháp

## 4.0. Tổng quan

```
mỗi batch:  01 plan → 02 download → 03 cut (+ SyncNet) → 04 identify → 05 review
            sau 04: 08 archive raw N → up Drive → --uploaded → 09 clean raw N
đủ người:   06 parts → 07 export N (dịch tiếng) → up Kaggle (private)
            → 08 archive part N → up Drive → --uploaded → 09 clean part N → 09 clean pool
bất cứ lúc: 10 status
```

Mọi bước chạy trên một máy Windows (CPU 16 luồng, RAM 7,8 GB, PyTorch bản CPU), đọc và ghi một **ledger** SQLite chung (`data/ledger/ledger.sqlite`): URL, video, lý do loại, batch, clip, độ lệch, speaker, quyết định duyệt, part, nơi lưu. Sau mỗi lệnh có ghi (kể cả khi lỗi hay Ctrl+C) ledger tự sao lưu (từ 20/10 giữ mọi bản; trước đó giữ 30 bản) và xuất CSV. Không có gì chạy ngầm: giới hạn batch áp khi chạy `01_plan`, part chỉ tạo khi chạy `06_parts`. Mỗi đơn vị việc (một video, một clip, một quyết định) commit riêng nên dừng giữa chừng rồi chạy lại là tiếp tục.

Phần mềm: Python 3.11; yt-dlp [57] (cần Node.js); ffmpeg / PyAV; ONNX Runtime (YuNet [50], Silero VAD [49], SFace [51], WeSpeaker [52]); PyTorch CPU (SyncNet [35]); SQLite; FastAPI (trang duyệt).

## 4.1. Chọn và tải video (01 plan, 02 download)

| Luật | Giá trị | Lý do / bằng chứng |
|---|---|---|
| Đầu vào | `urls.csv`: `url, speakers (1 / 2 / n), note` | `note` tự do; `speakers` của dòng đã plan không đổi được (chống đổi nhầm) |
| Nguồn 1 người | trải theo ngày đăng tới ≥ 20 phút, 3-10 video, ưu tiên ≤ 90 phút | ≥ 3 buổi quay cho đa dạng (§2.6); 20 phút đủ cho 30-40 clip (nguồn tốt giữ 60-100 % cửa sổ ●) |
| Nguồn 2 / n | 15 video mới nhất; link lẻ: chính video | |
| Chỉ tab `/videos`, chỉ video **ngang** | loại ở plan (metadata) và sau khi tải (file) | user chốt 17/10: video dọc hay có phụ đề to đè miệng, cắt cảnh 1-3 s, filter làm đẹp |
| Chất lượng | cạnh ngắn ≥ 720 px, **≥ 25 fps**, tiếng Việt gốc (loại track ghi rõ ngôn ngữ khác), không trùng | 25 fps và 24 fps: §2.9.2; trùng: YouTube hay đăng lại (VoxCeleb2 [21]) |
| Tải | bản 720p, chỉ HTTPS | luồng HLS tải tiếp có thể mất header mp4 ● |
| Batch raw | **≤ 20 giờ và ≤ 16 GB**, giới hạn nào đến trước; batch đã bắt đầu tải thì khoá | ổ máy không chứa hết raw |
| Lỗi tải | 403 / 429 / 5xx / timeout thử lại sau 5 / 20 / 60 s, mỗi lần xin link mới; "xác minh bot" thì **dừng ngay** (cả plan lẫn download) | 403 là link tạm bị từ chối ●; gọi tiếp khi bị chặn chỉ bị chặn lâu hơn |

URL nhập theo **từng link video riêng** nên mỗi video là một input (thiết kế ban đầu giả định một input là một kênh 3-10 video); điều này làm đa dạng buổi quay thấp (§5.2) và ảnh hưởng chỉ tiêu cắt (§4.2.4).

## 4.2. Cắt clip ứng viên (03 cut)

### 4.2.1. Đoạn nói và cửa sổ

- Silero VAD [49]: xác suất ≥ 0,5 là có tiếng, khúc 32 ms; nối đoạn cách nhau < 0,6 s; đệm 0,15 s hai đầu; bỏ đoạn < 0,25 s.
- Cửa sổ **4-7 s** trên lưới 25 fps, nằm gọn trong một đoạn nói, không chồng nhau, không vắt qua chuyển cảnh. 4 s là đoạn ngắn nhất VoxCeleb giữ [20] ✔; LRS3 giới hạn câu ở 6 s [19] ✔; FakeAVCeleb TB 7,8 s [4] ✔; đủ chỗ cho đoạn partial 0,3-3 s của bước sinh và vừa encoder hiện có ○.
- Đoạn nói dài hơn 7 s được chia **ở chỗ ngắt dài nhất** trong khoảng [đầu + 4 s, đầu + 7 s] (phần còn lại phải đủ ≥ 4 s; không đủ chỗ cho 2 cửa sổ thì lấy 1, bỏ đuôi), làm tròn theo lưới 25 fps; không có chỗ ngắt thì cắt ở mức dài nhất cho phép. "Chỗ ngắt" (`pause_spans`) = khúc xác suất < 0,5 nằm bên trong đoạn nói. Luật này có từ 19/10, thay luật chặt cứng mỗi 7 s (§4.2.5, §5.3).

### 4.2.2. Kiểm từng frame

| Lớp | Điều kiện | Lý do loại |
|---|---|---|
| Đúng 1 mặt | YuNet [50] độ tin ≥ 0,85; 0 hoặc ≥ 2 mặt đều loại | `face_frame` |
| Mặt đủ to | cạnh hộp mặt **≥ 48 px** ở khung cạnh dài 640 (≈ 96 px ở 720p, cỡ ảnh mặt của Wav2Lip [36]) | `face_frame` |
| Hình học mặt | hộp nằm trọn trong khung; hai khoé miệng ở nửa dưới mặt, cách nhau ≥ 12 % bề ngang mặt | `face_frame` |
| Mặt liền mạch | tâm không nhảy quá 0,6 lần cỡ mặt, cỡ không đổi quá 2 lần | `face_frame` |
| Hụt mặt ngắn | **≤ 3 frame hụt, không quá 2 liền nhau**; landmark nội suy qua chỗ hụt; cờ `face_gap` | |
| Không chuyển cảnh | tương quan histogram màu giữa hai frame ≥ 0,55 (cách dò của VoxCeleb [20]; ngưỡng chọn thực nghiệm ○) | `scene_change` |
| Đủ frame | đọc được ≥ 95 % frame | `coverage` |
| Jump cut | một cú giật đơn lẻ của điểm mốc mặt kèm thay đổi ảnh mặt hoặc âm thanh → loại; chỉ một dấu hiệu → giữ, cờ `jump_suspect` | `jump_cut` |
| Cùng người (nguồn 2 / n) | quét mặt sơ bộ 4 fps, ≥ 90 % điểm quét đạt; 5 điểm (đầu, ngẫu nhiên, giữa, ngẫu nhiên, cuối) SFace [51] cos ≥ 0,5 với mặt trung bình | `face_scan`, `identity_change` |
| Độ lệch tiếng-hình | §4.3 | `sync_unreliable` |

**Hai ngưỡng nới ngày 17/10, có số đo** ● (batch thử 5 video, 25 cửa sổ mỗi video):

| Thay đổi | Trước → sau | Bằng chứng |
|---|---|---|
| Cỡ mặt tối thiểu | 64 px (≈ 128 px ở 720p) → **48 px** | podcast quay vừa người có mặt ~110-125 px ra **0 clip**; ở 48 px frame đạt tăng 12 % → 76 %; hạ xuống 40 px chỉ cứu ô webcam nhỏ, quá nhỏ để vẽ môi → không hạ |
| Frame hụt | 1 frame hụt là bỏ → **≤ 3, không quá 2 liền** | 9/25 cửa sổ podcast chỉ hụt 1-2 frame (nghiêng đầu, mờ chuyển động); cửa sổ đạt 7 → 16/25 |

Cắt lại batch thử: 296 clip (podcast 0 → 241) ●; những gì vẫn bị loại là loại **đúng**: cảnh minh hoạ, lời dẫn đè hình, quay màn hình với ô webcam nhỏ, vlog giữa đám đông. Lời khuyên chọn nguồn: mặt nhìn camera, quay vừa người hoặc cận.

### 4.2.3. Những gì bước cắt không kiểm

| Không kiểm | Xử lý |
|---|---|
| Góc quay mặt (luật hình học chỉ chặn mặt gần nghiêng hẳn, ~60-70° trở lên ○) | người duyệt chọn clip lệch ≤ ~30 % (user chốt 18/10, không thêm code đo góc) |
| Người trong hình có phải người đang nói | bắt gián tiếp ở mức video bằng SyncNet (§4.3) |
| Nhạc nền to, chữ đè miệng, tay che miệng, filter làm đẹp, hình mờ | người duyệt loại |
| Độ to | không chuẩn hoá (§2.9.1); kiểm sau khi sinh |

Chủ ý từ v001: **không chọn từng clip theo điểm sync**, vì như vậy các detector dựa vào sync trông mạnh giả tạo.

### 4.2.4. Luật số lượng và rải đều

| Thời điểm | Chỉ tiêu nguồn 1 người | Lý do |
|---|---|---|
| đến 18/10 sáng | 30-50 clip / người | thiết kế 10 clip / người |
| 18/10 | 40-60 | lên 20 clip / người |
| **18/10 tối (hiện hành)** | **30-40 mỗi input**: 30 khi ≤ 5 phút tiếng nói, tăng tuyến tính tới 40 ở ≥ 20 phút | 9 video đầu batch 1: clip qua bộ lọc đã sạch ●; URL theo từng video nên người có 3-4 video có 160-240 ứng viên cho 20 clip |

- Chỉ tiêu là **mức tối đa** cố lấy; chỗ bắt buộc là duyệt (đủ 15 clip mới vào part).
- Thời lượng nguồn 20 phút là đủ: nguồn tốt giữ 60-100 % cửa sổ ●; 40-60 clip chỉ cần ~6-9 phút tiếng nói. (Có lúc tính nhầm cần 30 phút, đã sửa.)
- **Rải đều** (`spread_order`): trục thời gian chia N khúc (N = chỉ tiêu); mỗi vòng lấy cửa sổ gần tâm khúc nhất của các khúc chưa có clip; khúc hỏng thử cửa sổ kế; khi mọi khúc còn cửa sổ đã có clip thì lấy thêm từ khúc còn cửa sổ → một khúc không có clip tốt **không làm hụt** chỉ tiêu. Trần thử = 2,5 × chỉ tiêu + 5. Không làm chậm vì mỗi cửa sổ được ffmpeg nhảy thẳng tới (`-ss`) và giải mã riêng.
- **Nguồn `2`, từ batch 3 (20/10):** nguồn `n` cắt hết cho ra ~280 clip / video 60 phút (bảng dưới ●) trong khi hai người chỉ cần ~2 × 20 clip được chọn. `2` kiểm giống `n` nhưng lấy **tối đa 150 clip / video** rải đều (~75 / người, dư ~4 lần; nguồn `1` dư ~2 lần) - dư hơn vì clip hai người dễ bị loại khi duyệt (cắt sang người kia, nói chồng). Rủi ro: camera chia không đều (khách lên hình nhiều hơn MC); MC thường lặp lại qua nhiều tập nên identify cộng dồn clip. Batch 1-2 giữ nguyên luật cũ; user đánh dấu 28 link batch 3-4 là `2` (giữ 4 link `n`), ledger đổi nhãn đúng 28 input chưa cắt, có ghi sự kiện. Kiểm chứng: §5.4.

7 video `n` đầu của batch 2 (cắt hết) ●:

| Video | Dài | Clip / cửa sổ kiểm | Thời gian cắt |
|---|---|---|---|
| Video 1 | 81 phút | 362 / 370 | 32,5 phút |
| Video 2 | 64 phút | 432 / 479 | 34,6 phút |
| Video 3 | 56 phút | 321 / 333 | 28,9 phút |
| Video 4 | 57 phút | 299 / 321 | 27,4 phút |
| Video 5 | 58 phút | 343 / 360 | 29,7 phút |
| Video 6 | 33 phút | 146 / 151 | 14,3 phút |
| Video 7 | 26 phút | 61 / 138 | 10,6 phút |

### 4.2.5. Điểm cắt ở chỗ ngắt (từ 19/10)

**Vấn đề** ●: luật cũ nối chỗ lặng < 0,6 s rồi chặt đoạn nói dài **cứng mỗi 7 s**. Trên batch 1 (84 video, 2.392 clip): 84 % clip dài đúng 7 s; chỉ **8 %** clip có cả 2 mép rơi vào chỗ ngắt; **61 %** có cả 2 mép bên trong đoạn nói (có thể cắt ngang từ); 38 % đoạn nói sau VAD dài hơn 7 s. Không ảnh hưởng trực tiếp tới phát hiện (fake cùng mép với real), nhưng làm giọng AI đọc lại cả câu gặp từ cụt và real gần như không có khoảng lặng đầu clip.

**Thử trước khi áp** ● (6 video batch 1, 6 kênh, ~1,8 giờ tiếng nói; mép "sạch" = mọi khúc 32 ms quanh mép có xác suất < 0,5):

| | Luật cũ | Luật mới (bản thử) | Luật mới (code thật) |
|---|---|---|---|
| Cửa sổ | 930 | 1.187 | 1.176 |
| Cả 2 mép sạch | 74 (8 %) | 745 (63 %) | 723 (61 %) |
| Cả 2 mép giữa tiếng | 533 (57 %) | 77 (6 %) | 86 (7 %) |
| Độ dài | 87 % đúng 7 s | p10 / p50 / p90 = 4,2 / 5,4 / 6,6 s | 4,2 / 5,4 / 6,7 s |
| Tổng thời lượng phủ | 6.305 s | 6.398 s | |

- Phần chưa sạch là người **nói một mạch** suốt 4-7 s; nới chỗ ngắt từ ≥ 96 ms xuống ≥ 64 ms không đổi tỉ lệ (63 %).
- Chi phí gần như bằng 0 (xác suất VAD đã tính sẵn).
- Giới hạn: chỗ ngắt chưa chắc là hết câu; VAD có thể tụt ở phụ âm vô thanh. Cắt **tròn câu** cần ASR có mốc từ + dấu câu (PhoWhisper / Whisper / WhisperX), nặng hơn nhiều trên CPU → không làm.
- Tham khảo: code nhóm Capstone dùng Silero mặc định (ngắt ≥ 100 ms, đệm 30 ms), giữ nguyên đoạn 2-12 s làm một clip, đoạn dài hơn cắt cửa sổ 5 s bước 4 s (chồng 1 s).
- Áp dụng → đổi luật cắt → xoá toàn bộ dữ liệu batch 1 (đã duyệt 38 người), tải và cắt lại (19/10). Đo lại trên toàn bộ dữ liệu: §5.3.

### 4.2.6. Chuẩn hoá clip

25 fps CFR, cạnh ngắn ≤ 720, H.264 CRF 18 [53], AAC 48 kHz. Hình và tiếng trừ **cùng** một mốc thời gian → giữ nguyên độ lệch gốc để đo ở bước sau. Ảnh dải 5 mặt (`pool/faces/`) lưu để duyệt; trung bình embedding 5 mặt là template của clip cho identify.

## 4.3. Độ lệch tiếng-hình sẵn có của real

### 4.3.1. Vấn đề

Generator lip-sync vẽ môi khớp tiếng ở độ lệch 0 (theo "điểm 0" của bộ đo sync chúng dùng khi train); real của một kênh lệch d frame thì có độ lệch d. Hệ quả:
1. **Lối tắt "khớp quá mức":** fake lệch 0, real cùng clip lệch d; detector có thể học "khớp hoàn hảo → fake", tức học đặc điểm của **kênh**; báo nhầm dao động theo người.
2. **Nhãn mismatch nhiễu:** real lệch vài frame vẫn mang nhãn "không lệch".
3. **Ảnh hưởng không đều** giữa các họ model (học từ real coi độ lệch 0 là cặp đúng; dò độ lệch ±15 thì bền với lệch cố định nhưng có thể dùng "lệch 0, đỉnh nhọn" làm lối tắt).

Hai kết quả cũ **có thể** liên quan (giả thuyết, chưa kiểm): v001 Wav2Lip giữ tiếng gốc khớp tiếng hơn cả real (nhánh sync AUC 0,436 < 0,5); v002 một người có điểm real cao bất thường (0,73). Mục tiêu: **real và fake có cùng phân bố độ lệch**; độ lệch chỉ có hại khi đi kèm nhãn hoặc đi kèm người.

| Cách | Ưu | Nhược |
|---|---|---|
| Lọc video lệch ≥ N frame | đơn giản | mất dữ liệu; lệch 1-2 frame vẫn lọt; phần giữ thiên về kênh "sạch" |
| **Sửa theo video** (đo độ lệch cố định của video, dịch tiếng về 0) | giữ dữ liệu; real về 0 như fake; LRS3 / VoxCeleb dùng SyncNet [19, 20] | phụ thuộc độ chính xác bộ đo (±1 frame) |
| Sinh fake mang cùng độ lệch với real | không đụng real | phức tạp; real vẫn lệch; nhãn mismatch vẫn nhiễu |

### 4.3.2. Đo thử (17-18/10)

- Bộ đo: SyncNet [35] (code `syncnet_python`, MIT), crop mặt như `run_pipeline.py`, CPU, độ lệch −15…+15 frame.
- **Phép dịch biết trước** ●: dịch tiếng 30 clip k ∈ {−6, −3, −1, +1, +3, +6} frame → đo lại đúng k ở **180/180** lần (độ phân giải 1 frame; chỉ kiểm phần chênh, không kiểm "điểm 0" tuyệt đối).
- Theo video, batch thử ● (95 clip, 5 video):

| Video | Loại | Độ lệch cả video | Clip đồng ý ±1 frame | Độ nhọn đỉnh |
|---|---|---|---|---|
| Video A | podcast nhiều người | tiếng trễ 2 frame | 40/40 | 5,9 |
| Video B | 1 người | trễ 1 | 12/12 | 5,9 |
| Video C | 1 người | trễ 1 | 14/14 | 5,2 |
| Video D | 1 người (lời dẫn đè cảnh) | - | 8/22 | 1,6 |
| Video E | 1 người (vlog, ghép cảnh) | - | 2/7 | 1,9 |

- Batch 1, 9 video đầu ●: 7 `ok` (trễ 1-3 frame), 2 `unreliable`.
- Kết luận: video quay mặt người nói lệch **ổn định**; video lời dẫn đè hình cho độ lệch lung tung → bộ đo đồng thời **phát hiện mặt trên hình không phải người nói** ở mức video (lỗ hổng v001).

### 4.3.3. Chọn bộ đo

- SyncNet kém ở cửa sổ ngắn nhưng ngang các model mới khi gộp đủ dài (§2.9.3); pipeline gộp 15-20 clip × 100-175 frame mỗi video → **đủ dùng**, khớp kiểm 180/180.
- **Không dùng StableSyncNet làm bộ đo chính**: nó đi kèm một generator lip-sync (LatentSync [42]); sửa real theo "điểm 0" của nó có thể làm real giống fake của chính generator đó.
- **Không dùng AV-HuBERT [58] ở local**: fairseq bản cũ (Python 3.10, biên dịch extension), script cài chỉ cho Linux, bản Large ~300 triệu tham số nặng với máy CPU 7,8 GB RAM → để làm ý kiến thứ hai trên Kaggle.
- **Không hạ Python xuống 3.10**: không giúp AV-HuBERT (fairseq cần torch 2.0, phần còn lại cần torch ≥ 2.6); yt-dlp sắp bỏ 3.10 [57].
- **Đo trong bước cắt**, không đo trước: đo cần đoạn "có tiếng + đúng 1 mặt rõ", chính là việc bước cắt làm; đo trên raw gặp cảnh minh hoạ, quay màn hình, chuyển cảnh và ra kết quả rác.

### 4.3.4. Quyết định

1. **Đo trong `03_cut`**: sau mỗi video, SyncNet trên tối đa 20 clip rải đều; độ lệch video = đỉnh đường cong khoảng cách trung bình. Trạng thái:
   - `ok`: độ nhọn đỉnh (`conf`) ≥ 3,0, ≥ 80 % clip đồng ý ±1 frame (`agree`), ≥ 3 clip;
   - `few`: đạt nhưng < 3 clip → vẫn sửa, cờ `sync_few` (thêm vì test cho thấy video ra 2 clip bị loại oan);
   - `unreliable`: còn lại → mọi clip tự `reject`, cờ `sync_unreliable`; người duyệt vẫn chọn lại được (khi đó tiếng **không** được sửa, manifest ghi `av_sync_status: unreliable`).
   Ngưỡng 3,0 đặt giữa nhóm tin được (5,2-5,9) và nhóm hỏng (1,6-1,9) ●; kiểm độ nhạy trên toàn bộ dữ liệu: §5.5.
2. **Sửa khi export:** dịch tiếng mọi clip theo độ lệch video về 0 (chép luồng hình, encode lại tiếng, đệm lặng ≤ vài frame ở mép); manifest ghi `audio_lag_frames`, `av_sync_status`. Clip trong pool giữ bản gốc → đổi cách đo về sau chỉ cần export lại. Thử trên 8 clip ●: dịch đúng số frame, sau sửa còn trong ±1 frame quanh 0.
3. **Lệch lớn nhưng ổn định thì sửa, không bỏ.** Chỉ bỏ video **không đo được**.
4. **Không trái nguyên tắc v001**: chỉ dịch một **hằng số cho cả video**, không chọn từng clip theo điểm khớp.
5. **Augment khi train**: dịch tiếng so với hình k ∈ [−1, +2] frame (dương = tiếng trễ), đều cho real và fake, nhãn giữ nguyên. Căn cứ: (a) BT.1359 [45] ◐: −1…+2 frame (sớm 40 ms … trễ 80 ms) nằm trong vùng chưa nhận ra; (b) "điểm 0" giữa các công cụ lệch ~1 frame [39]; (c) sai số đo ±0,5-1 frame; (d) augment lớn có hại: DFD-Lab [43] augment lệch tới ±200 ms làm giảm hiệu năng ngoài miền ở cả 3 model, PVSync [41] coi clip lệch > 3 frame so với mode là dữ liệu bẩn. User yêu cầu con số phải có căn cứ nghiên cứu; thí nghiệm chọn biên độ để sau (v003_model).
6. **Không nghiên cứu lệch thời gian thuần** ("sham_shift") trong v003 (user chốt 17/10).

## 4.4. Gán người (04 identify)

- Xử lý theo batch, so khớp với **mọi** người đã có.
- Nguồn `1`: mọi clip mặc định cùng người; clip lệch mặt / giọng chỉ gắn cờ `face_outlier` / `voice_outlier`.
- Nguồn `2` / `n`: gom cụm theo mặt (SFace [51]); cụm < 3 clip không khớp ai thì để chưa gán.
- So với người đã có (ảnh và embedding trong registry): trong kênh trước, rồi toàn bộ. Rất giống → gán; hơi giống → speaker mới kèm **gợi ý gộp**; giọng (WeSpeaker [52]) mâu thuẫn thì không tự gộp.
- Ngưỡng ban đầu ước lượng (SFace cos 0,55 / 0,35) ○; `--calibrate` sau vài chục người để chỉnh theo quyết định gộp / tách thật.
- Batch thử ●: nhận ra cùng một người qua 2 nguồn khác nhau; podcast tách đúng 2 người (44 và 197 clip).
- Người đã duyệt xong mà còn < 15 clip được mở lại khi batch sau khớp thêm clip.

## 4.5. Duyệt thủ công (05 review)

- Trang duyệt chạy local (cổng 8001). Mỗi speaker: kiểm cụm đúng một người (gộp **G** / "Khác người" / **R** chuyển clip), **gán giới M / F**, **chọn 15-20 clip** (**S** chọn, **X** loại), **N** chốt, **E** loại người. Quyết định ghi ngay vào ledger.
- **Giới tính đánh tay**: đoán từ mặt / giọng dễ sai, mà part cần đúng 20 + 20; giới tính là **cảm nhận của người duyệt**, không phải thông tin tự khai.
- Tiêu chí chọn clip: góc mặt lệch ≤ ~30 %, không che miệng, không chữ đè miệng, không nhạc nền to, đúng người. Mặt nghiêng tới ~45° vẫn dùng được về kỹ thuật; 45-60° rủi ro cho generator; > 60° dò mặt hụt ○ (quan sát, chưa đo).
- Rủi ro đã ghi ○: mặt càng nghiêng generator càng hay lỗi và job lỗi bị bỏ, nên real có thể nhiều mặt nghiêng hơn fake → lối tắt "nghiêng → real"; kiểm sau khi sinh.
- Chỉ clip `keep` đi tiếp; `pending` (chưa xem) và `reject` (đã xem, hỏng) được xử lý như nhau.
- Người chưa đủ 15 clip: N là để **chờ**, không đánh dấu xong. "Done" chưa phải khoá: sửa được tới khi `06_parts` đưa vào part; muốn đưa về chờ dùng API `POST /api/speaker/<id>` với `{"review": "pending"}`.
- Nút "Khác người" chỉ bỏ **gợi ý gộp**; người đó vẫn là speaker riêng, vẫn có clip.
- Duyệt song song được với cắt / gán batch khác (ledger chờ khoá tối đa 60 s).

## 4.6. Tạo part và split (06 parts)

- Part tạo khi có **đủ 20 nam + 20 nữ** đã duyệt xong, đã gán giới, có 15-20 clip được chọn, theo thứ tự duyệt xong; đủ cho mấy part thì tạo bấy nhiêu; chưa đủ thì không tạo gì, không báo lỗi.
- Split theo người, cân giới 12+12 / 3+3 / 5+5; trong giới hạn đó cân tổng clip giữa các split. Người xuất hiện chung một video đi cùng nhau vào một part và một split (nhóm liên thông qua video chung).
- **Từ 22/10:** người chưa duyệt xong (pending) **không giữ nhóm lại**; người đã duyệt xong trong nhóm vẫn vào part. Nếu người pending sau này duyệt xong, nhóm của họ đã có người trong part cũ nên vào `blocked_groups` (không tự xếp, xử lý tay) → không có cảnh quay nào rơi vào split khác với nhóm. (Trước đó, một người pending làm cả nhóm chờ; xem sự cố §6.3.)
- Nhóm liên thông có thể rất lớn: một MC xuất hiện ở 13 video nối mọi khách mời của chương trình thành một nhóm. Part 3 có 3 nhóm 15 người (10 nam + 5 nữ), 10 người (6 nam + 4 nữ), 5 người (1 nam + 4 nữ); nhóm phải vào nguyên một split nên part 3 lệch nhẹ: validation 2 nam + 3 nữ, test 6 nam + 5 nữ. Nếu nhóm vượt 12 người một giới thì không vừa split nào → khi đó cân nhắc loại MC để tách nhóm.
- Trang duyệt đếm "sẵn cho part" theo người, **không** tính luật nhóm, nên có thể báo 20/20 trong khi `06_parts` chưa tạo được part.
- Part tạo xong bị **khoá** (`split-lock.json`). Nên nhập URL dư 25-30 % và cân nam / nữ ngay từ danh sách, vì người rơi rụng.

## 4.7. Một clip real hoàn chỉnh

| Thuộc tính | Giá trị / điều kiện | Bước bảo đảm |
|---|---|---|
| Nguồn | video YouTube ngang, ≥ 720p, ≥ 25 fps, tiếng Việt gốc, không trùng | 01 plan, 02 download |
| Độ dài | 4-7 s, trong một đoạn nói, mép ở chỗ ngắt khi có | 03 cut (VAD) |
| Hình | đúng một mặt người đang nói ở mọi frame (hụt ≤ 3, ≤ 2 liền), mặt ≥ 48 px ở khung 640, không chuyển cảnh, không jump cut chắc chắn; nguồn `2` / `n`: 5 điểm cùng một người | 03 cut |
| Định dạng | MP4, 25 fps CFR, cạnh ngắn ≤ 720, H.264 CRF 18, AAC 48 kHz | 03 cut, 07 export |
| Đồng bộ | tiếng đã dịch về 0 theo độ lệch SyncNet của video (trừ clip `unreliable` được chọn lại, ghi rõ `av_sync_status`) | 03 cut (đo), 07 export (sửa) |
| Người | thuộc một speaker; người duyệt xác nhận đúng người, gán giới | 04 identify, 05 review |
| Chất lượng | được chọn (keep) theo tiêu chí §4.5 | 05 review |
| Split | đúng một split theo người, khoá cùng part | 06 parts |
| Mô tả | một dòng `manifest.jsonl` (Phụ lục C) | 07 export |
| Truy vết | mọi thông tin trên + lý do loại các clip khác nằm trong ledger | ledger |

## 4.8. Export, lưu trữ, dọn máy, tái hiện

Thư mục `exports/vn-av-df-data/vn-av-df-data-partN/` (07 không zip) lên Kaggle dataset **private**; zip `vn-av-df-data-partN.zip` cho Drive do `08_archive part N` tạo.

| File | Nội dung |
|---|---|
| `clips/<clip_id>.mp4` | 600-800 clip real, tiếng đã dịch cho khớp môi |
| `manifest.jsonl` | mỗi dòng một clip (Phụ lục C) |
| `speakers.csv` | 40 người: giới, split, kênh, số video |
| `split-lock.json` | khoá split theo người |
| `dataset_info.json`, `quality_report.json` | thống kê part, hash manifest, kết quả kiểm định dạng |

- Export **chép** clip từ pool (độ lệch 0: chép nguyên file; khác 0: file mới, chép luồng hình, encode lại tiếng); pool giữ bản gốc.
- Khi export: kiểm đúng số clip đã ghi lúc tạo part, hash từng clip khớp ledger, kiểm định dạng và manifest.

| Lệnh | Việc | Điều kiện |
|---|---|---|
| `08_archive raw N` / `part N` | zip raw batch N (+ `SHA256SUMS.txt`) / zip thư mục part N (từ 20/10 cùng một kiểu; trước đó 07 tự zip part) | raw: tải + cắt + identify xong, đủ file raw; part: 07 xong |
| `08_archive … --uploaded` | ghi ledger: đã lên Drive | user tự up zip + thư mục `ledger/` |
| `09_clean raw N` | xoá raw + zip raw (clip trong pool giữ) | đã `--uploaded` |
| `09_clean part N` | xoá export + zip + **mọi** clip pool của 40 người trong part | đã `--uploaded` |
| `09_clean pool` | xoá clip **không chọn** của speaker đã duyệt xong / bị loại | chỉ clip thuộc batch raw đã lên Drive; clip của người còn chờ luôn giữ |

- Ba lệnh xoá tách riêng, đều bắt buộc đã đánh dấu upload (user đề xuất 18/10). Ledger, `registry/`, ảnh mặt, `urls.csv` không bao giờ bị lệnh nào xoá.
- **Cần up lên Drive**: zip raw, zip part, cả thư mục `data/ledger/` (sau mỗi lần ghi), `data/registry/` (ảnh đại diện, không tạo lại được), `data/inputs/urls.csv`, `data_pipeline/checkpoints/` (trọng số mô hình; hash nằm trong cut signature). `pool/clips/` không cần (cắt lại được); `pool/faces/` tuỳ chọn.
- **Tái hiện clip đã xoá:** ledger giữ mọi dòng clip (`clip_id` tính cố định từ video + mốc giây, `start_s`, `end_s`, hash, người, quyết định, cờ, nơi lưu) và video (URL, hash raw, zip raw trên Drive, `cut_info`) → cắt lại từ raw cùng mốc và cùng thông số; trùng nội dung và mốc, chưa chắc trùng hash từng byte nếu ffmpeg khác phiên bản. Chưa có lệnh tự động.

## 4.9. Vận hành: tốc độ, tuần tự, khoá luật

| Số đo ● | Giá trị |
|---|---|
| Batch thử (2,6 giờ video, có SyncNet) | ~45 phút; podcast 84 phút (quét mặt cả video 4 fps) chiếm 37 phút |
| SyncNet | ~4,5 s mỗi clip đo, ~1-2 phút mỗi video |
| VAD trước khi cắt | batch 1: 914 phút audio trong ~7,5 phút (không in tiến độ nên trông như đứng) |
| RAM | một tiến trình cao nhất ~0,75-1,05 GB (lúc SyncNet), phần lớn ~0,3 GB |
| Cắt song song 3 video | 459 s so với 593 s tuần tự: chỉ nhanh **1,3×** → giữ tuần tự (user chốt 17/10) |
| Toàn bộ (251 video) | ~30,6 giờ CPU; giây cắt / phút video (trung vị): `1` 13,3 · `2` 22,6 · `n` 29,6 |

- **Luật cắt bị khoá:** cut signature (config cắt + mã nguồn `cut.py`, `jumpcut.py`, `vad.py`, `face.py`, `syncnet.py` + hash mô hình) ghi vào ledger; đổi luật sau khi đã có clip thì lệnh từ chối chạy. Thử luật mới dùng workspace riêng (`VNAV_DATA=data_trial`). **Từ 20/10** chữ ký chỉ khoá cách làm ra một clip; luật số lượng (`single`, `pair`, `quota.py`) nằm ngoài (đổi chỉ tiêu không làm khác chất lượng clip); đóng dấu lại một lần khi tách (sự kiện `pair`: `b90ec5a90295 → 6ae7ba9ede0a`).
- Phần mềm: 40 test tự động cho pipeline dữ liệu; ruff sạch.

---

# Chương 5. Thí nghiệm và kết quả

## 5.1. Phương pháp chung

Năm nghiên cứu S00-S04 chạy ngày 22/10/2026 trên ledger ở chế độ **chỉ đọc**; kết quả chỉ chứa số tổng hợp. Thông số cũ được tái hiện bằng chính code hiện hành khi có thể (luật chặt cứng 7 s = hàm chia cửa sổ không truyền chỗ ngắt). Thước đo nào chỉ mô tả (dùng chính tín hiệu mà luật tối ưu) được ghi rõ. Code thí nghiệm đã được bỏ khỏi repo (22/10) theo yêu cầu; cách đo được mô tả đủ để viết lại.

## 5.2. S00 - Thống kê toàn bộ dữ liệu (trả lời RQ-D2)

**Phễu**

| Giai đoạn | Số ● |
|---|---|
| Input (`urls.csv`, đều là link video lẻ) | 288: `1` 239 · `2` 29 · `n` 20; input không còn video nào đạt: 29 |
| Video bị loại khi plan | 29: 24 fps 25 · track tiếng ngôn ngữ khác 2 · < 720p 2; trùng 0 |
| Video được chọn | **259 video, 115,6 giờ, 92 kênh**; `1` 212 / 69,1 giờ · `2` 29 / 28,4 · `n` 18 / 18,1 |
| Độ dài video p10 / p50 / p90 | 6,3 / 16,4 / 67,4 phút |
| Năm đăng | 2026: 187 (72 %) · 2025: 40 · 2024: 13 · 2023: 10 · 2022: 3 · 2021: 4 · 2020: 2 |
| Định dạng gốc (720p) | AV1 ≥ 84 % · H.264 ≥ 14 %; fps gốc 30 / 29,97 ~66 % · 25 ~20 % · 50 / 60 ~13 % |
| Đã tải và cắt (batch 1-6) | 251 video, 28,3 GB raw (≈ 0,25 GB / giờ); batch 7 (8 video, 2,1 giờ) chưa tải |
| Clip ứng viên | **16.371**, độ dài p10 / p50 / p90 = 4,16 / 5,36 / 6,72 s |
| Người tạo tự động | 156; khớp người đã có 144 lượt; gợi ý gộp 92 |
| Người (22/10) | duyệt xong 121 (120 trong part + 1 nữ dư) · loại 10 · chờ 25 |
| Clip được chọn trong part | 2.400 (đúng 20 / người) |

**Batch raw**

| Batch | Video | Giờ | GB |
|---|---|---|---|
| 1 | 80 | 18,5 | 5,4 |
| 2 | 32 | 19,0 | 4,7 |
| 3 | 36 | 19,6 | 4,5 |
| 4 | 17 | 17,7 | 3,8 |
| 5 | 22 | 19,7 | 4,3 |
| 6 | 64 | 18,9 | 5,6 |
| 7 | 8 | 2,1 | chưa tải |

Raw batch 1-6 đã zip và lên Drive, đã xoá ở máy (22/10).

**Cắt theo luật số lượng**

| Luật | Video | Phút video | Cửa sổ kiểm | Clip giữ | Tỉ lệ giữ | Clip / video | Giây cắt / phút video |
|---|---|---|---|---|---|---|---|
| `1` | 204 | 4.020 | 13.586 | 6.227 | 46 % | 31 | 13,3 |
| `2` | 29 | 1.705 | 4.410 | 4.032 | 91 % | 139 | 22,6 |
| `n` | 18 | 1.085 | 6.528 | 6.112 | 94 % | 340 | 29,6 |
| Tổng | 251 | 6.810 | | 16.371 | | | |

- Nguồn `2` / `n` lọc trước bằng quét mặt 4 fps (9.777 cửa sổ loại ở bước này, không tính vào "cửa sổ kiểm") nên tỉ lệ giữ cao; nguồn `1` không quét trước nên mọi cửa sổ có tiếng đều kiểm kỹ, kể cả cảnh minh hoạ.
- Lý do loại ở bước kiểm từng frame: mặt không đạt ở một frame 5.524 · chuyển cảnh 1.387 · jump cut 1.183 · đổi người giữa clip 59.
- Cờ trên clip ứng viên: `jump_suspect` 3.810 · `face_gap` 399 · `sync_unreliable` 198 · `face_outlier` 64 · `voice_outlier` 49 · `sync_few` 2.
- Cách gán clip: người mới 10.470 · khớp trong kênh 5.504 · khớp toàn cục 396 · chưa gán 1.

**Người trong part (120)**

| Số video có clip được chọn | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Người | **72 (60 %)** | 37 | 9 | 1 | 1 |

- 117 / 120 người chỉ từ 1 kênh; clip được chọn trên một cặp (người, video): trung vị 11, tối đa 20; ứng viên / người p10 / p50 / p90 = 32 / 77 / 236.

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
| Zip | 0,84 GB | 0,68 GB | 0,65 GB |

Tỉ lệ clip train / validation / test của part 1-2 đúng bằng tỉ lệ người 24 / 6 / 10 (mọi người 20 clip).

## 5.3. S01 - Luật đặt điểm cắt (trả lời RQ-D3)

**Thiết kế.** Với mỗi video đã cắt (251), lấy lại đúng đoạn nói và chỗ ngắt VAD đã lưu trong `cut_info`, chạy hàm chia cửa sổ hai lần: không truyền chỗ ngắt (= luật cũ, cắt ở mức dài nhất 7 s) và có chỗ ngắt (= luật mới); bỏ qua ranh giới chuyển cảnh cho cả hai. Hai thước đo mép:
- **A. Sạch theo VAD** (mọi video): mép trùng đầu / cuối đoạn nói (±0,04 s; đầu / cuối đoạn đã đệm 0,15 s lặng) hoặc nằm trong một chỗ ngắt. **Chỉ mô tả** vì gần với điều luật mới tối ưu.
- **B. Lặng theo năng lượng** (độc lập với VAD): năng lượng RMS của khúc 32 ms chứa mép và hai khúc kề (±1 khúc, cùng lưới 16 kHz) đều thấp hơn trung vị năng lượng các khúc có tiếng của chính video đó ít nhất 20 dB (hoặc 15 dB). Mẫu: 30 video raw chọn ngẫu nhiên (seed 0), 10 video mỗi loại nguồn.

| | Chặt cứng 7 s (cũ) | Cắt ở chỗ ngắt (mới) |
|---|---|---|
| Cửa sổ | 55.957 | 64.145 |
| Giờ tiếng nói được phủ | 98,1 | 97,1 |
| Dài đúng 7 s | 65,7 % | 4,8 % |
| Độ dài p10 / p50 / p90 (s) | 4,04 / 7,0 / 7,0 | 4,2 / 5,4 / 6,8 |
| A. Cả 2 mép sạch theo VAD (mô tả) | 15,0 % | 87,1 % |
| A. Không mép nào sạch | 43,5 % | 0,5 % |
| **B. Cả 2 mép lặng, −20 dB** | **10,2 %** | **41,4 %** |
| B. Cả 2 mép lặng, −15 dB | 12,2 % | 50,3 % |

**Kết luận.** Theo thước đo độc lập B, tỉ lệ clip có cả 2 mép lặng tăng ~4 lần trong khi tổng tiếng nói được phủ gần như không đổi. Số thử 19/10 (8 % → 61 %) dùng định nghĩa chặt hơn trên 6 video, không so trực tiếp được.

## 5.4. S02 - Giới hạn 150 clip cho nguồn hai người (trả lời RQ-D4)

**Thiết kế.** Mô phỏng trên 18 video nguồn nhiều người đã cắt hết (6.112 clip; clip / video p10 / p50 / p90 = 61 / 361 / 475): lấy clip đã giữ làm cửa sổ, chọn N clip bằng đúng thuật toán rải đều của pipeline, đếm ứng viên còn lại của từng cặp (người, video) theo speaker đã gán. Gần đúng: luật thật rải trên cửa sổ trước khi kiểm (tỉ lệ giữ nguồn nhiều người ~94 %).

| Giới hạn | Clip còn lại | Cặp (người duyệt xong, video) còn ≥ 20 ứng viên | Ứng viên / cặp p10 / p50 / p90 |
|---|---|---|---|
| không (cắt hết) | 6.112 (100 %) | - | - |
| 100 | 1.684 (27,6 %) | 32 / 36 | 18 / 37 / 80 |
| **150** | **2.480 (40,6 %)** | **34 / 36** | 32 / 56 / 114 |
| 200 | 3.230 (52,8 %) | 35 / 36 | 46 / 69 / 154 |

Kết quả thật của luật `2` (29 video): clip / video p10 / p50 / p90 = 85 / 150 / 150, 25 video chạm 150, ~22 phút cắt / video; 52 cặp (người duyệt xong, video) có ứng viên 27 / 70 / 121.

**Kết luận.** 150 giữ ~41 % số clip mà 34/36 cặp vẫn đủ 20 ứng viên (hai cặp hụt vẫn có thể đủ nhờ video khác của cùng người).

## 5.5. S03 - Độ lệch tiếng-hình (trả lời RQ-D5)

| Độ lệch (frame 25 fps, dương = tiếng trễ) | −4 | 0 | 1 | 2 | 3 | 4 | 7 |
|---|---|---|---|---|---|---|---|
| Số video (`ok` + `few`, 246) | 1 | 5 | 50 | 86 | 73 | 30 | 1 |

- Trạng thái: `ok` 245 · `few` 1 · `unreliable` 4 trên 250 video đo được (1 video không còn clip để đo).
- Tiếng trễ ở **97,6 %** video; ms p10 / p50 / p90 = 40 / **80** / 160; **13,0 %** video lệch > 125 ms (ngưỡng bắt đầu nhận ra khi tiếng trễ theo BT.1359 [45] ◐). Hai video ngoại lệ (−4, +7 frame) vẫn được sửa vì đo tin cậy.
- `conf` p10 / p50 / p90 = 4,76 / 6,16 / 7,00 (ngưỡng 3,0); `agree` = 0,90 / 1,00 / 1,00 (ngưỡng 0,8).
- Clip so với cả video (4.715 lần đo): trùng đúng 76,2 %, chênh 1 frame 21,5 %, 2-3 frame 1,1 %, > 3 frame 1,2 %.

| `min_conf` \\ `min_agree` | 0,7 | 0,8 | 0,9 |
|---|---|---|---|
| 2,5 | 1 | 4 | 17 |
| 3,0 (hiện hành) | 1 | **4** | 17 |
| 3,5 | 1 | 4 | 17 |
| 4,0 | 4 | 6 | 18 |

(Ô = số video `unreliable`.) Bốn video `unreliable`: (conf, agree, số clip đo) = (4,56; 0,75; 20), (3,84; 0,73; 11), (4,12; 0,75; 4), (2,83; 0,50; 20).

- 4 clip `unreliable` được chọn lại nằm trong part 1 (cùng một người; video có 15/20 clip đo trễ 2-3 frame, 5 clip ±15 frame ở mép khoảng tìm) → không được dịch tiếng, nhiều khả năng còn trễ ~3 frame; user quyết định giữ (20/10).

**Kết luận.** Gần như mọi video YouTube có tiếng trễ 1-4 frame; nếu không sửa, real khác fake một cách có hệ thống. Độ lệch ổn định trong video (98 % lần đo trong ±1 frame) → sửa bằng một hằng số là hợp lý. `conf` nằm trong vùng ổn định; `agree` là thông số nhạy. Dự án đo độ lớn và độ ổn định, không đo nguyên nhân của độ trễ.

## 5.6. S04 - Cờ cảnh báo so với quyết định duyệt (trả lời RQ-D6)

Trên 13.809 clip của 120 người đã duyệt xong (keep 2.400 · reject 708 · chưa xem 10.701):

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
- Người duyệt không gộp / chuyển clip lần nào (0 merge, 0 move) dù có 92 gợi ý gộp; 3.574 thao tác duyệt ghi trong ledger tới 21/10 (chọn 2.515 · loại 722 · bỏ chọn 64 · cập nhật người 273).

**Kết luận.** Cờ về mặt / giọng khớp với việc người duyệt bỏ clip; `jump_suspect` có tỉ lệ chọn ngang clip không cờ và có ở ~24 % clip được chọn (571 / 2.429) → ở mức hiện tại gần như không mang thông tin cho người duyệt (bộ dò quá nhạy, hoặc người duyệt không thấy jump cut nhỏ).

## 5.7. Trả lời câu hỏi nghiên cứu

| Mã | Trả lời ngắn |
|---|---|
| RQ-D1 | 40 người / part (20 + 20), 15-20 clip / người, split theo người 12 / 3 / 5; ưu tiên thêm người hơn thêm clip (§2.5, §3) |
| RQ-D2 | 120 người × 20 clip từ 251 video / 92 kênh; đa dạng buổi quay thấp (60 % người 1 video); giới lệch theo loại nguồn (§5.2) |
| RQ-D3 | có: mép lặng 10,2 % → 41,4 % theo thước đo độc lập (§5.3) |
| RQ-D4 | 150 giữ ~41 % clip, 34/36 cặp đủ ứng viên (§5.4) |
| RQ-D5 | trung vị 80 ms, 13 % > 125 ms; ổn định trong video; `conf` ổn định, `agree` nhạy (§5.5) |
| RQ-D6 | cờ mặt / giọng có ích; `jump_suspect` không (§5.6) |

## 5.8. Gợi ý thí nghiệm (chưa làm)

| Mã | Thiết kế | Ghi chú |
|---|---|---|
| S05 | ASR có mốc thời gian từng từ (PhoWhisper / WhisperX) trên ~20 video raw; đo % mép nằm bên trong một từ cho hai luật cắt | bằng chứng độc lập thứ hai cho S01; nên dùng GPU; cần raw (tải lại zip từ Drive) |
| S01 mở rộng | thước đo năng lượng trên ~90 video | ~30 phút CPU; cần raw |
| S06 | đo lại độ lệch bằng một mô hình đồng bộ khác (ví dụ AV-HuBERT [58]) trên cùng clip, so với SyncNet | kiểm "điểm 0"; nếu lệch hệ thống ~1 frame thì ghi thành giới hạn |
| S07 | gắn nhãn tay ~200 clip (100 có, 100 không `jump_suspect`) | precision / recall bộ dò jump cut (~2 giờ xem) |
| S08 | người thứ hai duyệt lại ~10 người | Cohen's kappa cho chọn clip và giới tính |
| S09 | kiểm lối tắt sau khi sinh: khoảng lặng đầu / cuối, độ to (BS.1770), độ dài, fps; AUC từng đặc điểm phải trong [0,3; 0,7] | bắt buộc trước khi train (v003_generation) |

---

# Chương 6. Quá trình thực hiện, lỗi và thay đổi

## 6.1. Dòng thời gian

| Ngày | Việc |
|---|---|
| 30/09 | quy ước thư mục: repo làm việc / nhật ký nghiên cứu |
| 14/10 | đóng v001 / v002; khảo sát bộ dữ liệu (mức S / M / L, ưu tiên số người) |
| 15/10 | thiết kế v003: sáu ô, part, split theo người |
| 17/10 | batch thử 5 video: nới cỡ mặt 64 → 48 px, cho hụt ≤ 3 frame; thêm SyncNet (kiểm 180/180); xử lý 403 / bot; nguồn 1 người ≥ 20 phút, 3-10 video, chỉ video ngang; quay lại cắt tuần tự |
| 18/10 | part 40 người, split 12 / 3 / 5; 20 clip / người; chỉ tiêu cắt 30-40 rải đều; batch 1 thật; xoá hết và tải lại từ đầu |
| 19/10 | đo mép clip (8 % sạch) → cắt ở chỗ ngắt; xoá và cắt lại batch 1; 15-20 clip / người + biện pháp cân theo người; bỏ 10 URL hỏng, thêm 8 |
| 20/10 | part 1; zip đồng bộ (07 chỉ export, 08 zip); lỡ xoá raw batch 1; plan dừng khi gặp bot; giữ mọi backup ledger; loại nguồn `2` từ batch 3; tách luật số lượng khỏi chữ ký; thêm 77 + 14 + 106 URL |
| 21/10 | cắt + gán batch 3-5; duyệt tới 120 người |
| 22/10 | batch 6; thí nghiệm S00-S04; sửa luật nhóm pending; part 2, 3; zip + lên Drive raw 2-6; dọn máy; viết báo cáo |

## 6.2. Các lần đổi luật

| Ngày | Thay đổi | Căn cứ |
|---|---|---|
| 17/10 | cỡ mặt 64 → 48 px; hụt ≤ 3 frame | podcast ra 0 clip; 9/25 cửa sổ chỉ hụt 1-2 frame ● |
| 17/10 | đo độ lệch bằng SyncNet trong bước cắt | real lệch sẵn; kiểm 180/180 ● |
| 17/10 | bỏ mục tiêu "đủ 200 / 300 speaker thì dừng" | chỉ là con số hiển thị |
| 18/10 | part 50 → 40 người; split 12 / 3 / 5 | chốt quy mô |
| 18/10 | 10 → 20 clip / người; chỉ tiêu cắt 40-60 → 30-40 | §3.4; clip qua bộ lọc đã sạch ● |
| 18/10 | tách 3 lệnh xoá (raw / part / pool), đều phải qua đánh dấu upload | user đề xuất |
| 19/10 | chặt cứng 7 s → cắt ở chỗ ngắt; cắt lại batch 1 | 8 % mép sạch ● (§4.2.5) |
| 19/10 | 20 cố định → 15-20 clip | không bỏ người tốt 15-19 clip; kèm biện pháp cân |
| 20/10 | 07 không zip, 08 zip cả raw và part | đồng bộ quy trình (user) |
| 20/10 | plan dừng ngay khi gặp bot check | gọi tiếp chỉ bị chặn lâu hơn |
| 20/10 | giữ mọi backup ledger (trước 30 bản) | không tự xoá lịch sử |
| 20/10 | loại nguồn `2` (≤ 150 / video) từ batch 3; luật số lượng ra khỏi cut signature | ~280 clip / video nhiều người ● (§5.4) |
| 20/10 | đóng gói raw bị chặn khi thiếu file raw | sự cố §6.3 |
| 22/10 | người pending không giữ nhóm; người đến sau vào `blocked_groups` | sự cố §6.3 |

## 6.3. Lỗi và sự cố đã gặp, cách xử lý, bài học

| # | Sự cố | Nguyên nhân | Xử lý | Bài học |
|---|---|---|---|---|
| 1 | Luật cắt chặt cứng 7 s (18-19/10) | chưa đo mép clip | đo (84 % đúng 7 s, 8 % mép sạch), đổi luật, xoá và cắt lại batch 1 (38 người đã duyệt) | đo trước khi tin một luật; đổi luật giữa chừng tốn thật → cut signature |
| 2 | Tính nhầm cần 30 phút nguồn / người (18/10) | ước sai tỉ lệ giữ | sửa: 20 phút là đủ | kiểm phép tính bằng số đo |
| 3 | Số người DFDC ghi 3.426 (khảo sát 14/10) | lẫn số người được quay với số phát hành | đính chính 960 người phát hành | đọc bảng của paper gốc |
| 4 | Lỡ chạy `09_clean raw 1` trước khi up Drive (20/10) | báo `--uploaded` sớm | khôi phục ledger từ bản sau export, tải lại 80 video, zip lại; thêm chặn đóng gói khi thiếu file raw; raw batch 1 trên Drive là lần tải thứ hai (hash khác lần đầu, mốc và clip không đổi), ledger ghi sự kiện `resync` | thao tác xoá cần chốt chặn; chỉ báo `--uploaded` sau khi kiểm file trên Drive |
| 5 | YouTube chặn bot khi plan (20/10), gọi tiếp 77 lần | plan chưa dừng khi gặp bot | plan dừng ngay như download | dừng sớm khi lỗi lặp lại hệ thống |
| 6 | Cookie Edge lỗi giải mã DPAPI | Chromium mã hoá cookie kiểu mới, yt-dlp không đọc được | dùng Firefox hoặc chờ hết chặn | |
| 7 | 4 clip `unreliable` vào part 1 không được sửa độ lệch | người duyệt chọn lại clip của video đo không tin cậy | ghi rõ trong manifest; user chốt không sửa code | ghi thành giới hạn |
| 8 | Part 3 không tạo được dù trang duyệt báo 20/20 (22/10) | người pending chung video (qua MC) giữ cả nhóm; trang duyệt không tính luật nhóm; "Khác người" chỉ bỏ gợi ý gộp | người pending không giữ nhóm nữa; người đến sau vào `blocked_groups` | luật đúng (chống rò cảnh quay) nhưng phải tính tới người chưa duyệt |
| 9 | Part 3 lệch split 1 nam | nhóm 10 người (6 nam) phải vào nguyên một split | chấp nhận; ghi giới hạn | nhóm lớn quanh MC làm split khó cân |
| 10 | Bộ đếm trang duyệt và `06_parts` khác nhau | trang duyệt đếm theo người | ghi rõ trong tài liệu | |
| 11 | Ledger không nằm trong zip | thiết kế | user tự up thư mục `ledger/` sau mỗi lần ghi | |

## 6.4. Nhật ký câu hỏi và quyết định

| Ngày | Câu hỏi / việc | Trả lời / quyết định |
|---|---|---|
| 14/10 | Khảo sát bộ dữ liệu, quy mô | mức S / M / L; ưu tiên số người hơn số giờ |
| 17/10 | Bỏ mục tiêu "đủ 200 / 300 speaker thì dừng" | bỏ; giữ giới hạn batch raw và part |
| 17/10 | Giới hạn raw / part áp khi nào | chỉ khi chạy lệnh |
| 17/10 | Nguồn 1 người lấy nhiều hơn 3 video? Video dọc? | ≥ 20 phút, 3-10 video; chỉ `/videos`, chỉ ngang (user) |
| 17/10 | Folder `data/`, ledger, batch, giới tính, Drive | một `data/` cho mọi batch / part; ledger là sổ cái; giới tính đánh tay |
| 17/10 | Speaker gán khi nào | máy gán ở 04, người sửa ở 05 |
| 17/10 | Lỗi 403 / "xác minh bot" | 403 thử lại; bot dừng, dùng cookie Firefox |
| 17/10 | Luật cắt quá chặt | 64 → 48 px, hụt ≤ 3 frame; user tự xoá kết quả cắt để cắt lại |
| 17/10 | YouTube lệch tiếng-môi sẵn có ảnh hưởng gì | lối tắt "khớp quá mức", nhãn nhiễu (§4.3.1) |
| 17/10 | AV-HuBERT đo độ lệch được không; vì sao không local | được nhưng nặng, chỉ Linux; chọn SyncNet (§4.3.3) |
| 17/10 | Lọc bỏ hay sửa video lệch | sửa theo video, chỉ bỏ video không đo được; augment không tạo file |
| 17/10 | "Augment +2 có chắc? Cần research, không phải vì tôi mà sửa số" | −1…+2 theo BT.1359 + 3 căn cứ (§4.3.4); thí nghiệm để sau |
| 17/10 | Sửa độ lệch có bỏ nhánh lệch thời gian? Hạ Python 3.10? | không nghiên cứu sham_shift; không hạ Python |
| 17/10 | Cắt lâu; song song tốn bao RAM | song song chỉ 1,3× → tuần tự |
| 18/10 | pending vs reject; video bị loại do sync | như nhau; không cần đánh gì |
| 18/10 | 06 / 07 / 08 làm gì; dọn gì | tách 3 lệnh xoá, đều qua đánh dấu upload |
| 18/10 | Có nên chia split ở bước dữ liệu? Benchmark là gì? | có (§2.1, §3.5) |
| 18/10 | Mặt nghiêng có dùng; code có đo góc | dùng được tới ~45°; không đo góc; user chọn lệch ≤ ~30 % |
| 18/10 | Nhiều người mỗi người 10 clip có tốt? Số người các bộ đúng không? | §2.5; đính chính DFDC 960 |
| 18/10 | "Tôi chọn 20; chỉ tiêu cắt 40-60"; 20-60 ngẫu nhiên? | chọn 20 cố định; bác 20-60 |
| 18/10 | Part 40 người, 20 nam 20 nữ | split 12 / 3 / 5 |
| 18/10 | `urls.csv` ở đâu; thêm part sau; không đủ part thì sao | `01_plan` tạo mẫu; thêm URL rồi chạy lại; part chỉ tạo khi đủ 20 + 20 |
| 18/10 | `"empty": [19, 57, 67, 71]` | 4 URL video lẻ bị loại vì 24 fps |
| 18/10 | Batch 20 GB / 16 giờ? | đính chính: 20 giờ / 16 GB |
| 18/10 | Hạ chỉ tiêu 30-40, rải khắp video; có hụt? có chậm? | không hụt (khúc khác bù); không chậm |
| 18/10 | Xoá hết, tải lại từ đầu (user) | đã xoá raw, ledger, pool |
| 19/10 | Speaker chưa đủ 20 clip thì sao | N để chờ; batch sau khớp thêm thì hiện lại |
| 19/10 | Cắt câu có tròn câu không? Cắt semantic? | không; cần ASR → thay bằng cắt ở chỗ ngắt VAD |
| 19/10 | 15-20 clip / người, "phải đảm bảo không ảnh hưởng" (user) | biện pháp cân (§3.4) |
| 19/10 | Áp luật cắt mới (user) | xoá hết, tải lại; bỏ 10 link (6 video `unreliable` + video của 4 speaker bị loại); thêm 8 link |
| 20/10 | Dư speaker khi tạo part | người dư chờ part sau, không duyệt lại |
| 20/10 | 07 tự zip, raw zip ở 08: đồng bộ (user) | 07 chỉ export; 08 zip cả hai |
| 20/10 | Lỡ xoá raw batch 1 | khôi phục ledger, tải lại, thêm chốt chặn |
| 20/10 | Plan bị chặn bot; cookie trình duyệt khác | plan dừng ngay; Edge lỗi DPAPI |
| 20/10 | Clip `unreliable` được chọn lại có tính là lệch 0? | không: export không dịch; user tự quyết |
| 20/10 | Backup ledger tự xoá? Ledger có trong zip? | giữ mọi bản; ledger up riêng |
| 20/10 | Thêm loại `2` cắt tối đa 150 clip từ batch 3 (user) | §4.2.4 |
| 20/10 | identify theo batch? Vừa duyệt vừa cắt? Dừng cắt giữa chừng? | theo batch, so với mọi người đã có; được; Ctrl+C rồi chạy lại |
| 20/10 | `09_clean raw` có xoá clip? | không; clip chỉ bị xoá bởi `09_clean part` / `pool` |
| 21/10 | Lỡ chốt 15 clip mà muốn thêm | "done" chưa khoá; đưa về pending qua API |
| 22/10 | Thống kê, thí nghiệm, báo cáo | S00-S04; báo cáo tới clip hoàn chỉnh; mọi nội dung trong file này |
| 22/10 | Part 3 không tạo được dù báo 20/20 | luật nhóm và người pending; sửa (§4.6) |
| 22/10 | Export lấy clip từ pool là chép hay cắt? Clip đã xoá tái hiện được? | chép; ledger giữ mốc giây + hash → cắt lại được |
| 22/10 | Cần up gì ngoài ledger? | zip raw / part, `ledger/`, `registry/`, `urls.csv`, `checkpoints/` (§4.8) |

---

# Chương 7. Thảo luận

## 7.1. Đối chiếu với tài liệu

- **Luật cắt.** Mép clip ở chỗ ngắt giúp clip ít cắt cụt từ và real có khoảng lặng tự nhiên ở mép. Liên hệ lối tắt khoảng lặng [31]: fake của các bộ cũ có 25-30 ms lặng đầu clip còn real thì không; ở v003 chính real có mép lặng, nên mép lặng tự nó khó thành dấu hiệu của fake - nhưng vẫn phải kiểm sau khi sinh (S09).
- **Độ lệch.** Kết quả S03 (trễ 1-4 frame ở hầu hết video) phù hợp với nhận xét của LRS3 rằng video trên mạng lệch tiếng-hình [19]; khác LRS3 / VoxCeleb dùng SyncNet chủ yếu để lọc / đồng bộ, v003 **giữ** video và sửa hằng số, vì mục tiêu là real và fake cùng phân bố độ lệch.
- **Chuyển cảnh, trùng, mặt đang nói** theo cách VoxCeleb [20, 21].
- **Số người.** 120 người cân giới là bước đầu theo hướng [25, 26, 28]; còn xa quy mô AV-Deepfake1M++ [8].
- **Nén.** Clip encode CRF 18 (gần như không mất, [53]); FF++ [1] cho thấy nén nặng làm giảm phát hiện, nên real và fake phải cùng chuỗi encode và nên có tập test nhiễu (việc mở).

## 7.2. Mối đe doạ tính hợp lệ (threats to validity)

- **Construct validity** (đo đúng thứ cần đo): "mép lặng theo năng lượng" là proxy của "không cắt ngang từ"; S05 (mốc từ ASR) sẽ đo trực tiếp hơn. Thước đo A dùng chính VAD nên chỉ mô tả. S04 là tỉ lệ thô vì clip chưa xem lẫn với clip bị loại. S02 rải trên clip đã giữ thay vì cửa sổ trước khi kiểm.
- **Internal validity**: một người duyệt (chưa đo đồng thuận, S08); góc mặt chọn bằng mắt; độ lệch đo theo video với độ phân giải ±1 frame, "điểm 0" chưa kiểm chéo (S06).
- **External validity**: một loại nguồn (YouTube), chủ yếu nói chuyện trước camera / phỏng vấn; 72 % video đăng năm 2026; 92 kênh nhưng 60 % người từ một video; lệch về kênh quay ≥ 25 fps (loại 24 fps); giới lệch theo loại nguồn (`2` 20 nam / 9 nữ; `n` 7 nam / 15 nữ).
- **Reliability** (tái lập): ledger + cut signature + hash mô hình cho phép tái lập; raw batch 1 trên Drive là lần tải thứ hai (hash khác lần đầu, mốc và clip không đổi).

## 7.3. Đạo đức và pháp lý

- Video công khai trên YouTube của người thật; không xin phép từng người → dữ liệu **chỉ dùng nghiên cứu nội bộ**, không phát hành video, không đẩy lên dịch vụ công khai (Kaggle để private).
- Báo cáo và kết quả chỉ có số tổng hợp; không tên người, kênh, tiêu đề, mã video, mã speaker.
- Nếu phát hành: chỉ mã video + mốc thời gian + script tái tạo, kèm datasheet / data card [54, 55] và điều khoản chỉ dùng nghiên cứu (như AV-Deepfake1M++ [8]); các bộ có quyền rõ hơn dùng người có đồng ý (DFDC [3], KoDF [5]).
- Giới tính là cảm nhận của người duyệt, dùng để cân bằng, không phải thông tin tự khai.
- Fake sinh từ dữ liệu này cũng không phát hành.

## 7.4. Hạn chế

1. Quy mô nhỏ hơn hàng chục lần so với bộ chuẩn; một loại nguồn.
2. Chưa có tập test nhiễu thực tế (nén lại, giảm độ phân giải, đổi fps).
3. Đa dạng buổi quay thấp (60 % người 1 video; 117 / 120 người 1 kênh).
4. Một người duyệt; góc mặt không có số đo.
5. Độ lệch đo theo video, ±1 frame; "điểm 0" chưa kiểm chéo; 4 clip `unreliable` trong part 1 không được sửa.
6. `jump_suspect` có ở ~24 % clip được chọn mà không ảnh hưởng quyết định duyệt.
7. Giới lệch theo loại nguồn.
8. Nguồn 30 / 50 / 60 fps chuyển về 25 fps bằng bỏ frame; nguồn 24 fps bị loại.
9. 72 % video đăng năm 2026.
10. Nhóm lớn quanh MC làm part 3 lệch split nhẹ.
11. Không chuẩn hoá độ to; chưa đo phân bố độ to.
12. Phương ngữ và tuổi chưa được ghi / cân bằng.

---

# Chương 8. Kết luận và hướng phát triển

## 8.1. Kết luận

- Pipeline đã tạo 2.400 clip real hoàn chỉnh của 120 người Việt (60 nam, 60 nữ, mỗi người 20 clip) từ 251 video YouTube (~113 giờ), đóng thành 3 part có split theo người, đã lưu trữ.
- Cắt ở chỗ ngắt tăng tỉ lệ clip có hai mép lặng ~4 lần (đo độc lập) mà không mất tiếng nói được phủ.
- Video YouTube thật hầu như đều có tiếng trễ (trung vị 80 ms, 13 % > 125 ms); đo theo video ổn định; pipeline sửa về 0 khi export.
- Giới hạn 150 clip cho nguồn hai người giảm ~60 % số clip ứng viên mà hầu hết người vẫn đủ ứng viên.
- Cờ cảnh báo về mặt / giọng có giá trị cho người duyệt; cờ nghi jump cut chưa.
- Quá trình cho thấy giá trị của việc đo trước khi đổi luật, khoá luật, chốt chặn thao tác xoá, và ghi đầy đủ quyết định.

## 8.2. Hướng phát triển và việc mở

| # | Việc | Ghi chú |
|---|---|---|
| 1 | Thí nghiệm S05-S09 (§5.8) | S05 / S01 mở rộng cần tải lại raw từ Drive |
| 2 | Batch 7 (8 video): tải, cắt, gán | |
| 3 | 25 người chờ duyệt; người chung video với part cũ sẽ vào `blocked_groups` | xử lý tay khi xảy ra (ví dụ theo split của nhóm) |
| 4 | Thu thập: 2-3 video / người; cân giới trong từng loại nguồn; ghi phương ngữ, tuổi | tăng đa dạng |
| 5 | Tập test nhiễu thực tế | chỉ cần ffmpeg |
| 6 | Đo phân bố độ to (BS.1770) của real | chuẩn bị cho S09 |
| 7 | Lệnh cắt lại clip theo ledger | khi cần tái hiện clip đã xoá |
| 8 | In tiến độ cho VAD trong `03_cut`; sửa link docstring `features/syncnet.py` | đổi code trong chữ ký → đóng dấu lại |
| 9 | Mở rộng tới ≥ 5 part | |

---

# Phụ lục A. Điền báo cáo theo template AIP491

Template: `Template_AIP491_CP_StudentsGuide.docx` (gốc thư mục Deepfake_VN, không sửa bản gốc). Phạm vi (user chốt 22/10): **chỉ tới khi tạo ra một clip real hoàn chỉnh**. Viết hai bản Việt / Anh cùng cấu trúc, cùng số (bản Việt dấu phẩy thập phân, bản Anh dấu chấm).

| Mục template | Lấy từ file này | Ghi chú |
|---|---|---|
| Bìa, thông tin nhóm, lời cảm ơn | - | user điền |
| Thuật ngữ và từ viết tắt | Phụ lục B | |
| I.1 Thông tin, tổng quan | Tóm tắt, §4.7 | phạm vi: clip real hoàn chỉnh |
| I.2 Bối cảnh | §1.1-§1.3 | |
| I.3 Mục tiêu | §1.4 | |
| I.4 Phát biểu vấn đề | §2.5 (số người), §2.7 (lối tắt), §4.2.3 (video lộn xộn), §7.3 (riêng tư) | |
| I.5 Ý nghĩa | §1.6, §2.12 | |
| I.6 Phạm vi và giới hạn | Cách đọc, §7.4 | |
| II.2.1 WBS | §4.0, §6.1, §8.2 | |
| II.2.2 Rủi ro | §6.3 + bảng dưới | |
| II.2.3 Chất lượng | 40 test, `quality_report.json`, S00-S04, duyệt tay | |
| II.2.5 Quản lý thay đổi | quy trình dưới + §6.2 | |
| III Liên quan | Chương 2 | |
| IV.1 Câu hỏi nghiên cứu | §1.5 | |
| IV.2 Thu thập và tiền xử lý | Chương 4 | |
| IV.3 Đặc trưng | §4.2, §4.3, §4.4 (VAD, YuNet, SFace, WeSpeaker, SyncNet, histogram) - chỉ để làm sạch / tổ chức dữ liệu | |
| IV.4 (đổi thành "Chia dữ liệu và kiểm tra tính đúng") | §3.5, §4.6, §4.8 | không huấn luyện mô hình ở phạm vi này |
| IV.5 Chỉ số | §5.3-§5.6 (tỉ lệ giữ, mép sạch / lặng, độ lệch, `conf`, `agree`, ứng viên / người, tỉ lệ chọn theo cờ) | |
| IV.6 Triển khai | §4.0, §4.9 | |
| IV.7 Đạo đức | §7.3 | |
| V Thiết kế hệ thống | §4.0, §4.8, `data_pipeline/README.md` | |
| VI.1 Kết quả | Chương 5 | |
| VI.2 Thảo luận | §7.1, §7.2 | |
| VI.3 Khuyến nghị | §5.8, §8.2 | |
| VI.4 Nhìn lại | §6.3 (cột bài học) | |
| VII Kết luận | §8.1 | |
| VIII Tài liệu | danh mục cuối file | kiểm đủ tác giả trước khi nộp; nguồn ◐ chỉ trích thông tin xuất bản |
| IX Phụ lục | Phụ lục B, C; §4.2.2, §4.3.4, §6.2 | |

**Rủi ro (mục II.2.2):**

| Rủi ro | Đã xảy ra? | Phòng / xử lý |
|---|---|---|
| YouTube chặn bot | có | dừng ngay; cookie trình duyệt đã đăng nhập hoặc chờ |
| Cookie bị mã hoá kiểu mới | có (Edge) | Firefox hoặc chờ |
| Xoá raw trước khi sao lưu | có | khôi phục ledger, tải lại; chặn đóng gói khi thiếu file; xoá chỉ sau `--uploaded` |
| Đổi luật giữa chừng | có | cut signature |
| Ổ đĩa đầy | nguy cơ | batch ≤ 20 giờ / 16 GB; raw lên Drive mới xoá |
| Mất ledger | nguy cơ | tự sao lưu mọi bản; up Drive cùng zip |
| Part kẹt vì nhóm chung video | có | người pending không giữ nhóm |
| Lộ dữ liệu người thật | nguy cơ | Kaggle private; báo cáo chỉ số tổng hợp |

**Quy trình thay đổi luật (mục II.2.5):** (1) nêu vấn đề bằng số đo; (2) đề xuất kèm bằng chứng, chờ chốt rồi mới sửa; (3) thử trên mẫu; (4) áp dụng - luật làm clip đổi thì cut signature đổi và phải cắt lại; (5) ghi quyết định, ngày, lý do vào §6.4 và vào ledger.

**Quy ước viết:** giọng sinh viên nghiên cứu; số lấy từ file này; trích dẫn chỉ từ danh mục cuối file; ghi mức bằng chứng; không nêu định danh người / kênh / video.

# Phụ lục B. Thuật ngữ (English → giải nghĩa)

| Thuật ngữ | Giải nghĩa |
|---|---|
| deepfake | media tạo hoặc sửa bằng mô hình sinh |
| lip-sync deepfake | video bị vẽ lại vùng miệng cho khớp một đoạn tiếng khác |
| face swap / voice cloning | thay mặt / nhân bản giọng |
| real / fake | video thật / video giả |
| benchmark | dữ liệu có nhãn + split chính thức + protocol chấm + baseline |
| split (train / validation / test) | cách chia dữ liệu để huấn luyện / chọn tham số / chấm |
| speaker-disjoint split | split theo người: một người chỉ thuộc một tập |
| identity leakage | mô hình học danh tính người thay vì dấu hiệu giả mạo |
| shortcut (lối tắt) | đặc điểm không liên quan tới giả mạo nhưng tương quan với nhãn |
| audio-visual (AV) | nghe-nhìn |
| AV offset / lip-sync offset | độ lệch thời gian giữa tiếng và hình |
| SyncNet | mạng hai nhánh đo độ khớp môi-tiếng [35] |
| VAD (voice activity detection) | dò đoạn có tiếng nói |
| pause | chỗ ngắt (khúc xác suất VAD thấp bên trong đoạn nói) |
| frame / fps / CFR | khung hình / số khung mỗi giây / tốc độ khung cố định |
| CRF (constant rate factor) | tham số chất lượng của x264, thấp hơn là đẹp hơn |
| codec (H.264, AV1, AAC) | chuẩn nén hình / tiếng |
| loudness (LUFS) | độ to cảm nhận theo ITU-R BS.1770 |
| narrowband / wideband | băng hẹp 300-3400 Hz / băng rộng 50-7000 Hz |
| MFCC | đặc trưng phổ tiếng theo thang mel |
| face detection / landmark / embedding | dò mặt / điểm mốc mặt / vector đặc trưng |
| shot boundary / scene change | ranh giới cảnh / chuyển cảnh |
| jump cut | chỗ cắt ghép làm hình nhảy trong cùng một cảnh |
| active speaker detection | xác định người đang nói trong khung |
| speaker / identify / review | người / gán người / duyệt |
| candidate clip | clip ứng viên (qua kiểm tự động, chưa duyệt) |
| keep / reject / pending | được chọn / bị loại / chưa xem |
| part | đơn vị phát hành cố định 40 người |
| ledger | sổ cái SQLite ghi mọi quyết định |
| cut signature | dấu vân tay của luật làm clip |
| manifest | file mô tả từng clip (`manifest.jsonl`) |
| datasheet / data card | tài liệu chuẩn mô tả bộ dữ liệu [54, 55] |
| threats to validity | các yếu tố có thể làm kết luận sai |
| code-switching | chêm ngôn ngữ khác trong lời nói |
| tone / phonation | thanh điệu / chất giọng |

# Phụ lục C. Cấu trúc dữ liệu

**Một dòng `manifest.jsonl`:** `clip_id`, `video` (đường dẫn clip), `sha256`, `source_id`, `speaker_id`, `gender`, `split`, `duration_s`, `source_start_s`, `source_end_s`, `review_decision`, `sync_status`, `url`, `channel`, `source_sha256`, `source_fps`, `source_width`, `source_height`, `source_codec`, `source_bitrate_kbps`, `batch`, `flags`, `audio_lag_frames`, `av_sync_status`.

**Bảng ledger:** `inputs` (URL, loại, trạng thái, lý do), `videos` (metadata, trạng thái tải / cắt / gán, hash, `cut_info` gồm đoạn nói, chỗ ngắt, độ lệch, luật số lượng), `duplicates`, `batches` (zip, nơi lưu), `clips` (mốc giây, hash, mặt, giọng, cờ, speaker, quyết định, part, split, nơi lưu), `speakers` (giới, trạng thái duyệt, gợi ý gộp, part), `parts` (thành viên, số clip, zip, nơi lưu), `events` (nhật ký mọi lệnh), `meta` (phiên bản schema, cut signature).

---

# Tài liệu tham khảo

Mức kiểm: ✔ đọc bản gốc / trang chính thức / mã nguồn; ◐ chỉ đọc tóm tắt, metadata hoặc nguồn thứ cấp.

**Bộ dữ liệu deepfake**
- [1] Rössler, A., et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. ICCV. [arXiv 1901.08971](https://arxiv.org/abs/1901.08971) ✔
- [2] Li, Y., Yang, X., Sun, P., Qi, H., Lyu, S. (2020). Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics. CVPR. [arXiv 1909.12962](https://arxiv.org/abs/1909.12962) ✔
- [3] Dolhansky, B., et al. (2020). The DeepFake Detection Challenge (DFDC) Dataset. [arXiv 2006.07397](https://arxiv.org/abs/2006.07397) ✔
- [4] Khalid, H., Tariq, S., Kim, M., Woo, S. S. (2021). FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset. NeurIPS Datasets and Benchmarks. [arXiv 2108.05080](https://arxiv.org/abs/2108.05080) ✔
- [5] Kwon, P., et al. (2021). KoDF: A Large-scale Korean DeepFake Detection Dataset. ICCV. [arXiv 2103.10094](https://arxiv.org/abs/2103.10094) ✔
- [6] Cai, Z., et al. (2022). Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization (LAV-DF). [arXiv 2204.06228](https://arxiv.org/abs/2204.06228) ✔
- [7] Cai, Z., et al. (2024). AV-Deepfake1M: A Large-Scale LLM-Driven Audio-Visual Deepfake Dataset. ACM MM. [arXiv 2311.15308](https://arxiv.org/abs/2311.15308) ✔
- [8] Cai, Z., et al. (2025). AV-Deepfake1M++. [arXiv 2507.20579](https://arxiv.org/abs/2507.20579) ✔
- [9] Jiang, L., et al. (2020). DeeperForensics-1.0. CVPR. [arXiv 2001.03024](https://arxiv.org/abs/2001.03024) ✔ / ◐
- [10] PolyGlotFake (2024). [arXiv 2405.08838](https://arxiv.org/abs/2405.08838) ◐
- [11] HAV-DF (2024). [arXiv 2411.15457](https://arxiv.org/abs/2411.15457) ◐
- [12] MAVOS-DD (2025). [arXiv 2505.11109](https://arxiv.org/abs/2505.11109) ✔
- [13] TalkingHeadBench (2025, WACV 2026). [arXiv 2505.24866](https://arxiv.org/abs/2505.24866) ✔
- [14] CHN-DF (2026). 中国图象图形学报, [DOI 10.11834/jig.250077](https://www.cjig.cn/en/article/doi/10.11834/jig.250077/) ◐
- [15] InDeepFake (2025). [IEEE DataPort](https://ieee-dataport.org/documents/indeepfake-novel-multimodal-multilingual-indian-deepfake-video-dataset) ◐
- [16] LipFD / AVLips (2024, NeurIPS). [arXiv 2401.15668](https://arxiv.org/abs/2401.15668) ◐
- [17] LipSyncTIMIT / LIPINC-V2 (2025). [arXiv 2504.01470](https://arxiv.org/abs/2504.01470) ◐
- [18] Chandra, N. A., et al. (2025). Deepfake-Eval-2024. [arXiv 2503.02857](https://arxiv.org/abs/2503.02857) ✔

**Bộ nghe-nhìn nền và tiếng Việt**
- [19] Afouras, T., Chung, J. S., Zisserman, A. (2018). LRS3-TED: a large-scale dataset for visual speech recognition. [arXiv 1809.00496](https://arxiv.org/abs/1809.00496) ✔
- [20] Nagrani, A., Chung, J. S., Zisserman, A. (2017). VoxCeleb: a large-scale speaker identification dataset. Interspeech. [arXiv 1706.08612](https://arxiv.org/abs/1706.08612) ✔
- [21] Chung, J. S., Nagrani, A., Zisserman, A. (2018). VoxCeleb2: Deep Speaker Recognition. Interspeech. [arXiv 1806.05622](https://arxiv.org/abs/1806.05622) ◐
- [22] ViCocktail (2025, Interspeech). [arXiv 2506.04635](https://arxiv.org/abs/2506.04635) ✔
- [23] VietPrism (2026). [arXiv 2609.30005](https://arxiv.org/abs/2609.30005) ✔
- [24] VoxVietnam (2025). [arXiv 2501.00328](https://arxiv.org/abs/2501.00328) ✔

**Số người, công bằng, lối tắt**
- [25] Scaling Laws for Deepfake Detection (2025). [arXiv 2510.16320](https://arxiv.org/abs/2510.16320) ✔
- [26] Dong, S., et al. (2023). Implicit Identity Leakage: The Stumbling Block to Improving Deepfake Detection Generalization. CVPR. [link](https://openaccess.thecvf.com/content/CVPR2023/html/Dong_Implicit_Identity_Leakage_The_Stumbling_Block_to_Improving_Deepfake_Detection_CVPR_2023_paper.html) ✔
- [27] Bansal, A., Castillo, C., Ranjan, R., Chellappa, R. (2017). The Do's and Don'ts for CNN-based Face Verification. ICCV Workshops. [arXiv 1705.07426](https://arxiv.org/abs/1705.07426) ✔
- [28] Nadimpalli, A. V., Rattani, A. (2022). GBDF: Gender Balanced DeepFake Dataset Towards Fair DeepFake Detection. ICPR Workshops. [arXiv 2207.10246](https://arxiv.org/abs/2207.10246) ✔
- [29] Trinh, L., Liu, Y. (2021). An Examination of Fairness of AI Models for Deepfake Detection. IJCAI, 567-574. [arXiv 2105.00558](https://arxiv.org/abs/2105.00558) ◐
- [30] Layton, S., et al. (2024). SoK: The Good, The Bad, and The Unbalanced: Measuring Structural Limitations of Deepfake Media Datasets. USENIX Security. [pdf](https://www.usenix.org/system/files/sec24fall-prepub-1479-layton.pdf) ✔
- [31] Smeu, S., Boldisor, D.-A., Oneata, D., Oneata, E. (2025). Circumventing Shortcuts in Audio-visual Deepfake Detection Datasets with Unsupervised Learning. CVPR. [arXiv 2412.00175](https://arxiv.org/abs/2412.00175) ✔
- [32] Referee (2025). [arXiv 2510.27475](https://arxiv.org/abs/2510.27475) ✔
- [33] Cozzolino, D., et al. (2021). ID-Reveal. ICCV ✔
- [34] POI-Forensics (2022). [arXiv 2204.03083](https://arxiv.org/abs/2204.03083) ✔

**Đồng bộ tiếng-hình và lip-sync**
- [35] Chung, J. S., Zisserman, A. (2016). Out of Time: Automated Lip Sync in the Wild. ACCV Workshops. [pdf](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf); [syncnet_python](https://github.com/joonson/syncnet_python) ✔
- [36] Prajwal, K. R., Mukhopadhyay, R., Namboodiri, V. P., Jawahar, C. V. (2020). A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild (Wav2Lip). ACM MM. [arXiv 2008.10010](https://arxiv.org/abs/2008.10010); [hparams.py](https://github.com/Rudrabha/Wav2Lip/blob/master/hparams.py) ✔
- [37] Perfect Match (2019). [arXiv 1809.08001](https://arxiv.org/abs/1809.08001) ✔
- [38] VocaLiST (2022). [arXiv 2204.02090](https://arxiv.org/abs/2204.02090) ✔
- [39] ModEFormer (2023). [arXiv 2303.11551](https://arxiv.org/abs/2303.11551) ✔
- [40] IC-SyncNet (2024). [arXiv 2409.00971](https://arxiv.org/abs/2409.00971) ✔
- [41] PVSync (2026). [arXiv 2610.09223](https://arxiv.org/abs/2610.09223) ✔
- [42] LatentSync / StableSyncNet (2024). [arXiv 2412.09262](https://arxiv.org/abs/2412.09262) ✔
- [43] DFD-Lab (2026). [arXiv 2609.23830](https://arxiv.org/abs/2609.23830) ✔
- [45] ITU-R BT.1359 (1998). Relative timing of sound and vision for broadcasting. Qua nguồn thứ cấp: [Fora Soft](https://www.forasoft.com/learn/audio-for-video/articles-audio/lip-sync-itu-r-bt-1359-tolerance-windows), [TV Technology](https://www.tvtechnology.com/opinions/av-synchronization-how-bad-is-bad), [arXiv 2212.01686](https://arxiv.org/abs/2212.01686) ◐

**Âm thanh, hình, công cụ**
- [46] ITU-T G.722: 7 kHz audio-coding within 64 kbit/s (50-7000 Hz, lấy mẫu 16 kHz); so với G.711 300-3400 Hz. Qua nguồn thứ cấp: [G.722](https://en.wikipedia.org/wiki/G.722) ◐
- [47] ITU-R BS.1770: Algorithms to measure audio programme loudness and true-peak audio level ◐
- [48] EBU R 128: Loudness normalisation and permitted maximum level of audio signals (−23 LUFS). [EBU](https://tech.ebu.ch/docs/r/r128-2014.pdf) ◐
- [49] Silero Team. Silero VAD. [snakers4/silero-vad](https://github.com/snakers4/silero-vad) (mặc định đọc trong `utils_vad.py`) ✔
- [50] Wu, W., Peng, H., Yu, S. (2023). YuNet: A Tiny Millisecond-level Face Detector. Machine Intelligence Research 20(5), 656-665 ◐
- [51] Zhong, Y., Deng, W., Hu, J., Zhao, D., Li, X., Wen, D. (2021). SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition. IEEE TIP. [arXiv 2205.12010](https://arxiv.org/abs/2205.12010) ◐
- [52] Wang, H., et al. (2023). Wespeaker: A Research and Production Oriented Speaker Embedding Learning Toolkit. ICASSP. [arXiv 2210.17016](https://arxiv.org/abs/2210.17016) ◐
- [53] FFmpeg Wiki: H.264 Video Encoding Guide (CRF 0-51, mặc định 23, 17-18 "visually lossless or nearly so"). [trac.ffmpeg.org](https://trac.ffmpeg.org/wiki/Encode/H.264) ◐
- [57] yt-dlp. [GitHub releases](https://github.com/yt-dlp/yt-dlp/releases) ◐
- [58] Shi, B., Hsu, W.-N., Lakhotia, K., Mohamed, A. (2022). Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction (AV-HuBERT). ICLR ◐

**Tài liệu hoá và ngôn ngữ**
- [54] Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., Crawford, K. (2021). Datasheets for Datasets. Communications of the ACM 64(12), 86-92 ◐
- [55] Pushkarna, M., Zaldivar, A., Kjartansson, O. (2022). Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI. ACM FAccT. [arXiv 2204.01075](https://arxiv.org/abs/2204.01075) ◐
- [56] Kirby, J. P. (2011). Vietnamese (Hanoi Vietnamese). Journal of the International Phonetic Association 41(3), 381-392 ◐
