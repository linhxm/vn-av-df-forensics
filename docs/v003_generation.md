> Bản sao từ nhật ký nghiên cứu (`RESEARCH_WORKLOG/v003_generation.md`, ngoài repo) ngày 20/10/2026. Sửa ở nhật ký rồi chép lại vào đây.

# v003 · Sinh fake: sáu ô, generator, giọng, số lượng

Cập nhật 19/10/2026. File 2/3 của v003:
- [v003_data.md](v003_data.md): dữ liệu real, số người và số clip, độ lệch tiếng-hình.
- **v003_generation.md** (file này): sáu ô, generator và giọng, số fake mỗi clip, cách gán cân bằng, chặn lối tắt, chạy sinh trên Kaggle. §0 là bản tóm tắt gửi đồng đội chạy bước sinh.
- [v003_model.md](v003_model.md): model, huấn luyện, đánh giá.

Code: `vn-av-df-forensics/src/vn_av_df/generation/`, script `generation/00-04_*.py`, notebook `notebooks/generate.ipynb`.

Quy ước mức tin: **●** đo trên dữ liệu của mình · **✔** đọc từ paper gốc · **○** suy luận / ước lượng.

---

## 0. Tóm tắt cho người chạy sinh fake

Một part có **600-800 clip real** (40 người × 15-20 clip) và sinh ra khoảng **5.850-7.800 file**: 9 file cho mỗi clip train / validation, 12 file cho mỗi clip test.

|  | **train** | **validation** | **test** | **Cả part** |
|---|---|---|---|---|
| Người (nam + nữ) | 24 (12+12) | 6 (3+3) | 10 (5+5) | 40 (20+20) |
| Clip real (15-20 clip / người) | 360-480 | 90-120 | 150-200 | 600-800 |
| ① real | 1 | 1 | 1 | |
| ② sham (full + partial) | 2 | 2 | 2 | |
| ③ voice | 2 (A, B) | 2 (A, B) | 3 (A, B, C) | |
| ④⑤⑥ lip-sync: generator mỗi clip × 2 (full + partial) | **2** × 2 = 4 | **2** × 2 = 4 | **3** × 2 = 6 | |
| **File mỗi clip real** | **9** | **9** | **12** | |
| ↳ có AI / không AI | 6 / 3 | 6 / 3 | 9 / 3 | |
| **Tổng file** | **3.240-4.320** | **810-1.080** | **1.800-2.400** | **5.850-7.800** |
| ↳ có AI / không AI | 2.160-2.880 / 1.080-1.440 | 540-720 / 270-360 | 1.350-1.800 / 450-600 | 4.050-5.400 / 1.800-2.400 (69 % / 31 %) |
| Generator (rút từ) | Wav2Lip, MuseTalk 1.5, IP-LAP, VideoReTalking | như train | như train + **LatentSync 1.5, Diff2Lip** | |
| Hệ giọng | VieNeu, F5-TTS VN, OmniVoice, Seed-VC | như train | như train + **VoxCPM2, viXTTS, F5 inpainting** | |

Phần in đậm ở cột test là generator và giọng **chưa thấy khi train**.

**Bạn sẽ nhận được gì:** một file zip `vn-av-df-data-partN.zip` hoặc một Kaggle dataset private cùng tên (nội dung: v003_data §8.1), link repo code, và weight IP-LAP (chỉ phát hành qua OneDrive) nếu bạn chạy IP-LAP. Đây là video người thật: **không chia sẻ lại, không để public**.

**Bạn chạy:** gắn part vào notebook `generate` (`CLEAN_PART`), chạy phiên giọng trước rồi các phiên lip-sync (§7). Phiên cuối tự finalize ra bộ dữ liệu.

## 1. Sáu ô

|  | Tiếng gốc | Tiếng thật của clip khác | Giọng AI |
|---|---|---|---|
| **Hình thật (môi gốc)** | ① real | ② sham | ③ voice |
| **Môi do generator vẽ lại** | ④ source | ⑤ donor | ⑥ both |

| Ô | Cách tạo | AI? | Phần bị giả | Tiếng khớp môi? |
|---|---|---|---|---|
| ① **real** | clip gốc, giữ nguyên | 0 | - | khớp |
| ② **sham** | giữ hình, thay tiếng bằng tiếng thật của **chính người đó** ở clip khác | **0** | - | **không** |
| ③ **voice** | giữ hình, thay tiếng bằng **giọng AI** nhân bản giọng người đó | 1 | tiếng | không |
| ④ **source** | generator vẽ lại môi theo **tiếng gốc**; tiếng không đổi | 1 | môi | khớp |
| ⑤ **donor** | generator vẽ lại môi theo **tiếng thật của clip khác** cùng người, rồi dùng tiếng đó cho video: người đó "nói" một câu chưa từng nói ở cảnh này | 1 | môi | khớp |
| ⑥ **both** | sửa **một cụm** trong câu thành giọng AI (track B), generator vẽ môi theo tiếng đã sửa | 1 | môi + cụm tiếng | khớp |

"Clip khác" ở ② và ⑤, và giọng mẫu để nhân bản, luôn là clip của **cùng người, cùng split** (cố định theo seed).

**Lý do từng thành phần:**
- **② sham** tách hai câu hỏi "có AI không" và "tiếng có khớp môi không". Thiếu nó, detector học "lệch = giả" (v001: sync-only báo nhầm 88 % sham ●). FakeAVCeleb và AV-Deepfake1M không có đối chứng này.
- **Tiếng / giọng tham chiếu cùng người**: khác người thì detector chỉ cần nhận ra "giọng này không phải người trong hình", quá dễ.
- **⑤ donor**: môi khớp tiếng hoàn toàn nhưng câu nói là mới; generator phải vẽ miệng khác hẳn miệng gốc (v001 / v002: B-AVH có donor > source).
- **⑥ dùng track B (sửa vài chữ: tên, số, "có" ↔ "không")** mô phỏng kiểu sửa lời khó phát hiện. Cặp ③ ↔ ⑥ cùng tiếng, chỉ khác hình: khó nhất và sát thực tế nhất. Chưa có kiểu "giọng AI cả câu + môi giả cả clip" (FVFA của FakeAVCeleb); làm được bằng cách cho ⑥ full dùng track A, cùng chi phí (đã nêu 18/10, chưa đổi).
- Tương ứng FakeAVCeleb: ① RVRA, ③ RVFA, ④⑤ FVRA, ⑥ FVFA.

## 2. Full, partial và các track giọng

### 2.1. Full / partial

- Mỗi ô (trừ ①) ra 2 bản: **full** giả cả clip; **partial** chỉ giả **một đoạn 0,3-3 s**, cách hai đầu ≥ 0,5 s, phần còn lại là gốc.
- Chỗ nối hình: 50 % cắt thẳng, 50 % trộn mượt 2-6 frame, để model không chỉ học phát hiện chỗ nối (lỗi detector partial-spoof bên audio đã mắc: bám vết overlap-add ✔).
- Full và partial lấy từ **cùng một lần chạy** generator (partial ghép đoạn giả vào video gốc), nên partial không tốn thêm GPU.
- ⑥ full: môi giả cả clip, tiếng chỉ có cụm track B là AI; ⑥ partial: môi giả đúng đoạn của cụm đó.

### 2.2. Vì sao đúng 1 đoạn, dài 0,3-3 s

| Bộ | Số đoạn mỗi video ✔ |
|---|---|
| FF++, Celeb-DF, DFDC, FakeAVCeleb, KoDF | giả cả video, không có đoạn |
| LAV-DF | 62,72 % video 1 đoạn, 10,55 % 2 đoạn; video < 10 s tối đa 1 đoạn; đoạn trung bình 0,65 s, 89 % < 1 s |
| AV-Deepfake1M | ⌈độ dài / 10 s⌉ đoạn → video < 10 s có 1 đoạn |

- Clip v003 dài 4-7 s → **1 đoạn đúng chuẩn**.
- Không tìm thấy nghiên cứu đối chứng cho thấy train nhiều đoạn giúp định vị. Thêm 2 đoạn có rủi ro: thói quen mới ("clip dài có 2 đoạn", vì clip 4 s không đủ chỗ), thêm vết ghép tiếng / hình, lệch thực tế (phần lớn deepfake ngoài đời giả cả video, Deepfake-Eval-2024). Nếu test chỉ có 1 đoạn thì không đo được lợi ích. → **2 đoạn là thí nghiệm phụ sau** (user chốt 18/10).
- Đổi kèm (18/10): độ dài **0,3-3 s** thay 0,5-3 s, để có cả đoạn sửa 1 từ như LAV-DF; thêm chỉ số định vị chuẩn AP / AR (v003_model §8).

### 2.3. Ba track của ô ③

| Track | Cách làm | Split |
|---|---|---|
| **A** (full) | PhoWhisper lấy lời; TTS đọc lại **cả câu** bằng giọng nhân bản; co giãn ≤ ±15 % cho vừa độ dài clip | mọi split |
| **B** (partial) | đổi **một cụm** (từ, số, tên, từ đệm), TTS cụm mới, co giãn vừa chỗ, ghép vào tiếng gốc: crossfade 15 ms, khớp độ to, phủ tiếng nền | mọi split |
| **C** (partial) | **sửa trực tiếp** bằng F5 inpainting: model sinh lại đoạn theo ngữ cảnh câu, chỉ lấy đoạn đã sửa nối lại vào tiếng gốc (nối 5 ms, vì vocoder tạo lại cả câu) | **chỉ test** |

- **Track B ghép giống hệt ② sham partial** (cùng hàm ghép, cùng crossfade, khớp độ to, tiếng nền): vết ghép không thành dấu hiệu riêng của giọng AI.
- **Track C chỉ ở test**: PartialEdit (Interspeech 2025) cho thấy detector train trên kiểu ghép phát hiện kém kiểu sửa bằng model ✔ → đo đúng điểm yếu đó.

## 3. Generator và hệ giọng

### 3.1. Danh sách

| | train / validation / test | chỉ test |
|---|---|---|
| Generator | **Wav2Lip-GAN**, **MuseTalk 1.5**, **IP-LAP**, **VideoReTalking** | **LatentSync 1.5**, **Diff2Lip** |
| Giọng (track A / B, ⑥) | **VieNeu-TTS v3**, **F5-TTS tiếng Việt**, **OmniVoice**, **TTS → Seed-VC** | **VoxCPM2**, **viXTTS** |
| Track C | - | **F5 inpainting** |

| Generator | Họ | Giấy phép | Thời gian ước / đo (2×T4) |
|---|---|---|---|
| Wav2Lip-GAN | GAN, vẽ 96 px | chỉ nghiên cứu | ~41 s / mẫu ● |
| MuseTalk 1.5 | latent VAE 1 bước, 256 px | MIT | ~68 s / mẫu ● |
| IP-LAP | landmark → render | Apache-2.0 (weight qua OneDrive) | ~100-200 s ○ |
| VideoReTalking | nhiều tầng, có làm nét mặt bên trong | Apache-2.0 | ~140-270 s ○ |
| LatentSync 1.5 | latent diffusion, 256 px | Apache-2.0 | ~200-400 s ○ |
| Diff2Lip | diffusion trên pixel | CC BY-NC 4.0 | ~140-270 s ○ |

| Giọng | Họ |
|---|---|
| VieNeu-TTS v3 | codec LM, ưu tiên tiếng Việt |
| F5-TTS tiếng Việt | flow matching + vocoder (CC-BY-NC-SA) |
| OmniVoice | diffusion language model |
| TTS → Seed-VC | chuyển giọng (voice conversion) |
| VoxCPM2 | không tokenizer, 30 ngôn ngữ (Apache 2.0) |
| viXTTS | GPT + HiFi-GAN |
| F5 inpainting | sửa trực tiếp (chế độ sửa chưa thấy) |

### 3.2. Lý do chọn

- **Mỗi generator / hệ giọng một họ khác nhau.** Chỉ dùng một họ thì detector học thuộc dấu vết của họ đó: v001 nhánh artifact train trên Wav2Lip đạt 0,979 trên Wav2Lip nhưng 0,553 trên MuseTalk ●. Số phương pháp giả là một trục làm detector tổng quát tốt hơn (Scaling Laws, α = 0,375 ✔).
- **Giữ riêng họ diffusion cho test** → đo được "chưa từng thấy diffusion thì có phát hiện được không". MAVOS-DD và TalkingHeadBench cũng giữ riêng generator cho test ✔. v002 §19 đã đề xuất LatentSync làm held-out.
- BioLip (baseline BL-2) có kết quả zero-shot trên đúng VideoReTalking, Diff2Lip, IP-LAP, Wav2Lip ✔ → có điểm tựa so sánh.
- **Loại ra:** PC-AVS (chỉ ra crop mặt 224 px, cũ), DINet (cần OpenFace Windows, train trên 363 video).
- **Sinh giữ nguyên 720p** (`max_side` 1280): v002 sinh ở 640 px, Wav2Lip phóng patch 96 px lên 2-3 lần → miệng mờ → quá dễ (AUC 0,99 ●).
- **Thay khi triển khai (15/10):** VieNeu v2 → **v3** (cùng tác giả); Higgs Audio v3 → **viXTTS** (Higgs cần server SGLang-Omni, benchmark trên H100, giấy phép phi thương mại; viXTTS khác họ với các hệ khi train).
- VietPrism (deepfake giọng tiếng Việt) đã có OmniVoice, Higgs v3, VoxCPM2: ai train trên VietPrism thì VoxCPM2 không còn "chưa thấy". Khó phát hiện nhất trong VietPrism: MiniMax (EER 37 %), VoxCPM2 (35 %); dễ nhất OmniVoice (15 %) ✔. MiniMax là lựa chọn thay thế (trả phí).
- **Giấy phép:** vài generator / TTS chỉ phi thương mại → repo để private tới khi rõ.

## 4. Bao nhiêu fake mỗi clip real

Số người và số clip mỗi người: v003_data §3. Ở đây: mỗi clip real sinh bao nhiêu file AI.

### 4.1. Bằng chứng

| Nghiên cứu / bộ dữ liệu | Kết quả ✔ | Hệ quả |
|---|---|---|
| GenD (WACV 2026) | fake cùng nguồn với real (paired) hơn không paired **+4,7 AUC** trung bình trên 14 bộ, cùng tổng số video; unpaired học vẹt nhanh hơn | mọi fake sinh từ chính clip real |
| Scaling Laws (2025) | số phương pháp giả là trục có lợi (α = 0,375); thêm mẫu từ cùng nguồn lợi ít | nhiều phương pháp, ít lặp mỗi clip |
| Layton et al. (USENIX Security 2024) | train 90 % fake → trên tập toàn real báo nhầm nhiều hơn báo đúng; "phải train với phân bố lớp giống tập đánh giá" | tỉ lệ vừa phải + ngưỡng theo FPR (v003_model §8) |
| AV-Deepfake1M | fake / real = **2,997 / 2,9996** ở train / test → suy ra mỗi real sinh đúng 3 fake (tiếng giả / hình giả / cả hai) ○; phương pháp **rút cho từng fake**; mỗi fake nhiều chỗ sửa (⌈độ dài / 10 s⌉) | ít fake mỗi real, phương pháp rút ngẫu nhiên |
| AV-Deepfake1M++ | 2,27 tổng (train 2,70; validation 2,82; testA 1,88; testB 1,03); phương pháp rút từ pool (3 lip-sync, 4 TTS) | test có real không sinh đủ 3 fake |
| FaceForensics++ / KoDF / LAV-DF / DFDC / Celeb-DF / FakeAVCeleb | 4 / ~2,8 / ~2,7 / ~5 / ~9,6 / 39 | FakeAVCeleb bị chê lệch quá nặng |

### 4.2. Các phương án (tính cho part 50 người cũ, 20 clip / người)

| | A. 10 sinh fake + 10 real_only | B. 20 clip, mọi generator | **C. 20 clip, một nửa generator** |
|---|---|---|---|
| Clip real có fake đi kèm | 500 | 1.000 | **1.000** |
| Tổng file | 7.600 | 14.200 | 9.720 |
| Có AI / không AI | 74 % / 26 % | 79 % / 21 % | **69 % / 31 %** |
| Lần chạy lip-sync | 2.240 | 4.480 | **2.240** |
| Một clip real lặp trong bao nhiêu file AI | 10-15 | 10-15 | **6-9** |
| GPU sinh (ước) ○ | 65-110 giờ | 130-220 giờ | 70-125 giờ |
| So mọi generator trên cùng clip | có | có | không |

**Chọn C** (18/10):
- **Cùng GPU lip-sync với A**, nhưng **gấp đôi clip real làm nền** cho fake; lớp cân hơn (2 : 1 ở train); mỗi clip real lặp ít hơn. Khớp "đa dạng hơn lặp lại" (Scaling Laws) và cách AV-Deepfake1M làm.
- Mọi real đều có fake cặp (GenD), không cần vai real_only. Real_only (A) có rủi ro lối tắt thấp nếu chia ngẫu nhiên theo video và encode như nhau, nhưng C tốt hơn mà đơn giản hơn.
- Mỗi generator vẫn chạy cùng số lần như A → tổng hợp theo generator / ô vẫn đủ mẫu.
- **Mất:** không còn "mọi generator trên cùng clip" (so cặp kiểu FaceForensics++). So sánh tổng hợp vẫn cân.
- **Chỉ giảm phần vẽ môi**; giọng vốn 1 hệ mỗi clip, giữ nguyên.

### 4.3. Con số cuối cho part 40 người

| | train | validation | test | Cả part |
|---|---|---|---|---|
| Lần chạy mỗi generator | 180-240 | 45-60 | 75-100 | |
| Lần chạy lip-sync | 720-960 | 180-240 | 450-600 | **1.350-1.800** |
| Job giọng | 360-480 | 90-120 | 150-200 | **600-800** |
| GPU sinh ○ | | | | **~40-100 giờ** (gần 2-4 tuần quota Kaggle) |

Khoảng là vì mỗi người có 15-20 clip (từ 19/10; v003_data §3.3). Mỗi clip vẫn sinh đúng 9 / 12 file nên tỉ lệ có AI / không AI không đổi.

- **69 % / 31 %** cùng vùng AV-Deepfake1M (75 % fake) và AV-Deepfake1M++ (69 %). Loss khi train đã cân trọng số lớp, nên không cần đổi tỉ lệ dữ liệu; chỗ cần sửa là ngưỡng (v003_model §8).
- Full factorial (mọi generator × mọi ô × mọi giọng) là 59 / 124 mẫu mỗi clip, gấp ~7 lần, phần lớn do ô ⑥ = generator × giọng → bỏ từ 15/10.

## 5. Gán generator vào ô: cân bằng thật sự

### 5.1. Lỗi của cách xoay vòng cũ (phát hiện 18/10)

User nghi xoay vòng lặp lại không cân. Mô phỏng trên part chuẩn (cấu hình lúc đó: 30 người train, 12 test, 10 clip / người) xác nhận ●:

| | Cũ: số lần [source, donor, both] mỗi generator | Mới |
|---|---|---|
| Train (4 generator) | 120 / 90 / 90 | **100 / 100 / 100** |
| Test (6 generator) | 48 / 36 / 36 | **40 / 40 / 40** |
| Validation (8 người) | lệch tương tự | 27 / 27 / 26 |

1. 10 clip mỗi người không chia hết cho 3 → ô đầu vòng dư 1 lần ở **mọi người** → lệch 33 %.
2. Ở test, 3 cặp generator (1 với 4, 2 với 5, 3 với 6) **luôn chung ô** trong mọi clip (6 là bội của 3).

### 5.2. Cách gán hiện hành (`plan.py`: `assign_generators`, `cell_index`)

- Gán theo **vòng** trên cả split: mỗi vòng, mỗi generator chạy đúng 1 clip → **số lần chạy mỗi generator bằng nhau**.
- Ô của mỗi generator xoay đều, lệch theo thứ tự người để phần dư bù trừ → **mỗi generator chia đều ④ / ⑤ / ⑥**.
- Trong một clip các generator luôn ở **ô khác nhau**; bộ generator chung clip rút theo seed và đổi theo vòng (nhóm sau lệch theo khối clip) → cặp cùng clip thay đổi dần.
- Hệ giọng 1 hệ mỗi clip, xoay theo clip, lệch theo người.
- Ví dụ 2 clip train liên tiếp: clip 1 → Wav2Lip ④ + IP-LAP ⑤; clip 2 → MuseTalk ⑥ + VideoReTalking ④.
- Có test kiểm cân bằng trên cấu hình part thật.

## 6. Chặn lối tắt ở bước sinh

| Lối tắt | Cách chặn / kiểm |
|---|---|
| Khác biệt nén, độ phân giải, fps (v002 E6: nhánh artifact học dấu vết xử lý) | **encode lại mọi file, kể cả real**, cùng một cấu hình (H.264 CRF 18, 25 fps, ≤ 720p, AAC 48 kHz) |
| Độ lệch tiếng-hình real ≠ fake | real đã dịch về 0 khi export (v003_data §6) |
| Khoảng lặng đầu / cuối, độ to, độ dài (Smeu CVPR 2025: chỉ nhìn khoảng lặng đầu > 98 % trên FakeAVCeleb, AV-Deepfake1M ✔) | `shortcut_audit.json` khi finalize: AUC của từng đặc trưng (lặng đầu, lặng cuối, tỉ lệ lặng, độ to dB, độ dài) một mình đoán nhãn AI; **ngoài [0,3; 0,7] là cờ đỏ**, phải sửa cách sinh trước khi train. Rủi ro riêng: track A co giãn TTS vừa độ dài clip nên khoảng lặng có thể khác real |
| Vết ghép tiếng | track B ghép giống hệt ② sham |
| Vết nối hình ở partial | 50 % trộn mượt 2-6 frame |
| Real và fake khác nguồn | mọi fake sinh từ chính clip real (paired) |
| Clip tham chiếu lẫn vào phép đo | ref-norm bỏ clip tham chiếu và mọi file sinh từ chúng (v003_model §8) |

Chưa có: lọc chất lượng fake (TalkingHeadBench loại > 60 % mẫu lộ lỗi ✔). Hiện chỉ job lỗi (không thấy mặt, generator báo lỗi) bị bỏ; fake "chạy được mà xấu" vẫn ở lại. Đề xuất ở bảng trắng 15/10: chấm LSE-C / D, độ nét miệng so với má, gắn mức khó.

## 7. Chạy sinh và đầu ra

### 7.1. Các bước

| Bước | Lệnh / phiên | Làm gì |
|---|---|---|
| 0 | `generation/00_setup.py` | tải source / weight generator |
| 1 | `generation/01_plan.py` | kế hoạch cả part (ô, generator, giọng, tiếng ghép), cố định theo seed |
| 2 | phiên **voice** (`PHASE = "voice"`, GPU) | ① ② ③ + lưu track B cho ⑥; **chạy trước** lip-sync |
| 3 | các phiên **lipsync** (`PHASE = "lipsync"`, `GENERATORS = [...]`) | ④ ⑤ ⑥, mỗi phiên vài generator; chạy song song trên nhiều notebook, phiên sau attach Output phiên trước vào `PREVIOUS_OUTPUT` |
| 4 | `generation/04_finalize.py` (phiên cuối tự chạy) | gom thành `manifest.jsonl` có nhãn + `shortcut_audit.json` |

- Mỗi job ghi `records/<key>.json`. Hết giờ (`TIME_BUDGET_HOURS`) hoặc lỗi thì chạy lại; job dở xoá và làm lại.
- Lỗi riêng một clip (không thấy mặt, generator báo lỗi) thì bỏ job đó, ghi lý do; bỏ 10 job liên tiếp thì dừng.
- IP-LAP: tải tay 2 file `.pth` từ OneDrive, upload thành Kaggle dataset, đặt `IPLAP_CHECKPOINTS`.
- Bước sinh **không ghi W&B** (user chốt 18/10); nhật ký ở `generation.json` / `records/`.

### 7.2. Đầu ra

| Đường dẫn | Nội dung |
|---|---|
| `datasets/vn-av-df-data/<part>/clips/` | toàn bộ mp4 (real + fake), encode cùng một cấu hình |
| `datasets/vn-av-df-data/<part>/manifest.jsonl` | mỗi dòng một file: split, người, ô, full / partial, generator, giọng, nhãn AI / hình / tiếng / lệch, các đoạn bị giả, hệ số co giãn tiếng |
| `datasets/vn-av-df-data/<part>/shortcut_audit.json` | kiểm lối tắt (§6) |

### 7.3. Demo dùng cùng code sinh

Tab "tạo fake" của demo (v003_model §10) gọi đúng các hàm của pipeline sinh (ghép tiếng, chọn cụm sửa, trộn mép partial, nhãn 6 ô), nên mẫu tạo trong demo giống hệt bộ dữ liệu.

## 8. Hạn chế và việc mở (phần sinh)

**Hạn chế:**
1. Mỗi clip chỉ 2-3 generator → không so được từng cặp generator trên cùng clip.
2. **Chưa chạy thật**: worker env của 4 generator mới và 6 hệ giọng chưa chạy trên Kaggle; test hiện dùng generator / TTS / ASR giả lập.
3. Chưa lọc chất lượng fake.
4. Rủi ro "mặt nghiêng → real" nếu generator hay lỗi ở mặt nghiêng (v003_data §7.2).

**Việc mở:**

| # | Việc | Ghi chú |
|---|---|---|
| 1 | Sinh part 1 thật trên Kaggle | sau khi export part |
| 2 | Kiểm `shortcut_audit.json` trước khi train | cờ đỏ → sửa cách sinh |
| 3 | Kiểm tỉ lệ job lỗi theo góc mặt | xem `records/` |
| 4 | Thí nghiệm partial 2 đoạn | train hai bản, test có cả 1 và 2 đoạn |
| 5 | Leave-one-generator-out | sinh thêm 2 generator test-only cho train / validation của một part (+~50 % GPU), split giữ nguyên |
| 6 | Lọc chất lượng fake | LSE-C / D, độ nét miệng / má |
| 7 | (Tuỳ chọn) ⑥ full dùng track A | thêm loại "cả hai giả toàn bộ", cùng chi phí |

## 9. Nhật ký câu hỏi và quyết định (phần sinh)

| Ngày | Câu hỏi / việc | Trả lời / quyết định |
|---|---|---|
| 15/10 | Chạy mọi generator × mọi ô × mọi giọng? | không (59 / 124 mẫu mỗi clip, ~7×); ô và giọng xoay vòng |
| 15/10 | Generator / giọng nào chỉ ở test | họ diffusion (LatentSync, Diff2Lip); VoxCPM2, Higgs v3 (sau thay viXTTS), F5 inpainting |
| 15/10 | Encode | encode lại mọi mẫu, kể cả real, cùng cấu hình |
| 16/10 | Code v003 nằm riêng `bench/` | user: gộp vào `generation/` + `training/`, xoá phần v002 không dùng |
| 18/10 | Source / donor / both có xoay vòng? Lip-sync đã xoay sao ra 8 file? Fake cả tiếng lẫn hình ở đâu? | xoay vòng chọn ô cho từng lần chạy, không bỏ generator; full + partial cùng một lần chạy; fake cả hai là ⑥ |
| 18/10 | Xoay vòng có cân bằng thật? | không (120 / 90 / 90; cặp cố định ở test) → sửa (§5) |
| 18/10 | Viết lại kế hoạch fake thành bảng, giải thích source / donor, cho đồng đội | bảng 1 trang + tài liệu (nay là §0-§2 file này) |
| 18/10 | Vì sao VideoReTalking tách dòng; test thì sao | 4 generator / 3 ô nên luôn có ô có 2 generator, ô đó đổi theo clip; test 6 generator: mỗi ô đúng 2, cặp đổi theo khối 3 clip |
| 18/10 | Họ fake mỗi real bao nhiêu, có fake hết real? | AV-Deepfake1M đúng 3 / real, phương pháp rút cho từng fake |
| 18/10 | 20 clip đều fake? Giảm còn ~6? | phương án **C**: một nửa generator, giọng giữ nguyên (§4) |
| 19/10 | Mỗi người 15-20 clip | số file mỗi part thành khoảng (5.850-7.800); mỗi clip vẫn 9 / 12 file |
| 18/10 | Full / partial có giúp định vị? Nhiều đoạn? | có; **giữ 1 đoạn** (user), 0,3-3 s |
| 18/10 | Thêm phần lý do, tài liệu, "bạn sẽ nhận được gì" cho đồng đội | §0 và §4 file này |

### Những điều đã đính chính

| Đã nói | Đúng là |
|---|---|
| Một clip test (thiết kế cũ) sinh 19 file | 18 (1 + 2 + 3 + 6 × 2); code vẫn sinh đúng |
| Bảng ví dụ ghi VideoReTalking như một dòng riêng | đúng hơn là "④ source × 2: Wav2Lip, VideoReTalking"; mỗi generator vẫn chia đều 3 ô |

## 10. Nguồn

- Yermakov et al., GenD, WACV 2026 - [arXiv 2508.06248](https://arxiv.org/abs/2508.06248)
- Scaling Laws for Deepfake Detection - [arXiv 2510.16320](https://arxiv.org/abs/2510.16320)
- Layton et al., USENIX Security 2024 - [pdf](https://www.usenix.org/system/files/sec24fall-prepub-1479-layton.pdf)
- Smeu et al., CVPR 2025 - [arXiv 2412.00175](https://arxiv.org/abs/2412.00175)
- Cai et al., LAV-DF, DICTA 2022 - [arXiv 2204.06228](https://arxiv.org/abs/2204.06228) · AV-Deepfake1M, ACM MM 2024 - [arXiv 2311.15308](https://arxiv.org/abs/2311.15308) · AV-Deepfake1M++, ACM MM 2025 - [arXiv 2507.20579](https://arxiv.org/abs/2507.20579)
- FakeAVCeleb [2108.05080](https://arxiv.org/abs/2108.05080) · FaceForensics++ · KoDF [2103.10094](https://arxiv.org/abs/2103.10094) · DFDC [2006.07397](https://arxiv.org/abs/2006.07397) · Celeb-DF [1909.12962](https://arxiv.org/abs/1909.12962)
- MAVOS-DD [2505.11109](https://arxiv.org/abs/2505.11109) · TalkingHeadBench [2505.24866](https://arxiv.org/abs/2505.24866) · Deepfake-Eval-2024 [2503.02857](https://arxiv.org/abs/2503.02857)
- PartialEdit (Interspeech 2025); survey partial-spoof 2025 (detector bám vết overlap-add)
- VietPrism [2609.30005](https://arxiv.org/abs/2609.30005) · SEA-Spoof
- Generator / giọng: Wav2Lip, MuseTalk, IP-LAP (CVPR 2023), VideoReTalking, LatentSync [2412.09262](https://arxiv.org/abs/2412.09262), Diff2Lip (WACV 2024), VieNeu-TTS, [F5-TTS Vietnamese](https://github.com/nguyenthienhy/F5-TTS-Vietnamese), OmniVoice, Seed-VC, [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), viXTTS, [PhoWhisper](https://github.com/VinAIResearch/PhoWhisper)
