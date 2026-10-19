> Bản sao từ nhật ký nghiên cứu (`RESEARCH_WORKLOG/v003_model.md`, ngoài repo) ngày 19/10/2026. Sửa ở nhật ký rồi chép lại vào đây.

# v003 · Model: phương pháp, huấn luyện, đánh giá

Cập nhật 19/10/2026 (khảo sát phương pháp ngày 15/10). File 3/3 của v003:
- [v003_data.md](v003_data.md): dữ liệu real, số người và số clip, độ lệch tiếng-hình.
- [v003_generation.md](v003_generation.md): sáu ô, generator và giọng, số fake mỗi clip.
- **v003_model.md** (file này): khảo sát phương pháp, lỗi của detector hiện nay, baseline, 3 kiến trúc đề xuất, protocol và cổng, huấn luyện, đánh giá, W&B, demo.

Phạm vi: detector cho benchmark lip-sync tiếng Việt v003, gồm 6 ô và 3 câu hỏi cho mỗi mẫu:
- **visual fake?** (hình bị AI vẽ)
- **audio fake?** (giọng AI)
- **AV mismatch?** (tiếng và môi lệch nhau)

cộng điểm tổng **AI?** = max(visual, audio) và định vị đoạn giả.

Quy ước:
- Số liệu lấy từ chính paper; khác điều kiện thì không so trực tiếp.
- "Chưa tìm thấy" nghĩa là chưa tìm thấy tính đến ngày ghi, không khẳng định là chưa ai làm.
- Kết quả v001 / v002 chỉ là **bằng chứng phụ**, vì test chỉ có 2-3 người.
- Mức tin: **●** đo trên dữ liệu của mình · **✔** đọc từ paper gốc · **○** suy luận / ước lượng.

Cách làm như v002: bài toán → lỗi X (lấy từ literature) → ngành khác giải X bằng Z → mang Z về, kèm đánh giá khả thi trên Kaggle 2×T4. Nguồn tìm: arXiv 2024-2026, CVPR / NeurIPS / ICME / ACM MM / EACL, Interspeech / ICASSP, tạp chí tiếng Trung (中国图象图形学报), luận văn tiếng Trung phồn thể (NCCU).

---

## 1. Toàn cảnh phương pháp

| Phương pháp | Năm · nơi | Ý chính | Train trên | Kết quả nổi bật (theo paper) | Code |
|---|---|---|---|---|---|
| **AVH-Align** | CVPR 2025 | AV-HuBERT (Large, fine-tune VSR) cho feature audio / visual riêng → MLP 1024→512→256→128→1 chấm độ khớp từng frame; InfoNCE trên ±15 frame; điểm video = log-sum-exp(−Φ_ii) | **Chỉ real** | Bền với lối tắt leading silence, tốt nhất AV-Deepfake1M ở nhóm real-only. Train FakeAVCeleb → test AV-1M chỉ 57,7 AUC | ✔ (bit-ml) |
| **SpeechForensics** | NeurIPS 2024 | AV-HuBERT Large **không fine-tune**; cosine audio / visual từng frame, lấy max theo độ lệch ±τ (τ≈15) | Chỉ real (không train detector) | FF++ 97,6, FakeAVCeleb 99,0 (Large) | ✔ |
| **BioLip** | arXiv 2026 | Chỉ landmark môi (64 điểm), y-coordinate, 4 bậc động học (vị trí, vận tốc, gia tốc, jerk) + thống kê + tỉ lệ vùng → CNN-1D 302K tham số | AVLips (W2L, TalkLip, SadTalker) | LipSyncTIMIT zero-shot TB **0,927** (W2L 0,97 · VideoReTalking 0,96 · Diff2Lip 0,93 · **IP-LAP 0,82**); CRF40 còn 0,76; 7 ngôn ngữ TB 0,83 | Chưa thấy |
| LIPINC / LIPINC-V2 | 2024 / 2025 | Bất nhất miệng theo thời gian; V2 dùng ViT + cross-attention | Lip-sync | LipSyncTIMIT TB 0,96 (CRF23) / 0,76 (CRF40) | ✔ |
| LoCC | ICME 2026 | So từng frame với "frame phản thực" dự đoán từ frame lân cận; teacher-student; định vị tới frame | LAV-DF, AV-1M | Theo paper hơn LIPINC trên LAV-DF / AV-1M / FakeAVCeleb / KoDF | ✔ |
| PIA | arXiv 2025 | Phoneme (WhisperX) + viseme (3D-CNN + EfficientNet) + độ mở môi + ArcFace theo thời gian | FakeAVCeleb, DeepSpeak | FakeAVCeleb AUC 99,8; DeepSpeak 98,1. **Pipeline phoneme chỉ tiếng Anh** | ✔ |
| LipFD | NeurIPS 2024 | Phổ âm + chuỗi frame miệng, hai đầu | AVLips | Trong miền > 95 %; **chéo bộ ~0,54** (X-AVDT đo lại) | ✔ |
| AV-Lip-Sync+ (≈ B-AVH của mình) | 2023 | AV-HuBERT + TCN đa tỉ lệ, có giám sát | Có fake | Mốc có giám sát | ✔ |
| X-AVDT | arXiv 2026 | Đảo ngược diffusion (Hallo) lấy cross-attention AV | MMDF | Chéo generator 95,3 AUROC, nhưng **~1 phút / clip 16 frame** | Trang dự án |
| KLASSify (hạng 3 AV-1M++ 2025) | 2025 | SSL audio + graph attention (kiểu AASIST) + đặc trưng hình thủ công | AV-1M++ | 92,8 AUC; định vị chỉ bằng audio IoU 0,35 | - |
| Pindrop (hạng 1 định vị AV-1M++) | 2025 | Ensemble mạng chuyên biệt + định vị kiểu ActionFormer | AV-1M++ | Điểm định vị 67,2 | - |
| XLS-R + AASIST (và XLSR-SLS, XLSR-Mamba) | 2022-2026 | SSL đa ngôn ngữ (128 ngôn ngữ, **có tiếng Việt**) + graph attention | ASVspoof | Mạnh trong miền, **yếu với hệ TTS chưa thấy** | ✔ |
| POI-Forensics | 2022 | Học danh tính AV từ real, so với video tham chiếu của chính người đó | Chỉ real | Bền với video chất lượng thấp | ✔ |
| AV-LMMDetect / Omni-Fake-R1 / ConLLM | 2026 | MLLM (Qwen2.5-Omni) fine-tune, có giải thích | FakeAVCeleb, MAVOS-DD | 85,1 % acc open-set (base 32,3 %) | Một phần |

## 2. Lỗi của detector hiện nay (bằng chứng chính từ literature)

| # | Lỗi | Bằng chứng | Phụ (v001 / v002) |
|---|---|---|---|
| X1 | **Học thuộc dấu vết pixel của generator đã thấy** | BioLip: Xception 0,93 → 0,67 khi nén; LipFD chéo bộ 0,54; MesoInception ≈ ngẫu nhiên với generator lạ | P2: W2L 0,98 → MuseTalk 0,55 |
| X2 | **Lối tắt của dữ liệu** (leading silence, khác biệt nén / kích thước) | Smeu CVPR25: chỉ nhìn im lặng đầu clip đạt > 98 %; Fake-or-JPEG ECCVW24 | E6: nhánh artifact báo nhầm 79-100 % clip chỉ bị xử lý |
| X3 | **Phụ thuộc ngôn ngữ** | BioLip: LipFD kém rõ trên tiếng Trung; PIA, TISAN chỉ chạy với phoneme tiếng Anh; AV-HuBERT pretrain bằng tiếng Anh | - |
| X4 | **Sụp khi nén / nhiễu** | BioLip: mọi phương pháp giảm 0,17-0,25 ở CRF40; SAVe 0,77 ở c40 | - |
| X5 | **Generator làm mượt theo thời gian** (IP-LAP, diffusion) | BioLip yếu nhất với IP-LAP (0,82 → 0,53 ở CRF40); X-AVDT: bộ dữ liệu cũ chủ yếu GAN | - |
| X6 | **Gộp thẳng điểm "lệch tiếng" vào điểm "AI"** | AVH-Align / SpeechForensics chỉ đo độ khớp, nên lệch tiếng thật (sham) cũng bị báo; BioLip không chạy được khi tiếng bị thay | v001: sync-only báo nhầm 88 % sham |
| X7 | **Không định vị được, hoặc định vị bằng vết nối** | Survey partial-spoof 2025 (bám vùng overlap-add); UMMAFormer từ LAV-DF sang AV-1M rơi 98,8 → 51,6 AP | - |
| X8 | **Phát hiện giọng giả không tổng quát sang TTS lạ** | XLS-R+AASIST yếu với hệ chưa thấy; VietPrism: MiniMax EER 37 %, VoxCPM2 35 % | - |
| X9 | **Điểm nền lệch theo người / danh tính** | POI-Forensics là cả một hướng dựa trên tham chiếu người; PIA dùng ArcFace theo thời gian | v002: AUC trong từng người 0,96, gộp 0,78 |
| X10 | **Quá đắt cho quy mô lớn** | X-AVDT ~1 phút / clip; MLLM 7B không fine-tune được trên T4 | - |

## 3. Cầu nối liên ngành

| Ý tưởng (Z) | Ngành gốc | Giải lỗi | Trạng thái tìm kiếm |
|---|---|---|---|
| **Cost volume theo độ trễ + gộp 2D** (học tương quan rồi cho CNN 2D chạy trên trục thời gian × độ trễ, ước độ trễ bằng soft-argmin) | Stereo matching (GC-Net, PSMNet), optical flow (PWC-Net, RAFT) | X6, X7 | Chưa tìm thấy cho lip-sync. SyncNet / AVH-Align chấm 1 độ trễ; SpeechForensics lấy max toàn clip; TriCue (v002) dùng hồ sơ cos thô → TCN |
| **Residual upsampling (NPR) ở độ phân giải gốc của miệng** | Image forensics (Tan et al., CVPR 2024) | X1, X4 | Chưa tìm thấy cho vùng miệng lip-sync |
| **Residual "phản thực" theo thời gian** (frame − nội suy từ hai frame lân cận) | Nội suy frame / phát hiện bất thường video | X5, X7 | LoCC (2026) đã làm dạng học teacher-student → v003 chỉ dùng như một luồng đặc trưng rẻ, **không nhận là ý mới** |
| **Động học landmark môi** (vận tốc, gia tốc, jerk) | Biomechanics / speech motor control | X3, X4 | BioLip đã làm → dùng làm **baseline** |
| **Học bất biến với xử lý lành tính** (augment đối xứng real / fake: nén lại, đổi cỡ, làm nét) | Domain generalization / anti-spoofing (RawBoost, codec augmentation) | X2, X4 | Phổ biến bên audio; v002 chưa làm (lỗi E6) |
| **Chuẩn hoá theo người bằng clip tham chiếu** | Speaker verification (enrollment, score normalization), POI-Forensics | X9 | Đã có cho face swap; **chưa tìm thấy cho lip-sync kèm 3 đầu ra** |
| **SSL đa ngôn ngữ cho audio** (XLS-R có tiếng Việt) | ASR / anti-spoofing | X3, X8 | Chuẩn; dùng làm nhánh audio + baseline |
| MLLM giải thích | Vision-language | Giải thích | Fine-tune 7B không vừa T4 → để sau, làm judge zero-shot |

Ghi chú riêng cho tiếng Việt:
- **Thanh điệu gần như không thấy trên môi.** Lỗi thanh điệu của TTS chỉ bắt được qua nhánh audio, không qua nhánh AV. Sync chủ yếu dựa vào nhịp đóng-mở miệng (môi khép ở /b/, /m/, mở ở nguyên âm), phần lớn không phụ thuộc ngôn ngữ.
- **AV-HuBERT pretrain bằng tiếng Anh** (rủi ro X3); bản "vox" pretrain trên VoxCeleb2 có nhiều ngôn ngữ. Cách giảm rủi ro: các nhánh không phụ thuộc ngôn ngữ (trace, landmark); XLS-R cho audio; về sau có thể thay bằng XLAVS-R (đa ngôn ngữ).

## 4. Baseline

| | **BL-1 · AVH-Align** (CVPR 2025) | **BL-2 · BioLip** (2026) |
|---|---|---|
| Vì sao | Real-only, có code, tốt nhất AV-1M nhóm real-only, bền với lối tắt; đo "AV mismatch" | Tốt nhất zero-shot trên **đúng các generator của v003** (VideoReTalking, Diff2Lip, IP-LAP, W2L); không phụ thuộc ngôn ngữ; rẻ |
| Cài lại trong repo | Đúng kiến trúc paper trên cache AV-HuBERT; bản `/sup` có giám sát | Đúng kiến trúc paper (302K tham số); landmark lấy từ dlib 20 điểm môi trong cache AV-HuBERT, thay cho MediaPipe 64 điểm của paper - **khác paper, ghi rõ** |
| Điểm yếu đã biết | X1 khi train có fake; X3 | IP-LAP, nén mạnh; không bắt được giọng giả |

Baseline tham chiếu (rẻ, chạy kèm):
- **SpeechForensics:** không cần train, chạy trên cùng cache.
- **AV-Lip-Sync+ (B-AVH):** có giám sát. v002: B-AVH là model **duy nhất** qua được các phép chẩn đoán xử lý (E6), vẫn là baseline khó vượt.
- **XLS-R + AASIST-lite:** chỉ audio, mốc cho đầu "audio fake?".

**Encoder AV-HuBERT:** mặc định **Large VoxCeleb2 + LRS3 fine-tune VSR** (`large_vox_433h`, đúng AVH-Align); Base giữ lại làm tuỳ chọn rẻ hơn. Lưu ý: fine-tune VSR chỉ dùng hình, nên đặc trưng **tiếng** của bản này có thể kém khớp hình hơn bản chỉ-pretrain (chưa kiểm).

**Lưu ý khi so BL-1** ○: AVH-Align gốc học chỉ từ real trên khối dữ liệu cỡ VoxCeleb2; v003 có ~0,9-1,2 giờ real mỗi part (600-800 clip × ~5,5 s), riêng train ~0,55-0,7 giờ. BL-1 có thể yếu vì thiếu dữ liệu chứ không vì phương pháp; ghi rõ trong báo cáo. Đây là một lý do chọn 20 clip / người thay vì 10 (v003_data §3.3).

## 5. Ba kiến trúc đề xuất

### A1 · AVCV: AV Cost-Volume Network

```
a, v (AV-HuBERT, 25 Hz) → chiếu học được Pa, Pv (Linear → LN, 256-d)
→ cost volume C[t, δ] = <Pa a_t, Pv v_{t+δ}> / √d,  δ ∈ [−15, 15]
→ CNN 2D trên (thời gian × độ trễ): 3 khối Conv2d 3×3, dilation (1, 2, 4) theo thời gian
→ soft-argmin theo δ: độ trễ ước lượng + độ nhọn đỉnh (entropy) + giá trị đỉnh
→ ghép [đặc trưng gộp theo δ ‖ a ‖ v chiếu] → TCN thời gian → 2 đầu:
   mismatch(t)  (lệch tiếng)   và   visual_fake(t) (miệng "khớp quá đều" của generator)
```

- **Giải:** X6 (tách "lệch" và "AI"), X7 (định vị theo ô 0,2 s, kèm độ trễ ước lượng).
- **Khác TriCue (v002):** tương quan được học (không phải cos thô); gộp 2D trên trục độ trễ (v002 làm 1D); có soft-argmin và độ nhọn đỉnh.
- Đầu độ trễ của A1 dùng được trong demo để ước lượng độ lệch toàn video.
- **Kiểm chứng:** ablation bỏ CNN 2D (giữ 1D như TriCue), bỏ chiếu học được (về cos thô).

### A2 · MTN: Native-resolution Mouth-Trace Network

```
crop miệng xám 96×96 ở ĐỘ PHÂN GIẢI GỐC (hộp từ cache AV-HuBERT, không phóng lên 224)
→ augment bất biến xử lý (JPEG q 30-95, đổi cỡ 0,5-1, làm nét, gamma), áp cho cả real và fake
→ 3 luồng residual: NPR  x − up(down(x))
                    temporal  x_t − (x_{t−1} + x_{t+1}) / 2
                    ảnh gốc chuẩn hoá
→ CNN 3D nhỏ (3 khối Conv3d 3×3×3, 32-64 kênh, stride không gian 2) → pool không gian
→ ghép động học môi (BioLip: 4 bậc trên landmark môi) → TCN → visual_fake(t)
```

- **Giải:** X1 (dấu vết upsampling chung cho mọi generator vẽ ở độ phân giải thấp rồi phóng), X2 / X4 (augment đối xứng), X5 (residual thời gian).
- **Rẻ:** cache crop khoảng 0,5 MB / mẫu, train end-to-end trên T4.
- Trả lời câu hỏi mở R1 của v001 / v002 (hộp miệng 45-85 px bị phóng lên 224 cho DINOv2).
- **Kiểm chứng:** bỏ NPR / bỏ temporal / bỏ augment / bỏ động học; so với DINOv2 224 (v002).

### A3 · TriHead-Ref: ba đầu ra + chuẩn hoá theo người

```
nhánh sync  = A1 (đặc trưng trước đầu ra)
nhánh trace = A2
nhánh audio = XLS-R-300M (trung bình lớp 6-12, 25 Hz) → AASIST-lite
             (graph attention theo thời gian × dải tần) → audio_fake(t)
hợp nhất muộn có cổng theo đầu ra:
  visual_fake ← trace + sync      audio_fake ← audio      mismatch ← sync (+ audio)
tham chiếu người: embedding trung bình của ≥ 3 clip real đã biết của cùng speaker → FiLM
  điều chỉnh 3 đầu (train: tham chiếu lấy từ clip real khác của cùng người trong cùng split)
```

- **Giải:** X9 (trong từng người v002 đạt 0,96, gộp chỉ 0,78), X8 (audio riêng, nhiều hệ TTS khi train), X6.
- 3 clip tham chiếu (~16 s) là đủ: Referee 1 clip 1-3 s đạt AUC 99,71 %, ID-Reveal 6 s ✔.
- **Protocol báo hai chế độ:** không tham chiếu / có tham chiếu (3 clip real của chính người đó giữ riêng làm tham chiếu ở test, không tính vào điểm).

### Bảng ablation

| Mã | Mô tả |
|---|---|
| A1 | AVCV đầy đủ · `avcv_no2d` · `avcv_cos` (cos thô) |
| A2 | MTN đầy đủ · `mtn_no_npr` · `mtn_no_temporal` · `mtn_no_kin` · `mtn_no_aug` |
| A3 | TriHead-Ref · `trihead_no_ref` · `trihead_no_audio` · `trihead_no_trace` · `trihead_no_sync` |
| BL | `avh_align`, `avh_align_sup`, `biolip`, `speechforensics`, `av_lipsync_plus`, `xlsr_aasist` |

Đội hình chính (`settings.MODELS`): 6 baseline + 3 đề xuất; 10 ablation ở `settings.ABLATIONS`; 3 seed (42, 43, 44).

## 6. Protocol và cổng

- Nhánh đánh giá: generator đã thấy / chưa thấy, giọng đã thấy / chưa thấy, người chưa thấy, định vị, lệch tiếng.
- Báo riêng: full / partial, độ dài đoạn giả, ô sát chỗ nối và ô giữa đoạn.
- Ba đầu ra chấm riêng: visual AUC, audio AUC, mismatch AUC, cộng điểm tổng "AI?" = max(visual, audio).
- **Cổng** (đặt trước, mỗi cổng có ΔAUC kèm CI bootstrap theo người):
  - A1 hơn AVH-Align trên đầu mismatch;
  - A2 hơn BioLip trên generator chưa thấy và ở CRF cao;
  - A3 có tham chiếu hơn không tham chiếu ≥ 0,03 AUC gộp.

  Trượt cổng nào thì báo đúng như vậy; không đổi cổng sau khi xem test.

## 7. Huấn luyện

| Hạng mục | Giá trị |
|---|---|
| Đặc trưng (bước prepare, Kaggle) | AV-HuBERT Large (25 Hz, kèm landmark dlib và hộp miệng), crop miệng độ phân giải gốc, XLS-R-300M |
| Nhãn | theo từng frame (0,04 s) cho mọi đầu ra; partial cho thấy ranh giới thật / giả trong cùng clip |
| Cân lớp | `pos_weights` cho từng tác vụ = số mẫu nhãn 0 / số mẫu nhãn 1 (kẹp trong 0,2-5); tác vụ "AI?" ở train là 3 / 6 = 0,5 → phần loss tương đương 50 / 50; không đổi tỉ lệ dữ liệu |
| Tối ưu | 30 epoch, patience 5, lr 3e-4 (AVH-Align 1e-5), weight decay 1e-4, batch 16 |
| Cân theo người | mỗi người 15-20 clip → lấy mẫu train theo 1 / số clip của người đó (`speaker_balance`), mỗi người đóng góp như nhau; mọi người cùng số clip thì giữ shuffle thường |
| Validation | AUC 4 tác vụ mỗi epoch (để chọn epoch và ghi log); một lượt quét validation thêm mỗi epoch |

### 7.1. Augment dịch tiếng −1…+2 frame

- Mỗi lần nạp mẫu, trượt chuỗi đặc trưng tiếng so với hình k ∈ [−1, +2] frame (dương = tiếng trễ), đều cho real và fake, **nhãn giữ nguyên**, frame ở mép bị bỏ. Không tạo file mới; mỗi epoch cùng một clip được nhìn với độ lệch hơi khác.
- Mục đích: model **không được dựa vào việc "khớp đúng 0 frame"** để đoán real / fake, che phần sai số còn lại sau khi sửa độ lệch real (±1 frame), và bền với video ngoài đời vốn hơi lệch. Augment không dạy model nhận ra lệch thật: việc đó do ② sham, ③ voice, ⑤ donor đảm nhận (lệch **nội dung**, không chỉ trượt vài frame).
- Căn cứ chọn biên độ (BT.1359, điểm 0 lệch ~1 frame, sai số đo, augment lớn có hại): v003_data §6.4.
- Áp cho mọi model dùng cả tiếng và hình, **trừ `avh_align` không giám sát**: nó học "độ lệch 0 = cặp đúng" từ real, dịch tiếng làm sai chính thứ nó học. (Lỗi cũ đã sửa 18/10: trước đó augment chỉ chạy cho model có ảnh miệng, nên `avcv`, `av_lipsync_plus`, `avh_align_sup` không được augment.)
- Thí nghiệm chọn biên độ để sau (§12).

## 8. Đánh giá

### 8.1. Chỉ số và phân rã

- 4 tác vụ (AI / visual / audio / mismatch); **AUC là chỉ số chính** (không phụ thuộc tỉ lệ lớp). Kèm AP, EER, F1, FAR.
- Phân rã theo ô, generator, giọng, full / partial, đã thấy / chưa thấy, theo người (AUC trong từng người).
- **AUC cân theo người** (`auc_speaker_balanced`, cột riêng trong `report.md`): AUC gộp với trọng số 1 / số file của người đó, để người 20 clip không lấn người 15 clip.
- **Ref-norm**: chuẩn hoá điểm theo 3 clip real tham chiếu của chính người đó; bỏ clip tham chiếu **và mọi file sinh từ chúng** khỏi phép đo.
- **CI bootstrap theo người** (người là đơn vị độc lập; full và partial cùng một lần chạy không độc lập).
- Test chạy **một lần**, khoá bằng `test-lock.json`; ngưỡng và checkpoint chọn trên validation trước khi mở test; mean ± std qua 3 seed.

### 8.2. Điểm vận hành: ngưỡng theo tỉ lệ báo nhầm

- Layton et al. ✔: không có tỉ lệ fake : real đúng cho mọi trường hợp; tỉ lệ train, ngưỡng và tỉ lệ fake ngoài thực tế phải khớp nhau. Tăng real khi train làm giảm báo nhầm nhưng cũng giảm bắt fake; tỉ lệ chỉ dịch điểm cân bằng, không làm model giỏi hơn. Khuyên báo nhiều chỉ số (ROC, PR, FPR / TPR, điểm thô) và nhiều base-rate. Giới hạn: thí nghiệm của họ trên giọng nói.
- Vấn đề cũ: ngưỡng **F1 tối đa trên validation** (~70 % fake) chỉnh cho môi trường "đa số là fake"; ngoài đời fake hiếm hơn nhiều → báo nhầm quá nhiều.
- Hiện hành:
  - ngưỡng **FPR 1 % / 5 %** trên real validation, báo TPR và FPR ở test;
  - **precision khi fake chiếm 50 % / 10 % / 1 %** ngoài thực tế (tính lại từ TPR / FPR, không cần sinh thêm);
  - FPR **riêng cho real và cho sham**;
  - demo dùng ngưỡng FPR 5 %.
- Độ chính xác FPR phụ thuộc số real ở test ○: 150-200 real / part; mỗi clip báo nhầm = 0,5-0,67 % ở 1 part, nên FPR 1 % chỉ đáng tin khi gộp nhiều part.

### 8.3. Định vị đoạn giả

- Trong clip (từ v002, thước đo sạch hơn vì so trong cùng clip): ô có điểm cao nhất có nằm trong đoạn không; AUC ô trong / ngoài đoạn; ô sát mép so với ô giữa đoạn.
- Chuẩn LAV-DF / AV-Deepfake1M (thêm 18/10, để so được với bài khác): **AP@IoU 0,5 / 0,75 / 0,95** trên mọi video test (video real chỉ góp báo nhầm), **AR@1/5/10** trung bình trên IoU 0,5-0,95. Đoạn ứng viên lấy từ các khúc frame vượt ngưỡng ở nhiều mức, bỏ trùng bằng NMS. Cách tạo ứng viên do mình chọn; muốn so số với một paper cụ thể thì kiểm lại cách họ tạo ứng viên.

## 9. Log W&B

Chốt 18/10: **chỉ cho quá trình train** (prepare, train, evaluate, report), project **private**, **không đẩy ảnh / video người thật**. Pipeline dữ liệu ghi ở ledger; bước sinh ghi ở `generation.json` / `records/`. Mục đích: log làm bằng chứng cho báo cáo, nên ghi đủ thứ quan trọng và tái lập được.

| Bước (job_type) | Ghi theo thời gian | Ghi lúc kết thúc / artifact |
|---|---|---|
| `prepare` | tiến độ, mẫu / phút theo encoder | tỉ lệ frame hợp lệ; **kiểm chéo độ lệch tiếng-hình bằng AV-HuBERT** trên clip real (đối chiếu SyncNet ở bước dữ liệu); `extraction.json` |
| `train` (mỗi model × seed) | mỗi epoch: loss tổng + từng đầu (frame / video), grad norm, lr, giây / epoch, VRAM, **AUC validation 4 tác vụ**, AUC validation theo ô / generator / giọng / kind | epoch tốt nhất, lý do dừng, giờ train, số tham số, có augment hay không; `best.pt` + `history.json` |
| `evaluate` (một lần) | - | bảng tổng (AUC, AP, EER, F1, FAR), phân rã, theo người, ref-norm, định vị, ROC, histogram điểm theo ô; `evaluation.json` + `test-lock.json` + `metrics.json` + `predictions-test.npz` |
| `report` | - | bảng AUC mean ± std theo seed, **các cổng + CI 95 %**; `report.md` + `gates.json` |

Mỗi run ghi config (cấu hình, seed, hash dữ liệu / test-lock); run id lưu cạnh kết quả, chạy lại sau khi bị ngắt ghi tiếp đúng run. Bật trong notebook: `USE_WANDB = True`, secret `WANDB_API_KEY`.

## 10. Demo (Kaggle, Gradio)

- **Chọn Kaggle** thay Colab: 2×T4 / P100 ổn định, quota 30 giờ / tuần, gắn sẵn dataset checkpoint và weight, dùng lại công thức cài worker; Colab hay bị ngắt.
- **Tab 1, tạo fake từ video real**: video ≤ 60 s → ô ③ / ④ / ⑤ (tải thêm tiếng thật khác) / ⑥, full hoặc partial (đoạn tự chọn), generator (mặc định Wav2Lip + LatentSync 1.5), giọng (mặc định F5-TTS VN), câu mới cho cụm sửa → video fake + nhãn thật. Dùng đúng hàm của pipeline sinh.
- **Tab 2, phân tích**: chuẩn hoá → **đo và sửa độ lệch tiếng-hình bằng SyncNet** (báo riêng "lệch thời gian ≈ X ms" nếu vượt ngưỡng BT.1359) → đặc trưng như lúc train → mỗi model chạy theo cửa sổ 7 s (model train trên clip 4-7 s) → điểm 4 tác vụ, kết luận theo ngưỡng FPR 5 %, timeline vẽ chồng nhãn thật. Thêm 1-3 video real cùng người làm tham chiếu (TriHead).
- Chưa có checkpoint: `RUN_DIR = None` → chỉ chạy `speechforensics` (không cần train) để thử luồng.
- Giới hạn 60 s (bộ trích AV-HuBERT). Link `gradio.live` công khai trong lúc chạy: chỉ dùng video được phép, tắt khi xong.
- Chạy ở máy Windows: cần xuất AV-HuBERT ra ONNX / TorchScript (không cần fairseq), crop miệng chuyển sang lấy vị trí từ YuNet (để sau).

## 11. Triển khai và trạng thái kiểm

- Code: `src/vn_av_df/training/` (prepare, data, models, train, evaluate, report), `features/encoders.py` (XLS-R, crop miệng), `features/avhubert*.py`, `tracking.py` (W&B), `demo/`; script `training/00-04_*.py`; notebook `notebooks/{prepare,train,demo}.ipynb` (sinh bằng `tools/make_notebooks.py`, không sửa tay). Môi trường worker: `environments/`.
- Khác paper khi cài lại: BioLip dùng 20 điểm môi dlib thay 64 điểm MediaPipe; AASIST-lite dùng self-attention thay HS-GAL đầy đủ.
- **Đã kiểm:** 78 test (data pipeline + sinh + train + demo) trên dữ liệu tổng hợp: plan → giọng → lip-sync → finalize đúng hợp đồng nhãn → train 1 epoch → test → report; forward / backward của 19 cấu hình; fbank khớp torchaudio; W&B với module giả; giao diện Gradio dựng thật.
- **Chưa kiểm** (cần GPU + Internet thật): AV-HuBERT Large, XLS-R, PhoWhisper thật; train thật; demo với checkpoint thật và video có mặt người.

## 12. Hạn chế và việc mở (phần model)

**Hạn chế:**
1. BL-1 có ít real hơn nhiều so với paper gốc (§4).
2. AV-HuBERT pretrain tiếng Anh; thanh điệu không thấy trên môi.
3. Chưa có bản test nhiễu thực tế (v003_data §11), nên cổng "A2 ở CRF cao" phải dựa vào bản đó khi có.
4. Ngưỡng FPR 1 % đo trên 150-200 real / part còn thô.

**Việc mở:**

| # | Việc | Ghi chú |
|---|---|---|
| 1 | Prepare + train + test part 1 thật | sau khi sinh xong |
| 2 | Thí nghiệm biên độ augment lệch tiếng (không / ±1 / −1…+2 / ±3) | AUC test + đường báo nhầm khi real bị dịch k = −3…+3; chọn mức nhỏ nhất mà báo nhầm trong vùng chấp nhận không tăng và AUC không giảm |
| 3 | Chấm trên bản test nhiễu thực tế | |
| 4 | Xuất AV-HuBERT ra ONNX cho demo local | |
| 5 | Hiệu chuẩn điểm (temperature scaling / isotonic) | việc mở từ v002 |
| 6 | Test ngoài (AVLips, FakeAVCeleb, LipSyncTIMIT) | sau khi khoá phương pháp |

## 13. Nhật ký câu hỏi và quyết định (phần model)

| Ngày | Câu hỏi / việc | Trả lời / quyết định |
|---|---|---|
| 15/10 | Chọn baseline, kiến trúc | BL-1 AVH-Align, BL-2 BioLip + 3 baseline tham chiếu; A1 AVCV, A2 MTN, A3 TriHead-Ref (§4-§5) |
| 17/10 | Demo sau này cần model đo độ lệch rồi dịch về 0? | đúng: SyncNet hoặc đầu độ trễ của A1; dịch về 0 rồi mới chấm "AI?" |
| 17/10 | AV-HuBERT cần Python 3.10, demo thì sao? | AV-HuBERT ở worker riêng; demo Windows thì xuất ONNX / TorchScript |
| 17/10 | Augment lệch tiếng tạo clip mới mỗi lần train? | không; trượt chuỗi đặc trưng khi nạp mẫu |
| 18/10 | Code đủ sinh / train / demo chưa? | demo đã bị xoá cùng v002 → làm lại (Kaggle, 2 tab) |
| 18/10 | W&B nên ghi gì | đề xuất ghi mọi bước; sau đó user: **chỉ train** (có prepare), không media |
| 18/10 | Tỉ lệ fake / real hợp lý? | không đổi tỉ lệ dữ liệu (loss đã cân); sửa **ngưỡng** sang FPR cố định và báo nhiều base-rate (§8.2) |
| 19/10 | Mỗi người 15-20 clip: đảm bảo không lệch | lấy mẫu train cân theo người; AUC cân theo người (§7, §8.1) |
| 18/10 | Full / partial có giúp model định vị? | có (nhãn theo frame); thêm AP@IoU + AR chuẩn LAV-DF / AV-1M (§8.3) |

## 14. Nguồn

- AVH-Align: arXiv 2412.00175 (CVPR 2025) · SpeechForensics: arXiv 2508.09913 (NeurIPS 2024) · BioLip: arXiv 2604.16808 · LoCC: arXiv 2606.22772 (ICME 2026) · PIA: arXiv 2510.14241 · LIPINC-V2: arXiv 2504.01470 · LipFD: arXiv 2401.15668 · X-AVDT: arXiv 2603.08483
- AV-Deepfake1M++: arXiv 2507.20579 · KLASSify: arXiv 2508.07337 · Pindrop: arXiv 2508.08141
- NPR: CVPR 2024, arXiv 2312.10461 · POI-Forensics: arXiv 2204.03083 · XLAVS-R: arXiv 2403.14402 · AV-LMMDetect: arXiv 2602.23393 · Omni-Fake: CVPR 2026
- Audio deepfake generalization: arXiv 2601.02914, 2604.26465
- Referee: arXiv 2510.27475 · ID-Reveal: ICCV 2021
- Layton et al., USENIX Security 2024 - [pdf](https://www.usenix.org/system/files/sec24fall-prepub-1479-layton.pdf) · Smeu et al.: arXiv 2412.00175 (CVPR 2025)
- LAV-DF: arXiv 2204.06228 · AV-Deepfake1M: arXiv 2311.15308 (chuẩn định vị AP / AR)
- Survey NCCU (thesis.lib.nccu.edu.tw) · 中国图象图形学报 survey (doi 10.11834/jig.230088)
