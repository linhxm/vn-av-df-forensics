> Bản sao từ nhật ký nghiên cứu (`RESEARCH_WORKLOG/v003_model.md`, ngoài repo) ngày 23/10/2026. Sửa ở nhật ký rồi chép lại vào đây.

# v003 · Model: phương pháp, huấn luyện, đánh giá

Cập nhật 23/10/2026 (khảo sát phương pháp 15/10, research theo lớp 22/10, đội hình chốt 23/10). File 3/3 của v003:
- [v003_data.md](v003_data.md): dữ liệu real, số người và số clip, độ lệch tiếng-hình.
- [v003_generation.md](v003_generation.md): sáu ô, generator và giọng, số fake mỗi clip.
- **v003_model.md** (file này): khảo sát phương pháp, lỗi của detector hiện nay, đội hình 4 baseline cài lại đúng code gốc + 2 model đề xuất, protocol và cổng, huấn luyện, đánh giá, W&B, demo.

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

## 4. Đội hình model (chốt 23/10/2026)

Đội hình cũ (15/10: 6 baseline + 3 đề xuất + 10 ablation, có bản cài lại khác paper như BioLip 20 điểm môi, AASIST-lite) đã thay bằng **4 baseline cài lại đúng code gốc + 2 model đề xuất + 3 ablation**: 9 cấu hình × 3 seed = 27 lượt train mỗi part. Căn cứ chọn: board kiến trúc `slides/v003/model_board.html` (research theo lớp 22/10) và trao đổi 23/10 (§13).

Mỗi baseline trả lời một câu hỏi, vì 6 ô của v003 có giả hình, giả tiếng, giả cả hai và lệch tiếng thật (② sham). Hai baseline hình + đồng bộ (bản đề xuất ban đầu) để hở hẳn câu hỏi "audio fake?" và không tách được ② / ③.

| Mã | Model | Câu hỏi | Code gốc | Ghi chú |
|---|---|---|---|---|
| BL-V | `lipinc_v2` · LIPINC-V2 (Datta et al. 2025) | visual | github.com/skrantidatta/LIPINC-V2 (Keras) | port sang PyTorch đúng kích thước từng lớp; chỉ cho điểm video |
| BL-A | `ssl_aasist` · XLS-R 300M + AASIST (Tak et al., Odyssey 2022) | audio | github.com/TakHemlata/SSL_Anti-spoofing (MIT) | fine-tune cả XLS-R như gốc, RawBoost algo 3 |
| BL-M | `avh_align` · AVH-Align (Smeu et al., CVPR 2025) | mismatch / AI | github.com/bit-ml/AVH-Align | chỉ học real; khởi tạo từ checkpoint công bố |
| BL-M/sup | `avh_align_sup` · AVH-Align/sup (cùng paper) | AI | như trên (avh_sup) | học có nhãn trên cả 6 ô: so công bằng về dữ liệu với M1 |
| BL-Ens | `bl_ens` = max(BL-V, BL-A) | AI | - | không train; chuẩn hoá z-score theo real validation rồi lấy max |
| M1 | `trihead_v2` | cả 3 + AI | - | model chính (§5) |
| M2 | `avh_align_plus` | mismatch + AI | - | nâng cấp trực tiếp BL-M trên cùng nền (§5) |

Bỏ khỏi đội hình: `biolip` (không có code gốc, bản cài khác paper), `av_lipsync_plus`, `xlsr_aasist` (AASIST-lite), `avcv`, `mtn` (thành nhánh của M1), 10 ablation cũ. `speechforensics` chỉ còn cho demo khi chưa có checkpoint.

### 4.1. Baseline: cài lại và khác biệt với code gốc

Đã tải và đọc code gốc của cả 3 repo (23/10) trước khi cài; khác biệt nào cũng ghi ở đây và trong docstring của lớp tương ứng (`training/models.py`).

**BL-M · AVH-Align** ✔ (code + checkpoint):
- Φ = `FusionModel` gốc: chiếu riêng hình / tiếng 1024 → 512, ghép, MLP 1024 → 512 → 256 → 128 → 1 (LayerNorm + ReLU). Bản cũ trong repo **thiếu 2 lớp chiếu** (ghép thẳng 2048 → 512), đã sửa.
- Loss InfoNCE: mỗi frame hình t so với 31 frame tiếng t−15…t+15, tiếng ngoài clip là vector 0 (như gốc, không mask). Điểm frame −Φ(v_t, a_t), điểm video log-sum-exp.
- Công thức gốc: Adam lr 1e-5, ReduceLROnPlateau (×0,1, patience 5), dừng sớm 10, tối đa 500 epoch, batch 512 frame (checkpoint ghi `batch_size=512`).
- Checkpoint công bố `AVH-Align_AV1M.pt` (train trên 45 k video real AV-1M, đặc trưng `self_large_vox_433h` = đúng encoder của v003). Đã kiểm 23/10: nạp vào `AlignMLP` của repo cho **điểm video giống hệt** `FusionModel` gốc, logit theo độ trễ lệch < 3·10⁻⁶ ●.
- Khác gốc: khởi tạo từ checkpoint công bố rồi học tiếp trên real của v003 (giải X12: v003 chỉ có 360-480 real train / part, paper cho thấy 8,7 k mẫu đã tụt AUC); batch theo clip (8 clip ≈ 1.100 frame); loss trung bình thay vì tổng (Adam bất biến tỉ lệ); bỏ frame mất mặt.

**BL-M/sup · AVH-Align/sup** ✔: cùng Φ; điểm frame Φ(v_t, a_t), điểm video s = log-sum-exp, loss cross-entropy trên (−s, s); Adam lr 1e-3, **batch 1 video**, 100 epoch, dừng sớm 10 (đúng `avh_sup`). Nhãn = AI? của video. Khởi tạo ngẫu nhiên như gốc. Thêm augment dịch tiếng −1…+2 theo protocol v003 (§7.1).

**BL-V · LIPINC-V2** ✔ (code Keras):
- Tiền xử lý đúng `utils.py` gốc: căn mặt `FaceAligner` 256 px (mắt trái ở 0,40), crop hộp môi 48-67 rồi resize 144 × 64; 5 frame liên tiếp có độ mở môi ≥ 1 % + 3 frame "cùng tư thế" (4 tỉ lệ môi / má lệch < 3 điểm phần trăm, sau mỗi frame chọn bỏ 3 frame); 7 ảnh hiệu |R_t+1 − R_t|.
- Model đúng `model.py`: 2 vision temporal transformer (patch 8, d 128, 4 head key_dim 128, 2 lớp spatial + 2 lớp temporal, post-norm, LN eps 1e-3) → bản đồ 8 × 8 × 3; 3 attention chéo (softmax(QKᵀ/√192)V trên 2 trục cuối); CNN 64-64-128-128 (BN, MaxPool 2 rồi 4) → Dense 1024 → 1024 → 2 (lớp 1 = real). 4,59 M tham số ●.
- Loss: cross-entropy + 5 × inconsistency loss. Paper: SSIM giữa mọi cặp frame trên lớp đầu của MSTIE, BCE với nhãn real. Bản công bố tính SSIM bằng numpy nên **không có gradient**; ở đây dùng SSIM toàn cục khả vi trên đặc trưng chiếu patch của nhánh RGB.
- Công thức gốc: Adam lr 1e-3, epsilon 0,1, batch 64, 100 epoch.
- Khác gốc: landmark lấy từ cache AV-HuBERT (cùng bộ dlib 68 điểm) thay vì chạy lại dlib trên mặt đã căn; hộp mặt để tính tỉ lệ lấy từ landmark (gốc dùng hộp dò dlib); clip không tìm được bộ frame theo luật gốc thì chọn theo luật dự phòng (gốc bỏ video, ghi `fallback` trong cache); ảnh chia 255 (gốc đưa 0-255); ảnh hiệu tính bằng số có dấu (gốc trừ uint8 nên tràn); nhãn = visual fake (BL-V là chuyên gia hình). Chỉ có điểm video → không có phép đo định vị.

**BL-A · XLS-R + AASIST** ✔ (code, MIT):
- Back-end AASIST đúng `model.py` gốc (LL 1024 → 128, max-pool 3 × 3, 6 residual block kiểu RawNet2, đồ thị phổ 42 nút + đồ thị thời gian, 4 HS-GAL có nút master, MGO 2 nhánh, readout). 0,45 M tham số + XLS-R 300M fine-tune toàn bộ.
- Train đúng script DF: cửa sổ 64.600 mẫu (~4 s, ngắn hơn thì lặp), RawBoost algo 3 (chép `RawBoost.py`, MIT), Adam lr 1e-6, weight decay 1e-4, batch 14, cross-entropy trọng số [0,1 giả; 0,9 thật], 100 epoch; tiếng không chuẩn hoá; không che thời gian (fairseq `mask=False`).
- Khác gốc: XLS-R qua `transformers` (cùng trọng số `facebook/wav2vec2-xls-r-300m`, tắt SpecAugment); clip 4-7 s nên khi đánh giá trượt cửa sổ 4 s bước 1 s, điểm frame = max các cửa sổ phủ nó, điểm video = max; nhãn cửa sổ = có frame tiếng giả; AMP fp16 trên GPU; checkpoint chọn theo val loss.

**BL-Ens**: hai model chuyên trách chạy riêng rồi lấy max sau khi chuẩn hoá theo real validation. Đây là cách gộp người đọc sẽ hỏi "sao không làm vậy cho đơn giản"; M1 phải thắng nó thì mới có lý do để gộp trong một model.

**Encoder AV-HuBERT** giữ **Large VoxCeleb2 + LRS3 fine-tune VSR** (`large_vox_433h`, lớp cuối): đúng đặc trưng của checkpoint AVH-Align. Lưu ý (○): fine-tune VSR có thể làm lớp trên chuyên cho hình (Pasad 2021 thấy điều tương tự ở wav2vec 2.0); chưa dò lớp.

## 5. Hai model đề xuất

### M1 · TriHead-v2 (`trihead_v2`, 1,94 M tham số)

```
nhánh sync  = cost volume ±15 (chiếu học được + cos) → CNN 2D → soft-argmin, đỉnh, entropy → TCN
nhánh trace = crop miệng 96×96 độ phân giải gốc → [ảnh, NPR, residual thời gian] → Conv3D ×3
              ‖ động học môi (y, vận tốc, gia tốc) + tư thế đầu (yaw / pitch / roll + vận tốc) → TCN
nhánh audio = XLS-R đóng băng, 3 nhóm lớp (1-4, 5-8, 9-12) trộn bằng softmax học được → 2 lớp attention
tham chiếu  = 3 clip real cùng người → FiLM  [h·(1+γ)+β ‖ h − ref]
đầu ra      : visual ← trace + sync · audio ← audio · mismatch ← sync + audio ; AI? = max(visual, audio)
```

So với TriHead cũ, gộp 6 sửa nhóm P1 trên board kiến trúc:
1. **NPR đúng paper** (Tan et al. CVPR 2024 ✔): hiệu với pixel tham chiếu của ô 2 × 2 (bản cũ dùng trung bình 2 × 2 = residual làm mờ).
2. **Luồng tư thế đầu** (LipDA, ICML 2026 ◐): lip-sync vẽ môi mới nhưng giữ đầu của câu gốc; tư thế từ solvePnP trên 6 landmark đã có trong cache (`mouth.pose`).
3. **Trọng số học được trên nhóm lớp XLS-R** (Lee et al. 2022 ✔: lớp 5 tốt nhất; SLS ◐). Cache `ssl` giờ là [T, 3, 1024].
4. **Gộp điểm top-5 frame** (`pool = "topk5"`): đoạn giả ngắn nhất 0,3 s ≈ 7 frame, top-10 % của 175 frame = 18 frame làm loãng ●.
5. **Pseudo-fake thay đoạn tiếng** khi nạp clip real (p = 0,25): đoạn 0,3-3 s lấy từ clip real khác cùng người; nhãn mismatch = 1, AI = 0 (Astrid et al. 2025 ✔: AUC chéo bộ 61 → 80 %).
6. **Gradient reversal lên đầu phân loại người** của nhánh audio và trace (Dao et al. 2026 ◐: EER −17 %); λ tăng 0 → 0,3 theo lịch DANN.

- Giải: X1, X4, X11 (trace + tư thế đầu), X6 (pseudo-fake, đầu mismatch riêng), X7 (top-5), X8 (nhóm lớp), X9 (tham chiếu + GRL).
- Ablation: `trihead_v2_no_sync`, `trihead_v2_no_trace`, `trihead_v2_no_ref`. Không bỏ nhánh audio vì nó là nhánh duy nhất nuôi đầu audio.

### M2 · AVH-Align+ (`avh_align_plus`, 1,92 M tham số)

```
Φ = AlignMLP của AVH-Align, khởi tạo từ checkpoint công bố (học tiếp với lr 1e-5)
cost volume C[t, δ] = Φ(v_t, a_t+δ), δ ∈ [−15, 15] ‖ kênh vị trí δ/15 → CNN 2D → soft-argmin, đỉnh, entropy
‖ vận tốc môi + tư thế đầu → TCN → 2 đầu: ai, mismatch ; pseudo-fake thay đoạn tiếng (p = 0,25)
```

- Học có giám sát trên **cùng dữ liệu** với BL-M/sup → kết luận sạch: cùng nền Φ, hơn được vì (1) đọc cả dải độ trễ thay vì một độ trễ, (2) môi–đầu, (3) pseudo-fake.
- Rẻ hơn M1, không cần crop miệng hay XLS-R.

### Bảng cấu hình

| Mã | Cấu hình | Tham số học ● |
|---|---|---|
| BL-V | `lipinc_v2` | 4,59 M |
| BL-A | `ssl_aasist` | 0,45 M + XLS-R 300M |
| BL-M | `avh_align`, `avh_align_sup` | 1,74 M |
| M1 | `trihead_v2` · `_no_sync` · `_no_trace` · `_no_ref` | 1,94 M · 0,86 M · 1,58 M · 1,76 M |
| M2 | `avh_align_plus` | 1,92 M |

`settings.MODELS` (6) + `settings.ABLATIONS` (3), 3 seed (42, 43, 44). Encoder cần trích suy ra từ model (`prepare.encoders_for`): avhubert, mouth, lipinc, xlsr, wave.

## 6. Protocol và cổng

- Nhánh đánh giá: generator đã thấy / chưa thấy, giọng đã thấy / chưa thấy, người chưa thấy, định vị, lệch tiếng.
- Báo riêng: full / partial, độ dài đoạn giả, ô sát chỗ nối và ô giữa đoạn.
- Ba đầu ra chấm riêng: visual AUC, audio AUC, mismatch AUC, cộng điểm tổng "AI?" = max(visual, audio). Model một điểm (BL-V, BL-M) dùng cùng điểm cho mọi tác vụ để thấy nó thực ra bắt cái gì.
- **Cổng** (đặt trước, ΔAUC kèm CI bootstrap theo người, `report.GATES`):

| # | So sánh | Tác vụ | Tập |
|---|---|---|---|
| 1 | M1 − BL-Ens | AI? | toàn bộ test |
| 2 | M1 − BL-V | visual | generator chưa thấy (LatentSync, Diff2Lip) |
| 3 | M1 − BL-A | audio | giọng chưa thấy (VoxCPM2, viXTTS, F5 inpainting) |
| 4 | M1 − BL-M/sup, M1 − BL-M | mismatch | toàn bộ |
| 5 | M1 − M1 no_ref | AI? | toàn bộ (≥ 0,03) |
| 6 | M1 − M1 no_sync (mismatch), M1 − M1 no_trace (visual) | | toàn bộ |
| 7 | M2 − BL-M/sup, M2 − BL-M | mismatch | toàn bộ |

  Báo nhầm trên ② sham báo riêng (`fpr_by_cell`). Trượt cổng nào thì báo đúng như vậy; không đổi cổng sau khi xem test.

## 7. Huấn luyện

| Hạng mục | Giá trị |
|---|---|
| Đặc trưng (bước prepare, Kaggle) | AV-HuBERT Large (25 Hz, kèm landmark dlib và hộp miệng), crop miệng độ phân giải gốc + tư thế đầu, 8 crop LIPINC-V2, XLS-R-300M 3 nhóm lớp, waveform 16 kHz |
| Nhãn | theo từng frame (0,04 s) cho mọi đầu ra; partial cho thấy ranh giới thật / giả trong cùng clip |
| Cân lớp | `pos_weights` cho từng tác vụ = số mẫu nhãn 0 / số mẫu nhãn 1 (kẹp trong 0,2-5) cho model dùng loss mặc định |
| Công thức train | **mỗi baseline dùng đúng công thức của code gốc** (`model.train_options`, §4.1); M1, M2: AdamW lr 3e-4, weight decay 1e-4, batch 16, 30 epoch, patience 5 (M2: Φ học với lr 1e-5) |
| Cân theo người | lấy mẫu train theo 1 / số clip của người đó (`speaker_balance`) |
| Validation | AUC 4 tác vụ mỗi epoch; checkpoint chọn theo val loss |
| Tài sản tải trước | XLS-R gốc (BL-A), checkpoint AVH-Align (commit `55401de`, BL-M và M2): `training/00_setup.py` hoặc `cli.model_assets` trong notebook train |
| Train tiếp khi bị ngắt | `last.pt` sau mỗi epoch (model, optimizer, scheduler, AMP scaler, lịch sử, trạng thái ngẫu nhiên kể cả thứ tự lấy mẫu); chạy lại thì tiếp từ epoch sau, đổi cấu hình thì báo lỗi thay vì trộn; xong thì xoá `last.pt` |
| Nguồn gốc | `runs/<RUN>/provenance.json` (dữ liệu, chữ ký cache, commit, tài sản) thay cho ledger: train không ghi vào ledger của data_pipeline |
| Đo trước | `training/05_benchmark.py`: giây / bước, bộ nhớ GPU, ước giờ mỗi epoch và tối đa mỗi seed, trên cache thật hoặc tensor tổng hợp (`--synthetic`) |

### 7.1. Augment dịch tiếng −1…+2 frame

- Mỗi lần nạp mẫu, trượt chuỗi đặc trưng tiếng so với hình k ∈ [−1, +2] frame (dương = tiếng trễ), đều cho real và fake, **nhãn giữ nguyên**, frame ở mép bị bỏ. Không tạo file mới; mỗi epoch cùng một clip được nhìn với độ lệch hơi khác.
- Mục đích: model **không được dựa vào việc "khớp đúng 0 frame"** để đoán real / fake, che phần sai số còn lại sau khi sửa độ lệch real (±1 frame), và bền với video ngoài đời vốn hơi lệch. Augment không dạy model nhận ra lệch thật: việc đó do ② sham, ③ voice, ⑤ donor đảm nhận (lệch **nội dung**, không chỉ trượt vài frame).
- Căn cứ chọn biên độ (BT.1359, điểm 0 lệch ~1 frame, sai số đo, augment lớn có hại): v003_data §4.3.4.
- Áp cho mọi model có giám sát dùng cả tiếng và hình (`avh_align_sup`, `trihead_v2` và ablation, `avh_align_plus`), **trừ `avh_align` không giám sát**: nó học "độ lệch 0 = cặp đúng" từ real, dịch tiếng làm sai chính thứ nó học. BL-V và BL-A chỉ một phía nên không áp. (Lỗi cũ đã sửa 18/10: trước đó augment chỉ chạy cho model có ảnh miệng.)
- Thí nghiệm chọn biên độ để sau (§12).

## 8. Đánh giá

### 8.1. Chỉ số và phân rã

- 4 tác vụ (AI / visual / audio / mismatch); **AUC là chỉ số chính** (không phụ thuộc tỉ lệ lớp). Kèm AP, EER, F1, FAR.
- BL-Ens tính ở bước test từ `predictions-val.npz` / `predictions-test.npz` của BL-V và BL-A cùng seed (`evaluate.evaluate_ensemble`), có đủ phép đo điểm video, không có định vị.
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
- Chỉ cho model có điểm theo frame (`frame_level`); LIPINC-V2 và BL-Ens chỉ có điểm video nên bỏ phần này. BL-A có điểm frame từ cửa sổ trượt 4 s (thô).
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

- Code: `src/vn_av_df/training/` (prepare, data, models, train, evaluate, report, rawboost), `features/encoders.py` (XLS-R nhóm lớp, crop miệng + tư thế đầu, khung LIPINC-V2, waveform), `features/avhubert*.py`, `tracking.py` (W&B), `demo/`; script `training/00-04_*.py`; notebook `notebooks/{prepare,train,demo}.ipynb` (sinh bằng `tools/make_notebooks.py`, không sửa tay). Môi trường worker: `environments/`.
- Khác code gốc khi cài lại: §4.1.
- **Đã kiểm (23/10):** 52 test trên dữ liệu tổng hợp (thêm: ngắt giữa epoch rồi train tiếp đúng epoch, provenance, benchmark), trong đó forward / backward cả 10 cấu hình (gồm XLS-R tí hon cho BL-A); train → test → BL-Ens → report end-to-end cho cả đội hình; luật chọn frame LIPINC-V2 (luật gốc + dự phòng); NPR theo định nghĩa pixel tham chiếu; gradient reversal; pseudo-fake; encoder LIPINC / waveform / tư thế đầu trên video tổng hợp; demo với LIPINC-V2 (khung không theo frame) và M2. **AVH-Align: nạp checkpoint công bố cho điểm giống hệt `FusionModel` gốc** ●. Ruff sạch.
- **Chưa kiểm** (cần GPU + Internet thật): AV-HuBERT Large, XLS-R thật (trích và fine-tune), train thật, thời gian / bộ nhớ của BL-A trên T4, tỉ lệ clip LIPINC rơi vào luật dự phòng trên dữ liệu thật, demo với checkpoint thật.

## 12. Hạn chế và việc mở (phần model)

**Hạn chế:**
1. BL-M có ít real hơn nhiều so với paper gốc; đã giảm bằng cách khởi tạo từ checkpoint công bố (45 k real AV-1M) (§4.1).
2. AV-HuBERT pretrain tiếng Anh; thanh điệu không thấy trên môi.
3. Chưa có bản test nhiễu thực tế (v003_data §7.4), nên chưa có phép đo M1 / BL-V ở CRF cao.
4. Ngưỡng FPR 1 % đo trên 150-200 real / part còn thô.
5. BL-V và BL-Ens chỉ có điểm video: không so định vị được với chúng.

**Việc mở:**

| # | Việc | Ghi chú |
|---|---|---|
| 1 | Prepare + train + test part 1 thật | sau khi sinh xong; ghi tỉ lệ `fallback` của cache lipinc |
| 2 | Thí nghiệm biên độ augment lệch tiếng (không / ±1 / −1…+2 / ±3) | AUC test + đường báo nhầm khi real bị dịch k = −3…+3; chọn mức nhỏ nhất mà báo nhầm trong vùng chấp nhận không tăng và AUC không giảm |
| 3 | Chấm trên bản test nhiễu thực tế | |
| 4 | Xuất AV-HuBERT ra ONNX cho demo local | |
| 5 | Hiệu chuẩn điểm (temperature scaling / isotonic) | việc mở từ v002 |
| 6 | Test ngoài (AVLips, FakeAVCeleb, LipSyncTIMIT) | sau khi khoá phương pháp |
| 7 | Đo bộ nhớ / thời gian thật (`training/05_benchmark.py`) trên T4 và RTX 4090 | BL-A không vừa T4 thì giảm batch, giữ lr gốc |
| 8 | Chấm thêm checkpoint LIPINC-V2 công bố (Keras, zero-shot) | cần chuyển trọng số hdf5 sang PyTorch |
| 9 | Dò lớp AV-HuBERT cho sync (board: "Dò lớp AV-HuBERT") | chỉ cần một lần trích |

## 13. Nhật ký câu hỏi và quyết định (phần model)

| Ngày | Câu hỏi / việc | Trả lời / quyết định |
|---|---|---|
| 15/10 | Chọn baseline, kiến trúc | BL-1 AVH-Align, BL-2 BioLip + 3 baseline tham chiếu; A1 AVCV, A2 MTN, A3 TriHead-Ref (đã thay bằng đội hình 23/10, §4) |
| 17/10 | Demo sau này cần model đo độ lệch rồi dịch về 0? | đúng: SyncNet hoặc đầu độ trễ của A1; dịch về 0 rồi mới chấm "AI?" |
| 17/10 | AV-HuBERT cần Python 3.10, demo thì sao? | AV-HuBERT ở worker riêng; demo Windows thì xuất ONNX / TorchScript |
| 17/10 | Augment lệch tiếng tạo clip mới mỗi lần train? | không; trượt chuỗi đặc trưng khi nạp mẫu |
| 18/10 | Code đủ sinh / train / demo chưa? | demo đã bị xoá cùng v002 → làm lại (Kaggle, 2 tab) |
| 18/10 | W&B nên ghi gì | đề xuất ghi mọi bước; sau đó user: **chỉ train** (có prepare), không media |
| 18/10 | Tỉ lệ fake / real hợp lý? | không đổi tỉ lệ dữ liệu (loss đã cân); sửa **ngưỡng** sang FPR cố định và báo nhiều base-rate (§8.2) |
| 19/10 | Mỗi người 15-20 clip: đảm bảo không lệch | lấy mẫu train cân theo người; AUC cân theo người (§7, §8.1) |
| 18/10 | Full / partial có giúp model định vị? | có (nhãn theo frame); thêm AP@IoU + AR chuẩn LAV-DF / AV-1M (§8.3) |
| 22/10 | Research kiến trúc theo từng lớp, đa ngành, đa ngôn ngữ | board `slides/v003/model_board.html`: 12 điểm yếu (thêm X11 generator tối ưu SyncNet, X12 ít real), ~35 thẻ, 9 phát hiện code ↔ paper |
| 23/10 | Quá nhiều model; chỉ giữ 2 baseline tốt nhất (cài lại đúng gốc) + 1-2 đề xuất | 2 baseline (AVH-Align, LIPINC-V2) không phủ giả tiếng → **3 baseline chuyên trách** (BL-V, BL-A, BL-M) + BL-Ens; M1 + M2 |
| 23/10 | Sao BL-M chỉ học real, sao không học full? | paper chọn real-only để tránh lối tắt; nhưng so với M1 (có giám sát) thì cần bản học full → chạy **cả hai** (`avh_align`, `avh_align_sup`) |
| 23/10 | "Khớp tiếng–môi" là gì khi generator vẽ đúng từng frame? | khớp **khẩu hình ↔ âm**, không phải thời gian; thời gian đã sửa bằng SyncNet, augment −1…+2 làm thời gian vô dụng với model |
| 23/10 | Chốt: 9 cấu hình × 3 seed | cài đặt đầy đủ (§4-§5); 50 test qua |

## 14. Nguồn

- AVH-Align: arXiv 2412.00175 (CVPR 2025) · SpeechForensics: arXiv 2508.09913 (NeurIPS 2024) · BioLip: arXiv 2604.16808 · LoCC: arXiv 2606.22772 (ICME 2026) · PIA: arXiv 2510.14241 · LIPINC-V2: arXiv 2504.01470 · LipFD: arXiv 2401.15668 · X-AVDT: arXiv 2603.08483
- AV-Deepfake1M++: arXiv 2507.20579 · KLASSify: arXiv 2508.07337 · Pindrop: arXiv 2508.08141
- NPR: CVPR 2024, arXiv 2312.10461 · POI-Forensics: arXiv 2204.03083 · XLAVS-R: arXiv 2403.14402 · AV-LMMDetect: arXiv 2602.23393 · Omni-Fake: CVPR 2026
- Audio deepfake generalization: arXiv 2601.02914, 2604.26465
- Referee: arXiv 2510.27475 · ID-Reveal: ICCV 2021
- Layton et al., USENIX Security 2024 - [pdf](https://www.usenix.org/system/files/sec24fall-prepub-1479-layton.pdf) · Smeu et al.: arXiv 2412.00175 (CVPR 2025)
- LAV-DF: arXiv 2204.06228 · AV-Deepfake1M: arXiv 2311.15308 (chuẩn định vị AP / AR)
- Code gốc đã đọc 23/10: github.com/bit-ml/AVH-Align (commit 55401de, checkpoint `AVH-Align_AV1M.pt`) · github.com/skrantidatta/LIPINC-V2 · github.com/TakHemlata/SSL_Anti-spoofing (MIT; RawBoost: Tak et al., ICASSP 2022)
- Tak et al., "Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation", Odyssey 2022 (arXiv 2202.12233) · Jung et al., AASIST, ICASSP 2022 (arXiv 2110.01200)
- Lee et al. 2022, XLSR theo lớp (arXiv 2204.02639) · Pasad et al. 2021 (arXiv 2107.04734) · Astrid et al. 2025 (arXiv 2501.08137) · LipDA (ICML 2026 poster) · Dao et al. 2026 (arXiv 2602.20805) · Chandrasekaran et al. 2009 (PLoS Comput Biol)
- Survey NCCU (thesis.lib.nccu.edu.tw) · 中国图象图形学报 survey (doi 10.11834/jig.230088)
