# VN-AV-DF Forensics

Phát hiện và định vị lip-sync deepfake trong video tiếng Việt: **video YouTube → cắt/duyệt → sinh fake → train → so sánh → demo**.
Điểm 0–1 là mức nghi ngờ, chưa hiệu chuẩn thành xác suất; đoạn không quan sát được bị mask, không coi là real.

## Tổng quan

| Giai đoạn | Chạy ở đâu | Entrypoint | Đầu ra |
|---|---|---|---|
| 1. Thu thập, cắt, duyệt | Local (cắt có thể chạy Kaggle) | `data_pipeline/steps/01…05`, [cut.ipynb](notebooks/cut.ipynb) | Part clip sạch |
| 2. Sinh fake | Kaggle | [generate.ipynb](notebooks/generate.ipynb) | Part real/fake chờ duyệt |
| 2b. Duyệt fake | Local | `generation/04…07` | Part đã chốt (ZIP) |
| 3. Train, báo cáo | Kaggle | [train.ipynb](notebooks/train.ipynb) | `runs/<RUN_NAME>/` |
| 4. Demo | Local | `training/05_demo.py` | http://127.0.0.1:8000 |

Dữ liệu chia theo **part** (`vn-av-df-data-part1`, `-part2`, …), mỗi part là một đợt bổ sung, xử lý và đóng ZIP riêng. Local chọn part bằng `PART` trong [data_settings.py](data_settings.py); trên Kaggle chỉnh trong cell cấu hình của notebook.

Ba notebook Kaggle ghi kết quả lên **W&B** (project `vn-av-df`): cut (thống kê, bảng clip, clip mẫu), generate (số mẫu theo ô 2×2, video mẫu), train (mỗi detector/seed một run, ghi **ngay sau từng epoch** của stage A, S và detector cùng GPU/RAM theo thời gian; sau báo cáo bổ sung metric, bảng ô 2×2, ảnh vào đúng run đó). Cần Kaggle secret `WANDB_API_KEY`; đặt `USE_WANDB = False` để tắt.

## Phương pháp

| Mã | Tên | Mô tả |
|---|---|---|
| `fate_gru` | **B-FATE** | PE-AV Small + FATE (frozen) → projection 128 → BiGRU 2 tầng |
| `avh_tcn` | **B-AVH** | AV-HuBERT Base LRS3 (frozen) → concat A/V → TCN; cùng encoder với P2 nhưng không mô hình hoá consistency |
| `p2_syncartifact` | **P2** (đề xuất) | Nhánh sync (residual A→V) + nhánh artifact (DINOv2 ViT-S/14, crop miệng RGB 224) + gate fusion |

P2 học 3 stage:
- **A:** R học A→V chỉ trên real.
- **S:** head sync học real (0) với sham và real dịch lệch ±3–15 frame (1), rồi đóng băng; **không bao giờ thấy fake**.
- **C:** nhánh artifact + fusion học nhãn AI (sham = 0), thêm head phụ artifact.

Ablation: `avh_realrecon`, `p2_sync_only`, `p2_artifact_only`, `p2_concat`, `p2_sync_seen_fake`; `fate_linear` để sanity check. Residual A→V lấy cảm hứng từ [AuViRe](https://github.com/mever-team/auvire), không phải bản tái hiện.

## Cài đặt local

Python 3.10, chạy từ thư mục `vn-av-df-forensics`. Notebook Kaggle tự cài thư viện và tải model, không cần bước này.

```powershell
python -m pip install -e ./data_pipeline -e .      # Data, duyệt fake
python -m pip install -e ".[training]"             # Thêm thư viện inference cho demo
python data_pipeline/steps/00_setup.py             # YuNet + Silero VAD
python training/00_setup.py --encoder fate         # Demo B-FATE
python training/00_setup.py --encoder avhubert     # Demo B-AVH/P2
python training/00_setup.py --encoder dinov2       # Demo P2
```

`python 00_install.py` (`PROFILE = "demo"` hoặc `"data"`) thay được hai lệnh pip. `training/00_setup.py` không có `--encoder` thì tải theo `ARCHITECTURES` trong `settings.py`. Không cần tải Wav2Lip/MuseTalk về local.

## 1. Dữ liệu

Điền `data_pipeline/data/sources/vn-av-df-data/<part>/videos.csv` (xem [mẫu](data_pipeline/configs/videos.example.csv)):

```csv
url,speaker_id
https://www.youtube.com/playlist?list=PLxxxxxxxx,speaker_01
https://www.youtube.com/watch?v=xxxxxxxxxxx,speaker_02
```

`url` là video hoặc playlist (mọi video trong playlist nhận `speaker_id` của dòng). Giữ `speaker_id` nhất quán giữa các part. Sau đó chạy lần lượt trong `data_pipeline/steps/`:

| Bước | Việc làm | Kết quả chính |
|---|---|---|
| `01_collect.py` | Bung playlist, bỏ trùng, hỏi metadata YouTube, kiểm chất lượng | `selected_videos.csv` (để tải), `video_metadata.csv` (thống kê), `skipped_videos.csv` (video bị bỏ và lý do) |
| `02_download.py` | Tải bản ≤1080 (cạnh ngắn), kiểm lại file | `data/raw/<part>/` |
| `03_cut.py` | VAD + dò mặt, cắt clip 5–8 s, chuẩn hoá 25 fps CFR, cạnh ngắn ≤1080 | `data/candidates/<part>/` |
| `04_review.py` | Duyệt tại http://127.0.0.1:8001 (keep, tiếng khớp người trên hình), Ctrl+C khi xong | `review.csv` |
| `05_export.py` | Dựng part sạch từ clip keep | `exports/<part>/` |

- **Kiểm chất lượng (01):** loại video private/đã xoá/livestream, fps gốc <25, cạnh ngắn <720 px, dài <5 s hoặc >12 giờ. Video 4K lấy bản 1080 có sẵn của YouTube.
- **Video bị bỏ (01):** `skipped_videos.csv` ghi mọi video không được chọn, cột `type` = `same_part` (trùng trong part, giữ lần đầu) / `other_part` (đã thuộc part khác) / `rejected` (không đạt kiểm) / `error` (lỗi mạng, chạy lại để hỏi lại), kèm `reason`. `speaker_conflict = yes` → sửa `videos.csv` trước khi tải.
- **Mạng:** lỗi IPv6 thì thêm `--force-ipv4`; bị chặn 403 thì thêm `--cookies-from-browser firefox`.
- **Bổ sung video:** thêm URL vào `videos.csv` rồi chạy lại 01 → 05. Mỗi bước chỉ xử lý phần mới và giữ quyết định đã duyệt. Đổi `speaker_id` của video đã chọn sẽ báo lỗi.
- **Đổi luật cắt** (`configs/data.yaml`, code hay model) thì bước 03 báo lỗi. Xoá `data/candidates/<part>` để cắt và duyệt lại.
- VAD/YuNet chỉ tạo ứng viên, không thay được bước duyệt. Clip chồng nhau không làm tăng số mẫu độc lập.

```text
data/raw/<dataset>/<part>/            data/candidates/<dataset>/<part>/
├── videos/        video gốc          ├── clips/       clip .mp4
├── sources.jsonl  manifest           ├── review.csv   manifest + quyết định duyệt
└── logs/          nhật ký tải        └── logs/        nhật ký cắt (để chạy tiếp)
```

**Cắt trên Kaggle** (tuỳ chọn, khi máy local chậm):
1. Upload thư mục raw của part thành Kaggle dataset.
2. Chạy [cut.ipynb](notebooks/cut.ipynb).
3. Giải nén ZIP vào `data/candidates/vn-av-df-data/`, rồi chạy `04_review.py`.

Code và config cắt phải giống local (có kiểm chữ ký). Hết phiên Kaggle thì attach output cũ vào `PREVIOUS_CUT` để cắt tiếp.

## 2. Sinh fake

Dữ liệu theo **thiết kế 2×2**:

| | Không dấu vết AI | Có dấu vết AI |
|---|---|---|
| **Tiếng khớp miệng** | real | fake `source`: miệng vẽ theo tiếng gốc (chỉ có artifact) |
| **Tiếng lệch miệng** | sham: ghép tiếng thật khác của cùng người, nhãn AI = 0 | fake `donor`: miệng vẽ theo tiếng thật khác của cùng người (mối đe doạ chính) |

Mặc định trong [generate.ipynb](notebooks/generate.ipynb):
- Generator: Wav2Lip GAN cho mọi split; MuseTalk 1.5 chỉ dùng ở test.
- Chế độ fake: train chỉ `donor`; validation/test có `donor` + `source`.
- Fake cục bộ dài 0,4 / 0,8 / 1,6 / 2,4 s.
- Chia split 80/10/10 theo nhóm người/nguồn.
- `CLIPS_PER_SPLIT = 2` để chạy thử; `0` lấy toàn bộ.

Clip không tìm được clip donor cùng `speaker_id` sẽ bị bỏ qua và ghi vào plan, nên cần điền `speaker_id` đầy đủ.

Nhãn mỗi mẫu:
- `label` (AI), `fake_intervals`, `audio_mode` (`donor` / `source` / null);
- `av_mismatch_intervals`: chỉ head sync của P2 dùng, không phải nhãn deepfake.

Sau khi tải output về local, chạy trong `generation/`:
- `04_review.py`: duyệt tại http://127.0.0.1:8002.
- `05_finalize.py`: chốt clip keep.
- `07_export.py`: tạo ZIP để train.

`06_import_external.py` nhập output từ generator khác. Đổi plan, code hay weights thì đặt tên dataset mới.

## 3. Train và báo cáo

Chỉnh trong cell cấu hình của [train.ipynb](notebooks/train.ipynb):

```python
DATASET_PARTS = ["vn-av-df-data-part1"]  # hoặc nhiều part, hoặc "all"
ARCHITECTURES = ["fate_gru"]             # hoặc ["p2_syncartifact"], hoặc "all"
SEEDS = [42]                             # thí nghiệm: [42, 43, 44]
RUN_NAME = "train_part1_gru_run01"
RESUME = False                           # True chỉ khi cùng data/config/code/cache
```

Notebook chạy lần lượt: setup encoder → cache đặc trưng → train → báo cáo → test (tuỳ chọn) → ZIP.
- **Train:** `best.pt` của mỗi detector chọn theo validation loss; notebook không tự chọn kiến trúc tốt nhất. Thêm part vào tập train thì dùng `RUN_NAME` mới.
- **Báo cáo** (validation, mỗi model/seed): loss/AUC theo epoch, ROC/PR, confusion matrix, timeline mẫu. Mỗi model có bảng theo 4 ô 2×2 (`by_condition`); P2 thêm AUC của từng nhánh (`branches`) và biểu đồ stage S.
- **Test:** `RUN_TEST = True` để chạy; ghi `test.json`, `evaluation.json` và `test-lock.json`.

[compare.ipynb](notebooks/compare.ipynb) mới là khung, chưa so sánh được.

## 4. Demo

1. **Chỉ cần cho B-AVH/P2:** cài worker AV-HuBERT theo [environments/README.md](environments/README.md), rồi điền `AVH_PYTHON` trong `settings.py`.
2. **Lấy detector đã train:** giải nén output `train.ipynb` sao cho có `runs/<RUN_NAME>/training.json`. Trong `settings.py`, chọn `RUN_NAME`, `DEMO_METHOD`, `SEEDS`, hoặc trỏ `CHECKPOINT` tới `best.pt`. Demo cần cả encoder lẫn detector tương ứng.
3. **Build frontend và chạy:**

   ```powershell
   cd demo; npm ci; npm run build; cd ..
   python training/05_demo.py
   ```

Demo nhận clip một người nói, tối đa 60 s. Kết quả gồm timeline AI và ngưỡng (lấy từ validation). Với P2, demo hiện thêm điểm và timeline riêng của nhánh sync và nhánh artifact. Hai nhánh này chưa có ngưỡng riêng; nhánh sync cao cũng có thể chỉ do lồng tiếng thông thường.

## Kiểm thử

```powershell
python -m pytest -q
python -m ruff check src tests settings.py
```

Test dùng generator giả và feature fixture, nên **không chứng minh checkpoint thật chạy được hay detector chính xác**. Cần pilot với dữ liệu thật.

## Phiên bản và checkpoint được chọn

Bảng dưới mô tả tài sản được cấu hình sử dụng; không có nghĩa tất cả đã được tải về local. Checkpoint generator được tải trên Kaggle.

| Thành phần | Phiên bản/checkpoint được chọn |
|---|---|
| YuNet | Bản March 2023 của OpenCV Zoo: `face_detection_yunet_2023mar.onnx` |
| Silero VAD | **v6.0**: `silero_vad.onnx` |
| PE-AV Small | `facebook/pe-av-small`, file `model.safetensors`; revision mặc định `dd050762bb9704ae9cd996ca45532a98f81d817e` |
| FATE adapter | `Guan123/fate`, file `adapter_model.safetensors`; revision mặc định `8463ab93a644a22bd85db91e7e77d99ebe1ec5e0` |
| FATE source | `guankaisi/FATE`; commit `beae95aeb6f72cf1751d06d1428931016a7a1867` |
| AV-HuBERT | **Base, LRS3, clean-pretrain, iteration 4**, chưa fine-tune: `base_lrs3_iter4.pt` |
| Dlib landmark cho AV-HuBERT | Landmark 68 điểm: `shape_predictor_68_face_landmarks.dat`, tải dạng `.dat.bz2` rồi giải nén |
| Mean-face cho AV-HuBERT | `20words_mean_face.npy` từ `mpc001/Lipreading_using_Temporal_Convolutional_Networks`; dữ liệu tham chiếu tiền xử lý |
| DINOv2 cho nhánh artifact P2 | `facebook/dinov2-small` (ViT-S/14), `model.safetensors`; revision `ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| Wav2Lip | **Wav2Lip GAN**, lưu thành `Wav2Lip-SD-GAN.pt` |
| Face detector của Wav2Lip | **S3FD**, nguồn `s3fd-619a316812.pth`, lưu thành `s3fd.pth` |
| MuseTalk | **MuseTalk 1.5**, repository `TMElyralab/MuseTalk`: `musetalkV15/unet.pth` và cấu hình `musetalkV15/musetalk.json` |
| VAE của MuseTalk | `stabilityai/sd-vae-ft-mse`: `diffusion_pytorch_model.bin` |
| Audio encoder của MuseTalk | `openai/whisper-tiny`: `pytorch_model.bin` |
| DWPose của MuseTalk | `yzd-v/DWPose`: `dw-ll_ucoco_384.pth` |
| Face parsing của MuseTalk | `79999_iter.pth` và `resnet18-5c106cde.pth` |
| Detector của project | `best.pt` riêng cho từng kiến trúc/seed, tạo sau khi train trên Kaggle; demo dùng kèm encoder tương ứng |

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `src/vn_av_df/` | Code chung |
| `data_pipeline/` | Thu thập dữ liệu |
| `generation/`, `training/` | Entrypoint |
| `notebooks/` | Notebook Kaggle |
| `environments/` | Worker |
| `demo/` | Frontend |
| `tests/` | Kiểm thử |

Không commit weights, external, cache, datasets, runs, outputs. Nhật ký nghiên cứu nằm trong `RESEARCH_WORKLOG`.
