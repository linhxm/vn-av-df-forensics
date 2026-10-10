# VN-AV-DF Forensics

Phát hiện và định vị lip-sync deepfake trong video tiếng Việt: **video YouTube → cắt/duyệt → chia split → sinh fake → trích feature → train + test → so sánh → demo**.
Điểm 0-1 là mức nghi ngờ, chưa hiệu chuẩn thành xác suất; đoạn không quan sát được bị mask, không coi là real.

## Tổng quan

| # | Bước | Chạy ở đâu | Entrypoint | Đầu ra (đường dẫn trong repo) |
|---|---|---|---|---|
| 1 | Thu thập → tải → cắt → duyệt → export (gán split) | Local (cắt có thể Kaggle) | `data_pipeline/steps/01…05`, [cut.ipynb](notebooks/cut.ipynb) | `data_pipeline/exports/vn-av-df-data/<part>/` |
| 2 | Sinh real/fake/sham | Kaggle | [generate.ipynb](notebooks/generate.ipynb) | `datasets/vn-av-df-data/<part>/` + `generation/plans/<part>.json` |
| 3 | Duyệt mẫu sinh → chốt → ZIP | Local | `generation/04…06` | `datasets/vn-av-df-data/<part>.zip` |
| 4 | Trích feature (cache) | Kaggle | [prepare.ipynb](notebooks/prepare.ipynb) | `cache/vn-av-df-data/<encoder>/` |
| 5 | Train + test | Kaggle (hoặc local) | [train.ipynb](notebooks/train.ipynb), `training/02…03` | `runs/<RUN_NAME>/` |
| 6 | So sánh kiến trúc theo part | Local hoặc Kaggle | [compare.ipynb](notebooks/compare.ipynb) | `reports/compare/` |
| 7 | Demo | Local | `training/04_demo.py` | http://127.0.0.1:8000 |

**Part** (`vn-av-df-data-part1`, `-part2`, …) là một đợt dữ liệu bổ sung, xử lý và đóng gói riêng. Local chọn part bằng `PART` trong [data_settings.py](data_settings.py); Kaggle chỉnh trong cell cấu hình của notebook.

### Quy ước tên và đường dẫn (Local ↔ Kaggle)

Mọi notebook đọc input bằng **một đường dẫn đầy đủ** (copy từ panel *Input* của Kaggle; nếu ZIP giữ thêm một thư mục bên trong, notebook tự tìm vào) và ghi Output **dưới `vn-av-df-forensics/` đúng đường dẫn như ở local**, chỉ giữ thư mục kết quả. Tải Output về là đặt thẳng vào repo.

| Dữ liệu | Thư mục local | Kaggle dataset = tên ZIP | Tạo ZIP |
|---|---|---|---|
| Raw part | `data_pipeline/data/raw/vn-av-df-data/vn-av-df-data-partN/` | `vn-av-df-raw-partN` | nén thư mục part (có `videos/`, `sources.jsonl`, `logs/`) |
| Part sạch (có split) | `data_pipeline/exports/vn-av-df-data/vn-av-df-data-partN/` | `vn-av-df-clean-partN` | nén thư mục part |
| Part đã sinh, đã chốt | `datasets/vn-av-df-data/vn-av-df-data-partN/` | `vn-av-df-data-partN` | `generation/06_export.py` (giữ thư mục `vn-av-df-data-partN/` bên trong) |
| Cache feature | `cache/vn-av-df-data/{fate,avhubert,dinov2}/` | `vn-av-df-features-partN` | nén các thư mục encoder (lệnh ở mục 4) |

- **Tên thư mục part (`vn-av-df-data-partN`) đi vào hash lựa chọn dữ liệu.** Luôn upload part đã sinh bằng ZIP của `06_export.py`; đổi tên thư mục part làm cache/checkpoint không khớp.
- `PREVIOUS_OUTPUT` (cut, generate) trỏ tới thư mục `vn-av-df-forensics` trong Output phiên trước để chạy tiếp.
- Log W&B nằm ở `/kaggle/working/wandb` (vẫn trong Output, `wandb sync` lại được).

### Giới hạn Kaggle cần nhớ

| Giới hạn | Ảnh hưởng | Cách làm |
|---|---|---|
| 12 giờ/phiên, ~30 giờ GPU/tuần, Output ≤ 20 GB | Prepare cả 3 encoder (~10,7 h part1) sát giới hạn | Tách phiên (FATE / AV-HuBERT+DINOv2); generate có `TIME_BUDGET_HOURS`, chạy tiếp bằng `PREVIOUS_OUTPUT` |
| Dataset tạo từ Output chỉ lấy **500 file** | Cache ~6000 file/encoder, part đã sinh hàng nghìn clip | Tải Output về, ZIP ở local rồi upload thành dataset |
| `kaggle kernels output` chỉ lấy version mới nhất, skip chậm, đứt mạng là dừng | Tải Output lớn hay lỗi | `python tools/kaggle_pull.py <owner/slug> .. "<regex>"` (bỏ qua file đã có, tự thử lại) |
| Kaggle CLI mới cần Python ≥ 3.11 | Env dự án là 3.10 | Cài CLI ở Python base: `D:\anaconda3\python.exe -m pip install -U kaggle`, `kaggle auth login` |

## Cài đặt local

Python 3.10 (env conda `vn-av-df`), chạy từ thư mục `vn-av-df-forensics`. Notebook Kaggle tự cài thư viện và tải model.

```powershell
python -m pip install -e ./data_pipeline -e ".[training]"   # data, duyệt, train/test CPU, demo
python data_pipeline/steps/00_setup.py                       # YuNet + Silero VAD
python training/00_setup.py --encoder fate                   # chỉ cần cho demo B-FATE
python training/00_setup.py --encoder avhubert               # demo B-AVH/P2 (cần worker, mục 7)
python training/00_setup.py --encoder dinov2                 # demo P2
```

`python 00_install.py` (`PROFILE = "data"` hoặc `"demo"`) thay được lệnh pip đầu. Không cần Wav2Lip/MuseTalk ở local.

## 1. Dữ liệu real và split

Điền `data_pipeline/data/sources/vn-av-df-data/<part>/videos.csv` (xem [mẫu](data_pipeline/configs/videos.example.csv)):

```csv
url,speaker_id
https://www.youtube.com/playlist?list=PLxxxxxxxx,speaker_01
https://www.youtube.com/watch?v=xxxxxxxxxxx,speaker_02
```

`url` là video hoặc playlist (mọi video trong playlist nhận `speaker_id` của dòng). Giữ `speaker_id` nhất quán giữa các part. Chạy lần lượt trong `data_pipeline/steps/`:

| Bước | Việc làm | Kết quả chính |
|---|---|---|
| `01_collect.py` | Bung playlist, bỏ trùng, hỏi metadata YouTube, kiểm chất lượng | `selected_videos.csv`, `video_metadata.csv`, `skipped_videos.csv` |
| `02_download.py` | Tải bản ≤1080 (cạnh ngắn), kiểm lại file | `data/raw/<dataset>/<part>/` |
| `03_cut.py` | VAD + dò mặt, cắt clip 5-8 s, 25 fps CFR, cạnh ngắn ≤1080 | `data/candidates/<dataset>/<part>/` |
| `04_review.py` | Duyệt tại http://127.0.0.1:8001 | `review.csv` |
| `05_export.py` | Dựng part sạch từ clip keep **và gán split** train/validation/test | `exports/<dataset>/<part>/` (+ `split-lock.json`) |

- **Split** gán ở `05_export` theo nhóm liên thông người/nguồn/hash (cả nhóm cùng một split), tỷ lệ `SPLIT_RATIOS` (tạm 70/15/15), `SPLIT_SEED` trong `data_settings.py`; cần ≥3 nhóm độc lập. Mỗi part chia độc lập; muốn part mới kế thừa split part cũ thì điền `SPLIT_HISTORY` bằng `split-lock.json` của các part trước. Đổi split = export lại và sinh lại part.
- **Trang duyệt** (04): danh sách có lọc theo quyết định/speaker/nguồn, phím tắt `K` giữ, `R` loại, `U` chưa rõ, `←/→`, `N` tới clip chưa chắc, `Space` phát/dừng, `L` lặp; chỉnh tốc độ phát. Clip mới cắt ở trạng thái *chưa rõ*.
- **Kiểm chất lượng (01):** loại video private/đã xoá/livestream, fps gốc <25, cạnh ngắn <720 px, dài <5 s hoặc >12 giờ. `skipped_videos.csv` ghi lý do; `speaker_conflict = yes` → sửa `videos.csv` trước khi tải.
- **Mạng:** lỗi IPv6 thì thêm `--force-ipv4`; bị chặn 403 thì thêm `--cookies-from-browser firefox`.
- **Bổ sung video:** thêm URL rồi chạy lại 01 → 05; mỗi bước chỉ xử lý phần mới, giữ quyết định đã duyệt. **Đổi luật cắt** (`configs/data.yaml`, code, model) thì 03 báo lỗi: xoá `data/candidates/<dataset>/<part>` để cắt lại.

**Cắt trên Kaggle** (khi máy chậm): upload thư mục raw thành `vn-av-df-raw-partN` → chạy [cut.ipynb](notebooks/cut.ipynb) (`RAW_PART`) → tải `vn-av-df-forensics/data_pipeline/data/candidates/...` về đúng chỗ → `04_review.py`. Code/config cắt phải giống local (có chữ ký); hết phiên thì đặt `PREVIOUS_OUTPUT` để cắt tiếp.

## 2. Sinh fake (Kaggle)

Thiết kế **2×2**:

| | Không dấu vết AI | Có dấu vết AI |
|---|---|---|
| **Tiếng khớp miệng** | real | fake `source`: miệng vẽ theo tiếng gốc (chỉ có artifact) |
| **Tiếng lệch miệng** | sham: ghép tiếng thật khác của cùng người, nhãn AI = 0 | fake `donor`: miệng vẽ theo tiếng thật khác của cùng người (mối đe doạ chính) |

1. Upload part sạch thành `vn-av-df-clean-partN`; trong [generate.ipynb](notebooks/generate.ipynb) đặt `PART`, `CLEAN_PART`.
2. **Một generator mỗi phiên** (nhẹ hơn, chỉ cài worker cần dùng): phiên 1 `GENERATORS = ["wav2lip_gan"]` (sinh luôn real/sham), phiên 2 `GENERATORS = ["musetalk_1_5"]` + `PREVIOUS_OUTPUT`. Hoặc `"all"` trong một phiên. Hết giờ/còn generator chưa chạy → `status: partial` kèm `remaining_by_generator`.
3. Mặc định: Wav2Lip GAN cho mọi split, MuseTalk 1.5 chỉ ở test; train chỉ `donor`, validation/test có `donor` + `source`; fake cục bộ 0,4/0,8/1,6/2,4 s; `CLIPS_PER_SPLIT = 0` lấy toàn bộ (2 = chạy thử). Clip không có donor cùng `speaker_id` bị bỏ và ghi vào plan.
4. Thời gian mỗi phiên (giây/cặp theo generator) ghi vào `generation.json` → `sessions` (dùng cho bảng chi phí).

Nhãn mỗi mẫu: `label` (AI), `fake_intervals`, `audio_mode` (`donor`/`source`/null), `av_mismatch_intervals` (chỉ head sync của P2 dùng, không phải nhãn deepfake).

## 3. Duyệt, chốt, đóng gói (local)

Tải `vn-av-df-forensics/datasets/vn-av-df-data/<part>/` và `vn-av-df-forensics/generation/plans/<part>.json` về đúng chỗ, đặt `PART` trong `data_settings.py`, chạy trong `generation/`:

- `04_review.py`: http://127.0.0.1:8002. **Mặc định mọi mẫu `keep`**, chỉ đánh `reject`/`uncertain` mẫu lỗi. Real gốc chiếu cạnh mẫu đang duyệt; timeline vạch đoạn fake (đỏ) và đoạn ghép tiếng của sham (cam); lọc theo split/variant/generator/audio_mode.
- `05_finalize.py`: chốt mẫu keep (bất biến; đổi thì làm part mới).
- `06_export.py`: tạo `datasets/vn-av-df-data/<part>.zip` → upload thành `vn-av-df-data-partN`.

## 4. Trích feature (Kaggle, [prepare.ipynb](notebooks/prepare.ipynb))

```python
DATA_PARTS = [Path("/kaggle/input/datasets/<user>/vn-av-df-data-part1")]
ENCODERS = "all"     # hoặc ["fate"], ["avhubert", "dinov2"], ["dinov2"] (cần avhubert/ trong CACHE_INPUTS)
CACHE_INPUTS = []    # cache đã có: trích tiếp, hoặc lấy hộp miệng AV-HuBERT cho DINOv2
```

Part1 (2954 mẫu, 2×T4): FATE ~5,6 h, AV-HuBERT ~4,7 h (4 worker), DINOv2 ~0,4 h → nên 2 phiên. Output chỉ giữ `cache/vn-av-df-data/<encoder>/` đã chọn. Vì giới hạn 500 file, tạo dataset `vn-av-df-features-partN` ở local:

```powershell
python tools/kaggle_pull.py <user>/<prepare-slug> .. "/cache/"      # tải Output về repo (thư mục cha của vn-av-df-forensics)
cd cache/vn-av-df-data; tar -a -cf D:\vn-av-df-features-part1.zip fate avhubert dinov2   # chỉ thư mục encoder
```

## 5. Train + test ([train.ipynb](notebooks/train.ipynb))

```python
DATA_PARTS = [Path("/kaggle/input/datasets/<user>/vn-av-df-data-part1")]   # trùng lúc prepare
CACHE_INPUTS = [Path("/kaggle/input/datasets/<user>/vn-av-df-features-part1")]
ARCHITECTURES = "all"   # hoặc danh sách, vd. ["fate_gru"], ["p2_sync_only", "p2_concat"]
SEEDS = [42, 43, 44]
RUN_NAME = "train_part1_run01"
RUN_TEST = True         # test ngay sau train (ngưỡng/checkpoint chọn trên validation, khoá test-lock.json)
```

- Không tải encoder, không cài worker; gắn cache bằng symlink và kiểm cache đủ encoder, đúng hash dữ liệu. Output chỉ giữ `runs/<RUN_NAME>/`.
- `best.pt` mỗi detector chọn theo validation loss; báo cáo validation/test từng model/seed (theo ô 2×2, theo nhánh P2, theo generator); không tự xếp hạng. Đổi dữ liệu/model/hyperparameter thì dùng `RUN_NAME` mới.
- Tải về: `python tools/kaggle_pull.py <user>/<train-slug> .. "/runs/"`. Hai lần train cùng dữ liệu có thể gộp vào một `runs/<RUN_NAME>` (gộp `training.json`/`evaluation.json`, dựng lại `test-lock.json`) hoặc để riêng và liệt kê cả hai trong `compare.ipynb`.
- **Local (CPU):** đặt `RUN_NAME`, `ARCHITECTURES`, `DEVICE = "cpu"` trong `settings.py`, cache ở `cache/vn-av-df-data/`, rồi `python training/02_train.py` → `python training/03_evaluate.py`.

| Kiến trúc (`"all"`) | Vai trò |
|---|---|
| `fate_gru` **B-FATE** | PE-AV Small + FATE (frozen) → projection 128 → BiGRU 2 tầng |
| `avh_tcn` **B-AVH** | AV-HuBERT Base LRS3 (frozen) → concat A/V → TCN; cùng encoder với P2 nhưng không mô hình hoá consistency |
| `p2_syncartifact` **P2** (đề xuất) | Nhánh sync (residual A→V, học không thấy fake) + nhánh artifact (DINOv2 ViT-S/14, crop miệng RGB 224) + gate fusion |
| `p2_sync_only`, `p2_artifact_only` | Chỉ một nhánh của P2 |
| `p2_concat` | Hai nhánh ghép thẳng, không gate |
| `p2_sync_seen_fake` | Nhánh sync không đóng băng, học cả fake |
| `avh_realrecon` | Residual A→V + visual với gate (thiết kế P1 cũ) |

P2 học 3 stage: **A** R học A→V chỉ trên real; **S** head sync học real (0) với sham và real dịch lệch ±3-15 frame (1) rồi đóng băng, **không thấy fake**; **C** nhánh artifact + fusion học nhãn AI (sham = 0), thêm head phụ artifact. Residual A→V lấy cảm hứng từ [AuViRe](https://github.com/mever-team/auvire), không phải bản tái hiện.

## 6. So sánh ([compare.ipynb](notebooks/compare.ipynb))

Đọc `runs/` đã test, **mỗi part một bộ bảng** (+ bảng gộp khi nhiều part): video (mean ± std qua seed, seed-ensemble), theo generator, theo ô 2×2 (+ false alarm real/sham), theo nhánh, định vị thời gian, **paired ΔAUC** (bootstrap theo clip gốc, có CI), chi phí train/trích/sinh và dự tính GPU cho part mới. Ghi `reports/compare/<split>_<part>.md` + PNG. Ở local chạy bằng env `vn-av-df` (cài `ipykernel` một lần, xem đầu notebook); trên Kaggle attach Output train rồi sửa `RUNS`.

## 7. Demo (local)

1. **B-AVH/P2:** cài worker AV-HuBERT theo [environments/README.md](environments/README.md), điền `AVH_PYTHON` trong `settings.py`.
2. Đặt `runs/<RUN_NAME>/` (có `training.json`), chọn `RUN_NAME`, `DEMO_METHOD`, `SEEDS` (hoặc `CHECKPOINT` tới `best.pt`) trong `settings.py`.
3. `cd demo; npm ci; npm run build; cd ..` rồi `python training/04_demo.py`.

Demo nhận clip một người nói, tối đa 60 s: timeline AI + ngưỡng (từ validation); P2 thêm điểm/timeline nhánh sync và artifact (chưa có ngưỡng riêng; nhánh sync cao cũng có thể chỉ do lồng tiếng thông thường). Bảng "Kết quả test" lấy từ `runs/<RUN_NAME>/evaluation.json`.

## Kiểm thử

```powershell
python -m pytest -q tests data_pipeline/tests
python -m ruff check src tests settings.py data_settings.py data_pipeline/src data_pipeline/tests data_pipeline/steps tools
```

Test dùng generator giả và feature fixture, nên **không chứng minh checkpoint thật chạy được hay detector chính xác**.

## Phiên bản và checkpoint được chọn

Bảng mô tả tài sản được cấu hình sử dụng; không có nghĩa tất cả đã có ở local. Checkpoint generator được tải trên Kaggle.

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
| Detector của project | `best.pt` riêng cho từng kiến trúc/seed, tạo sau khi train; demo dùng kèm encoder tương ứng |

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `src/vn_av_df/` | Generation, feature, model, train/test, demo |
| `data_pipeline/` | Thu thập, cắt, duyệt, export + split (`steps/01…05`) |
| `generation/`, `training/` | Entrypoint local (`00…06`, `00…04`) |
| `notebooks/` | `cut`, `generate`, `prepare`, `train` (Kaggle) và `compare` |
| `tools/` | `kaggle_pull.py` |
| `environments/` | Worker MuseTalk / AV-HuBERT |
| `demo/` | Frontend |
| `tests/`, `data_pipeline/tests/` | Kiểm thử |

Không commit weights, external, cache, datasets, runs, reports. Nhật ký nghiên cứu nằm trong `RESEARCH_WORKLOG`.
