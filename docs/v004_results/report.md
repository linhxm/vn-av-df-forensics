# Kết quả v004 (report.md sinh từ Output Kaggle P1d + P2; tiêu đề gốc in nhầm "v003")

Generator đã thấy khi train: iplap, musetalk_1_5, videoretalking, wav2lip_gan
Giọng đã thấy khi train: f5_vi, omnivoice, seedvc, vieneu

## Cách đọc

- Số seed mỗi model: 1. ± là độ lệch chuẩn giữa các seed (ddof 0); một seed thì không có ±.
- CI 95 % của cổng: bootstrap theo speaker (1000 lần, phân vị 2,5-97,5 %), trên seed đầu tiên: chỉ gồm biến động do mẫu người ở test, không gồm biến động do khởi tạo / thứ tự dữ liệu (chạy 1 seed).
- Không dò siêu tham số: baseline theo công thức của code gốc, M1 / M2 đặt trước; checkpoint và ngưỡng chọn trên validation; số lần mỗi model dùng test: `test-log.json`.
- Môi trường, commit, dữ liệu, cache: `provenance.json`; thời gian từng lần train: `<model>_seed<seed>/history.json`.

## AUC test (mean ± std theo seed)

| Model | ai | ai (cân theo người) | visual | audio | mismatch | ai (ref-norm) | định vị visual: ô cao nhất trong đoạn | định vị mismatch: AUC trong / ngoài đoạn |
|---|---|---|---|---|---|---|---|---|
| avh_align | 0.382 | 0.383 | 0.225 | 0.602 | 0.898 | 0.436 | 0.702 | 0.904 |
| avh_align_plus | 0.927 | 0.927 | - | - | 0.957 | 0.789 | - | 0.973 |
| avh_align_sup | 0.871 | 0.871 | 0.912 | 0.530 | 0.432 | 0.833 | 0.897 | 0.646 |
| bl_ens_av | 0.930 | 0.930 | 0.912 | 0.963 | - | 0.844 | - | - |
| lipinc_v2 | 0.608 | 0.608 | 0.662 | 0.472 | 0.437 | 0.620 | - | - |
| speechforensics | 0.490 | 0.490 | 0.456 | 0.513 | 0.668 | 0.522 | 0.320 | 0.814 |
| trihead_v4 | 0.968 | 0.968 | 0.992 | 0.959 | 0.968 | 0.943 | 0.986 | 0.985 |
| trihead_v4_no_prosody | 0.971 | 0.972 | 0.992 | 0.959 | 0.961 | 0.941 | 0.985 | 0.982 |
| xlsr_frozen | 0.742 | 0.742 | - | 0.963 | - | 0.737 | - | - |

## Cổng (ΔAUC, bootstrap theo speaker, seed đầu tiên)

| So sánh | Tác vụ | ΔAUC | CI 95% | Speaker |
|---|---|---|---|---|
| M2 hơn BL-M/sup (cùng dữ liệu) (avh_align_plus − avh_align_sup) | mismatch | +0.525 | [+0.510; +0.539] | 31 |
| M2 hơn BL-M (avh_align_plus − avh_align) | mismatch | +0.059 | [+0.052; +0.066] | 31 |
| V2 M3 hơn ghép muộn BL-M/sup + BL-A2 (trihead_v4 − bl_ens_av) | ai | +0.038 | [+0.031; +0.045] | 31 |
| V3 M3 hơn M2 (trihead_v4 − avh_align_plus) | ai | +0.041 | [+0.033; +0.050] | 31 |
| V6 M3 hơn BL-A2, giọng chưa thấy (trihead_v4 − xlsr_frozen) | audio | -0.005 | [-0.012; +0.001] | 31 |
| V7 M3 hơn BL-M ở mismatch (trihead_v4 − avh_align) | mismatch | +0.070 | [+0.063; +0.077] | 31 |
| V9 nhánh ngữ điệu ↔ đầu giúp (≥ 0,01) (trihead_v4 − trihead_v4_no_prosody) | ai | -0.003 | [-0.006; -0.001] | 31 |
| V10 nhánh ngữ điệu ↔ đầu giúp ở ⑤ donor (trihead_v4 − trihead_v4_no_prosody) | visual | -0.000 | [-0.001; +0.000] | 31 |
| V11 nhánh ngữ điệu ↔ đầu giúp mismatch (trihead_v4 − trihead_v4_no_prosody) | mismatch | +0.007 | [+0.005; +0.009] | 31 |

## Điểm vận hành, tác vụ ai (seed đầu; ngưỡng = FPR 5 % trên real validation)

Precision ở tỉ lệ fake π ngoài thực tế = TPR·π / (TPR·π + FPR·(1 − π)) (Layton et al. 2024).

| Model | TPR | FPR test | precision π=50 % | π=10 % | π=1 % | báo nhầm / 1000 real |
|---|---|---|---|---|---|---|
| avh_align | 0.012 | 0.065 | 0.157 | 0.020 | 0.002 | 65 |
| avh_align_plus | 0.774 | 0.059 | 0.930 | 0.595 | 0.118 | 59 |
| avh_align_sup | 0.632 | 0.030 | 0.955 | 0.702 | 0.176 | 30 |
| bl_ens_av | 0.620 | 0.026 | 0.960 | 0.725 | 0.194 | 26 |
| lipinc_v2 | 0.131 | 0.059 | 0.691 | 0.199 | 0.022 | 59 |
| speechforensics | 0.017 | 0.039 | 0.299 | 0.045 | 0.004 | 39 |
| trihead_v4 | 0.847 | 0.012 | 0.985 | 0.883 | 0.407 | 12 |
| trihead_v4_no_prosody | 0.845 | 0.016 | 0.981 | 0.852 | 0.344 | 16 |
| xlsr_frozen | 0.438 | 0.026 | 0.944 | 0.652 | 0.145 | 26 |

## Định vị đoạn giả, tác vụ ai (seed đầu; chuẩn LAV-DF / AV-Deepfake1M)

| Model | AP@0.5 | AP@0.75 | AP@0.95 | AR@1 | AR@5 | AR@10 |
|---|---|---|---|---|---|---|
| avh_align | 0.005 | 0.002 | 0.000 | 0.003 | 0.062 | 0.137 |
| avh_align_plus | 0.022 | 0.011 | 0.000 | 0.011 | 0.162 | 0.286 |
| avh_align_sup | 0.001 | 0.000 | 0.000 | 0.002 | 0.026 | 0.049 |
| speechforensics | 0.001 | 0.000 | 0.000 | 0.001 | 0.022 | 0.065 |
| trihead_v4 | 0.017 | 0.011 | 0.001 | 0.012 | 0.134 | 0.276 |
| trihead_v4_no_prosody | 0.017 | 0.011 | 0.001 | 0.012 | 0.137 | 0.275 |
| xlsr_frozen | 0.010 | 0.006 | 0.001 | 0.003 | 0.066 | 0.170 |

## Phép thử trễ: real test bị trượt tiếng k frame (AI = 0; seed đầu)

Báo nhầm ở ngưỡng FPR 5 % của real validation; AUC real trượt so với real gốc (0,5 = không bị lừa bởi độ trễ).

| Model | báo nhầm k=0 | k=2 | AUC k=2 | k=3 | AUC k=3 | k=4 | AUC k=4 |
|---|---|---|---|---|---|---|---|
| avh_align | 0.029 | 0.994 | 0.992 | 1.000 | 0.995 | 1.000 | 0.997 |
| avh_align_plus | 0.052 | 0.055 | 0.508 | 0.045 | 0.432 | 0.044 | 0.419 |
| avh_align_sup | 0.023 | 0.027 | 0.504 | 0.026 | 0.509 | 0.027 | 0.511 |
| speechforensics | 0.031 | 0.032 | 0.504 | 0.032 | 0.513 | 0.034 | 0.522 |
| trihead_v4 | 0.011 | 0.013 | 0.520 | 0.011 | 0.530 | 0.011 | 0.530 |
| trihead_v4_no_prosody | 0.015 | 0.015 | 0.471 | 0.015 | 0.468 | 0.013 | 0.463 |

## Đầu loại thao tác: trihead_v4 (seed đầu, test)

Độ chính xác 0.818 trên 7368 mẫu; recall real 0.91, sham 0.97, voice 0.73, source 0.80, donor 0.73, both 0.85.

| ô thật \ dự đoán | real | sham | voice | source | donor | both |
|---|---|---|---|---|---|---|
| real | 562 | 28 | 3 | 18 | 8 | 1 |
| sham | 1 | 1190 | 26 | 0 | 5 | 0 |
| voice | 34 | 443 | 1367 | 3 | 3 | 10 |
| source | 146 | 5 | 1 | 987 | 95 | 4 |
| donor | 4 | 10 | 5 | 271 | 884 | 34 |
| both | 2 | 0 | 36 | 14 | 134 | 1034 |

## Đầu loại thao tác: trihead_v4_no_prosody (seed đầu, test)

Độ chính xác 0.823 trên 7368 mẫu; recall real 0.90, sham 0.96, voice 0.78, source 0.75, donor 0.74, both 0.87.

| ô thật \ dự đoán | real | sham | voice | source | donor | both |
|---|---|---|---|---|---|---|
| real | 557 | 23 | 8 | 21 | 10 | 1 |
| sham | 1 | 1169 | 43 | 1 | 7 | 1 |
| voice | 23 | 367 | 1448 | 2 | 4 | 16 |
| source | 147 | 5 | 3 | 928 | 143 | 12 |
| donor | 2 | 15 | 5 | 235 | 896 | 55 |
| both | 2 | 1 | 21 | 2 | 127 | 1067 |

## Phân rã (seed đầu tiên của mỗi model, tác vụ ai)

- **avh_align**: cell/both 0.809, cell/donor 0.856, cell/source 0.724, cell/voice 0.955, generator/diff2lip 0.731, generator/iplap 0.809, generator/latentsync_1_5 0.531, generator/musetalk_1_5 0.877, generator/videoretalking 0.880, generator/wav2lip_gan 0.946, generator_seen/False 0.631, generator_seen/True 0.879, kind/full 0.856, kind/partial 0.844, voice/f5_inpaint 0.922, voice/f5_vi 0.862, voice/omnivoice 0.903, voice/seedvc 0.905, voice/vieneu 0.887, voice/vixtts 0.897, voice/voxcpm2 0.892, voice_seen/False 0.906, voice_seen/True 0.889, gender/F 0.854, gender/M 0.844
- **avh_align_plus**: cell/both 0.983, cell/donor 0.989, cell/source 0.938, cell/voice 0.950, generator/diff2lip 0.979, generator/iplap 0.987, generator/latentsync_1_5 0.866, generator/musetalk_1_5 0.994, generator/videoretalking 0.996, generator/wav2lip_gan 0.997, generator_seen/False 0.922, generator_seen/True 0.994, kind/full 0.982, kind/partial 0.948, voice/f5_inpaint 0.929, voice/f5_vi 0.967, voice/omnivoice 0.978, voice/seedvc 0.988, voice/vieneu 0.977, voice/vixtts 0.957, voice/voxcpm2 0.963, voice_seen/False 0.946, voice_seen/True 0.977, gender/F 0.959, gender/M 0.967
- **avh_align_sup**: cell/both 0.975, cell/donor 0.974, cell/source 0.911, cell/voice 0.749, generator/diff2lip 0.972, generator/iplap 0.983, generator/latentsync_1_5 0.802, generator/musetalk_1_5 0.985, generator/videoretalking 0.992, generator/wav2lip_gan 0.987, generator_seen/False 0.887, generator_seen/True 0.987, kind/full 0.941, kind/partial 0.840, voice/f5_inpaint 0.650, voice/f5_vi 0.886, voice/omnivoice 0.893, voice/seedvc 0.934, voice/vieneu 0.904, voice/vixtts 0.850, voice/voxcpm2 0.850, voice_seen/False 0.763, voice_seen/True 0.904, gender/F 0.883, gender/M 0.889
- **bl_ens_av**: cell/both 0.996, cell/donor 0.971, cell/source 0.904, cell/voice 0.937, generator/diff2lip 0.970, generator/iplap 0.981, generator/latentsync_1_5 0.836, generator/musetalk_1_5 0.982, generator/videoretalking 0.989, generator/wav2lip_gan 0.985, generator_seen/False 0.903, generator_seen/True 0.984, kind/full 0.967, kind/partial 0.937, voice/f5_inpaint 0.878, voice/f5_vi 0.994, voice/omnivoice 0.994, voice/seedvc 0.998, voice/vieneu 0.998, voice/vixtts 0.994, voice/voxcpm2 0.910, voice_seen/False 0.920, voice_seen/True 0.996, gender/F 0.951, gender/M 0.951
- **lipinc_v2**: cell/both 0.662, cell/donor 0.665, cell/source 0.658, cell/voice 0.500, generator/diff2lip 0.645, generator/iplap 0.701, generator/latentsync_1_5 0.643, generator/musetalk_1_5 0.672, generator/videoretalking 0.657, generator/wav2lip_gan 0.655, generator_seen/False 0.644, generator_seen/True 0.671, kind/full 0.740, kind/partial 0.501, voice/f5_inpaint 0.500, voice/f5_vi 0.541, voice/omnivoice 0.579, voice/seedvc 0.599, voice/vieneu 0.586, voice/vixtts 0.598, voice/voxcpm2 0.582, voice_seen/False 0.551, voice_seen/True 0.576, gender/F 0.592, gender/M 0.624
- **speechforensics**: cell/both 0.787, cell/donor 0.828, cell/source 0.756, cell/voice 0.818, generator/diff2lip 0.774, generator/iplap 0.835, generator/latentsync_1_5 0.654, generator/musetalk_1_5 0.822, generator/videoretalking 0.812, generator/wav2lip_gan 0.848, generator_seen/False 0.714, generator_seen/True 0.829, kind/full 0.896, kind/partial 0.723, voice/f5_inpaint 0.732, voice/f5_vi 0.816, voice/omnivoice 0.829, voice/seedvc 0.824, voice/vieneu 0.847, voice/vixtts 0.824, voice/voxcpm2 0.808, voice_seen/False 0.779, voice_seen/True 0.829, gender/F 0.817, gender/M 0.805
- **trihead_v4**: cell/both 0.999, cell/donor 0.998, cell/source 0.976, cell/voice 0.927, generator/diff2lip 0.997, generator/iplap 1.000, generator/latentsync_1_5 0.953, generator/musetalk_1_5 0.997, generator/videoretalking 1.000, generator/wav2lip_gan 1.000, generator_seen/False 0.975, generator_seen/True 0.999, kind/full 0.982, kind/partial 0.959, voice/f5_inpaint 0.861, voice/f5_vi 0.990, voice/omnivoice 0.994, voice/seedvc 0.998, voice/vieneu 0.997, voice/vixtts 0.997, voice/voxcpm2 0.898, voice_seen/False 0.910, voice_seen/True 0.995, gender/F 0.962, gender/M 0.976
- **trihead_v4_no_prosody**: cell/both 0.999, cell/donor 0.998, cell/source 0.976, cell/voice 0.938, generator/diff2lip 0.995, generator/iplap 0.999, generator/latentsync_1_5 0.955, generator/musetalk_1_5 0.997, generator/videoretalking 1.000, generator/wav2lip_gan 0.999, generator_seen/False 0.975, generator_seen/True 0.999, kind/full 0.985, kind/partial 0.963, voice/f5_inpaint 0.890, voice/f5_vi 0.989, voice/omnivoice 0.994, voice/seedvc 0.997, voice/vieneu 0.997, voice/vixtts 0.997, voice/voxcpm2 0.906, voice_seen/False 0.924, voice_seen/True 0.994, gender/F 0.968, gender/M 0.978
- **xlsr_frozen**: cell/both 0.993, cell/donor 0.748, cell/source 0.503, cell/voice 0.974, generator/diff2lip 0.756, generator/iplap 0.754, generator/latentsync_1_5 0.745, generator/musetalk_1_5 0.739, generator/videoretalking 0.750, generator/wav2lip_gan 0.737, generator_seen/False 0.750, generator_seen/True 0.745, kind/full 0.778, kind/partial 0.860, voice/f5_inpaint 0.954, voice/f5_vi 0.995, voice/omnivoice 0.997, voice/seedvc 0.999, voice/vieneu 0.999, voice/vixtts 0.998, voice/voxcpm2 0.942, voice_seen/False 0.963, voice_seen/True 0.997, gender/F 0.821, gender/M 0.825
