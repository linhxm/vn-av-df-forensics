# Kết quả v005

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
| avh_align | 0.373 | 0.373 | 0.206 | 0.608 | 0.909 | 0.427 | 0.655 | 0.903 |
| avh_align_plus | 0.930 | 0.930 | - | - | 0.964 | 0.823 | - | 0.975 |
| avh_align_sup | 0.876 | 0.876 | 0.903 | 0.542 | 0.443 | 0.858 | 0.893 | 0.640 |
| bl_ens_av | 0.935 | 0.934 | 0.903 | 0.964 | - | 0.888 | - | - |
| lipinc_v2 | 0.633 | 0.633 | 0.700 | 0.466 | 0.438 | 0.625 | - | - |
| speechforensics | 0.490 | 0.490 | 0.456 | 0.513 | 0.668 | 0.522 | 0.320 | 0.814 |
| trihead_v5 | 0.973 | 0.973 | 0.992 | 0.966 | 0.972 | 0.945 | 0.980 | 0.987 |
| trihead_v5_no_pair | 0.972 | 0.972 | 0.992 | 0.964 | 0.971 | 0.944 | 0.980 | 0.987 |
| xlsr_frozen | 0.744 | 0.744 | - | 0.964 | - | 0.742 | - | - |

## Cổng (ΔAUC, bootstrap theo speaker, seed đầu tiên)

| So sánh | Tác vụ | Bản | ΔAUC | CI 95% | Speaker |
|---|---|---|---|---|---|
| M2 hơn BL-M/sup (cùng dữ liệu) (avh_align_plus − avh_align_sup) | mismatch | gốc | +0.522 | [+0.508; +0.535] | 31 |
| M2 hơn BL-M (avh_align_plus − avh_align) | mismatch | gốc | +0.055 | [+0.049; +0.061] | 31 |
| R1 M3-R hơn BL-Ens-AV (trihead_v5 − bl_ens_av) | ai | gốc | +0.039 | [+0.033; +0.044] | 31 |
| R2 M3-R hơn BL-Ens-AV, video nén 480p (trihead_v5 − bl_ens_av) | ai | c480 | +0.050 | [+0.043; +0.056] | 31 |
| R2b M3-R hơn BL-Ens-AV, video nén 360p (trihead_v5 − bl_ens_av) | ai | c360 | +0.065 | [+0.051; +0.079] | 31 |
| R3 M3-R hơn M2, video nén 480p (trihead_v5 − avh_align_plus) | ai | c480 | +0.055 | [+0.046; +0.065] | 31 |
| R6 cặp sạch ↔ nén giúp ở 480p (trihead_v5 − trihead_v5_no_pair) | ai | c480 | +0.004 | [+0.001; +0.006] | 31 |
| R8 M3-R không kém BL-A2, giọng chưa thấy, 480p (≥ −0,01) (trihead_v5 − xlsr_frozen) | audio | c480 | +0.004 | [-0.004; +0.013] | 31 |

## Điểm vận hành, tác vụ ai (seed đầu; ngưỡng = FPR 5 % trên real validation)

Precision ở tỉ lệ fake π ngoài thực tế = TPR·π / (TPR·π + FPR·(1 − π)) (Layton et al. 2024).

| Model | TPR | FPR test | precision π=50 % | π=10 % | π=1 % | báo nhầm / 1000 real |
|---|---|---|---|---|---|---|
| avh_align | 0.013 | 0.078 | 0.146 | 0.019 | 0.002 | 78 |
| avh_align_plus | 0.767 | 0.052 | 0.936 | 0.620 | 0.129 | 52 |
| avh_align_sup | 0.633 | 0.033 | 0.951 | 0.684 | 0.164 | 33 |
| bl_ens_av | 0.747 | 0.023 | 0.970 | 0.780 | 0.244 | 23 |
| lipinc_v2 | 0.155 | 0.044 | 0.779 | 0.282 | 0.034 | 44 |
| speechforensics | 0.017 | 0.039 | 0.299 | 0.045 | 0.004 | 39 |
| trihead_v5 | 0.817 | 0.015 | 0.982 | 0.861 | 0.360 | 15 |
| trihead_v5_no_pair | 0.828 | 0.017 | 0.979 | 0.841 | 0.325 | 17 |
| xlsr_frozen | 0.461 | 0.024 | 0.950 | 0.677 | 0.160 | 24 |

## Video nén lại (cfg["variants"]), tác vụ ai

Cùng mẫu test, nén lại video + tiếng (settings.VARIANTS). Ngưỡng FPR 5 % đặt trên real validation **sạch** (lúc dùng không biết chất lượng video) → FPR thực tế cho biết điểm real bị đẩy lên bao nhiêu khi nén; cột cuối: ngưỡng đặt trên validation cùng mức nén.

| Model | biến thể | AUC sạch (seed đầu) | AUC nén (mean ± std) | TPR (ngưỡng sạch) | FPR (ngưỡng sạch) | TPR (ngưỡng cùng mức) | FPR (ngưỡng cùng mức) |
|---|---|---|---|---|---|---|---|
| avh_align | c360 | 0.373 | 0.385 | 0.003 | 0.012 | 0.022 | 0.092 |
| avh_align | c480 | 0.373 | 0.380 | 0.008 | 0.042 | 0.013 | 0.065 |
| avh_align_plus | c360 | 0.930 | 0.851 | 0.604 | 0.092 | 0.438 | 0.029 |
| avh_align_plus | c480 | 0.930 | 0.907 | 0.697 | 0.056 | 0.634 | 0.039 |
| avh_align_sup | c360 | 0.876 | 0.771 | 0.555 | 0.156 | 0.296 | 0.034 |
| avh_align_sup | c480 | 0.876 | 0.848 | 0.570 | 0.055 | 0.508 | 0.026 |
| bl_ens_av | c360 | 0.935 | 0.848 | 0.523 | 0.043 | - | - |
| bl_ens_av | c480 | 0.935 | 0.913 | 0.628 | 0.022 | - | - |
| speechforensics | c360 | 0.490 | 0.467 | 0.094 | 0.115 | 0.007 | 0.024 |
| speechforensics | c480 | 0.490 | 0.461 | 0.023 | 0.050 | 0.017 | 0.037 |
| trihead_v5 | c360 | 0.973 | 0.914 | 0.773 | 0.085 | 0.409 | 0.015 |
| trihead_v5 | c480 | 0.973 | 0.962 | 0.800 | 0.021 | 0.744 | 0.012 |
| trihead_v5_no_pair | c360 | 0.972 | 0.911 | 0.742 | 0.074 | 0.436 | 0.013 |
| trihead_v5_no_pair | c480 | 0.972 | 0.959 | 0.800 | 0.026 | 0.742 | 0.010 |
| xlsr_frozen | c360 | 0.744 | 0.735 | 0.429 | 0.031 | 0.428 | 0.031 |
| xlsr_frozen | c480 | 0.744 | 0.744 | 0.457 | 0.023 | 0.460 | 0.024 |

## Định vị đoạn giả, tác vụ ai (seed đầu; chuẩn LAV-DF / AV-Deepfake1M)

| Model | AP@0.5 | AP@0.75 | AP@0.95 | AR@1 | AR@5 | AR@10 |
|---|---|---|---|---|---|---|
| avh_align | 0.006 | 0.002 | 0.000 | 0.003 | 0.065 | 0.137 |
| avh_align_plus | 0.025 | 0.012 | 0.000 | 0.012 | 0.177 | 0.298 |
| avh_align_sup | 0.001 | 0.000 | 0.000 | 0.002 | 0.025 | 0.053 |
| speechforensics | 0.001 | 0.000 | 0.000 | 0.001 | 0.022 | 0.065 |
| trihead_v5 | 0.023 | 0.015 | 0.000 | 0.012 | 0.180 | 0.315 |
| trihead_v5_no_pair | 0.022 | 0.015 | 0.001 | 0.012 | 0.177 | 0.314 |
| xlsr_frozen | 0.011 | 0.006 | 0.001 | 0.004 | 0.077 | 0.177 |

## Phép thử trễ: real test bị trượt tiếng k frame (AI = 0; seed đầu)

Báo nhầm ở ngưỡng FPR 5 % của real validation; AUC real trượt so với real gốc (0,5 = không bị lừa bởi độ trễ).

| Model | báo nhầm k=0 | k=2 | AUC k=2 | k=3 | AUC k=3 | k=4 | AUC k=4 |
|---|---|---|---|---|---|---|---|
| avh_align | 0.045 | 0.997 | 0.995 | 1.000 | 0.997 | 1.000 | 0.998 |
| avh_align_plus | 0.024 | 0.031 | 0.538 | 0.034 | 0.515 | 0.027 | 0.530 |
| avh_align_sup | 0.048 | 0.047 | 0.518 | 0.048 | 0.534 | 0.048 | 0.533 |
| speechforensics | 0.031 | 0.032 | 0.504 | 0.032 | 0.513 | 0.034 | 0.522 |
| trihead_v5 | 0.005 | 0.005 | 0.518 | 0.005 | 0.521 | 0.005 | 0.529 |
| trihead_v5_no_pair | 0.013 | 0.013 | 0.573 | 0.013 | 0.602 | 0.013 | 0.625 |

## Đầu loại thao tác: trihead_v5 (seed đầu, test)

Độ chính xác 0.833 trên 7368 mẫu; recall real 0.94, sham 0.95, voice 0.82, source 0.74, donor 0.74, both 0.87.

| ô thật \ dự đoán | real | sham | voice | source | donor | both |
|---|---|---|---|---|---|---|
| real | 580 | 22 | 4 | 13 | 1 | 0 |
| sham | 3 | 1166 | 49 | 0 | 4 | 0 |
| voice | 43 | 281 | 1520 | 3 | 3 | 10 |
| source | 199 | 3 | 2 | 918 | 111 | 5 |
| donor | 8 | 18 | 8 | 242 | 888 | 44 |
| both | 1 | 1 | 57 | 1 | 93 | 1067 |

## Đầu loại thao tác: trihead_v5_no_pair (seed đầu, test)

Độ chính xác 0.821 trên 7368 mẫu; recall real 0.90, sham 0.96, voice 0.80, source 0.72, donor 0.73, both 0.88.

| ô thật \ dự đoán | real | sham | voice | source | donor | both |
|---|---|---|---|---|---|---|
| real | 560 | 37 | 11 | 11 | 1 | 0 |
| sham | 1 | 1170 | 50 | 0 | 1 | 0 |
| voice | 26 | 342 | 1481 | 2 | 3 | 6 |
| source | 201 | 8 | 1 | 886 | 134 | 8 |
| donor | 7 | 25 | 7 | 234 | 881 | 54 |
| both | 0 | 2 | 50 | 1 | 97 | 1070 |

## Phân rã (seed đầu tiên của mỗi model, tác vụ ai)

- **avh_align**: cell/both 0.777, cell/donor 0.824, cell/source 0.689, cell/voice 0.958, generator/diff2lip 0.676, generator/iplap 0.765, generator/latentsync_1_5 0.478, generator/musetalk_1_5 0.857, generator/videoretalking 0.856, generator/wav2lip_gan 0.942, generator_seen/False 0.577, generator_seen/True 0.856, kind/full 0.828, kind/partial 0.829, voice/f5_inpaint 0.926, voice/f5_vi 0.845, voice/omnivoice 0.888, voice/seedvc 0.889, voice/vieneu 0.874, voice/vixtts 0.888, voice/voxcpm2 0.874, voice_seen/False 0.901, voice_seen/True 0.874, gender/F 0.835, gender/M 0.821
- **avh_align_plus**: cell/both 0.983, cell/donor 0.984, cell/source 0.936, cell/voice 0.954, generator/diff2lip 0.974, generator/iplap 0.989, generator/latentsync_1_5 0.859, generator/musetalk_1_5 0.992, generator/videoretalking 0.994, generator/wav2lip_gan 0.997, generator_seen/False 0.916, generator_seen/True 0.993, kind/full 0.980, kind/partial 0.949, voice/f5_inpaint 0.933, voice/f5_vi 0.968, voice/omnivoice 0.980, voice/seedvc 0.988, voice/vieneu 0.981, voice/vixtts 0.961, voice/voxcpm2 0.964, voice_seen/False 0.950, voice_seen/True 0.979, gender/F 0.958, gender/M 0.967
- **avh_align_sup**: cell/both 0.979, cell/donor 0.974, cell/source 0.913, cell/voice 0.733, generator/diff2lip 0.966, generator/iplap 0.986, generator/latentsync_1_5 0.817, generator/musetalk_1_5 0.985, generator/videoretalking 0.989, generator/wav2lip_gan 0.988, generator_seen/False 0.891, generator_seen/True 0.987, kind/full 0.941, kind/partial 0.832, voice/f5_inpaint 0.641, voice/f5_vi 0.881, voice/omnivoice 0.892, voice/seedvc 0.931, voice/vieneu 0.903, voice/vixtts 0.837, voice/voxcpm2 0.821, voice_seen/False 0.747, voice_seen/True 0.902, gender/F 0.877, gender/M 0.884
- **bl_ens_av**: cell/both 0.997, cell/donor 0.973, cell/source 0.907, cell/voice 0.926, generator/diff2lip 0.966, generator/iplap 0.985, generator/latentsync_1_5 0.846, generator/musetalk_1_5 0.984, generator/videoretalking 0.986, generator/wav2lip_gan 0.987, generator_seen/False 0.906, generator_seen/True 0.986, kind/full 0.968, kind/partial 0.931, voice/f5_inpaint 0.854, voice/f5_vi 0.989, voice/omnivoice 0.994, voice/seedvc 0.997, voice/vieneu 0.997, voice/vixtts 0.994, voice/voxcpm2 0.901, voice_seen/False 0.907, voice_seen/True 0.994, gender/F 0.947, gender/M 0.949
- **lipinc_v2**: cell/both 0.699, cell/donor 0.702, cell/source 0.697, cell/voice 0.500, generator/diff2lip 0.688, generator/iplap 0.712, generator/latentsync_1_5 0.653, generator/musetalk_1_5 0.716, generator/videoretalking 0.722, generator/wav2lip_gan 0.707, generator_seen/False 0.670, generator_seen/True 0.714, kind/full 0.793, kind/partial 0.504, voice/f5_inpaint 0.500, voice/f5_vi 0.561, voice/omnivoice 0.612, voice/seedvc 0.599, voice/vieneu 0.591, voice/vixtts 0.615, voice/voxcpm2 0.617, voice_seen/False 0.565, voice_seen/True 0.590, gender/F 0.618, gender/M 0.647
- **speechforensics**: cell/both 0.787, cell/donor 0.828, cell/source 0.756, cell/voice 0.818, generator/diff2lip 0.774, generator/iplap 0.835, generator/latentsync_1_5 0.654, generator/musetalk_1_5 0.822, generator/videoretalking 0.812, generator/wav2lip_gan 0.848, generator_seen/False 0.714, generator_seen/True 0.829, kind/full 0.896, kind/partial 0.723, voice/f5_inpaint 0.732, voice/f5_vi 0.816, voice/omnivoice 0.829, voice/seedvc 0.824, voice/vieneu 0.847, voice/vixtts 0.824, voice/voxcpm2 0.808, voice_seen/False 0.779, voice_seen/True 0.829, gender/F 0.817, gender/M 0.805
- **trihead_v5**: cell/both 0.999, cell/donor 0.995, cell/source 0.964, cell/voice 0.949, generator/diff2lip 0.987, generator/iplap 0.996, generator/latentsync_1_5 0.941, generator/musetalk_1_5 0.997, generator/videoretalking 0.998, generator/wav2lip_gan 0.998, generator_seen/False 0.964, generator_seen/True 0.997, kind/full 0.981, kind/partial 0.967, voice/f5_inpaint 0.911, voice/f5_vi 0.995, voice/omnivoice 0.998, voice/seedvc 0.999, voice/vieneu 1.000, voice/vixtts 0.998, voice/voxcpm2 0.905, voice_seen/False 0.934, voice_seen/True 0.998, gender/F 0.968, gender/M 0.978
- **trihead_v5_no_pair**: cell/both 0.999, cell/donor 0.995, cell/source 0.965, cell/voice 0.940, generator/diff2lip 0.987, generator/iplap 0.997, generator/latentsync_1_5 0.939, generator/musetalk_1_5 0.998, generator/videoretalking 0.998, generator/wav2lip_gan 0.999, generator_seen/False 0.963, generator_seen/True 0.998, kind/full 0.981, kind/partial 0.962, voice/f5_inpaint 0.889, voice/f5_vi 0.994, voice/omnivoice 0.996, voice/seedvc 0.998, voice/vieneu 0.998, voice/vixtts 0.997, voice/voxcpm2 0.907, voice_seen/False 0.925, voice_seen/True 0.996, gender/F 0.968, gender/M 0.974
- **xlsr_frozen**: cell/both 0.994, cell/donor 0.733, cell/source 0.500, cell/voice 0.973, generator/diff2lip 0.754, generator/iplap 0.747, generator/latentsync_1_5 0.751, generator/musetalk_1_5 0.730, generator/videoretalking 0.737, generator/wav2lip_gan 0.729, generator_seen/False 0.753, generator_seen/True 0.736, kind/full 0.765, kind/partial 0.862, voice/f5_inpaint 0.953, voice/f5_vi 0.995, voice/omnivoice 0.997, voice/seedvc 0.999, voice/vieneu 0.999, voice/vixtts 0.999, voice/voxcpm2 0.939, voice_seen/False 0.962, voice_seen/True 0.997, gender/F 0.817, gender/M 0.821
