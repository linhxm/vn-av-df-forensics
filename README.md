# VN-AV-DF Forensics (v004)

**A Vietnamese lip-sync deepfake benchmark: real speakers, six audio-visual cells, unseen generators and voices at test time.**

| Version | Branch | What it is |
|---|---|---|
| v001-v002 | `v001`, `v002` | earlier study on one pilot part (closed 14 October 2026) |
| v003 | `v003` | the benchmark rebuilt from scratch; detectors M1, M2 and baselines on part 1 |
| **v004** | `v004` | measured mismatch labels, the M3 lip–audio–head detector, all three parts (closed 7 November 2026): **[docs/v004_report.md](docs/v004_report.md)** |
| v005 | `main` | M3-R, a detector trained for re-compressed video (in progress) |

## Benchmark design

| | Original audio | Spliced real audio | AI voice |
|---|---|---|---|
| **Real video** | ① real | ② sham (full + partial) | ③ AI voice: track A (whole sentence), B (one phrase, spliced); test adds C (inpainting) |
| **Lip-synced video** | ④ source | ⑤ donor | ⑥ both (lips driven by track B) |

- **Real data:** parts of 600-800 clips (20 men + 20 women × 15-20 clips, 4-7 s, 720p, audio-video offset corrected), split by speaker.
- **Generators:** Wav2Lip, MuseTalk 1.5, IP-LAP, VideoReTalking for train/validation/test; LatentSync 1.5 and Diff2Lip appear **only at test time**.
- **Voices:** VieNeu-TTS, F5-TTS (Vietnamese), OmniVoice, Seed-VC for train/validation/test; VoxCPM2, viXTTS and F5 inpainting appear **only at test time**.
- **Labels:** visual / audio / audio-visual mismatch intervals; `fake_intervals` is their union.
- **Per real clip:** each clip goes through half of the generators (2 of 4 in train/validation, 3 of 6 in test, balanced); 9 files in train/validation (6 with AI), 12 in test (9 with AI). About 5,850-7,800 files per part, 69% with AI. Rationale: [docs/v003_generation.md](docs/v003_generation.md) §4.

## Detectors

Baselines are re-implemented from the authors' released code (differences listed in the model docstrings and in [docs/v003_model.md](docs/v003_model.md)). Names in **bold** were trained in v004.

| Name | Role |
|---|---|
| `lipinc_v2` | **BL-V**, LIPINC-V2 (Datta et al. 2025): mouth-region vision temporal transformer (visual fake) |
| `ssl_aasist` | BL-A, XLS-R 300M fine-tuned with AASIST + RawBoost (Tak et al. 2022) (audio fake); did not learn in v003, optional in v004 |
| `xlsr_frozen` | **BL-A2**, frozen XLS-R 300M with a learned mix of 5 layer groups (v004) |
| `avh_align` | **BL-M**, AVH-Align (CVPR 2025): audio-visual alignment learned from reals only, initialized from the released AV-1M checkpoint |
| `avh_align_sup` | **BL-M/sup**, AVH-Align/sup: same network trained with labels |
| `speechforensics` | **BL-SF**, SpeechForensics (no training) |
| `bl_ens`, `bl_ens_av` | BL-Ens = max(BL-V, BL-A); **BL-Ens-AV** = max(BL-M/sup, BL-A2); z-normalized on validation reals (no training) |
| `trihead_v2` | M1 (v003): sync (lag cost volume) + mouth trace (NPR, head pose) + audio (learned XLS-R layer weights) branches, three outputs, per-speaker reference clips, pseudo-fake audio replacement, speaker gradient reversal; deferred in v004 |
| `avh_align_plus` | **M2**: AVH-Align scorer over a ±15-frame lag volume + lip–head motion + pseudo-fakes |
| `trihead_v4` | **M3** (v004): relative / absolute sync streams, AVH-Align Φ channel, appearance, prosody ↔ head-motion branch, mismatch / offset / cell-type heads |
| `trihead_v4_no_prosody` | **M3 without the prosody ↔ head branch**; other ablations (`_no_phi`, `_abs`, `_ref`, `_no_type`) are in `models.SPECS` |

Evaluation covers four tasks (ai / visual / audio / mismatch), per-cell, per-generator, per-voice and seen/unseen breakdowns, per-speaker AUC, localization inside partial segments (visual and mismatch), reference-normalized scores, a delayed-audio stress test, operating points at 5% validation FPR, and pre-registered gates with speaker-level bootstrap CIs. Since v004 the mismatch label of lip-synced clips is **measured** (SyncNet at lag 0 against real clips) instead of set by construction, and each model is tested right after training (every test use is logged in `test-log.json`).

## Repository layout

| Folder | Contents |
|---|---|
| `data_pipeline/` | Real data workspace: step scripts (plan → download → cut → identify → review → parts → export → archive), `configs/`, `checkpoints/`, `data/` (ledger, raw, pool), `exports/` |
| `generation/` | Steps 00-04: setup, plan, voice session, lip-sync sessions, finalize |
| `training/` | Steps 00-04: encoder setup, feature extraction, train, evaluate, report |
| `notebooks/` | `generate`, `prepare`, `train`, `demo` (Kaggle; W&B logging in `prepare` and `train`) |
| `src/vn_av_df/` | `generation/` (cells, plan, voice, generator adapters), `training/` (data, models, train, evaluate, report), `features/`, `demo/` (make a fake from a real video, analyze a video; Gradio), `tracking.py` (W&B) |
| `src/vn_av_data/` | Real data pipeline code (ledger, cut, face / VAD / SyncNet / voice, review page, split, export), run by `data_pipeline/steps/` |
| `docs/` | Vietnamese design notes with rationale and sources: `v003_data.md` (real data pipeline, people and clip counts, audio-visual offset), `v003_generation.md` (six cells, generators and voices, counts per part), `v003_model.md` (methods, baselines, proposed models, evaluation); English **`v004_report.md`** with its generated tables in `v004_results/` |
| `environments/` | Isolated worker environments for generators, TTS and AV-HuBERT |
| `tests/` | Synthetic end-to-end and component tests; `tests/data_pipeline/` for the real data pipeline |

How to run each step is described in **[PIPELINE.md](PIPELINE.md)** (written in Vietnamese).

## Status (v004, 7 November 2026)

- Three parts (120 speakers, 23,242 clips) generated and featurized on Kaggle; M3, M3 without prosody, M2 and five baselines trained and tested once (seed 42).
- Test, 31 speakers: **M3 AI? AUC 0.968**, +0.038 [+0.031; +0.045] over the BL-Ens-AV late fusion and +0.041 [+0.033; +0.050] over M2; 0.953 on the unseen LatentSync. The new prosody ↔ head branch does not help (−0.003).
- A post-hoc review found that the visual head relies partly on sharpness and a whole-frame trace of the generation pipeline, and that every detector loses much of its accuracy on re-compressed video (M3-lean AI? 0.965 → 0.900 at 480p → 0.761 at 360p). Details and limitations: [docs/v004_report.md](docs/v004_report.md).

## Ethics and usage limits

- Only the authors' own videos or videos used with permission. **Raw videos and fake samples of real people are not redistributed**; the data is for research only.
- Several generator and TTS checkpoints are restricted to non-commercial use.
- Detector scores are suspicion levels in 0-1 and are **not calibrated** probabilities.
