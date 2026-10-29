# VN-AV-DF Forensics (v003)

**A Vietnamese lip-sync deepfake benchmark: real speakers, six audio-visual cells, unseen generators and voices at test time.**

v003 is rebuilt from scratch. The earlier study (v001-v002, closed 14 October 2026) lives in the archived repository and its `v002` release.

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

Baselines are re-implemented from the authors' released code (differences listed in the model docstrings and in [docs/v003_model.md](docs/v003_model.md)).

| Name | Role |
|---|---|
| `lipinc_v2` | **BL-V**, LIPINC-V2 (Datta et al. 2025): mouth-region vision temporal transformer (visual fake) |
| `ssl_aasist` | **BL-A**, XLS-R 300M fine-tuned with AASIST + RawBoost (Tak et al. 2022) (audio fake) |
| `avh_align` | **BL-M**, AVH-Align (CVPR 2025): audio-visual alignment learned from reals only, initialized from the released AV-1M checkpoint |
| `avh_align_sup` | **BL-M/sup**, AVH-Align/sup: same network trained with labels |
| `bl_ens` | **BL-Ens**, max(BL-V, BL-A) after z-normalization on validation reals (no training) |
| `trihead_v2` | **M1**: sync (lag cost volume) + mouth trace (NPR, head pose) + audio (learned XLS-R layer weights) branches, three outputs, per-speaker reference clips, pseudo-fake audio replacement, speaker gradient reversal |
| `avh_align_plus` | **M2**: AVH-Align scorer over a ±15-frame lag volume + lip–head motion + pseudo-fakes |
| `trihead_v2_no_{sync,trace,ref}` | M1 ablations |

Evaluation covers four tasks (ai / visual / audio / mismatch), per-cell, per-generator, per-voice and seen/unseen breakdowns, per-speaker AUC, localization inside partial segments, reference-normalized scores, and pre-registered gates with speaker-level bootstrap CIs. The test split is evaluated once and locked.

## Repository layout

| Folder | Contents |
|---|---|
| `data_pipeline/` | Real data workspace: step scripts (plan → download → cut → identify → review → parts → export → archive), `configs/`, `checkpoints/`, `data/` (ledger, raw, pool), `exports/` |
| `generation/` | Steps 00-04: setup, plan, voice session, lip-sync sessions, finalize |
| `training/` | Steps 00-04: encoder setup, feature extraction, train, evaluate, report |
| `notebooks/` | `generate`, `prepare`, `train`, `demo` (Kaggle; W&B logging in `prepare` and `train`) |
| `src/vn_av_df/` | `generation/` (cells, plan, voice, generator adapters), `training/` (data, models, train, evaluate, report), `features/`, `demo/` (make a fake from a real video, analyze a video; Gradio), `tracking.py` (W&B) |
| `src/vn_av_data/` | Real data pipeline code (ledger, cut, face / VAD / SyncNet / voice, review page, split, export), run by `data_pipeline/steps/` |
| `docs/` | Vietnamese design notes with rationale and sources: `v003_data.md` (real data pipeline, people and clip counts, audio-visual offset), `v003_generation.md` (six cells, generators and voices, counts per part), `v003_model.md` (methods, baselines, proposed models, evaluation) |
| `environments/` | Isolated worker environments for generators, TTS and AV-HuBERT |
| `tests/` | Synthetic end-to-end and component tests; `tests/data_pipeline/` for the real data pipeline |

How to run each step is described in **[PIPELINE.md](PIPELINE.md)** (written in Vietnamese).

## Status

- The code passes the synthetic end-to-end tests, which use mocked TTS, generators and ASR.
- The worker environments for the new generators and voices have **not been run on Kaggle yet**.
- No results yet.

## Ethics and usage limits

- Only the authors' own videos or videos used with permission. **Raw videos and fake samples of real people are not redistributed**; the data is for research only.
- Several generator and TTS checkpoints are restricted to non-commercial use.
- Detector scores are suspicion levels in 0-1 and are **not calibrated** probabilities.
