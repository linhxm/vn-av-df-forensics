"""Đánh giá trên test (cả 3 part), ngưỡng chọn trên validation. v004: test từng model ngay
sau khi train xong (notebook train, mỗi bước một model); mỗi checkpoint chấm một lần, gọi lại
thì đọc kết quả đã có; mọi lần chấm ghi vào test-log.json (số lần mỗi model dùng test).

Tác vụ (điểm video; điểm cao = nghi ngờ):
- ai: sample có bị AI không (label). Model 3 đầu: max(visual, audio).
- visual / audio / mismatch: cờ synthetic_visual / synthetic_audio / có av_mismatch.
Model một điểm (baseline) dùng cùng điểm cho mọi tác vụ, để thấy nó thực ra bắt cái gì.

Báo cáo:
- tổng: AUC, AP, EER, F1 / FAR tại ngưỡng validation;
- phân rã: AUC từng nhóm fake so với real (theo cell, generator, voice_method, kind, đã thấy / chưa
  thấy khi train), AUC trong từng speaker;
- partial: AUC frame trong đoạn vs ngoài đoạn trong CÙNG clip, ô sát chỗ nối (±2 frame) vs ô giữa;
- định vị chuẩn LAV-DF / AV-Deepfake1M: AP theo IoU 0,5 / 0,75 / 0,95 và AR@1/5/10 (IoU 0,5-0,95);
- ref-norm: z-score điểm theo R clip real tham chiếu của chính người đó (bỏ các clip tham chiếu VÀ
  mọi file sinh từ chúng khỏi phép đo; điểm thô báo trên đúng tập còn lại để so công bằng);
- điểm vận hành theo tỉ lệ báo nhầm cố định trên real validation (FPR 1 %, 5 %) và precision / báo
  nhầm ở nhiều tỉ lệ fake ngoài thực tế (50 %, 10 %, 1 %), theo khuyến nghị Layton et al. (USENIX
  Security 2024): F1 tối đa trên validation (~70 % fake) cho ngưỡng hợp với môi trường nhiều fake.
- BL-Ens (không train): max(BL-V, BL-A) sau khi chuẩn hoá điểm mỗi model theo clip real validation
  (z-score), cho tác vụ ai; visual = BL-V, audio = BL-A. Model chỉ có điểm video (frame_level =
  False, LIPINC-V2) không có phép đo định vị.
Kết quả: `<run>/<model>_seed<seed>/metrics.json`, `predictions-{val,test}.npz`; `evaluation.json`
gộp mọi model đã chấm; `test-log.json`.
"""

import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from vn_av_df import tracking
from vn_av_df.common.runtime import fingerprint, read_json, sha, write_json
from vn_av_df.dataset import training_dataset
from vn_av_df.training.data import BenchData, collate, enrollment, to_device
from vn_av_df.training.models import pool_scores
from vn_av_df.training.train import load_model

TASKS = ("ai", "visual", "audio", "mismatch")
TARGET_KEY = {"ai": "label", "visual": "synthetic_visual", "audio": "synthetic_audio"}


def target(row, task):
    """Nhãn video của một tác vụ. Mismatch v004 (training/sync_labels.py): None = không xác định
    (mask) → mọi phép đo của tác vụ đó bỏ mẫu này."""
    if task == "mismatch":
        if "_mismatch" in row:
            return row["_mismatch_eval"]
        return int(bool(row.get("av_mismatch_intervals")))
    return int(bool(row.get(TARGET_KEY[task])))


def known(rows, task):
    return [r for r in rows if target(r, task) is not None]


def auc(labels, scores):
    labels = np.asarray(labels)
    return float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) == 2 else None


def speaker_balanced_auc(rows, labels, scores):
    """AUC gộp nhưng mỗi người nặng như nhau (trọng số 1 / số file của người đó): người có 20 clip
    không lấn người có 15 clip (docs/v003_model.md §8)."""
    if len(set(labels)) < 2:
        return None
    n = Counter(r.get("speaker_id") for r in rows)
    weights = [1.0 / n[r.get("speaker_id")] for r in rows]
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def eer(labels, scores):
    if len(set(labels)) != 2:
        return None
    fpr, tpr, _ = roc_curve(labels, scores)
    i = np.nanargmin(np.abs(fpr - (1 - tpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def best_threshold(labels, scores):
    """Max F1 trên validation, cho điểm thực bất kỳ (không cần trong [0, 1])."""
    labels, scores = np.asarray(labels), np.asarray(scores, float)
    order = np.argsort(-scores)
    sorted_scores, sorted_labels = scores[order], labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    f1 = 2 * tp / np.maximum(1, tp + labels.sum() + fp)
    last = np.r_[np.diff(sorted_scores) != 0, True]
    candidates = np.flatnonzero(last)
    if not len(candidates):
        return float(scores.mean())
    return float(sorted_scores[candidates[np.argmax(f1[candidates])]])


def at_threshold(labels, scores, threshold):
    labels, predicted = np.asarray(labels), np.asarray(scores) >= threshold
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())
    return {
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "far": fp / max(1, int((labels == 0).sum())),
        "recall": tp / max(1, int((labels == 1).sum())),
    }


@torch.no_grad()
def predict(model, cfg, rows, device, lag=None):
    """lag: trượt tiếng cố định k frame cho mọi mẫu (phép thử trễ); None = như dữ liệu."""
    data = BenchData(
        cfg,
        rows,
        model.needs,
        references=model.references,
        fixed_references=True,
        fixed_lag=lag,
    )
    loader = torch.utils.data.DataLoader(
        data, batch_size=cfg["training"]["eval_batch_size"], collate_fn=collate
    )
    out = {}
    model.eval()
    for batch in loader:
        batch = to_device(batch, device)
        heads = model(batch)
        mask = batch["mask"]
        for i, index in enumerate(batch["index"].tolist()):
            sid = rows[index]["sample_id"]
            length = int(mask[i].sum()) if mask[i].any() else mask.shape[1]
            frames = {
                h: v[i, :length].float().cpu().numpy()
                for h, v in heads.items()
                if not h.startswith("_")
            }
            video = {
                h: float(pool_scores(v[i : i + 1], mask[i : i + 1], model.pool)[0])
                for h, v in heads.items()
                if not h.startswith("_")
            }
            out[sid] = {"frames": frames, "video": video, "mask": mask[i, :length].cpu().numpy()}
            if "_cell" in heads:  # M3: xác suất loại thao tác (6 ô)
                out[sid]["cell"] = heads["_cell"][i].float().softmax(-1).cpu().numpy().tolist()
    return out


def manipulation_type(model, rows, predictions):
    """Đầu loại thao tác của M3 (đầu ra cho suy luận loại trừ): độ chính xác, recall từng ô và
    ma trận nhầm lẫn (hàng = ô thật, cột = ô dự đoán) trên tập đưa vào."""
    classes = list(getattr(model, "cell_classes", ()))
    pairs = [
        (classes.index(r["cell"]), int(np.argmax(predictions[r["sample_id"]]["cell"])))
        for r in rows
        if r.get("cell") in classes and "cell" in predictions.get(r["sample_id"], {})
    ]
    if not classes or not pairs:
        return None
    matrix = np.zeros((len(classes), len(classes)), int)
    for truth, guess in pairs:
        matrix[truth, guess] += 1
    recall = {
        c: float(matrix[i, i] / matrix[i].sum()) for i, c in enumerate(classes) if matrix[i].sum()
    }
    return {
        "classes": classes,
        "samples": len(pairs),
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "recall": recall,
        "confusion": matrix.tolist(),
    }


def task_scores(model, prediction):
    """Điểm video theo tác vụ từ các đầu của model."""
    video = prediction["video"]
    by_task = {}
    for head, task in model.heads.items():
        by_task[task] = video[head]
    if "score" in video:
        return {t: video["score"] for t in TASKS}
    if "ai" not in by_task:
        parts = [by_task[t] for t in ("visual", "audio") if t in by_task]
        if parts:
            by_task["ai"] = max(parts)
    return by_task


def frame_head(model, prediction, task):
    frames = prediction["frames"]
    if "score" in frames:
        return frames["score"]
    for head, t in model.heads.items():
        if t == task:
            return frames[head]
    if task == "ai":
        parts = [frames[h] for h, t in model.heads.items() if t in ("visual", "audio")]
        return np.maximum.reduce(parts) if parts else None
    return None


def localization(model, rows, predictions, task):
    """Partial: AUC frame trong / ngoài đoạn (cùng clip) và ô sát chỗ nối vs ô giữa đoạn."""
    key = {
        "visual": "visual_fake_intervals",
        "audio": "audio_fake_intervals",
        "ai": "fake_intervals",
    }[task]
    inside_auc, top_inside, edge, middle = [], [], [], []
    for row in rows:
        if row.get("kind") != "partial" or not row.get(key):
            continue
        scores = frame_head(model, predictions[row["sample_id"]], task)
        if scores is None:
            continue
        n = len(scores)
        truth = np.zeros(n, bool)
        for a, b in row[key]:
            truth[int(round(a * 25)) : int(round(b * 25))] = True
        if truth.all() or not truth.any():
            continue
        inside_auc.append(roc_auc_score(truth, scores))
        top_inside.append(bool(truth[int(np.argmax(scores))]))
        idx = np.flatnonzero(truth)
        near = np.zeros(n, bool)
        for e in (idx[0], idx[-1]):
            near[max(0, e - 2) : e + 3] = True
        edge.append(float(scores[near & truth].mean()))
        if (truth & ~near).any():
            middle.append(float(scores[truth & ~near].mean()))
    if not inside_auc:
        return None
    return {
        "clips": len(inside_auc),
        "auc_inside_vs_outside": float(np.mean(inside_auc)),
        "top_frame_inside": float(np.mean(top_inside)),
        "mean_score_edge": float(np.mean(edge)),
        "mean_score_middle": float(np.mean(middle)) if middle else None,
    }


IOU_AP = (0.5, 0.75, 0.95)
IOU_AR = tuple(np.round(np.arange(0.5, 0.951, 0.05), 2))
AR_AT = (1, 5, 10)


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def proposals(scores, step=0.04, levels=(50, 60, 70, 80, 90, 95), keep=10):
    """Đoạn ứng viên từ điểm frame: các khúc liên tiếp vượt ngưỡng ở nhiều mức phân vị của chính
    video; điểm đoạn = điểm frame trung bình trong đoạn; bỏ trùng bằng NMS (IoU ≥ 0,7)."""
    scores = np.asarray(scores, float)
    found = []
    for level in levels:
        above = scores >= np.percentile(scores, level)
        edges = np.flatnonzero(np.diff(np.r_[0, above.astype(int), 0]))
        for start, end in zip(edges[::2], edges[1::2]):
            if end - start >= 2:
                found.append((float(scores[start:end].mean()), start * step, end * step))
    found.sort(reverse=True)
    kept = []
    for score, a, b in found:
        if all(iou((a, b), (x, y)) < 0.7 for _, x, y in kept):
            kept.append((score, a, b))
        if len(kept) >= keep:
            break
    return kept


def temporal_metrics(model, rows, predictions, task):
    """Định vị kiểu LAV-DF / AV-Deepfake1M: AP theo IoU (0,5 / 0,75 / 0,95) trên mọi video test
    (video real chỉ góp báo nhầm) và AR@N (recall với N đoạn đầu, trung bình IoU 0,5-0,95)."""
    key = {
        "visual": "visual_fake_intervals",
        "audio": "audio_fake_intervals",
        "ai": "fake_intervals",
    }[task]
    pool, truth = [], {}
    for row in rows:
        scores = frame_head(model, predictions[row["sample_id"]], task)
        if scores is None:
            return None
        truth[row["sample_id"]] = [tuple(span) for span in row.get(key) or []]
        for score, a, b in proposals(scores):
            pool.append((score, row["sample_id"], a, b))
    total = sum(len(v) for v in truth.values())
    if not total:
        return None
    pool.sort(key=lambda p: -p[0])
    out = {"segments": total}
    for threshold in IOU_AP:
        used, hits = set(), []
        for _, sid, a, b in pool:
            best, match = threshold, None
            for k, span in enumerate(truth[sid]):
                overlap = iou((a, b), span)
                if (sid, k) not in used and overlap >= best:
                    best, match = overlap, (sid, k)
            hits.append(match is not None)
            if match is not None:
                used.add(match)
        hits = np.asarray(hits, float)
        if not len(hits):
            out[f"ap@{threshold:g}"] = 0.0
            continue
        tp = np.cumsum(hits)
        precision, recall = tp / np.arange(1, len(hits) + 1), tp / total
        # AP kiểu VOC: precision bao (giảm dần) lấy tại mỗi điểm recall tăng.
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        steps = np.diff(np.r_[0.0, recall])
        out[f"ap@{threshold:g}"] = float((envelope * steps).sum())
    by_video = {}
    for score, sid, a, b in pool:
        by_video.setdefault(sid, []).append((a, b))
    for n in AR_AT:
        recalls = []
        for threshold in IOU_AR:
            found = 0
            for sid, spans in truth.items():
                top = by_video.get(sid, [])[:n]
                found += sum(any(iou(p, span) >= threshold for p in top) for span in spans)
            recalls.append(found / total)
        out[f"ar@{n}"] = float(np.mean(recalls))
    return out


def breakdown(rows, scores, task, seen):
    """AUC từng nhóm sample dương so với clip real (cell real) cùng tập."""
    reals = [r for r in rows if r.get("cell") == "real"]
    real_scores = [scores[r["sample_id"]] for r in reals]
    groups = defaultdict(list)
    for r in rows:
        if not target(r, task):
            continue
        groups[f"cell/{r['cell']}"].append(r)
        groups[f"kind/{r.get('kind')}"].append(r)
        if r.get("generator") not in (None, "none"):
            groups[f"generator/{r['generator']}"].append(r)
            groups[f"generator_seen/{r['generator'] in seen['generators']}"].append(r)
        if r.get("voice_method"):
            groups[f"voice/{r['voice_method']}"].append(r)
            groups[f"voice_seen/{r['voice_method'] in seen['voices']}"].append(r)
    out = {}
    for name, members in sorted(groups.items()):
        labels = [1] * len(members) + [0] * len(reals)
        values = [scores[r["sample_id"]] for r in members] + real_scores
        out[name] = {"n": len(members), "auc": auc(labels, values)}
    # Theo giới (Model Cards: kết quả phân rã theo nhóm nhân khẩu học): dương và real cùng giới.
    for gender in sorted({r.get("gender") for r in rows if r.get("gender")}):
        positives = [r for r in rows if r.get("gender") == gender and target(r, task)]
        negatives = [r for r in reals if r.get("gender") == gender]
        labels = [1] * len(positives) + [0] * len(negatives)
        values = [scores[r["sample_id"]] for r in positives + negatives]
        out[f"gender/{gender}"] = {"n": len(positives), "auc": auc(labels, values)}
    return out


def per_speaker(rows, scores, task):
    by = defaultdict(list)
    for r in rows:
        by[r.get("speaker_id")].append(r)
    values = {}
    for speaker, members in by.items():
        value = auc([target(r, task) for r in members], [scores[r["sample_id"]] for r in members])
        if value is not None:
            values[speaker] = value
    return {
        "speakers": len(values),
        "mean_within_speaker_auc": float(np.mean(list(values.values()))) if values else None,
        "min": float(min(values.values())) if values else None,
    }


def ref_norm(rows, scores, task, count):
    """z-score theo clip real tham chiếu của từng người; trả (metric ref, metric thô cùng tập)."""
    references = enrollment(rows, count)
    used = {r["source_clip_id"] for refs in references.values() for r in refs}
    stats = {}
    for key, refs in references.items():
        values = np.array([scores[r["sample_id"]] for r in refs])
        stats[key] = (values.mean(), values.std() + 1e-3)
    kept = [
        r
        for r in rows
        if r.get("source_clip_id") not in used and (r["split"], r.get("speaker_id")) in stats
    ]
    labels = [target(r, task) for r in kept]
    raw = [scores[r["sample_id"]] for r in kept]
    normed = [
        (scores[r["sample_id"]] - stats[(r["split"], r["speaker_id"])][0])
        / stats[(r["split"], r["speaker_id"])][1]
        for r in kept
    ]
    return {
        "samples": len(kept),
        "auc_raw_same_set": auc(labels, raw),
        "auc_ref_norm": auc(labels, normed),
    }


FPR_TARGETS = (0.01, 0.05)
BASE_RATES = (0.5, 0.1, 0.01)


def operating_points(val_labels, val_scores, labels, scores):
    """Ngưỡng = phân vị điểm real validation (FPR mục tiêu); TPR / FPR test và precision theo tỉ lệ
    fake π: TPR·π / (TPR·π + FPR·(1 − π))."""
    val_labels, val_scores = np.asarray(val_labels), np.asarray(val_scores, float)
    labels, scores = np.asarray(labels), np.asarray(scores, float)
    reals = val_scores[val_labels == 0]
    out = {}
    if not len(reals) or len(set(labels.tolist())) < 2:
        return out
    for target in FPR_TARGETS:
        threshold = float(np.quantile(reals, 1 - target))
        predicted = scores > threshold
        tpr = float(predicted[labels == 1].mean())
        fpr = float(predicted[labels == 0].mean())
        out[f"fpr_{target:g}"] = {
            "threshold": threshold,
            "tpr": tpr,
            "fpr": fpr,
            "base_rates": {
                f"{rate:g}": {
                    "precision": tpr * rate / max(1e-12, tpr * rate + fpr * (1 - rate)),
                    "false_alarms_per_1000_real": fpr * 1000,
                }
                for rate in BASE_RATES
            },
        }
    return out


def task_metrics(cfg, val_rows, test_rows, val_scores, test_scores, task, seen):
    """Mọi phép đo điểm video cho một tác vụ (ngưỡng chọn trên validation); None nếu không đo được."""
    val_rows, test_rows = known(val_rows, task), known(test_rows, task)
    if any(test_scores.get(r["sample_id"]) is None for r in test_rows):
        return None
    v_labels = [target(r, task) for r in val_rows]
    if len(set(v_labels)) < 2:
        return None
    v_values = [val_scores[r["sample_id"]] for r in val_rows]
    threshold = best_threshold(v_labels, v_values)
    labels = [target(r, task) for r in test_rows]
    values = [test_scores[r["sample_id"]] for r in test_rows]
    return {
        "auc": auc(labels, values),
        "auc_speaker_balanced": speaker_balanced_auc(test_rows, labels, values),
        "ap": float(average_precision_score(labels, values)) if any(labels) else None,
        "eer": eer(labels, values),
        "threshold": threshold,
        **at_threshold(labels, values, threshold),
        "operating_points": operating_points(v_labels, v_values, labels, values),
        "fpr_by_cell": {
            cell: float(
                np.mean(
                    [
                        test_scores[r["sample_id"]] >= threshold
                        for r in test_rows
                        if r.get("cell") == cell and not target(r, task)
                    ]
                )
            )
            for cell in ("real", "sham")
            if any(r.get("cell") == cell and not target(r, task) for r in test_rows)
        },
        "breakdown": breakdown(test_rows, test_scores, task, seen),
        "per_speaker": per_speaker(test_rows, test_scores, task),
        "ref_norm": ref_norm(test_rows, test_scores, task, cfg["generation"]["references"]),
        "localization": None,
        "temporal": None,
    }


def save_predictions(path, scores):
    """scores: {tác vụ: {sample_id: điểm}} → npz (sample_id, video_<tác vụ>)."""
    ids = sorted({sid for values in scores.values() for sid in values})
    np.savez_compressed(
        path,
        sample_id=np.array(ids),
        **{
            f"video_{t}": np.array([scores.get(t, {}).get(sid, np.nan) for sid in ids], float)
            for t in TASKS
        },
    )


SHIFT_LAGS = (2, 3, 4)


def shift_applicable(model):
    """Model nhận cả tiếng lẫn hình (trượt tiếng mới có nghĩa)."""
    needs = set(model.needs)
    return "av" in needs or {"ssl", "mouth"} <= needs


def shift_robustness(model, cfg, val_rows, test_rows, device, val_pred, test_pred, lags=SHIFT_LAGS):
    """Phép thử trễ (RESEARCH_WORKLOG/v004_model.md §7): clip real test bị trượt tiếng k frame (trượt
    đặc trưng như lúc train). AI = 0 với mọi k nên điểm AI? không nên tăng. Báo tỉ lệ báo nhầm ở
    ngưỡng FPR 5 % của real validation (k = 0) và AUC real trượt so với real gốc (0,5 = không bị
    lừa); đầu mismatch (nếu có) thì ngược lại nên tăng khi k lớn."""
    val_real = [r for r in val_rows if r.get("cell") == "real"]
    test_real = [r for r in test_rows if r.get("cell") == "real"]
    if not val_real or not test_real:
        return None

    def scores(pred, task):
        values = [task_scores(model, pred[r["sample_id"]]).get(task) for r in test_real]
        return None if any(v is None for v in values) else values

    reference = np.array([task_scores(model, val_pred[r["sample_id"]])["ai"] for r in val_real])
    threshold = float(np.quantile(reference, 0.95))
    base = scores(test_pred, "ai")
    base_mm = scores(test_pred, "mismatch")
    out = {
        "real_test_clips": len(test_real),
        "threshold_fpr_0.05": threshold,
        "fpr_lag_0": float(np.mean(np.asarray(base) > threshold)),
    }
    for k in lags:
        pred = predict(model, cfg, test_real, device, lag=k)
        shifted = scores(pred, "ai")
        entry = {
            "fpr": float(np.mean(np.asarray(shifted) > threshold)),
            "auc_vs_unshifted": auc([1] * len(shifted) + [0] * len(base), shifted + base),
        }
        mm = scores(pred, "mismatch")
        if mm is not None and base_mm is not None:
            entry["mismatch_auc_vs_unshifted"] = auc(
                [1] * len(mm) + [0] * len(base_mm), mm + base_mm
            )
        out[f"lag_{k}"] = entry
    return out


def evaluate_model(cfg, model_dir, rows, device, seen):
    model = load_model(Path(model_dir) / "best.pt", device)
    val_rows = [r for r in rows if r["split"] == "validation"]
    test_rows = [r for r in rows if r["split"] == "test"]
    val_pred = predict(model, cfg, val_rows, device)
    test_pred = predict(model, cfg, test_rows, device)
    metrics = {"model": Path(model_dir).name, "frame_level": model.frame_level, "tasks": {}}
    val_all, test_all = {}, {}
    for task in TASKS:
        val_scores = {s: task_scores(model, p).get(task) for s, p in val_pred.items()}
        test_scores = {s: task_scores(model, p).get(task) for s, p in test_pred.items()}
        if all(v is not None for v in val_scores.values()):
            val_all[task] = val_scores
        if all(v is not None for v in test_scores.values()):
            test_all[task] = test_scores
        result = task_metrics(cfg, val_rows, test_rows, val_scores, test_scores, task, seen)
        if result is None:
            continue
        if model.frame_level and task != "mismatch":
            result["localization"] = localization(model, test_rows, test_pred, task)
            result["temporal"] = temporal_metrics(model, test_rows, test_pred, task)
        metrics["tasks"][task] = result
    kinds = manipulation_type(model, test_rows, test_pred)
    if kinds is not None:
        metrics["manipulation_type"] = kinds
    if shift_applicable(model):
        metrics["shift_robustness"] = shift_robustness(
            model, cfg, val_rows, test_rows, device, val_pred, test_pred
        )
    save_predictions(Path(model_dir) / "predictions-val.npz", val_all)
    save_predictions(Path(model_dir) / "predictions-test.npz", test_all)
    write_json(Path(model_dir) / "metrics.json", metrics)
    return metrics


# Ghép muộn (không train): ai = max(z_visual, z_audio). bl_ens: hai baseline đúng gốc (v003);
# bl_ens_av (v004): hai chuyên gia học được trên cùng dữ liệu (BL-M/sup chấm AI? từ độ khớp, BL-A2
# giọng) → trả lời câu "một model hay ghép muộn chuyên gia tốt" (v003 §5.9 RQ-M1 bỏ ngỏ).
ENSEMBLES = (
    {"name": "bl_ens", "visual": "lipinc_v2", "audio": "ssl_aasist"},
    {"name": "bl_ens_av", "visual": "avh_align_sup", "audio": "xlsr_frozen"},
)
ENSEMBLE = ENSEMBLES[0]


def load_predictions(path, task):
    with np.load(path, allow_pickle=False) as z:
        return dict(zip(z["sample_id"].tolist(), z[f"video_{task}"].tolist()))


def zscore(val, test, val_rows):
    """Chuẩn hoá theo TB / độ lệch điểm của clip real validation (cùng thang cho hai model)."""
    reals = np.array([val[r["sample_id"]] for r in val_rows if r.get("cell") == "real"])
    mean, std = (reals.mean(), reals.std() + 1e-6) if len(reals) else (0.0, 1.0)
    return (
        {k: (v - mean) / std for k, v in val.items()},
        {k: (v - mean) / std for k, v in test.items()},
    )


def evaluate_ensemble(cfg, runs, rows, seen):
    """Mỗi ghép muộn trong ENSEMBLES, mỗi seed có đủ hai thành viên."""
    results = {}
    for spec in ENSEMBLES:
        results.update(evaluate_one_ensemble(cfg, runs, rows, seen, spec))
    return results


def evaluate_one_ensemble(cfg, runs, rows, seen, spec):
    """ai = max(z_V, z_A), visual = z_V, audio = z_A (z-score theo real validation)."""
    val_rows = [r for r in rows if r["split"] == "validation"]
    test_rows = [r for r in rows if r["split"] == "test"]
    results = {}
    for visual_dir in sorted(runs.glob(f"{spec['visual']}_seed*")):
        seed = visual_dir.name.rsplit("_seed", 1)[1]
        audio_dir = runs / f"{spec['audio']}_seed{seed}"
        if not (audio_dir / "predictions-test.npz").is_file():
            continue
        parts = {}
        for task, folder in (("visual", visual_dir), ("audio", audio_dir)):
            parts[task] = zscore(
                load_predictions(folder / "predictions-val.npz", task),
                load_predictions(folder / "predictions-test.npz", task),
                val_rows,
            )
        val_scores = {t: parts[t][0] for t in ("visual", "audio")}
        test_scores = {t: parts[t][1] for t in ("visual", "audio")}
        for scores in (val_scores, test_scores):
            scores["ai"] = {
                k: max(scores["visual"][k], scores["audio"][k]) for k in scores["visual"]
            }
        out = runs / f"{spec['name']}_seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        metrics = {"model": out.name, "frame_level": False, "tasks": {}}
        metrics["members"] = {"visual": visual_dir.name, "audio": audio_dir.name}
        for task in ("ai", "visual", "audio"):
            result = task_metrics(
                cfg, val_rows, test_rows, val_scores[task], test_scores[task], task, seen
            )
            if result is not None:
                metrics["tasks"][task] = result
        save_predictions(out / "predictions-val.npz", val_scores)
        save_predictions(out / "predictions-test.npz", test_scores)
        write_json(out / "metrics.json", metrics)
        results[out.name] = metrics
    return results


def evaluated(folder, stamp):
    """metrics.json đã có của đúng checkpoint + dữ liệu này (không phải chấm lại)."""
    path = Path(folder) / "metrics.json"
    if not path.is_file():
        return None
    metrics = read_json(path)
    return metrics if metrics.get("stamp") == stamp else None


def evaluate_all(cfg, force=False, models=None):
    """Test model đã train xong. `models` (tên model, vd. ["trihead_v4"]): chỉ chấm các model này
    (bước train vừa xong); None: mọi model trong runs. Model khác đã chấm thì đọc lại kết quả.
    Checkpoint đã chấm (cùng sha best.pt, cùng dữ liệu) không chạy lại trừ khi force; checkpoint
    mới của model đã từng chấm (train lại) được chấm và ghi là lần thứ n trong test-log.json.
    Ghép muộn tính lại từ dự đoán đã lưu khi đủ hai thành viên (không cần GPU)."""
    rows, selection = training_dataset(cfg)
    runs = Path(cfg["runs"])
    device = cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu"
    names = {m for m in models} if models else None
    ensembles = {spec["name"] for spec in ENSEMBLES}
    folders = sorted(
        p
        for p in runs.iterdir()
        if (p / "best.pt").is_file() and p.name.rsplit("_seed", 1)[0] not in ensembles
    )
    data = fingerprint(selection)
    stamps = {p.name: {"checkpoint": sha(p / "best.pt"), "dataset": data} for p in folders}
    log_path = runs / "test-log.json"
    log = read_json(log_path) if log_path.is_file() else []
    train_rows = [r for r in rows if r["split"] == "train"]
    seen = {
        "generators": {
            r["generator"] for r in train_rows if r.get("generator") not in (None, "none")
        },
        "voices": {r["voice_method"] for r in train_rows if r.get("voice_method")},
    }
    results, fresh = {}, []
    began = time.monotonic()
    test = sum(r["split"] == "test" for r in rows)
    chosen = [p for p in folders if names is None or p.name.rsplit("_seed", 1)[0] in names]
    for folder in folders:
        old = None if force else evaluated(folder, stamps[folder.name])
        if old is not None:
            results[folder.name] = old
        elif folder not in chosen:
            print(f"    {folder.name}: chưa chấm test (không thuộc lượt này)", flush=True)
    for index, folder in enumerate(chosen, 1):
        if folder.name in results:
            print(f"=== evaluate {folder.name}: đã chấm checkpoint này, đọc lại", flush=True)
            continue
        print(f"=== evaluate [{index}/{len(chosen)}] {folder.name}: {test} mẫu test", flush=True)
        tick = time.monotonic()
        metrics = evaluate_model(cfg, folder, rows, device, seen)
        metrics["stamp"] = stamps[folder.name]
        write_json(folder / "metrics.json", metrics)
        results[folder.name] = metrics
        fresh.append(folder)
        times = 1 + sum(e["model"] == folder.name for e in log)
        log.append(
            {
                "model": folder.name,
                **stamps[folder.name],
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "session": cfg.get("session"),
                "test_use": times,
            }
        )
        if times > 1:
            print(f"    CHÚ Ý: {folder.name} dùng test lần {times} (checkpoint mới)", flush=True)
        aucs = " ".join(
            f"{t} {m['auc']:.3f}"
            for t, m in results[folder.name]["tasks"].items()
            if m.get("auc") is not None
        )
        print(
            f"    {folder.name}: AUC test {aucs} | {time.monotonic() - tick:.0f} s "
            f"(tổng {(time.monotonic() - began) / 60:.1f} phút)",
            flush=True,
        )
    results.update(evaluate_ensemble(cfg, runs, rows, seen))
    write_json(log_path, log)
    write_json(
        runs / "evaluation.json",
        {"seen": {k: sorted(v) for k, v in seen.items()}, "results": results},
    )
    stamp = {"dataset": selection, "checkpoints": {k: v["checkpoint"] for k, v in stamps.items()}}
    log_evaluation(cfg, runs, rows, results, seen, stamp, fresh)
    return {name: {t: m["tasks"][t]["auc"] for t in m["tasks"]} for name, m in results.items()}


def log_evaluation(cfg, runs, rows, results, seen, stamp, models):
    """Run "evaluate" mỗi lần chấm (theo phiên): bảng test của mọi model đã chấm tới lúc đó +
    ROC + histogram điểm + artifact; `models` = các model vừa chấm ở lần này."""
    session = cfg.get("session")
    group = cfg.get("run_group") or runs.name  # lane ghi vào <runs>-gpu<i>, group vẫn là <runs>
    # Tên run (và file wandb-<tên>.json): một model thì tên model, nhiều model thì số lượng; danh
    # sách đủ ở config evaluated_now (ghép mọi tên vượt giới hạn 260 ký tự đường dẫn Windows).
    now = Path(models[0]).name if len(models) == 1 else (f"{len(models)}models" if models else "")
    run = tracking.open_run(
        cfg,
        "evaluate",
        f"{group}-test" + (f"-{session}" if session else "") + (f"-{now}" if now else ""),
        group=group,
        config={
            "seen": {k: sorted(v) for k, v in seen.items()},
            "stamp": fingerprint(stamp),
            "evaluated_now": [Path(m).name for m in models],
            "session": session,
            "train_group": cfg.get("train_group"),
        },
        folder=runs,
        tags=[f"group:{cfg['train_group']}"] if cfg.get("train_group") else (),
    )
    if not run.active:
        return
    summary, parts, speakers, refs, locs = [], [], [], [], []
    for folder, metrics in results.items():
        model, seed = folder.rsplit("_seed", 1)
        for task, m in metrics["tasks"].items():
            summary.append(
                [model, seed, task, m["auc"], m["ap"], m["eer"], m["f1"], m["far"], m["recall"]]
            )
            for group, value in m["breakdown"].items():
                parts.append([model, seed, task, group, value["n"], value["auc"]])
            ps = m["per_speaker"]
            speakers.append(
                [model, seed, task, ps["speakers"], ps["mean_within_speaker_auc"], ps["min"]]
            )
            rn = m["ref_norm"]
            refs.append(
                [model, seed, task, rn["samples"], rn["auc_raw_same_set"], rn["auc_ref_norm"]]
            )
            lo = m.get("localization")
            if lo:
                locs.append(
                    [
                        model,
                        seed,
                        task,
                        lo["clips"],
                        lo["auc_inside_vs_outside"],
                        lo["top_frame_inside"],
                        lo["mean_score_edge"],
                        lo["mean_score_middle"],
                    ]
                )
    run.table(
        "test/summary",
        ["model", "seed", "task", "auc", "ap", "eer", "f1", "far", "recall"],
        summary,
    )
    run.table("test/breakdown", ["model", "seed", "task", "group", "n", "auc"], parts)
    run.table(
        "test/per_speaker", ["model", "seed", "task", "speakers", "mean_auc", "min_auc"], speakers
    )
    run.table(
        "test/ref_norm", ["model", "seed", "task", "samples", "auc_raw", "auc_ref_norm"], refs
    )
    run.table(
        "test/localization",
        ["model", "seed", "task", "clips", "auc_in_vs_out", "top_frame_inside", "edge", "middle"],
        locs,
    )
    test_rows = {r["sample_id"]: r for r in rows if r["split"] == "test"}
    first = {}
    for folder in sorted(results):
        first.setdefault(folder.rsplit("_seed", 1)[0], folder)
    series = {}
    for model, folder in first.items():
        with np.load(runs / folder / "predictions-test.npz", allow_pickle=False) as z:
            ids, scores = z["sample_id"].tolist(), z["video_ai"]
        keep = [i for i, s in enumerate(ids) if s in test_rows and np.isfinite(scores[i])]
        labels = [target(test_rows[ids[i]], "ai") for i in keep]
        if len(set(labels)) == 2:
            fpr, tpr, _ = roc_curve(labels, scores[keep])
            step = max(1, len(fpr) // 200)
            series[model] = (fpr[::step], tpr[::step])
        for cell in ("real", "sham", "voice", "source", "donor", "both"):
            run.hist(
                f"test_scores_ai/{model}/{cell}",
                [scores[i] for i in keep if test_rows[ids[i]].get("cell") == cell],
            )
    tracking.curves(run, "test/roc_ai", series, title="ROC test, tác vụ ai (seed đầu)")
    run.artifact(
        f"evaluation-{runs.name}",
        "evaluation",
        [runs / "evaluation.json", runs / "test-log.json"]
        + [runs / f / "metrics.json" for f in results]
        + [runs / f / "predictions-test.npz" for f in results],
        {"stamp": fingerprint(stamp)},
    )
    run.finish()
