"""Báo cáo v003 từ `<run>/evaluation.json`: bảng mean ± std theo seed, so sánh có cặp.

So sánh có cặp: ΔAUC (A − B) trên cùng sample test, khoảng tin cậy 95% bằng bootstrap theo speaker
(người là đơn vị độc lập). Các cặp mặc định là các cổng ở docs/v003_model.md §6.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from vn_av_df import tracking
from vn_av_df.common.runtime import atomic_bytes, read_json, write_json
from vn_av_df.dataset import training_dataset
from vn_av_df.training.evaluate import TASKS, target

# (model A, model B, tác vụ, ý nghĩa, tập con): tập con "unseen_generator" / "unseen_voice" giữ
# mọi mẫu âm tính và chỉ các mẫu dương sinh bằng generator / giọng chưa thấy khi train.
GATES = [
    ("trihead_v2", "bl_ens", "ai", "M1 hơn BL-Ens (max BL-V, BL-A)", None),
    ("trihead_v2", "lipinc_v2", "visual", "M1 hơn BL-V, generator chưa thấy", "unseen_generator"),
    ("trihead_v2", "ssl_aasist", "audio", "M1 hơn BL-A, giọng chưa thấy", "unseen_voice"),
    ("trihead_v2", "avh_align_sup", "mismatch", "M1 hơn BL-M/sup ở mismatch", None),
    ("trihead_v2", "avh_align", "mismatch", "M1 hơn BL-M ở mismatch", None),
    ("trihead_v2", "trihead_v2_no_ref", "ai", "tham chiếu người giúp (≥ 0,03)", None),
    ("trihead_v2", "trihead_v2_no_sync", "mismatch", "nhánh sync giúp mismatch", None),
    ("trihead_v2", "trihead_v2_no_trace", "visual", "nhánh trace giúp visual", None),
    ("avh_align_plus", "avh_align_sup", "mismatch", "M2 hơn BL-M/sup (cùng dữ liệu)", None),
    ("avh_align_plus", "avh_align", "mismatch", "M2 hơn BL-M", None),
]


def subset(rows, task, kind, seen):
    """Giữ mẫu âm tính + mẫu dương thuộc generator / giọng chưa thấy khi train."""
    if kind is None:
        return rows
    key, known = (
        ("generator", seen["generators"])
        if kind == "unseen_generator"
        else ("voice_method", seen["voices"])
    )
    return [
        r
        for r in rows
        if not target(r, task) or (r.get(key) not in (None, "none") and r[key] not in known)
    ]


def model_name(folder):
    return folder.rsplit("_seed", 1)[0]


def load_scores(runs, folder, task):
    path = Path(runs) / folder / "predictions-test.npz"
    with np.load(path, allow_pickle=False) as z:
        return dict(zip(z["sample_id"].tolist(), z[f"video_{task}"].tolist()))


def paired_delta(rows, first, second, task, replicates=1000, seed=42):
    common = [
        r
        for r in rows
        if r["sample_id"] in first
        and r["sample_id"] in second
        and np.isfinite(first[r["sample_id"]])
        and np.isfinite(second[r["sample_id"]])
    ]
    speakers = defaultdict(list)
    for r in common:
        speakers[r.get("speaker_id")].append(r)
    groups = list(speakers.values())

    def delta(members):
        labels = [target(r, task) for r in members]
        if len(set(labels)) != 2:
            return None
        return roc_auc_score(labels, [first[r["sample_id"]] for r in members]) - roc_auc_score(
            labels, [second[r["sample_id"]] for r in members]
        )

    point = delta(common)
    values = []
    if len(groups) >= 5:
        rng = np.random.default_rng(seed)
        for _ in range(replicates):
            value = delta(
                [r for i in rng.integers(len(groups), size=len(groups)) for r in groups[i]]
            )
            if value is not None:
                values.append(value)
    ci = np.quantile(values, [0.025, 0.975]).tolist() if values else None
    return {"delta": point, "ci95": ci, "speakers": len(groups), "samples": len(common)}


def fmt(values):
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    return (
        f"{np.mean(values):.3f} ± {np.std(values):.3f}" if len(values) > 1 else f"{values[0]:.3f}"
    )


def build_report(cfg):
    runs = Path(cfg["runs"])
    evaluation = read_json(runs / "evaluation.json")
    results = evaluation["results"]
    by_model = defaultdict(list)
    for folder, metrics in results.items():
        by_model[model_name(folder)].append((folder, metrics))
    seeds = sorted({len(items) for items in by_model.values()})
    lines = [
        "# Kết quả v003",
        "",
        f"Generator đã thấy khi train: {', '.join(evaluation['seen']['generators'])}",
        f"Giọng đã thấy khi train: {', '.join(evaluation['seen']['voices'])}",
        "",
        "## Cách đọc",
        "",
        f"- Số seed mỗi model: {', '.join(map(str, seeds))}. "
        "± là độ lệch chuẩn giữa các seed (ddof 0); một seed thì không có ±.",
        f"- CI 95 % của cổng: bootstrap theo speaker ({cfg.get('bootstrap_replicates', 1000)} lần, "
        "phân vị 2,5-97,5 %), trên seed đầu tiên: chỉ gồm biến động do mẫu người ở test, "
        "không gồm biến động do khởi tạo / thứ tự dữ liệu"
        + (" (chạy 1 seed)." if seeds == [1] else "."),
        "- Không dò siêu tham số: baseline theo công thức của code gốc, M1 / M2 đặt trước; "
        "checkpoint và ngưỡng chọn trên validation; test mở một lần (`test-lock.json`).",
        "- Môi trường, commit, dữ liệu, cache: `provenance.json`; thời gian từng lần train: "
        "`<model>_seed<seed>/history.json`.",
        "",
        "## AUC test (mean ± std theo seed)",
        "",
        "| Model | ai | ai (cân theo người) | visual | audio | mismatch | ai (ref-norm) "
        "| định vị visual: ô cao nhất trong đoạn |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, items in sorted(by_model.items()):

        def col(task, key="auc"):
            return fmt([m["tasks"].get(task, {}).get(key) for _, m in items])

        ref = fmt(
            [m["tasks"].get("ai", {}).get("ref_norm", {}).get("auc_ref_norm") for _, m in items]
        )
        loc = fmt(
            [
                ((m["tasks"].get("visual", {}) or {}).get("localization") or {}).get(
                    "top_frame_inside"
                )
                for _, m in items
            ]
        )
        lines.append(
            f"| {name} | {col('ai')} | {col('ai', 'auc_speaker_balanced')} | {col('visual')} "
            f"| {col('audio')} | {col('mismatch')} | {ref} | {loc} |"
        )
    rows, _ = training_dataset(cfg)
    test_rows = [r for r in rows if r["split"] == "test"]
    lines += [
        "",
        "## Cổng (ΔAUC, bootstrap theo speaker, seed đầu tiên)",
        "",
        "| So sánh | Tác vụ | ΔAUC | CI 95% | Speaker |",
        "|---|---|---|---|---|",
    ]
    seen = {k: set(v) for k, v in evaluation["seen"].items()}
    gates = []
    for first, second, task, note, kind in GATES:
        a = sorted(f for f, _ in by_model.get(first, []))
        b = sorted(f for f, _ in by_model.get(second, []))
        if not a or not b:
            continue
        result = paired_delta(
            subset(test_rows, task, kind, seen),
            load_scores(runs, a[0], task),
            load_scores(runs, b[0], task),
            task,
        )
        gates.append(
            {"first": first, "second": second, "task": task, "note": note, "subset": kind, **result}
        )
        ci = (
            "-"
            if result["ci95"] is None
            else f"[{result['ci95'][0]:+.3f}; {result['ci95'][1]:+.3f}]"
        )
        delta = "-" if result["delta"] is None else f"{result['delta']:+.3f}"
        lines.append(
            f"| {note} ({first} − {second}) | {task} | {delta} | {ci} | {result['speakers']} |"
        )
    lines += [
        "",
        "## Điểm vận hành, tác vụ ai (seed đầu; ngưỡng = FPR 5 % trên real validation)",
        "",
        "Precision ở tỉ lệ fake π ngoài thực tế = TPR·π / (TPR·π + FPR·(1 − π)) (Layton et al. 2024).",
        "",
        "| Model | TPR | FPR test | precision π=50 % | π=10 % | π=1 % | báo nhầm / 1000 real |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, items in sorted(by_model.items()):
        point = (
            sorted(items)[0][1]["tasks"].get("ai", {}).get("operating_points", {}).get("fpr_0.05")
        )
        if point:
            rates = point["base_rates"]
            lines.append(
                f"| {name} | {point['tpr']:.3f} | {point['fpr']:.3f} | "
                f"{rates['0.5']['precision']:.3f} | {rates['0.1']['precision']:.3f} | "
                f"{rates['0.01']['precision']:.3f} | "
                f"{rates['0.5']['false_alarms_per_1000_real']:.0f} |"
            )
    lines += [
        "",
        "## Định vị đoạn giả, tác vụ ai (seed đầu; chuẩn LAV-DF / AV-Deepfake1M)",
        "",
        "| Model | AP@0.5 | AP@0.75 | AP@0.95 | AR@1 | AR@5 | AR@10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, items in sorted(by_model.items()):
        temporal = sorted(items)[0][1]["tasks"].get("ai", {}).get("temporal")
        if temporal:
            values = [
                temporal.get(k) for k in ("ap@0.5", "ap@0.75", "ap@0.95", "ar@1", "ar@5", "ar@10")
            ]
            lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += ["", "## Phân rã (seed đầu tiên của mỗi model, tác vụ ai)", ""]
    for name, items in sorted(by_model.items()):
        folder, metrics = sorted(items)[0]
        detail = metrics["tasks"].get("ai", {}).get("breakdown", {})
        if detail:
            parts = ", ".join(
                f"{k} {v['auc']:.3f}" for k, v in detail.items() if v["auc"] is not None
            )
            lines.append(f"- **{name}**: {parts}")
    text = "\n".join(lines) + "\n"
    atomic_bytes(runs / "report.md", text.encode("utf-8"))
    write_json(runs / "gates.json", gates)
    log_report(cfg, runs, by_model, gates)
    return {"report": str(runs / "report.md"), "gates": gates, "tasks": TASKS}


def log_report(cfg, runs, by_model, gates):
    """Run "report": bảng mean ± std theo seed, các cổng kèm CI, report.md làm artifact."""
    run = tracking.open_run(cfg, "report", f"{runs.name}-report", group=runs.name, folder=runs)
    if not run.active:
        return
    rows = []
    for name, items in sorted(by_model.items()):
        for task in TASKS:
            values = [m["tasks"].get(task, {}).get("auc") for _, m in items]
            values = [v for v in values if v is not None]
            if values:
                rows.append(
                    [name, task, float(np.mean(values)), float(np.std(values)), len(values)]
                )
    run.table("report/auc_mean_std", ["model", "task", "mean_auc", "std_auc", "seeds"], rows)
    run.table(
        "report/gates",
        [
            "first",
            "second",
            "task",
            "note",
            "delta_auc",
            "ci_low",
            "ci_high",
            "speakers",
            "samples",
        ],
        [
            [
                g["first"],
                g["second"],
                g["task"],
                g["note"],
                g["delta"],
                (g["ci95"] or [None, None])[0],
                (g["ci95"] or [None, None])[1],
                g["speakers"],
                g["samples"],
            ]
            for g in gates
        ],
    )
    run.artifact(f"report-{runs.name}", "report", [runs / "report.md", runs / "gates.json"])
    run.finish()
