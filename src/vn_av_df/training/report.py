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

# (model A, model B, tác vụ, ý nghĩa, tập con[, biến thể nén]): tập con "unseen_generator" /
# "unseen_voice" giữ mọi mẫu âm tính và chỉ các mẫu dương sinh bằng generator / giọng chưa thấy khi
# train; biến thể (vd "c480", cfg["variants"]) = so trên predictions-test-<biến thể>.npz.
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
    # v004 (RESEARCH_WORKLOG/v004_model.md §7, đặt trước khi train lại); thiếu model thì bỏ qua.
    ("trihead_v4", "trihead_v2", "ai", "V1 M3 không kém M1 (≥ −0,01)", None),
    ("trihead_v4", "bl_ens_av", "ai", "V2 M3 hơn ghép muộn BL-M/sup + BL-A2", None),
    ("trihead_v4", "avh_align_plus", "ai", "V3 M3 hơn M2", None),
    ("trihead_v4", "trihead_v4_no_phi", "ai", "V4 kênh Φ giúp (≥ 0,01)", None),
    ("trihead_v4_ref", "trihead_v4", "ai", "V5 tham chiếu người giúp (≥ 0,01)", None),
    ("trihead_v4", "xlsr_frozen", "audio", "V6 M3 hơn BL-A2, giọng chưa thấy", "unseen_voice"),
    ("trihead_v4", "avh_align", "mismatch", "V7 M3 hơn BL-M ở mismatch", None),
    ("trihead_v4", "trihead_v4_abs", "ai", "V8 tách luồng tương đối không làm kém (≥ −0,01)", None),
    ("trihead_v4", "trihead_v4_no_prosody", "ai", "V9 nhánh ngữ điệu ↔ đầu giúp (≥ 0,01)", None),
    (
        "trihead_v4",
        "trihead_v4_no_prosody",
        "visual",
        "V10 nhánh ngữ điệu ↔ đầu giúp ở ⑤ donor",
        "cell:donor",
    ),
    (
        "trihead_v4",
        "trihead_v4_no_prosody",
        "mismatch",
        "V11 nhánh ngữ điệu ↔ đầu giúp mismatch",
        None,
    ),
    # v005: M3-R (RESEARCH_WORKLOG/v005_model.md, đặt trước khi train thật).
    ("trihead_v5", "bl_ens_av", "ai", "R1 M3-R hơn BL-Ens-AV", None),
    ("trihead_v5", "bl_ens_av", "ai", "R2 M3-R hơn BL-Ens-AV, video nén 480p", None, "c480"),
    # v005 §9 (đặt 08/11 sau thử ở máy, trước khi train thật): 360p có khi train; VP9 360p chưa thấy.
    ("trihead_v5", "bl_ens_av", "ai", "R2b M3-R hơn BL-Ens-AV, video nén 360p", None, "c360"),
    ("trihead_v5", "bl_ens_av", "ai", "R2c M3-R hơn BL-Ens-AV, VP9 360p (codec chưa thấy)", None, "w360"),
    ("trihead_v5", "avh_align_plus", "ai", "R3 M3-R hơn M2, video nén 480p", None, "c480"),
    ("trihead_v5", "trihead_v5_clean", "ai", "R4 học nén giúp ở 480p (≥ 0,01)", None, "c480"),
    ("trihead_v5_quality", "trihead_v5", "ai", "R5 nhánh chất lượng giúp ở 480p (≥ 0,01)", None, "c480"),
    ("trihead_v5", "trihead_v5_no_pair", "ai", "R6 cặp sạch ↔ nén giúp ở 480p", None, "c480"),
    (
        "trihead_v5",
        "xlsr_frozen",
        "audio",
        "R8 M3-R không kém BL-A2, giọng chưa thấy, 480p (≥ −0,01)",
        "unseen_voice",
        "c480",
    ),
]


def subset(rows, task, kind, seen):
    """Giữ mẫu âm tính + mẫu dương thuộc generator / giọng chưa thấy khi train; "cell:<ô>" giữ mẫu
    âm tính + mẫu dương của một ô (vd. cell:donor cho nhánh ngữ điệu ↔ đầu của M3)."""
    if kind is None:
        return rows
    if kind.startswith("cell:"):
        cell = kind.split(":", 1)[1]
        return [r for r in rows if not target(r, task) or r.get("cell") == cell]
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


def load_scores(runs, folder, task, variant=None):
    path = Path(runs) / folder / (f"predictions-test-{variant}.npz" if variant else "predictions-test.npz")
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as z:
        return dict(zip(z["sample_id"].tolist(), z[f"video_{task}"].tolist()))


def paired_delta(rows, first, second, task, replicates=1000, seed=42):
    common = [
        r
        for r in rows
        if target(r, task) is not None  # v004: mismatch mask không vào phép so
        and r["sample_id"] in first
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


def num(value, digits=3):
    """Số trong bảng report; thiếu (None / NaN) thì '-' thay vì làm hỏng cả report cuối run."""
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    return f"{value:.{digits}f}"


def build_report(cfg, run=None):
    runs = Path(cfg["runs"])
    evaluation = read_json(runs / "evaluation.json")
    results = evaluation["results"]
    by_model = defaultdict(list)
    for folder, metrics in results.items():
        by_model[model_name(folder)].append((folder, metrics))
    seeds = sorted({len(items) for items in by_model.values()})
    lines = [
        "# Kết quả v005",
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
        "checkpoint và ngưỡng chọn trên validation; số lần mỗi model dùng test: `test-log.json`.",
        "- Môi trường, commit, dữ liệu, cache: `provenance.json`; thời gian từng lần train: "
        "`<model>_seed<seed>/history.json`.",
        "",
        "## AUC test (mean ± std theo seed)",
        "",
        "| Model | ai | ai (cân theo người) | visual | audio | mismatch | ai (ref-norm) "
        "| định vị visual: ô cao nhất trong đoạn | định vị mismatch: AUC trong / ngoài đoạn |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, items in sorted(by_model.items()):

        def col(task, key="auc"):
            return fmt([m["tasks"].get(task, {}).get(key) for _, m in items])

        ref = fmt(
            [m["tasks"].get("ai", {}).get("ref_norm", {}).get("auc_ref_norm") for _, m in items]
        )

        def located(task, key):
            return fmt(
                [
                    ((m["tasks"].get(task, {}) or {}).get("localization") or {}).get(key)
                    for _, m in items
                ]
            )

        loc = located("visual", "top_frame_inside")
        loc_mismatch = located("mismatch", "auc_inside_vs_outside")
        lines.append(
            f"| {name} | {col('ai')} | {col('ai', 'auc_speaker_balanced')} | {col('visual')} "
            f"| {col('audio')} | {col('mismatch')} | {ref} | {loc} | {loc_mismatch} |"
        )
    rows, _ = training_dataset(cfg)
    test_rows = [r for r in rows if r["split"] == "test"]
    lines += [
        "",
        "## Cổng (ΔAUC, bootstrap theo speaker, seed đầu tiên)",
        "",
        "| So sánh | Tác vụ | Bản | ΔAUC | CI 95% | Speaker |",
        "|---|---|---|---|---|---|",
    ]
    seen = {k: set(v) for k, v in evaluation["seen"].items()}
    gates = []
    for first, second, task, note, kind, *rest in GATES:
        variant = rest[0] if rest else None
        a = sorted(f for f, _ in by_model.get(first, []))
        b = sorted(f for f, _ in by_model.get(second, []))
        if not a or not b:
            continue
        first_scores = load_scores(runs, a[0], task, variant)
        second_scores = load_scores(runs, b[0], task, variant)
        if not first_scores or not second_scores:
            continue
        result = paired_delta(subset(test_rows, task, kind, seen), first_scores, second_scores, task)
        gates.append(
            {"first": first, "second": second, "task": task, "note": note, "subset": kind,
             "variant": variant, **result}
        )
        ci = (
            "-"
            if result["ci95"] is None
            else f"[{result['ci95'][0]:+.3f}; {result['ci95'][1]:+.3f}]"
        )
        delta = "-" if result["delta"] is None else f"{result['delta']:+.3f}"
        lines.append(
            f"| {note} ({first} − {second}) | {task} | {variant or 'gốc'} | {delta} | {ci} "
            f"| {result['speakers']} |"
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
                f"| {name} | {num(point['tpr'])} | {num(point['fpr'])} | "
                f"{num(rates['0.5']['precision'])} | {num(rates['0.1']['precision'])} | "
                f"{num(rates['0.01']['precision'])} | "
                f"{num(rates['0.5']['false_alarms_per_1000_real'], 0)} |"
            )
    variant_rows = []
    for name, items in sorted(by_model.items()):
        clean_ai = sorted(items)[0][1]["tasks"].get("ai", {})
        for variant, entry in sorted((sorted(items)[0][1].get("variants") or {}).items()):
            task = (entry.get("tasks") or {}).get("ai") or {}
            if not task:
                continue
            clean_point = (task.get("operating_points") or {}).get("fpr_0.05") or {}
            same = (task.get("operating_points_same_quality") or {}).get("fpr_0.05") or {}
            aucs = [m.get("variants", {}).get(variant, {}).get("tasks", {}).get("ai", {}).get("auc") for _, m in items]
            variant_rows.append(
                f"| {name} | {variant} | {num(clean_ai.get('auc'))} | {fmt(aucs)} | "
                f"{num(clean_point.get('tpr'))} | {num(clean_point.get('fpr'))} | "
                f"{num(same.get('tpr'))} | {num(same.get('fpr'))} |"
            )
    if variant_rows:
        lines += [
            "",
            "## Video nén lại (cfg[\"variants\"]), tác vụ ai",
            "",
            "Cùng mẫu test, nén lại video + tiếng (settings.VARIANTS). Ngưỡng FPR 5 % đặt trên real "
            "validation **sạch** (lúc dùng không biết chất lượng video) → FPR thực tế cho biết điểm "
            "real bị đẩy lên bao nhiêu khi nén; cột cuối: ngưỡng đặt trên validation cùng mức nén.",
            "",
            "| Model | biến thể | AUC sạch (seed đầu) | AUC nén (mean ± std) | TPR (ngưỡng sạch) "
            "| FPR (ngưỡng sạch) | TPR (ngưỡng cùng mức) | FPR (ngưỡng cùng mức) |",
            "|---|---|---|---|---|---|---|---|",
            *variant_rows,
        ]
    pipeline_rows = []
    for name, items in sorted(by_model.items()):
        for variant, entry in sorted((sorted(items)[0][1].get("variants") or {}).items()):
            for task, m in sorted((entry.get("pipeline") or {}).items()):
                aucs = [x.get("variants", {}).get(variant, {}).get("pipeline", {}).get(task, {}).get("auc_pipeline_matched") for _, x in items]
                pipeline_rows.append(
                    f"| {name} | {task} | {num(m.get('auc_clean_negatives'))} | {fmt(aucs)} | "
                    f"{num(m.get('fpr_clean'))} | {num(m.get('fpr_processed'))} | "
                    f"{num(m.get('fpr_seen_family'))} | {num(m.get('fpr_unseen_family'))} |"
                )
    if pipeline_rows:
        lines += [
            "",
            "## Phát hiện AI thật hay dấu vết đường ống? (biến thể PR)",
            "",
            "Dương = lip-sync test (bản sạch); âm = real / sham (/ voice với visual) test đã qua xử lý "
            "lành tính giống đường ống generator, nhãn giữ nguyên. AUC khớp đường ống tụt nhiều so với âm "
            "sạch = model dựa vào vết xử lý. FPR ở ngưỡng FPR 5 % của validation sạch; họ chưa thấy = "
            "không có khi train (settings.VARIANTS[\"pr\"][\"heldout_families\"]).",
            "",
            "| Model | tác vụ | AUC âm sạch (seed đầu) | AUC khớp đường ống (mean ± std) | FPR sạch | FPR PR "
            "| FPR họ đã thấy | FPR họ chưa thấy |",
            "|---|---|---|---|---|---|---|---|",
            *pipeline_rows,
        ]
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
            lines.append(f"| {name} | " + " | ".join(num(v) for v in values) + " |")
    shift_rows = []
    for name, items in sorted(by_model.items()):
        shift = sorted(items)[0][1].get("shift_robustness")
        if shift:
            cells = [num(shift.get("fpr_lag_0"))]
            for k in (2, 3, 4):
                entry = shift.get(f"lag_{k}") or {}
                cells += [num(entry.get("fpr")), num(entry.get("auc_vs_unshifted"))]
            shift_rows.append(f"| {name} | " + " | ".join(cells) + " |")
    if shift_rows:
        lines += [
            "",
            "## Phép thử trễ: real test bị trượt tiếng k frame (AI = 0; seed đầu)",
            "",
            "Báo nhầm ở ngưỡng FPR 5 % của real validation; AUC real trượt so với real gốc "
            "(0,5 = không bị lừa bởi độ trễ).",
            "",
            "| Model | báo nhầm k=0 | k=2 | AUC k=2 | k=3 | AUC k=3 | k=4 | AUC k=4 |",
            "|---|---|---|---|---|---|---|---|",
            *shift_rows,
        ]
    for name, items in sorted(by_model.items()):
        kinds = sorted(items)[0][1].get("manipulation_type")
        if kinds:
            recall = ", ".join(f"{c} {v:.2f}" for c, v in kinds["recall"].items())
            lines += [
                "",
                f"## Đầu loại thao tác: {name} (seed đầu, test)",
                "",
                f"Độ chính xác {kinds['accuracy']:.3f} trên {kinds['samples']} mẫu; recall {recall}.",
                "",
                "| ô thật \\ dự đoán | " + " | ".join(kinds["classes"]) + " |",
                "|---" * (len(kinds["classes"]) + 1) + "|",
                *[
                    f"| {c} | " + " | ".join(str(x) for x in row) + " |"
                    for c, row in zip(kinds["classes"], kinds["confusion"])
                ],
            ]
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
    log_report(cfg, runs, by_model, gates, run=run)
    return {"report": str(runs / "report.md"), "gates": gates, "tasks": TASKS}


def log_report(cfg, runs, by_model, gates, run=None):
    """Run "report": bảng mean ± std theo seed, các cổng kèm CI, report.md làm artifact."""
    own = run is None
    if own:
        run = tracking.open_run(
            cfg, "report", f"{runs.name}-report", group=runs.name, folder=runs, with_setup=True
        )
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
    if own:
        run.finish()
