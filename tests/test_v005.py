"""v005: biến thể dữ liệu (c480, c360, PR, mức chỉ test) và phép đo khớp đường ống. Dữ liệu tổng hợp."""

import numpy as np


def _rows():
    rows = []
    for k in range(40):
        split = ("train", "validation", "test")[k % 3]
        cid = f"clip{k:02d}"
        for cell in ("real", "sham", "voice", "source"):
            rows.append({"sample_id": f"{cid}_{cell}", "source_clip_id": cid, "split": split, "cell": cell,
                         "synthetic_visual": int(cell == "source"), "label": int(cell in ("voice", "source"))})
    return rows


def test_variant_specs_and_selection():
    """Đặc tả: c480 / c360 học khi train; PR chỉ validation / test (real / sham / voice); w360 / c288 chỉ test."""
    import settings
    from vn_av_df.training.prepare import variant_selection

    specs = settings.VARIANTS
    assert {n for n, s in specs.items() if s.get("train")} == {"c480", "c360"}
    assert set(specs) >= {"w360", "c288"}
    rows = _rows()
    pr = variant_selection(specs["pr"], rows)
    assert pr and {r["cell"] for r in pr} == {"real", "sham", "voice"}
    assert {r["split"] for r in pr} == {"validation", "test"}
    only_test = variant_selection(specs["w360"], rows)
    assert only_test and {r["split"] for r in only_test} == {"test"}
    assert len(variant_selection(specs["c480"], rows)) == len(rows)
    assert settings.config()["variant_specs"] is specs


def test_processed_real_chain_shared_and_heldout():
    """Chuỗi PR cố định theo clip nguồn (real / sham / voice chung); train / val chỉ họ đã thấy, test
    có cả họ chưa thấy; mọi họ có tham số ffmpeg."""
    import settings
    from vn_av_df.training.prepare import processed_real_chain, processed_real_middle

    spec = settings.VARIANTS["pr"]
    rows = _rows()
    by_clip = {}
    for r in rows:
        by_clip.setdefault(r["source_clip_id"], set()).add(processed_real_chain(spec, r))
    assert all(len(v) == 1 for v in by_clip.values())
    fams = {r["split"]: set() for r in rows}
    for r in rows:
        fams[r["split"]].add(processed_real_chain(spec, r)[0])
    assert fams["train"] <= set(spec["families"]) and fams["validation"] <= set(spec["families"])
    assert fams["test"] & set(spec["heldout_families"])
    for fam in spec["families"] + spec["heldout_families"]:
        ext, args = processed_real_middle(fam, 0.5, 1280, 720)
        assert ext.startswith(".") and "-c:v" in args


def test_variant_names_train_and_checkpoint():
    """Train học biến thể "train"; chọn checkpoint trung bình bản sạch + mức nén có nhãn (không PR,
    không mức chỉ test). Biến thể PR cho train vẫn chạy được nếu bật."""
    import settings
    from vn_av_df.training.train import variant_names

    cfg = {"variants": {n: ["/x"] for n in settings.VARIANTS}, "variant_specs": settings.VARIANTS}
    assert set(variant_names(cfg, train=True)) == {"c480", "c360"}
    on = {**settings.VARIANTS, "pr": {**settings.VARIANTS["pr"], "train": True}}
    assert set(variant_names({**cfg, "variant_specs": on}, train=True)) == {"c480", "c360", "pr"}
    assert set(variant_names(cfg, labelled=True)) == {"c480", "c360"}
    assert variant_names({"variants": {"c480": ["/x"]}}, train=True) == ["c480"]  # thiếu đặc tả: như cũ


def test_pipeline_metrics_detects_trace_reliance():
    """Model dựa vào vết xử lý: điểm âm PR bị đẩy lên → AUC khớp đường ống tụt, FPR PR tăng; model
    bất biến: hai AUC bằng nhau."""
    import settings
    from vn_av_df.training.evaluate import pipeline_metrics

    spec = settings.VARIANTS["pr"]
    rows = _rows()
    val = [r for r in rows if r["split"] == "validation"]
    test = [r for r in rows if r["split"] == "test"]
    pr_rows = [r for r in test if r["cell"] in ("real", "sham", "voice")]
    rng = np.random.default_rng(0)
    clean = {r["sample_id"]: float(r["synthetic_visual"] * 2 + rng.normal(0, 0.5)) for r in rows}
    val_all = {"visual": {r["sample_id"]: clean[r["sample_id"]] for r in val}}
    test_all = {"visual": {r["sample_id"]: clean[r["sample_id"]] for r in test}}
    same = pipeline_metrics(spec, val, test, pr_rows, val_all, test_all,
                            {"visual": {r["sample_id"]: clean[r["sample_id"]] for r in pr_rows}})
    shifted = pipeline_metrics(spec, val, test, pr_rows, val_all, test_all,
                               {"visual": {r["sample_id"]: clean[r["sample_id"]] + 2.0 for r in pr_rows}})
    v = same["visual"]
    assert v["auc_pipeline_matched"] == v["auc_clean_negatives"] and v["fpr_processed"] == v["fpr_clean"]
    s = shifted["visual"]
    assert s["auc_pipeline_matched"] < v["auc_pipeline_matched"] - 0.2 and s["fpr_processed"] > 0.5
    assert s["samples"]["negative"] == len(pr_rows) and set(s["fpr_by_family"]) <= set(spec["families"] + spec["heldout_families"])
    assert "ai" not in same  # thiếu điểm ai → bỏ qua, không lỗi


def test_resolution_gate():
    """Dưới mức train thấp nhất: AI? = audio, không xét hình; đủ độ phân giải hoặc thiếu audio: giữ nguyên."""
    from vn_av_df.training.evaluate import resolution_gate

    scores = {"ai": 0.9, "visual": 0.95, "audio": 0.2, "mismatch": 0.7}
    gated, on = resolution_gate(scores, 288, 360)
    assert on and gated["ai"] == 0.2 and gated["visual"] is None and gated["mismatch"] is None
    assert resolution_gate(scores, 360, 360) == (scores, False)
    assert resolution_gate({"ai": 0.9, "visual": 0.9}, 288, 360)[1] is False
    assert resolution_gate(scores, 288, None)[1] is False
