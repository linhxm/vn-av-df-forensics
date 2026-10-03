"""Persistent full/partial AI lip-sync edits with frame-exact manipulation intervals."""

import csv
import io
import math
import random
import tempfile
import time
from collections import Counter
from pathlib import Path

from vn_av_data.contract import validate_bundle

from vn_av_df.common.runtime import atomic_bytes, fingerprint, read_json, sha, write_json
from vn_av_df.data.groups import connected_groups, read_manifest, write_manifest
from vn_av_df.data.media import decode, probe
from vn_av_df.data.render import encode
from vn_av_df.dataset import SCHEMA, validate_rows

DEFAULT_SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
AUDIO_MODES = ("source", "donor")


def fake_audio_modes(options, split):
    """source: vẽ miệng theo đúng tiếng gốc (chỉ artifact); donor: tiếng thật khác của cùng người.

    Cấu hình cũ chỉ có audio_mode chung cho mọi split vẫn được đọc như trước.
    """
    if "fake_audio_modes_by_split" in options:
        modes = options["fake_audio_modes_by_split"].get(split)
    else:
        modes = [options.get("audio_mode", "source")]
    if not modes or len(set(modes)) != len(modes) or set(modes) - set(AUDIO_MODES):
        raise ValueError("Fake audio modes must be a nonempty unique subset of source/donor")
    return list(modes)


def split_ratios(value=None):
    """Tỷ lệ theo số clip sạch; giữ nguyên nhóm nên số thực tế có thể lệch mục tiêu."""
    value = DEFAULT_SPLIT_RATIOS if value is None else value
    if (
        not isinstance(value, dict)
        or set(value) != set(DEFAULT_SPLIT_RATIOS)
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < 1
            for v in value.values()
        )
        or not math.isclose(sum(value.values()), 1.0, abs_tol=1e-9)
    ):
        raise ValueError("split_ratios must contain positive train/validation/test summing to 1")
    return {key: float(value[key]) for key in DEFAULT_SPLIT_RATIOS}


def csv_write(path, rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, text.getvalue().encode("utf-8-sig"))


def prior_assignments(cfg):
    """Danh sách rỗng chia part độc lập; điền lịch sử để bật lại kế thừa split."""
    return prior_registry(cfg)[0]


def prior_registry(cfg):
    """Gộp lịch sử tùy chọn, kiểm đủ part khi bật lại sau nhiều part chia độc lập."""
    paths = cfg.get("split_history", [])
    if not paths:
        return [], []
    prior = {}
    versions = []
    covered = set()
    for path in paths:
        lock = read_json(path)
        if lock.get("schema") != "vn-av-df-split-registry-v1":
            raise ValueError("Need a cumulative split-lock with complete source identities")
        versions.append(lock.get("data_part", 0))
        # Registry cũ luôn tích lũy; registry mới ghi rõ part thực sự có trong file.
        covered.update(lock.get("covered_parts", range(1, lock.get("data_part", 0) + 1)))
        for row in lock["assignments"]:
            key = row["clip_id"]
            if key in prior and prior[key] != row:
                raise ValueError("Conflicting split history for the same clip")
            prior[key] = row
    if paths and not prior:
        raise ValueError("Empty split history")
    if cfg.get("data_part", 1) > 1 and max(versions) != cfg["data_part"] - 1:
        raise ValueError("Use the cumulative split-lock of the immediately preceding part")
    if cfg.get("data_part", 1) > 1 and covered != set(range(1, cfg["data_part"])):
        raise ValueError("Split history must cover all preceding parts; include independent locks")
    return list(prior.values()), sorted(covered)


def split_clean(rows, seed=42, history=(), ratios=None):
    """Mặc định 80/10/10 trong part; chỉ giữ split part cũ khi có history."""
    ratios = split_ratios(ratios)
    current = [{**r, "sample_id": r["clip_id"]} for r in rows]
    ids = {r["clip_id"] for r in current}
    if len(ids) != len(current) or ids & {r["clip_id"] for r in history}:
        raise ValueError("Duplicate clean clip across parts; keep each clip in one part")
    old_hashes = {r["sha256"] for r in history if r.get("sha256")}
    if any(r.get("sha256") in old_hashes for r in current):
        raise ValueError("Duplicate clean media across parts")
    if any(r.get("split") not in {"train", "validation", "test"} for r in history):
        raise ValueError("Invalid split history")
    groups = connected_groups([*history, *current])
    if len(groups) < 3 and not history:
        raise ValueError("Need three independent speaker/source groups before generation")
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    total = len(rows) + len(history)
    target = {split: ratio * total for split, ratio in ratios.items()}
    counts = Counter()
    result = []
    fresh = []
    for group in groups:
        locked = {r["split"] for r in group if r["clip_id"] not in ids}
        if len(locked) > 1:
            raise ValueError("New part connects previously separated splits; review identities")
        if locked:
            split = next(iter(locked))
            counts[split] += len(group)
            result.extend({**r, "split": split} for r in group if r["clip_id"] in ids)
        else:
            fresh.append(group)
    for i, group in enumerate(fresh):
        empty = [s for s in target if not counts[s]]
        choices = empty if len(fresh) - i == len(empty) else list(target)
        split = min(
            choices,
            key=lambda s: sum(
                (counts[t] + (len(group) if t == s else 0) - target[t]) ** 2 for t in target
            ),
        )
        counts[split] += len(group)
        result.extend({**r, "split": split} for r in group)
    return result


def make_plan(cfg):
    """Khóa parent/split/generator, hỗ trợ generator test-only và giữ audio gốc."""
    dest = Path(cfg["plan"])
    if dest.exists():
        raise FileExistsError("Plan exists; reuse it to resume or choose a new run name")
    rows, info = validate_bundle(cfg["clean_dataset"], probe=probe)
    history, covered = prior_registry(cfg)
    ratios = split_ratios(cfg.get("split_ratios"))
    rows = split_clean(rows, cfg["seed"], history, ratios)
    count = int(cfg["generation"]["clips_per_split"])
    if count < 0:
        raise ValueError("clips_per_split is 0 for all, otherwise positive")
    selected = []
    for split in ("train", "validation", "test"):
        candidates = [r for r in rows if r["split"] == split and r["duration_s"] >= 5]
        random.Random(cfg["seed"]).shuffle(candidates)
        selected.extend(candidates[:count] if count else candidates)
    jobs, skipped = [], []
    for row in selected:
        # Same-speaker donors stay in split and do not overlap the same source interval.
        donors = [
            d
            for d in rows
            if d["split"] == row["split"]
            and d["clip_id"] != row["clip_id"]
            and d.get("speaker_id")
            and d["speaker_id"] == row.get("speaker_id")
            and d["duration_s"] >= row["duration_s"]
            and (
                d["source_id"] != row["source_id"]
                or d["source_end_s"] <= row["source_start_s"]
                or row["source_end_s"] <= d["source_start_s"]
            )
        ]
        modes = fake_audio_modes(cfg["generation"], row["split"])
        sham_donor = sorted(donors, key=lambda r: r["clip_id"])[0] if donors else None
        if "donor" in modes and not donors:
            # Bỏ cả parent để mỗi parent giữ đủ các ô 2×2 đã chọn; ghi lại để báo cáo.
            skipped.append(row["clip_id"])
            continue
        donor = (
            random.Random(cfg["seed"] + len(jobs)).choice(
                sorted(donors, key=lambda r: r["clip_id"])
            )
            if "donor" in modes
            else row
        )
        generators = (
            cfg["generation"].get("generators_by_split", {}).get(row["split"], ["wav2lip_gan"])
        )
        if not generators or len(set(generators)) != len(generators):
            raise ValueError("Need unique generators per split")
        first = True
        for name in generators:
            for mode in modes:
                jobs.append(
                    {
                        "original": row,
                        "donor": donor if mode == "donor" else row,
                        "generator": name,
                        "audio_mode": mode,
                        "emit_real": first,
                        "sham_donor": sham_donor
                        if first and cfg["generation"].get("include_sham", False)
                        else None,
                    }
                )
                first = False
    if not jobs or (
        not history and {j["original"]["split"] for j in jobs} != {"train", "validation", "test"}
    ):
        raise ValueError("No eligible source/donor pair in a split; add longer independent clips")
    plan = {
        "schema": "vn-av-df-plan-v1",
        "data_part": cfg.get("data_part", 1),
        "split_ratios": ratios,
        "covered_parts": sorted({*covered, cfg.get("data_part", 1)}),
        "manifest_sha256": info["manifest_sha256"],
        "seed": cfg["seed"],
        "prior_assignments": history,
        "assignments": rows,
        "jobs": jobs,
        "config": cfg["generation"],
        "split_counts": dict(Counter(j["original"]["split"] for j in jobs)),
        "skipped_without_donor": skipped,
    }
    write_json(dest, plan)
    registry = {
        "schema": "vn-av-df-split-registry-v1",
        "data_part": cfg.get("data_part", 1),
        "split_ratios": ratios,
        "covered_parts": plan["covered_parts"],
        "manifest_sha256": info["manifest_sha256"],
        "seed": cfg["seed"],
        "assignments": [*history, *rows],
        "groups": len(connected_groups([*history, *rows])),
        "counts": dict(Counter(r["split"] for r in [*history, *rows])),
        "note": "Source/identity grouped; channel-disjointness and near-duplicate review required separately",
    }
    write_json(
        dest.with_name(dest.stem + "_split-lock.json"),
        registry,
    )
    total = sum(
        2 + int(j.get("emit_real", True)) + int(j.get("sham_donor") is not None) for j in jobs
    )
    return {
        "pairs": len(jobs),
        "videos": total,
        "splits": plan["split_counts"],
        "skipped_without_donor": len(skipped),
    }


def compose_partial(original, fake, span):
    """Identical decoded timeline and dimensions; no stretched/looped donors."""
    if (
        original["frames"].shape != fake["frames"].shape
        or original["pcm"].shape != fake["pcm"].shape
    ):
        raise ValueError("Generated video must preserve source geometry and timeline")
    a, b = span
    n = len(original["frames"])
    if not 0 <= a < b <= n:
        raise ValueError("Partial bounds must be valid frame indices")
    frames, pcm = original["frames"].copy(), original["pcm"].copy()
    frames[a:b] = fake["frames"][a:b]
    pcm[a * 1920 : b * 1920] = fake["pcm"][a * 1920 : b * 1920]
    return frames, pcm, [[a / 25, b / 25]]


def validate_plan_settings(cfg, plan):
    """Không âm thầm đổi tỷ lệ/lịch sử của plan đã khóa khi chạy lại notebook."""
    # Plan trước khi có cấu hình tỷ lệ dùng 70/15/15; không diễn giải lại thành 80/10/10.
    ratios = plan.get("split_ratios", {"train": 0.7, "validation": 0.15, "test": 0.15})
    if (
        plan["config"] != cfg["generation"]
        or plan["seed"] != cfg["seed"]
        or plan.get("data_part", 1) != cfg.get("data_part", 1)
        or ratios != split_ratios(cfg.get("split_ratios"))
        or plan.get("prior_assignments", []) != prior_assignments(cfg)
    ):
        raise ValueError("Generation settings differ from frozen plan; use a new plan/output")


def generate(cfg, generator=None):
    """Sinh bằng worker thực; giữ audio chung cho V-only, provenance và resume nghiêm ngặt."""
    from vn_av_df.generators import generator_adapter

    plan = read_json(cfg["plan"])
    validate_plan_settings(cfg, plan)
    root, out = Path(cfg["clean_dataset"]), Path(cfg["generated_dataset"])
    originals, info = validate_bundle(root, probe=probe)
    if (
        plan.get("schema") != "vn-av-df-plan-v1"
        or plan["manifest_sha256"] != info["manifest_sha256"]
    ):
        raise ValueError("Clean dataset differs from plan")
    assigned = split_clean(
        originals, plan["seed"], plan.get("prior_assignments", []), cfg.get("split_ratios")
    )
    if "assignments" in plan and assigned != plan["assignments"]:
        raise ValueError("Changed split assignments in plan")
    locked = {r["clip_id"]: r for r in assigned}
    for job in plan["jobs"]:
        if (
            any(locked.get(job[k]["clip_id"]) != job[k] for k in ("original", "donor"))
            or job["original"]["split"] != job["donor"]["split"]
        ):
            raise ValueError("Changed source, donor or split")
        if job.get("sham_donor") and (
            locked.get(job["sham_donor"]["clip_id"]) != job["sham_donor"]
            or job["sham_donor"]["split"] != job["original"]["split"]
        ):
            raise ValueError("Changed sham donor or split")
    adapters, provenance = {}, {}
    for name in {j.get("generator", "wav2lip_gan") for j in plan["jobs"]}:
        if generator is None:
            adapter = generator_adapter(name)
            adapters[name], provenance[name] = adapter.synthesize, adapter.provenance(cfg)
        else:
            adapters[name] = generator
            provenance[name] = {"generator": "test_fixture", "version": "test-only"}
    signature = fingerprint(
        {
            "plan": plan,
            "generator": provenance,
            "implementation": sha(__file__),
            "render": sha(Path(__file__).parent / "data/render.py"),
        }
    )
    lock = out / "generation.json"
    if lock.exists():
        if read_json(lock)["signature"] != signature:
            raise ValueError("Changed generation run; choose new output")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Output has no generation lock")
    write_json(lock, {"signature": signature, "provenance": provenance, "plan": plan})
    # Giữ biên bản chia part trong ZIP; chỉ tích lũy khi chủ động cung cấp history.
    write_json(
        out / "split-lock.json",
        {
            "schema": "vn-av-df-split-registry-v1",
            "data_part": plan.get("data_part", 1),
            "split_ratios": split_ratios(cfg.get("split_ratios")),
            "covered_parts": plan.get(
                "covered_parts", list(range(1, plan.get("data_part", 1) + 1))
            ),
            "assignments": [*plan.get("prior_assignments", []), *assigned],
        },
    )
    rows = []
    total = len(plan["jobs"])
    print(
        f"Generate {total} cặp: theo split {plan['split_counts']}, "
        f"{len(plan.get('skipped_without_donor', []))} parent bỏ vì không có donor",
        flush=True,
    )
    began = time.monotonic()
    for number, job in enumerate(plan["jobs"]):
        tick = time.monotonic()
        r, d = job["original"], job["donor"]
        name = job.get("generator", "wav2lip_gan")
        head = (
            f"Pair {number + 1}/{total} [{r['split']}] {name}/{job.get('audio_mode', 'source')}: "
            f"parent {r['clip_id']} ({r.get('speaker_id') or '-'})"
        )
        gen_info = provenance[name]
        key = fingerprint(job)[:20]
        record = out / "records" / f"{key}.json"
        if record.exists():
            previous = read_json(record)
            if any(sha(out / p["video"]) != p["sha256"] for p in previous):
                raise ValueError("Generated file changed")
            rows.extend(previous)
            print(f"{head}: đã sinh ở lượt trước, bỏ qua", flush=True)
            continue
        kinds = (["real"] if job.get("emit_real", True) else []) + ["full", "partial"]
        if job.get("sham_donor"):
            kinds.append("sham")
        dests = {kind: out / "clips" / f"{key}_{kind}.mp4" for kind in kinds}
        if any(p.exists() for p in dests.values()):
            raise ValueError(
                "Orphan output from interrupted pair; move that pair out before resuming"
            )
        original = decode(root / r["video"], max_side=plan["config"]["max_side"], sample_rate=48000)
        with tempfile.TemporaryDirectory(dir=out) as temp:
            target = Path(temp) / "generated.mkv"
            adapters[name](
                cfg,
                root / r["video"],
                root / d["video"],
                target,
                len(original["frames"]),
                original["frames"].shape[2],
                original["frames"].shape[1],
            )
            fake = decode(target, max_side=plan["config"]["max_side"], sample_rate=48000)
        mode = job.get("audio_mode", plan["config"].get("audio_mode", "source"))
        # Không lấy PCM generator (đã qua 16kHz/codec) làm dấu phân biệt fake.
        if mode == "source":
            fake["pcm"] = original["pcm"].copy()
        else:
            donor_audio = decode(
                root / d["video"], max_side=plan["config"]["max_side"], sample_rate=48000
            )["pcm"]
            if len(donor_audio) < len(original["pcm"]):
                raise ValueError("Donor audio shorter than parent")
            fake["pcm"] = donor_audio[: len(original["pcm"])].copy()
        n = len(original["frames"])
        rng = random.Random(plan["seed"] + number)
        # Cover short and longer events; manipulation extent is measured at 25 fps.
        length = min(n - 2, round(rng.choice(plan["config"]["partial_seconds"]) * 25))
        if length < 1:
            raise ValueError("partial_seconds must be positive")
        a = rng.randrange(1, n - length)
        partial_frames, partial_audio, spans = compose_partial(original, fake, (a, a + length))
        sham_audio = original["pcm"].copy()
        if job.get("sham_donor"):
            # Ghép âm thanh thu thật, hình không AI: đối chứng dấu vết splice và lệch A/V.
            genuine = decode(
                root / job["sham_donor"]["video"],
                max_side=plan["config"]["max_side"],
                sample_rate=48000,
            )["pcm"]
            if len(genuine) < n * 1920:
                raise ValueError("Sham donor too short")
            sham_audio[a * 1920 : (a + length) * 1920] = genuine[a * 1920 : (a + length) * 1920]
        generated = []
        for kind, frames, pcm, intervals in (
            ("real", original["frames"], original["pcm"], []),
            ("full", fake["frames"], fake["pcm"], [[0, n / 25]]),
            ("partial", partial_frames, partial_audio, spans),
            ("sham", original["frames"], sham_audio, []),
        ):
            if kind not in dests:
                continue
            is_fake = kind in {"full", "partial"}
            encode(frames, pcm, dests[kind], plan["config"]["crf"])
            if abs(probe(dests[kind])["duration_s"] - n / 25) > 0.1:
                raise ValueError("Encoded duration differs")
            generated.append(
                {
                    "sample_id": f"{key}_{kind}",
                    "video": dests[kind].relative_to(out).as_posix(),
                    "sha256": sha(dests[kind]),
                    "duration_s": n / 25,
                    "label": int(is_fake),
                    "fake_intervals": intervals,
                    "split": r["split"],
                    "source_id": r["source_id"],
                    "source_clip_id": r["clip_id"],
                    "speaker_id": r.get("speaker_id"),
                    **{
                        field: r[field]
                        for field in (
                            "global_speaker_ids",
                            "source_sha256",
                            "duplicate_group_id",
                        )
                        if r.get(field)
                    },
                    "audio_source_id": job["sham_donor"]["source_id"]
                    if kind == "sham"
                    else d["source_id"]
                    if kind != "real"
                    else r["source_id"],
                    "parent_ids": sorted(
                        {
                            r["source_id"],
                            d["source_id"],
                            *(([job["sham_donor"]["source_id"]]) if kind == "sham" else []),
                        }
                    ),
                    "speaker_ids": list(filter(None, {r.get("speaker_id"), d.get("speaker_id")})),
                    "generator": gen_info["generator"] if is_fake else "none",
                    "generator_version": gen_info["version"] if is_fake else "matched-encode-v2",
                    "synthetic_audio": False,
                    "synthetic_visual": is_fake,
                    "audio_fake_intervals": [],
                    "visual_fake_intervals": intervals,
                    "audio_mode": mode if is_fake else None,
                    # Lệch tiếng–miệng đã biết: sham = đoạn ghép; fake chưa đo được nên null.
                    "av_mismatch_intervals": spans
                    if kind == "sham"
                    else []
                    if kind == "real"
                    else None,
                    "audio_edit_kind": "splice"
                    if kind == "sham"
                    else "none"
                    if mode == "source" or kind == "real"
                    else "donor_replace"
                    if kind == "full"
                    else "splice",
                    "processing_profile": "common-pcm48k-h264-aac-crf" + str(plan["config"]["crf"]),
                    "control_type": "conventional_audio_splice" if kind == "sham" else None,
                    "sync_status": "reviewed_match" if kind == "real" else "unknown",
                    "generator_checkpoint_sha256": gen_info.get("checkpoint_sha256")
                    if is_fake
                    else None,
                    "variant": kind,
                    "review_status": "pending",
                    "group_id": key,
                }
            )
        write_json(record, generated)
        rows.extend(generated)
        spent = time.monotonic() - began
        print(
            f"{head}, donor {d['clip_id'] if mode == 'donor' else '(tiếng gốc)'}; "
            f"{'/'.join(dests)}; partial {length / 25:.2f}s tại {a / 25:.2f}s | "
            f"{time.monotonic() - tick:.0f}s, đã chạy {spent / 60:.1f} phút, "
            f"còn ~{spent / (number + 1) * (total - number - 1) / 60:.1f} phút",
            flush=True,
        )
    write_manifest(out / "candidates.jsonl", rows)
    if not (out / "review.csv").exists():
        csv_write(
            out / "review.csv", [{"sample_id": r["sample_id"], "decision": "pending"} for r in rows]
        )
    return {
        "samples": len(rows),
        "next": "Run review then finalize; pending media is not training data",
    }


def finalize(cfg):
    """Chỉ keep được đi vào dataset; chốt xong không sửa nhãn/split tại chỗ."""
    root = Path(cfg["generated_dataset"])
    rows = read_manifest(root / "candidates.jsonl")
    with (root / "review.csv").open(encoding="utf-8-sig", newline="") as f:
        decisions = list(csv.DictReader(f))
    if len({r["sample_id"] for r in decisions}) != len(decisions) or {
        r["sample_id"] for r in decisions
    } != {r["sample_id"] for r in rows}:
        raise ValueError("Review IDs differ from candidates")
    review = {r["sample_id"]: r["decision"] for r in decisions}
    if set(review.values()) - {"keep", "reject", "uncertain", "pending"}:
        raise ValueError("Invalid review decision")
    kept = [{**r, "review_status": "keep"} for r in rows if review[r["sample_id"]] == "keep"]
    validate_rows(kept, root, verify_media=True)
    generation = read_json(root / "generation.json")
    # Part bổ sung có thể chỉ thuộc train; kiểm đủ train/validation ở dataset gộp.
    splits = (
        {r["split"] for r in kept}
        if generation["plan"].get("prior_assignments")
        else {"train", "validation", "test"}
    )
    for split in splits:
        if {r["label"] for r in kept if r["split"] == split} != {0, 1}:
            raise ValueError(f"Review needs real and fake in {split}")
    manifest = root / "manifest.jsonl"
    if manifest.exists():
        if read_manifest(manifest) == kept:
            return read_json(root / "dataset_info.json")
        raise FileExistsError("Finalized dataset immutable; use a new dataset version")
    write_manifest(manifest, kept)
    info = {
        "schema_version": SCHEMA,
        "samples": len(kept),
        "manifest_sha256": sha(manifest),
        "review_sha256": sha(root / "review.csv"),
    }
    write_json(root / "dataset_info.json", info)
    return info
