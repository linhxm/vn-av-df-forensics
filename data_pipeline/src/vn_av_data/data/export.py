"""Export portable reviewed clips of one part, without training splits or controls.

Chạy lại sau khi duyệt thêm: dựng lại toàn bộ part, giữ nguyên nếu danh sách clip không đổi.
"""

import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path, PureWindowsPath

from vn_av_data.common.runtime import sha, write_json
from vn_av_data.contract import SCHEMA, valid_id, validate_annotations, validate_bundle
from vn_av_data.data.manifest import read_manifest, write_manifest
from vn_av_data.data.media import probe


def review_path(root, value):
    root = Path(root).resolve()
    portable = value.replace("\\", "/")
    candidate = Path(portable)
    if candidate.is_absolute() or PureWindowsPath(portable).is_absolute():
        candidate = (
            candidate
            if candidate.is_absolute() and candidate.is_relative_to(root)
            else root / portable.rsplit("/", 1)[-1]
        )
    else:
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError(f"Review media missing/outside root: {value}")
    return candidate


def export_dataset(review, root, output, dataset_id, annotations=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not valid_id(dataset_id):
        raise ValueError("dataset_id must contain letters, digits, underscore or hyphen")
    annotation_map = {}
    if annotations:
        for row in read_manifest(annotations):
            if row["clip_id"] in annotation_map:
                raise ValueError("Duplicate annotation clip_id")
            annotation_map[row["clip_id"]] = row["relation_annotations"]
    with Path(review).open(encoding="utf-8-sig", newline="") as stream:
        entries = list(csv.DictReader(stream))
    rows, media, ids, digests, duplicates = [], [], set(), set(), []
    for entry in entries:
        if entry.get("decision", "").strip().lower() != "keep":
            continue
        sid = entry["clip_id"]
        if not valid_id(sid) or sid in ids:
            raise ValueError(f"Invalid/duplicate clip_id: {sid}")
        ids.add(sid)
        if entry.get("sync_status", "reviewed_match") != "reviewed_match":
            raise ValueError(f"{sid}: confirm synchrony in the review UI before export")
        source = entry.get("source_id", "").strip()
        if not source:
            raise ValueError(f"{sid}: source_id required for leakage-safe training")
        video = review_path(root, entry["file_path"])
        digest = sha(video)
        if entry.get("sha256") and digest != entry["sha256"]:
            raise ValueError(f"{sid}: reviewed video has changed")
        if digest in digests:
            if sid in annotation_map or entry.get("relation_annotations"):
                raise ValueError(
                    f"{sid}: duplicate with annotations; reconcile labels before export"
                )
            duplicates.append(sid)
            continue
        digests.add(digest)
        duration = probe(video)["duration_s"]
        start = float(entry.get("source_start_s") or 0)
        end = float(entry.get("source_end_s") or start + duration)
        labels = annotation_map.get(sid, json.loads(entry.get("relation_annotations") or "{}"))
        validate_annotations(labels, duration)
        row = {
            "clip_id": sid,
            "video": f"clips/{sid}{video.suffix.lower()}",
            "sha256": digest,
            "source_id": source,
            "speaker_id": entry.get("speaker_id") or None,
            "duration_s": duration,
            "source_start_s": start,
            "source_end_s": end,
            "review_decision": "keep",
            "sync_status": "reviewed_match",
            "relation_annotations": labels,
        }
        for key in ("url", "source_sha256", "global_speaker_ids"):
            if entry.get(key):
                row[key] = entry[key]
        # Thông số nguồn cho nghiên cứu fps/độ phân giải/độ nén (R1–R3).
        for key, kind in (
            ("source_fps", float),
            ("source_width", int),
            ("source_height", int),
            ("source_codec", str),
            ("source_bitrate_kbps", float),
        ):
            if entry.get(key):
                row[key] = kind(entry[key])
        rows.append(row)
        media.append(video)
    if set(annotation_map) - {r["clip_id"] for r in rows}:
        raise ValueError("Annotations reference clips not included in the reviewed export")
    if not rows:
        raise ValueError("No reviewed keep clips to export")
    # Dựng ở thư mục tạm để bản export dở dang không bao giờ trông như bộ dữ liệu hoàn chỉnh.
    stage = output.with_name(output.name + ".partial")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    (stage / "clips").mkdir()
    for row, video in zip(rows, media):
        shutil.copy2(video, stage / row["video"])
    write_manifest(stage / "manifest.jsonl", rows)
    info = {
        "schema_version": SCHEMA,
        "dataset_id": dataset_id,
        "clips": len(rows),
        "manifest_sha256": sha(stage / "manifest.jsonl"),
        "review_sha256": sha(review),
        "annotations_sha256": sha(annotations) if annotations else None,
        "exporter_sha256": sha(__file__),
    }
    write_json(stage / "dataset_info.json", info)
    _, report = validate_bundle(stage, probe=probe)
    report.update(
        decisions=dict(Counter(e.get("decision") for e in entries)),
        duplicate_clips_skipped=duplicates,
    )
    write_json(stage / "quality_report.json", report)
    if output.exists():
        if sha(output / "manifest.jsonl") == info["manifest_sha256"]:
            shutil.rmtree(stage)
            return {**report, "output": str(output), "unchanged": True}
        # Danh sách clip đổi: thay cả bản cũ; ZIP và generation của part phải làm lại.
        old = output.with_name(output.name + ".old")
        if old.exists():
            shutil.rmtree(old)
        os.rename(output, old)
        os.rename(stage, output)
        shutil.rmtree(old)
        return {**report, "output": str(output), "replaced": True}
    os.rename(stage, output)
    return {**report, "output": str(output)}
