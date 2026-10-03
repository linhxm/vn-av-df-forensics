"""Import reviewed media from other generators, retaining the clean split lock."""

import shutil
from pathlib import Path

from vn_av_data.contract import validate_bundle

from vn_av_df.common.runtime import sha, write_json
from vn_av_df.data.groups import read_manifest, write_manifest
from vn_av_df.data.media import probe
from vn_av_df.dataset import SCHEMA, load_dataset, validate_rows
from vn_av_df.generation import prior_assignments, split_clean


def import_external(cfg):
    source = Path(cfg["external_manifest"]).resolve()
    out = Path(cfg["import_output"]).resolve()
    if out.exists():
        raise FileExistsError("Use a new import output; existing datasets are immutable")
    clean, _ = validate_bundle(cfg["clean_dataset"])
    clean = {
        r["clip_id"]: r
        for r in split_clean(clean, cfg["seed"], prior_assignments(cfg), cfg.get("split_ratios"))
    }
    # Optional existing binary dataset supplies matched real examples and prior generators.
    rows = []
    files = []
    if cfg.get("import_base"):
        for row in load_dataset(cfg["import_base"], verify_media=True):
            rows.append(dict(row))
            files.append(Path(cfg["import_base"]) / row["video"])
    for index, item in enumerate(read_manifest(source)):
        required = {
            "video",
            "label",
            "fake_intervals",
            "source_clip_id",
            "generator",
            "generator_version",
            "review_status",
        }
        if not required <= set(item) or item["review_status"] != "keep":
            raise ValueError(
                "External record requires reviewed binary label, explicit intervals/null and generator provenance"
            )
        lineage = [item["source_clip_id"]] + [
            item[k] for k in ("audio_source_clip_id", "identity_source_clip_id") if item.get(k)
        ]
        if any(k not in clean for k in lineage):
            raise ValueError("Unknown parent; add it to the clean bundle before locking splits")
        parents = [clean[k] for k in lineage]
        if len({p["split"] for p in parents}) != 1:
            raise ValueError("External generator donor crosses splits")
        path = (source.parent / item["video"]).resolve()
        info = probe(path)
        digest = sha(path)
        row = {
            **{
                k: item[k]
                for k in (
                    "label",
                    "fake_intervals",
                    "generator",
                    "generator_version",
                    "review_status",
                    "synthetic_audio",
                    "synthetic_visual",
                    "audio_fake_intervals",
                    "visual_fake_intervals",
                    "audio_edit_kind",
                    "control_type",
                    "processing_profile",
                )
                if k in item
            },
            "sample_id": "external_" + digest[:24],
            "video": f"clips/external_{digest[:24]}.mp4",
            "sha256": digest,
            "duration_s": info["duration_s"],
            "source_id": parents[0]["source_id"],
            "source_clip_id": parents[0]["clip_id"],
            "speaker_id": parents[0].get("speaker_id"),
            "speaker_ids": sorted({p["speaker_id"] for p in parents if p.get("speaker_id")}),
            "parent_ids": sorted({p["source_id"] for p in parents}),
            "split": parents[0]["split"],
        }
        rows.append(row)
        files.append(path)
    validate_rows(rows)
    # Validate completely before creating the output directory.
    for row, path in zip(rows, files):
        dest = out / row["video"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    validate_rows(rows, out, verify_media=True)
    write_manifest(out / "manifest.jsonl", rows)
    receipt = {
        "schema_version": SCHEMA,
        "samples": len(rows),
        "manifest_sha256": sha(out / "manifest.jsonl"),
        "external_manifest_sha256": sha(source),
    }
    write_json(out / "dataset_info.json", receipt)
    return receipt
