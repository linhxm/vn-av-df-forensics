"""Manifest import, connected provenance groups, split and label validation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from vn_av_df.common.runtime import atomic_bytes

SPLITS = {"train", "validation", "test"}


def identity_list(value):
    """CSV uses semicolon-separated IDs; JSONL may use a list."""
    if value is None:
        return []
    values = value.split(";") if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)):
        raise ValueError("Identity fields must be a list or semicolon-separated string")
    return sorted(
        {
            str(x).strip()
            for x in values
            if str(x).strip().lower() not in ("", "unknown", "nan", "none", "-1")
        }
    )


def identity_nodes(row):
    """Local IDs are dataset-scoped; hashes and global speaker IDs join datasets."""
    namespace = str(row.get("dataset", ""))
    nodes = [(namespace, "source", str(row["source_id"]))]
    # Hash trùng và cụm repost do review gán phải theo cùng split dù khác URL.
    if row.get("sha256"):
        nodes.append(("global", "media_sha256", row["sha256"]))
    if row.get("source_sha256"):
        nodes.append(("global", "source_sha256", row["source_sha256"]))
    if row.get("duplicate_group_id"):
        nodes.append(("global", "duplicate", str(row["duplicate_group_id"])))
    nodes.extend((namespace, "source", x) for x in identity_list(row.get("parent_ids")))
    speakers = identity_list(row.get("speaker_id")) + identity_list(row.get("speaker_ids"))
    nodes.extend((namespace, "speaker", x) for x in speakers)
    nodes.extend(("global", "speaker", x) for x in identity_list(row.get("global_speaker_ids")))
    if row.get("group_id"):
        nodes.append((namespace, "locked_group", str(row["group_id"])))
    return nodes


def connected_groups(rows):
    """Resolve transitive source, donor, speaker and repost connections."""
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while node != root:
            parent[node], node = root, parent[node]
        return root

    for row in rows:
        nodes = identity_nodes(row)
        for node in nodes[1:]:
            parent[find(node)] = find(nodes[0])
    members = defaultdict(list)
    for row in rows:
        members[find(identity_nodes(row)[0])].append(row)
    return [
        sorted(group, key=lambda r: r["sample_id"])
        for group in sorted(members.values(), key=lambda group: min(r["sample_id"] for r in group))
    ]


def read_manifest(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_manifest(path, rows):
    atomic_bytes(
        path,
        (
            "\n".join(json.dumps(x, ensure_ascii=False, allow_nan=False) for x in rows) + "\n"
        ).encode(),
    )
