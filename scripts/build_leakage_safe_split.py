#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


LABELS = ("0", "1", "2")
CLASS_NAMES = {"0": "Recyclable", "1": "Electronic", "2": "Organic"}


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        self.parent[max(left_root, right_root)] = min(left_root, right_root)


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_pair(left, right):
    return tuple(sorted((left, right)))


def require_file(path):
    if not path.exists():
        raise SystemExit(f"missing required input: {path}")


def val_targets(class_counts, fraction):
    return {label: round(class_counts[label] * fraction) for label in LABELS}


def add_edge(union_find, manifest_paths, edges, left, right, source):
    if left in manifest_paths and right in manifest_paths:
        union_find.union(left, right)
        edges.append({"path_a": left, "path_b": right, "source": source})
        return True
    return False


def load_group_edges(args, rows, union_find):
    manifest_paths = {row["path"] for row in rows}
    edges = []
    counts = Counter()

    exact_rows = read_csv(args.exact_duplicates)
    for row in exact_rows:
        paths = [path for path in row["paths"].split("|") if path]
        retained = [path for path in paths if path in manifest_paths]
        counts["exact_duplicate_groups_total"] += int(len(paths) > 1)
        counts["exact_duplicate_groups_retained_multi_path"] += int(len(retained) > 1)
        for path in retained[1:]:
            add_edge(union_find, manifest_paths, edges, retained[0], path, "exact_duplicate")

    explicit_pairs = set()
    group_pair_rows = read_csv(args.group_pairs)
    for row in group_pair_rows:
        pair = normalized_pair(row["path_a"], row["path_b"])
        explicit_pairs.add(pair)
        counts["dedup_group_pairs_total"] += 1
        if add_edge(
            union_find,
            manifest_paths,
            edges,
            row["path_a"],
            row["path_b"],
            "dedup_group_pair",
        ):
            counts["dedup_group_pairs_retained"] += 1

    unsafe_pairs = []
    near_rows = read_csv(args.near_pairs)
    for row in near_rows:
        left = row["path_a"]
        right = row["path_b"]
        if left not in manifest_paths or right not in manifest_paths:
            continue
        counts["retained_near_duplicate_pairs"] += 1
        if normalized_pair(left, right) not in explicit_pairs:
            unsafe_pairs.append((left, right))

    if unsafe_pairs:
        examples = "\n".join(f"- {left} | {right}" for left, right in unsafe_pairs[:20])
        raise SystemExit(
            "retained near-duplicate pairs are not recorded in "
            f"{args.group_pairs}:\n{examples}"
        )

    return edges, counts


def build_groups(rows, union_find, edges):
    members_by_root = defaultdict(list)
    for row in rows:
        members_by_root[union_find.find(row["path"])].append(row)

    sources_by_root = defaultdict(set)
    for edge in edges:
        root = union_find.find(edge["path_a"])
        sources_by_root[root].add(edge["source"])

    groups = []
    for index, root in enumerate(sorted(members_by_root), start=1):
        members = members_by_root[root]
        labels = Counter(row["current_label"] for row in members)
        paths = sorted(row["path"] for row in members)
        sources = sorted(sources_by_root[root]) or ["singleton"]
        groups.append({
            "group_id": f"G{index:05d}",
            "root": root,
            "paths": paths,
            "members": members,
            "label_counts": labels,
            "size": len(members),
            "sources": sources,
        })
    return groups


def choose_single_label_groups(groups, target, rng):
    shuffled = groups[:]
    rng.shuffle(shuffled)
    selected = set()
    selected_count = 0

    remaining = []
    for group in shuffled:
        if selected_count + group["size"] <= target:
            selected.add(group["group_id"])
            selected_count += group["size"]
        else:
            remaining.append(group)

    if remaining:
        current_gap = abs(target - selected_count)
        best = min(
            remaining,
            key=lambda group: (abs(target - (selected_count + group["size"])), group["group_id"]),
        )
        best_gap = abs(target - (selected_count + best["size"]))
        if best_gap < current_gap:
            selected.add(best["group_id"])
            selected_count += best["size"]

    return selected, selected_count


def assign_splits(groups, targets, seed, val_fraction):
    rng = random.Random(seed)
    val_group_ids = set()
    val_counts = Counter()

    mixed_groups = [
        group for group in groups
        if sum(count > 0 for count in group["label_counts"].values()) > 1
    ]
    rng.shuffle(mixed_groups)
    for group in mixed_groups:
        if rng.random() < val_fraction:
            val_group_ids.add(group["group_id"])
            val_counts.update(group["label_counts"])

    for label in LABELS:
        target = max(0, targets[label] - val_counts[label])
        candidates = [
            group for group in groups
            if group["group_id"] not in val_group_ids
            and len(group["label_counts"]) == 1
            and group["label_counts"][label] == group["size"]
        ]
        selected, _ = choose_single_label_groups(candidates, target, rng)
        val_group_ids.update(selected)
        for group in candidates:
            if group["group_id"] in selected:
                val_counts.update(group["label_counts"])

    return {
        group["group_id"]: "val" if group["group_id"] in val_group_ids else "train"
        for group in groups
    }


def count_labels(rows):
    return dict(sorted(Counter(row["current_label"] for row in rows).items()))


def validate_rows(rows):
    required = ["image_id", "path", "current_label", "class_name", "review_status", "reason"]
    missing_columns = [column for column in required if column not in rows[0]]
    if missing_columns:
        raise SystemExit(f"manifest missing columns: {', '.join(missing_columns)}")

    path_counts = Counter(row["path"] for row in rows)
    duplicate_paths = [path for path, count in path_counts.items() if count > 1]
    if duplicate_paths:
        raise SystemExit(f"duplicate manifest paths: {len(duplicate_paths)}")

    bad_labels = sorted({row["current_label"] for row in rows} - set(LABELS))
    if bad_labels:
        raise SystemExit(f"invalid labels in manifest: {bad_labels}")

    missing_files = [row["path"] for row in rows if not Path(row["path"]).is_file()]
    if missing_files:
        examples = "\n".join(f"- {path}" for path in missing_files[:20])
        raise SystemExit(f"missing image files: {len(missing_files)}\n{examples}")

    return required


def build_summary(args, rows, train_rows, val_rows, groups, split_by_group, edge_counts):
    class_counts = Counter(row["current_label"] for row in rows)
    target_counts = val_targets(class_counts, args.val_fraction)
    val_counts = Counter(row["current_label"] for row in val_rows)
    train_counts = Counter(row["current_label"] for row in train_rows)
    group_split_counts = Counter(split_by_group.values())

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": args.manifest.as_posix(),
        "source_manifest_sha256": sha256_file(args.manifest),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "row_counts": {
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
        },
        "class_counts": {
            "total": dict(sorted(class_counts.items())),
            "target_val": dict(sorted(target_counts.items())),
            "actual_val": dict(sorted(val_counts.items())),
            "actual_train": dict(sorted(train_counts.items())),
        },
        "class_names": CLASS_NAMES,
        "group_counts": {
            "total": len(groups),
            "train": group_split_counts["train"],
            "val": group_split_counts["val"],
            "non_singleton": sum(group["size"] > 1 for group in groups),
            "mixed_label": sum(len(group["label_counts"]) > 1 for group in groups),
        },
        "constraint_counts": dict(sorted(edge_counts.items())),
        "outputs": {
            "train_manifest": (args.out_dir / "train_manifest.csv").as_posix(),
            "val_manifest": (args.out_dir / "val_manifest.csv").as_posix(),
            "split_groups": (args.out_dir / "split_groups.csv").as_posix(),
            "split_summary": (args.out_dir / "split_summary.json").as_posix(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build a fixed leakage-safe train/validation split.")
    parser.add_argument("--manifest", type=Path, default=Path("manifest_clean.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--exact-duplicates", type=Path, default=Path("outputs/audit_kept/exact_duplicates.csv"))
    parser.add_argument("--group-pairs", type=Path, default=Path("outputs/duplicate_review/dedup_group_pairs.csv"))
    parser.add_argument("--near-pairs", type=Path, default=Path("outputs/audit_kept/near_duplicate_pairs.csv"))
    args = parser.parse_args()

    for path in (args.manifest, args.exact_duplicates, args.group_pairs, args.near_pairs):
        require_file(path)

    rows = read_csv(args.manifest)
    if not rows:
        raise SystemExit("manifest is empty")
    manifest_fields = validate_rows(rows)

    union_find = UnionFind(row["path"] for row in rows)
    edges, edge_counts = load_group_edges(args, rows, union_find)
    groups = build_groups(rows, union_find, edges)

    class_counts = Counter(row["current_label"] for row in rows)
    targets = val_targets(class_counts, args.val_fraction)
    split_by_group = assign_splits(groups, targets, args.seed, args.val_fraction)

    split_by_path = {}
    group_by_path = {}
    for group in groups:
        split = split_by_group[group["group_id"]]
        for row in group["members"]:
            split_by_path[row["path"]] = split
            group_by_path[row["path"]] = group

    train_rows = [row for row in rows if split_by_path[row["path"]] == "train"]
    val_rows = [row for row in rows if split_by_path[row["path"]] == "val"]

    group_rows = []
    for row in rows:
        group = group_by_path[row["path"]]
        group_rows.append({
            "group_id": group["group_id"],
            "split": split_by_path[row["path"]],
            "group_size": group["size"],
            "group_sources": "|".join(group["sources"]),
            "path": row["path"],
            "current_label": row["current_label"],
            "class_name": row["class_name"],
        })

    write_csv(args.out_dir / "train_manifest.csv", train_rows, manifest_fields)
    write_csv(args.out_dir / "val_manifest.csv", val_rows, manifest_fields)
    write_csv(
        args.out_dir / "split_groups.csv",
        group_rows,
        ["group_id", "split", "group_size", "group_sources", "path", "current_label", "class_name"],
    )
    summary = build_summary(args, rows, train_rows, val_rows, groups, split_by_group, edge_counts)
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"train rows: {len(train_rows)} {count_labels(train_rows)}")
    print(f"val rows: {len(val_rows)} {count_labels(val_rows)}")
    print(f"groups: {len(groups)}")
    print(f"wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
