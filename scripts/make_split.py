#!/usr/bin/env python3
"""Build a fixed group-aware train/validation split for BDC 2026."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold


LABELS = {
    0: "Recyclable",
    1: "Electronic",
    2: "Organic",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitRecord:
    path: Path
    rel_path: str
    label: int
    class_name: str
    width: int | None = None
    height: int | None = None
    file_sha256: str = ""
    pixel_sha256: str = ""
    ahash: int | None = None
    dhash: int | None = None
    group_id: int = -1
    valid: bool = False
    error: str = ""


class UnionFind:
    def __init__(self, items: Iterable[int]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--train-dir", type=Path, default=Path("train"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--near-threshold", type=int, default=6)
    parser.add_argument("--max-bucket-size", type=int, default=700)
    return parser.parse_args()


def class_dirs(train_dir: Path) -> list[tuple[int, str, Path]]:
    dirs: list[tuple[int, str, Path]] = []
    for path in sorted(train_dir.iterdir()):
        if not path.is_dir():
            continue
        prefix = path.name.split("_", 1)[0]
        if not prefix.isdigit():
            continue
        label = int(prefix)
        class_name = path.name.split("_", 1)[1] if "_" in path.name else LABELS.get(label, path.name)
        dirs.append((label, class_name, path))
    found = {label for label, _, _ in dirs}
    if found != set(LABELS):
        raise SystemExit(f"Expected class labels {sorted(LABELS)}, found {sorted(found)} in {train_dir}")
    return dirs


def iter_records(root: Path, train_dir: Path) -> list[SplitRecord]:
    records: list[SplitRecord] = []
    for label, class_name, directory in class_dirs(train_dir):
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    SplitRecord(
                        path=path,
                        rel_path=path.relative_to(root).as_posix(),
                        label=label,
                        class_name=class_name,
                    )
                )
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.astype(bool):
        value = (value << 1) | int(bit)
    return value


def image_hashes(rgb: Image.Image) -> tuple[str, int, int]:
    arr = np.asarray(rgb, dtype=np.uint8)
    pixel_sha = hashlib.sha256(arr.tobytes()).hexdigest()
    gray_8 = ImageOps.grayscale(rgb).resize((8, 8), Image.Resampling.LANCZOS)
    gray_arr = np.asarray(gray_8, dtype=np.float32)
    ahash = bits_to_int((gray_arr >= float(gray_arr.mean())).reshape(-1))

    gray_d = ImageOps.grayscale(rgb).resize((9, 8), Image.Resampling.LANCZOS)
    d_arr = np.asarray(gray_d, dtype=np.int16)
    dhash = bits_to_int((d_arr[:, 1:] >= d_arr[:, :-1]).reshape(-1))
    return pixel_sha, ahash, dhash


def hamming(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def audit_record(record: SplitRecord) -> SplitRecord:
    try:
        record.file_sha256 = sha256_file(record.path)
        with Image.open(record.path) as image:
            image.load()
            record.width, record.height = image.size
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            record.pixel_sha256, record.ahash, record.dhash = image_hashes(rgb)
        record.valid = True
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        record.error = f"{type(exc).__name__}: {exc}"
    return record


def hash_buckets(records: list[SplitRecord], attr: str, band_bits: int = 16) -> dict[tuple[str, int, int], list[int]]:
    buckets: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    mask = (1 << band_bits) - 1
    for idx, record in enumerate(records):
        value = getattr(record, attr)
        if value is None:
            continue
        for band in range(64 // band_bits):
            chunk = (value >> (band * band_bits)) & mask
            buckets[(attr, band, chunk)].append(idx)
    return buckets


def build_similarity_groups(
    records: list[SplitRecord],
    threshold: int,
    max_bucket_size: int,
) -> tuple[int, int, int]:
    uf = UnionFind(range(len(records)))

    pixel_groups: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        if record.pixel_sha256:
            pixel_groups[record.pixel_sha256].append(idx)
    exact_group_count = 0
    for members in pixel_groups.values():
        if len(members) < 2:
            continue
        exact_group_count += 1
        first = members[0]
        for member in members[1:]:
            uf.union(first, member)

    candidates: set[tuple[int, int]] = set()
    for attr in ("ahash", "dhash"):
        for members in hash_buckets(records, attr).values():
            if len(members) < 2 or len(members) > max_bucket_size:
                continue
            for pos, left in enumerate(members):
                for right in members[pos + 1 :]:
                    candidates.add((left, right) if left < right else (right, left))

    near_pair_count = 0
    for left, right in sorted(candidates):
        if not records[left].valid or not records[right].valid:
            continue
        ah = hamming(records[left].ahash or 0, records[right].ahash or 0)
        dh = hamming(records[left].dhash or 0, records[right].dhash or 0)
        if min(ah, dh) <= threshold:
            uf.union(left, right)
            near_pair_count += 1

    roots = {}
    next_group_id = 0
    for idx, record in enumerate(records):
        root = uf.find(idx)
        if root not in roots:
            roots[root] = next_group_id
            next_group_id += 1
        record.group_id = roots[root]
    return exact_group_count, near_pair_count, next_group_id


def choose_validation_fold(
    records: list[SplitRecord],
    val_fraction: float,
    seed: int,
) -> tuple[set[int], dict[str, object]]:
    if not 0.05 <= val_fraction <= 0.5:
        raise SystemExit("--val-fraction must be between 0.05 and 0.5")

    labels = np.asarray([record.label for record in records])
    groups = np.asarray([record.group_id for record in records])
    n_splits = max(2, round(1.0 / val_fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    total_counts = Counter(labels.tolist())
    total_n = len(records)
    target_counts = {label: total_counts[label] * val_fraction for label in LABELS}

    best_score = float("inf")
    best_val_indices: set[int] = set()
    best_info: dict[str, object] = {}
    dummy_x = np.zeros((len(records), 1))
    for fold_id, (_, val_idx) in enumerate(splitter.split(dummy_x, labels, groups)):
        val_labels = labels[val_idx]
        val_counts = Counter(val_labels.tolist())
        size_error = abs((len(val_idx) / total_n) - val_fraction)
        class_error = sum(abs(val_counts[label] - target_counts[label]) / max(1.0, target_counts[label]) for label in LABELS)
        score = size_error * 10.0 + class_error
        if score < best_score:
            best_score = score
            best_val_indices = set(int(idx) for idx in val_idx)
            best_info = {
                "fold_id": fold_id,
                "score": score,
                "n_splits": n_splits,
                "val_fraction_actual": len(val_idx) / total_n,
                "val_counts": {str(label): int(val_counts[label]) for label in sorted(LABELS)},
            }
    return best_val_indices, best_info


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_rows(records: list[SplitRecord], val_indices: set[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, record in enumerate(records):
        split = "val" if idx in val_indices else "train"
        rows.append(
            {
                "path": record.rel_path,
                "label": record.label,
                "class_name": record.class_name,
                "split": split,
                "group_id": record.group_id,
                "width": record.width,
                "height": record.height,
                "valid": record.valid,
                "error": record.error,
            }
        )
    return rows


def leakage_check(records: list[SplitRecord], val_indices: set[int]) -> None:
    group_sides: dict[int, set[str]] = defaultdict(set)
    for idx, record in enumerate(records):
        group_sides[record.group_id].add("val" if idx in val_indices else "train")
    leaking = {group: sides for group, sides in group_sides.items() if len(sides) > 1}
    if leaking:
        sample = sorted(leaking)[:10]
        raise SystemExit(f"Group leakage detected for groups: {sample}")


def write_report(
    output_dir: Path,
    records: list[SplitRecord],
    val_indices: set[int],
    exact_group_count: int,
    near_pair_count: int,
    group_count: int,
    fold_info: dict[str, object],
    seed: int,
    val_fraction: float,
) -> None:
    train_counts = Counter(record.label for idx, record in enumerate(records) if idx not in val_indices)
    val_counts = Counter(record.label for idx, record in enumerate(records) if idx in val_indices)
    lines = [
        "# Fixed Split Report",
        "",
        f"- Seed: `{seed}`",
        f"- Requested validation fraction: `{val_fraction}`",
        f"- Actual validation fraction: `{fold_info['val_fraction_actual']:.6f}`",
        f"- Selected fold: `{fold_info['fold_id']}` of `{fold_info['n_splits']}`",
        f"- Total images: `{len(records)}`",
        f"- Similarity groups: `{group_count}`",
        f"- Exact pixel duplicate groups merged: `{exact_group_count}`",
        f"- Near-duplicate pairs merged: `{near_pair_count}`",
        "",
        "## Class Distribution",
        "",
        "| label | class | train | val |",
        "| --- | --- | ---: | ---: |",
    ]
    for label in sorted(LABELS):
        lines.append(f"| {label} | {LABELS[label]} | {train_counts[label]} | {val_counts[label]} |")
    lines.extend(
        [
            "",
            "## Leakage Guard",
            "",
            "All images in the same exact/near-duplicate group are assigned to the same split side.",
        ]
    )
    (output_dir / "split_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    train_dir = (root / args.train_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    if not train_dir.exists():
        raise SystemExit(f"Train directory not found: {train_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    records = iter_records(root, train_dir)
    for idx, record in enumerate(records, start=1):
        audit_record(record)
        if idx % 1000 == 0:
            print(f"Hashed {idx}/{len(records)} images")

    invalid = [record for record in records if not record.valid]
    if invalid:
        raise SystemExit(f"Refusing to split with {len(invalid)} invalid images. Run audit first.")

    exact_group_count, near_pair_count, group_count = build_similarity_groups(
        records,
        threshold=args.near_threshold,
        max_bucket_size=args.max_bucket_size,
    )
    val_indices, fold_info = choose_validation_fold(records, args.val_fraction, args.seed)
    leakage_check(records, val_indices)

    rows = split_rows(records, val_indices)
    fieldnames = ["path", "label", "class_name", "split", "group_id", "width", "height", "valid", "error"]
    write_csv(output_dir / "split_manifest.csv", rows, fieldnames)
    write_csv(output_dir / "train.csv", [row for row in rows if row["split"] == "train"], fieldnames)
    write_csv(output_dir / "val.csv", [row for row in rows if row["split"] == "val"], fieldnames)
    write_report(
        output_dir,
        records,
        val_indices,
        exact_group_count,
        near_pair_count,
        group_count,
        fold_info,
        args.seed,
        args.val_fraction,
    )

    summary = {
        "seed": args.seed,
        "val_fraction_requested": args.val_fraction,
        "val_fraction_actual": fold_info["val_fraction_actual"],
        "selected_fold": fold_info["fold_id"],
        "total_images": len(records),
        "train_images": len(records) - len(val_indices),
        "val_images": len(val_indices),
        "similarity_groups": group_count,
        "exact_pixel_duplicate_groups_merged": exact_group_count,
        "near_duplicate_pairs_merged": near_pair_count,
        "val_counts": fold_info["val_counts"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
