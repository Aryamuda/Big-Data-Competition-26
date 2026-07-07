#!/usr/bin/env python3
"""Train-only audit for the BDC 2026 image classification dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


LABELS = {
    0: "Recyclable",
    1: "Electronic",
    2: "Organic",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageRecord:
    path: Path
    rel_path: str
    label: int
    class_name: str
    filename: str
    width: int | None = None
    height: int | None = None
    mode: str = ""
    aspect_ratio: float | None = None
    pixels: int | None = None
    file_sha256: str = ""
    pixel_sha256: str = ""
    ahash: int | None = None
    dhash: int | None = None
    border_brightness: float | None = None
    border_rgb: tuple[float, float, float] | None = None
    foreground_ratio: float | None = None
    bbox_area_ratio: float | None = None
    color_hist: np.ndarray | None = None
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/audit"))
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--samples-per-class", type=int, default=36)
    parser.add_argument("--near-threshold", type=int, default=6)
    parser.add_argument("--max-bucket-size", type=int, default=700)
    parser.add_argument("--outliers-per-class", type=int, default=36)
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
    expected = set(LABELS)
    found = {label for label, _, _ in dirs}
    if found != expected:
        raise SystemExit(f"Expected class labels {sorted(expected)}, found {sorted(found)} in {train_dir}")
    return dirs


def iter_train_images(root: Path, train_dir: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for label, class_name, directory in class_dirs(train_dir):
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    ImageRecord(
                        path=path,
                        rel_path=path.relative_to(root).as_posix(),
                        label=label,
                        class_name=class_name,
                        filename=path.name,
                    )
                )
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_hashes(rgb: Image.Image) -> tuple[str, int, int]:
    arr = np.asarray(rgb, dtype=np.uint8)
    pixel_sha = hashlib.sha256(arr.tobytes()).hexdigest()

    gray_8 = ImageOps.grayscale(rgb).resize((8, 8), Image.Resampling.LANCZOS)
    gray_arr = np.asarray(gray_8, dtype=np.float32)
    avg = float(gray_arr.mean())
    ahash_bits = gray_arr >= avg
    ahash = bits_to_int(ahash_bits.reshape(-1))

    gray_d = ImageOps.grayscale(rgb).resize((9, 8), Image.Resampling.LANCZOS)
    d_arr = np.asarray(gray_d, dtype=np.int16)
    dhash_bits = d_arr[:, 1:] >= d_arr[:, :-1]
    dhash = bits_to_int(dhash_bits.reshape(-1))
    return pixel_sha, ahash, dhash


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.astype(bool):
        value = (value << 1) | int(bit)
    return value


def hamming(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def color_histogram(rgb: Image.Image, bins: int = 4) -> np.ndarray:
    small = rgb.resize((96, 96), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.uint8)
    quantized = np.clip(arr // (256 // bins), 0, bins - 1)
    flat = quantized[:, :, 0] * (bins * bins) + quantized[:, :, 1] * bins + quantized[:, :, 2]
    hist = np.bincount(flat.reshape(-1), minlength=bins**3).astype(np.float32)
    total = float(hist.sum())
    if total:
        hist /= total
    return hist


def border_and_object_proxies(rgb: Image.Image) -> tuple[float, tuple[float, float, float], float, float]:
    small = rgb.resize((128, 128), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32)
    h, w, _ = arr.shape
    border = max(4, int(min(h, w) * 0.1))
    mask = np.zeros((h, w), dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True

    border_pixels = arr[mask]
    border_rgb = tuple(float(x) for x in border_pixels.mean(axis=0))
    border_brightness = float((0.299 * border_pixels[:, 0] + 0.587 * border_pixels[:, 1] + 0.114 * border_pixels[:, 2]).mean())

    bg_median = np.median(border_pixels, axis=0)
    bg_std = float(np.mean(np.std(border_pixels, axis=0)))
    distance = np.linalg.norm(arr - bg_median.reshape(1, 1, 3), axis=2)
    threshold = max(25.0, 1.5 * bg_std)
    fg = distance > threshold
    foreground_ratio = float(fg.mean())

    if fg.any():
        ys, xs = np.where(fg)
        bbox_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        bbox_area_ratio = bbox_area / float(h * w)
    else:
        bbox_area_ratio = 0.0
    return border_brightness, border_rgb, foreground_ratio, bbox_area_ratio


def audit_image(record: ImageRecord) -> ImageRecord:
    try:
        record.file_sha256 = sha256_file(record.path)
        with Image.open(record.path) as image:
            image.load()
            record.width, record.height = image.size
            record.mode = image.mode
            record.aspect_ratio = record.width / record.height if record.height else None
            record.pixels = record.width * record.height
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            record.pixel_sha256, record.ahash, record.dhash = image_hashes(rgb)
            (
                record.border_brightness,
                record.border_rgb,
                record.foreground_ratio,
                record.bbox_area_ratio,
            ) = border_and_object_proxies(rgb)
            record.color_hist = color_histogram(rgb)
        record.valid = True
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        record.error = f"{type(exc).__name__}: {exc}"
    return record


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ["min", "p05", "p25", "median", "mean", "p75", "p95", "max"]}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "mean": float(np.mean(arr)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def hash_buckets(records: list[ImageRecord], attr: str, band_bits: int = 16) -> dict[tuple[str, int, int], list[int]]:
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


def find_near_duplicate_pairs(
    records: list[ImageRecord],
    threshold: int,
    max_bucket_size: int,
) -> list[dict[str, object]]:
    candidates: set[tuple[int, int]] = set()
    for attr in ("ahash", "dhash"):
        for members in hash_buckets(records, attr).values():
            if len(members) < 2 or len(members) > max_bucket_size:
                continue
            for pos, left in enumerate(members):
                for right in members[pos + 1 :]:
                    candidates.add((left, right) if left < right else (right, left))

    pairs: list[dict[str, object]] = []
    for left, right in sorted(candidates):
        left_record = records[left]
        right_record = records[right]
        if not left_record.valid or not right_record.valid:
            continue
        ah = hamming(left_record.ahash or 0, right_record.ahash or 0)
        dh = hamming(left_record.dhash or 0, right_record.dhash or 0)
        if min(ah, dh) <= threshold:
            pairs.append(
                {
                    "left_path": left_record.rel_path,
                    "right_path": right_record.rel_path,
                    "left_label": left_record.label,
                    "right_label": right_record.label,
                    "left_class": left_record.class_name,
                    "right_class": right_record.class_name,
                    "ahash_distance": ah,
                    "dhash_distance": dh,
                    "cross_label": left_record.label != right_record.label,
                }
            )
    return pairs


def duplicate_groups(
    records: list[ImageRecord],
    key: str,
    min_size: int = 2,
) -> list[dict[str, object]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        value = getattr(record, key)
        if value:
            grouped[value].append(record)
    rows: list[dict[str, object]] = []
    group_id = 0
    for value, members in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(members) < min_size:
            continue
        group_id += 1
        labels = sorted({member.label for member in members})
        rows.append(
            {
                "group_id": group_id,
                "hash": value,
                "size": len(members),
                "labels": " ".join(str(label) for label in labels),
                "cross_label": len(labels) > 1,
                "paths": " ".join(member.rel_path for member in members),
            }
        )
    return rows


def near_duplicate_groups(records: list[ImageRecord], pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    path_to_index = {record.rel_path: idx for idx, record in enumerate(records)}
    uf = UnionFind(range(len(records)))
    for pair in pairs:
        uf.union(path_to_index[str(pair["left_path"])], path_to_index[str(pair["right_path"])])

    grouped: dict[int, list[ImageRecord]] = defaultdict(list)
    for idx, record in enumerate(records):
        grouped[uf.find(idx)].append(record)

    rows: list[dict[str, object]] = []
    group_id = 0
    for members in sorted(grouped.values(), key=lambda value: (-len(value), value[0].rel_path)):
        if len(members) < 2:
            continue
        group_id += 1
        labels = sorted({member.label for member in members})
        rows.append(
            {
                "group_id": group_id,
                "size": len(members),
                "labels": " ".join(str(label) for label in labels),
                "cross_label": len(labels) > 1,
                "paths": " ".join(member.rel_path for member in members),
            }
        )
    return rows


def feature_matrix(records: list[ImageRecord]) -> tuple[np.ndarray, list[int]]:
    features: list[np.ndarray] = []
    indices: list[int] = []
    for idx, record in enumerate(records):
        if record.color_hist is None or not record.valid:
            continue
        numeric = np.asarray(
            [
                math.log1p(record.width or 0) / 10.0,
                math.log1p(record.height or 0) / 10.0,
                min(record.aspect_ratio or 0.0, 5.0) / 5.0,
                record.foreground_ratio or 0.0,
                record.bbox_area_ratio or 0.0,
                (record.border_brightness or 0.0) / 255.0,
            ],
            dtype=np.float32,
        )
        features.append(np.concatenate([record.color_hist.astype(np.float32), numeric]))
        indices.append(idx)
    if not features:
        return np.zeros((0, 0), dtype=np.float32), []
    matrix = np.vstack(features).astype(np.float32)
    return matrix, indices


def diversity_and_outliers(
    records: list[ImageRecord],
    outliers_per_class: int,
) -> tuple[list[dict[str, object]], dict[int, list[ImageRecord]]]:
    matrix, indices = feature_matrix(records)
    if matrix.size == 0:
        return [], {}

    rows: list[dict[str, object]] = []
    outliers: dict[int, list[ImageRecord]] = {}
    labels = np.asarray([records[idx].label for idx in indices])
    for label in sorted(set(labels.tolist())):
        positions = np.where(labels == label)[0]
        class_matrix = matrix[positions]
        centroid = class_matrix.mean(axis=0)
        distances = np.linalg.norm(class_matrix - centroid.reshape(1, -1), axis=1)
        order = np.argsort(-distances)
        selected = [records[indices[positions[pos]]] for pos in order[:outliers_per_class]]
        outliers[label] = selected
        rows.append(
            {
                "label": label,
                "class_name": LABELS[label],
                "n": len(positions),
                "feature_distance_mean": float(distances.mean()),
                "feature_distance_p95": float(np.quantile(distances, 0.95)),
                "feature_distance_max": float(distances.max()),
                "unique_ahash_ratio": unique_ratio(records, label, "ahash"),
                "unique_dhash_ratio": unique_ratio(records, label, "dhash"),
            }
        )
    return rows, outliers


def unique_ratio(records: list[ImageRecord], label: int, attr: str) -> float:
    values = [getattr(record, attr) for record in records if record.label == label and getattr(record, attr) is not None]
    if not values:
        return 0.0
    return len(set(values)) / len(values)


def class_summary(records: list[ImageRecord], attr: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in sorted(LABELS):
        values = [
            float(getattr(record, attr))
            for record in records
            if record.label == label and getattr(record, attr) is not None
        ]
        row: dict[str, object] = {"label": label, "class_name": LABELS[label], "n": len(values)}
        row.update(quantiles(values))
        rows.append(row)
    return rows


def dimension_summary(records: list[ImageRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in sorted(LABELS):
        class_records = [record for record in records if record.label == label and record.valid]
        for attr in ("width", "height", "aspect_ratio", "pixels"):
            values = [float(getattr(record, attr)) for record in class_records if getattr(record, attr) is not None]
            row: dict[str, object] = {
                "label": label,
                "class_name": LABELS[label],
                "metric": attr,
                "n": len(values),
            }
            row.update(quantiles(values))
            rows.append(row)
    return rows


def background_summary(records: list[ImageRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in sorted(LABELS):
        class_records = [record for record in records if record.label == label and record.border_rgb is not None]
        colors: Counter[str] = Counter()
        brightness = [float(record.border_brightness) for record in class_records if record.border_brightness is not None]
        for record in class_records:
            assert record.border_rgb is not None
            bucket = tuple(int(channel // 32) * 32 for channel in record.border_rgb)
            colors[f"{bucket[0]}-{bucket[1]}-{bucket[2]}"] += 1
        row: dict[str, object] = {
            "label": label,
            "class_name": LABELS[label],
            "n": len(class_records),
            "top_border_rgb_buckets": "; ".join(f"{key}:{count}" for key, count in colors.most_common(8)),
        }
        row.update({f"brightness_{key}": value for key, value in quantiles(brightness).items()})
        rows.append(row)
    return rows


def manifest_rows(records: list[ImageRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "path": record.rel_path,
                "label": record.label,
                "class_name": record.class_name,
                "filename": record.filename,
                "valid": record.valid,
                "error": record.error,
                "width": record.width,
                "height": record.height,
                "mode": record.mode,
                "aspect_ratio": record.aspect_ratio,
                "pixels": record.pixels,
                "file_sha256": record.file_sha256,
                "pixel_sha256": record.pixel_sha256,
                "ahash": f"{record.ahash:016x}" if record.ahash is not None else "",
                "dhash": f"{record.dhash:016x}" if record.dhash is not None else "",
                "border_brightness": record.border_brightness,
                "foreground_ratio_proxy": record.foreground_ratio,
                "bbox_area_ratio_proxy": record.bbox_area_ratio,
            }
        )
    return rows


def make_contact_sheet(records: list[ImageRecord], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    thumb_w, thumb_h = 160, 160
    caption_h = 34
    cols = min(6, max(1, math.ceil(math.sqrt(len(records)))))
    rows = math.ceil(len(records) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + caption_h) + 28), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for idx, record in enumerate(records):
        row, col = divmod(idx, cols)
        x = col * thumb_w
        y = 28 + row * (thumb_h + caption_h)
        try:
            with Image.open(record.path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                px = x + (thumb_w - image.width) // 2
                py = y + (thumb_h - image.height) // 2
                sheet.paste(image, (px, py))
        except (OSError, UnidentifiedImageError):
            draw.rectangle([x, y, x + thumb_w - 1, y + thumb_h - 1], outline=(220, 0, 0))
        caption = Path(record.rel_path).name[:28]
        draw.text((x + 4, y + thumb_h + 3), caption, fill=(0, 0, 0))
        draw.text((x + 4, y + thumb_h + 17), record.class_name[:28], fill=(80, 80, 80))
    sheet.save(output_path, quality=90)


def write_report(
    output_dir: Path,
    records: list[ImageRecord],
    exact_pixel_groups: list[dict[str, object]],
    exact_file_groups: list[dict[str, object]],
    near_pairs: list[dict[str, object]],
    near_groups: list[dict[str, object]],
    diversity_rows: list[dict[str, object]],
) -> None:
    counts = Counter(record.label for record in records)
    invalid = [record for record in records if not record.valid]
    cross_exact = [row for row in exact_pixel_groups if row["cross_label"]]
    cross_near = [row for row in near_groups if row["cross_label"]]
    electronic_diversity = next((row for row in diversity_rows if row["label"] == 1), None)

    lines = [
        "# Train Audit Report",
        "",
        "This audit uses train images only. Test images are not inspected.",
        "",
        "## Class Counts",
        "",
    ]
    for label in sorted(LABELS):
        lines.append(f"- `{label}_{LABELS[label]}`: {counts[label]}")
    lines.extend(
        [
            "",
            "## Data Integrity",
            "",
            f"- Total train images discovered: {len(records)}",
            f"- Invalid/unreadable images: {len(invalid)}",
            f"- Exact file duplicate groups: {len(exact_file_groups)}",
            f"- Exact pixel duplicate groups: {len(exact_pixel_groups)}",
            f"- Cross-label exact pixel duplicate groups: {len(cross_exact)}",
            f"- Near-duplicate pairs: {len(near_pairs)}",
            f"- Near-duplicate groups: {len(near_groups)}",
            f"- Cross-label near-duplicate groups: {len(cross_near)}",
            "",
            "## Bias and Diversity Proxies",
            "",
            "- Background bias is approximated from border color and border brightness.",
            "- Object-size bias is approximated from foreground and bounding-box ratios against border color.",
            "- Potential label noise is based on cross-label exact/near duplicates and class-feature outliers.",
        ]
    )
    if electronic_diversity:
        lines.extend(
            [
                "",
                "## Electronic Diversity",
                "",
                f"- Electronic samples: {electronic_diversity['n']}",
                f"- Feature-distance mean: {electronic_diversity['feature_distance_mean']:.4f}",
                f"- Feature-distance p95: {electronic_diversity['feature_distance_p95']:.4f}",
                f"- Unique dHash ratio: {electronic_diversity['unique_dhash_ratio']:.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Key Output Files",
            "",
            "- `train_manifest.csv`",
            "- `dimension_summary.csv`",
            "- `exact_pixel_duplicate_groups.csv`",
            "- `near_duplicate_pairs.csv`",
            "- `near_duplicate_groups.csv`",
            "- `potential_label_noise.csv`",
            "- `background_bias_summary.csv`",
            "- `object_size_proxy_summary.csv`",
            "- `intra_class_diversity_summary.csv`",
            "- `samples/`",
        ]
    )
    (output_dir / "audit_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    train_dir = (root / args.train_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    samples_dir = output_dir / "samples"

    if not train_dir.exists():
        raise SystemExit(f"Train directory not found: {train_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    records = iter_train_images(root, train_dir)
    for idx, record in enumerate(records, start=1):
        audit_image(record)
        if idx % 1000 == 0:
            print(f"Audited {idx}/{len(records)} images")

    valid_records = [record for record in records if record.valid]
    exact_file_groups = duplicate_groups(valid_records, "file_sha256")
    exact_pixel_groups = duplicate_groups(valid_records, "pixel_sha256")
    near_pairs = find_near_duplicate_pairs(valid_records, args.near_threshold, args.max_bucket_size)
    near_groups = near_duplicate_groups(valid_records, near_pairs)
    diversity_rows, outliers = diversity_and_outliers(valid_records, args.outliers_per_class)

    label_noise_rows: list[dict[str, object]] = []
    for row in exact_pixel_groups:
        if row["cross_label"]:
            label_noise_rows.append({"source": "cross_label_exact_pixel_duplicate", **row})
    for row in near_groups:
        if row["cross_label"]:
            label_noise_rows.append({"source": "cross_label_near_duplicate", **row})
    for label, members in outliers.items():
        for record in members:
            label_noise_rows.append(
                {
                    "source": "class_feature_outlier_review",
                    "group_id": "",
                    "size": "",
                    "labels": str(label),
                    "cross_label": "",
                    "paths": record.rel_path,
                }
            )

    write_csv(
        output_dir / "train_manifest.csv",
        manifest_rows(records),
        [
            "path",
            "label",
            "class_name",
            "filename",
            "valid",
            "error",
            "width",
            "height",
            "mode",
            "aspect_ratio",
            "pixels",
            "file_sha256",
            "pixel_sha256",
            "ahash",
            "dhash",
            "border_brightness",
            "foreground_ratio_proxy",
            "bbox_area_ratio_proxy",
        ],
    )
    write_csv(
        output_dir / "class_counts.csv",
        [{"label": label, "class_name": LABELS[label], "count": sum(r.label == label for r in records)} for label in sorted(LABELS)],
        ["label", "class_name", "count"],
    )
    write_csv(
        output_dir / "dimension_summary.csv",
        dimension_summary(valid_records),
        ["label", "class_name", "metric", "n", "min", "p05", "p25", "median", "mean", "p75", "p95", "max"],
    )
    write_csv(
        output_dir / "exact_file_duplicate_groups.csv",
        exact_file_groups,
        ["group_id", "hash", "size", "labels", "cross_label", "paths"],
    )
    write_csv(
        output_dir / "exact_pixel_duplicate_groups.csv",
        exact_pixel_groups,
        ["group_id", "hash", "size", "labels", "cross_label", "paths"],
    )
    write_csv(
        output_dir / "near_duplicate_pairs.csv",
        near_pairs,
        [
            "left_path",
            "right_path",
            "left_label",
            "right_label",
            "left_class",
            "right_class",
            "ahash_distance",
            "dhash_distance",
            "cross_label",
        ],
    )
    write_csv(
        output_dir / "near_duplicate_groups.csv",
        near_groups,
        ["group_id", "size", "labels", "cross_label", "paths"],
    )
    write_csv(
        output_dir / "potential_label_noise.csv",
        label_noise_rows,
        ["source", "group_id", "hash", "size", "labels", "cross_label", "paths"],
    )
    write_csv(
        output_dir / "background_bias_summary.csv",
        background_summary(valid_records),
        [
            "label",
            "class_name",
            "n",
            "top_border_rgb_buckets",
            "brightness_min",
            "brightness_p05",
            "brightness_p25",
            "brightness_median",
            "brightness_mean",
            "brightness_p75",
            "brightness_p95",
            "brightness_max",
        ],
    )
    write_csv(
        output_dir / "object_size_proxy_summary.csv",
        class_summary(valid_records, "foreground_ratio"),
        ["label", "class_name", "n", "min", "p05", "p25", "median", "mean", "p75", "p95", "max"],
    )
    write_csv(
        output_dir / "bbox_area_proxy_summary.csv",
        class_summary(valid_records, "bbox_area_ratio"),
        ["label", "class_name", "n", "min", "p05", "p25", "median", "mean", "p75", "p95", "max"],
    )
    write_csv(
        output_dir / "intra_class_diversity_summary.csv",
        diversity_rows,
        [
            "label",
            "class_name",
            "n",
            "feature_distance_mean",
            "feature_distance_p95",
            "feature_distance_max",
            "unique_ahash_ratio",
            "unique_dhash_ratio",
        ],
    )

    by_label: dict[int, list[ImageRecord]] = defaultdict(list)
    for record in valid_records:
        by_label[record.label].append(record)
    for label in sorted(LABELS):
        sample_count = min(args.samples_per_class, len(by_label[label]))
        sampled = random.sample(by_label[label], sample_count) if sample_count else []
        make_contact_sheet(sampled, samples_dir / f"random_{label}_{LABELS[label]}.jpg", f"Random samples: {label}_{LABELS[label]}")
        make_contact_sheet(outliers.get(label, []), samples_dir / f"outliers_{label}_{LABELS[label]}.jpg", f"Review outliers: {label}_{LABELS[label]}")

    summary = {
        "total_images": len(records),
        "valid_images": len(valid_records),
        "invalid_images": len(records) - len(valid_records),
        "class_counts": {f"{label}_{LABELS[label]}": sum(record.label == label for record in records) for label in sorted(LABELS)},
        "exact_file_duplicate_groups": len(exact_file_groups),
        "exact_pixel_duplicate_groups": len(exact_pixel_groups),
        "cross_label_exact_pixel_duplicate_groups": sum(bool(row["cross_label"]) for row in exact_pixel_groups),
        "near_duplicate_pairs": len(near_pairs),
        "near_duplicate_groups": len(near_groups),
        "cross_label_near_duplicate_groups": sum(bool(row["cross_label"]) for row in near_groups),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(output_dir, records, exact_pixel_groups, exact_file_groups, near_pairs, near_groups, diversity_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
