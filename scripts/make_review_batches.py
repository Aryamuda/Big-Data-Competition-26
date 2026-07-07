#!/usr/bin/env python3
"""Create deterministic human-review batches for the train dataset."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from pathlib import Path


LABELS = [
    (0, "Recyclable", "0_Recyclable"),
    (1, "Electronic", "1_Electronic"),
    (2, "Organic", "2_Organic"),
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ENCODE_GUIDE = "0=keep / label acceptable; 1=remove / clearly inappropriate; 2=unsure / needs second review"
REVIEW_COLUMNS = ["image_id", "path", "current_label", "class_name", "review_status", "reason", "encode"]
MANIFEST_COLUMNS = ["image_id", "path", "current_label", "class_name", "review_batch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--train-dir", type=Path, default=Path("train"))
    parser.add_argument("--output-dir", type=Path, default=Path("review_batches"))
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing review CSVs even if they contain filled review decisions.",
    )
    return parser.parse_args()


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def contiguous_batch_for_index(index: int, total: int, batch_count: int) -> str:
    base_size, remainder = divmod(total, batch_count)
    start = 0
    for batch_index in range(batch_count):
        size = base_size + (1 if batch_index < remainder else 0)
        end = start + size
        if start <= index < end:
            return batch_name(batch_index)
        start = end
    raise ValueError(f"Index {index} is out of range for total {total}")


def batch_name(index: int) -> str:
    return f"batch_{index + 1}"


def known_csv_names(batch_count: int) -> set[str]:
    return {f"{batch_name(idx)}_review.csv" for idx in range(batch_count)} | {
        "review_assignment_manifest.csv"
    }


def known_dirs(batch_count: int) -> set[tuple[str, ...]]:
    dirs: set[tuple[str, ...]] = set()
    for idx in range(batch_count):
        batch = batch_name(idx)
        dirs.add((batch,))
        for _, _, folder_name in LABELS:
            dirs.add((batch, folder_name))
    return dirs


def read_review_csv_has_decisions(path: Path) -> bool:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        allowed_fieldnames = [
            REVIEW_COLUMNS,
            ["image_id", "path", "current_label", "class_name", "review_status", "reason"],
        ]
        if reader.fieldnames not in allowed_fieldnames:
            return True
        for row in reader:
            if row.get("review_status", "").strip() or row.get("reason", "").strip():
                return True
    return False


def prepare_output_dir(output_dir: Path, batch_count: int, force: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    if not output_dir.is_dir():
        raise SystemExit(f"Output path exists but is not a directory: {output_dir}")

    allowed_csvs = known_csv_names(batch_count)
    allowed_dirs = known_dirs(batch_count)

    for csv_name in sorted(allowed_csvs):
        csv_path = output_dir / csv_name
        if not csv_path.exists():
            continue
        if not csv_path.is_file():
            raise SystemExit(f"Refusing to overwrite non-file output: {csv_path}")
        if csv_name.endswith("_review.csv") and not force and read_review_csv_has_decisions(csv_path):
            raise SystemExit(
                f"Refusing to overwrite review decisions in {csv_path}. "
                "Use --force only after backing up human review work."
            )

    for path in sorted(output_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        rel_parts = path.relative_to(output_dir).parts
        if path.is_symlink():
            path.unlink()
        elif path.is_file():
            rel_parent = path.parent.relative_to(output_dir).parts
            if path.parent == output_dir and path.name in allowed_csvs:
                path.unlink()
            elif rel_parent in allowed_dirs and path.suffix.lower() in IMAGE_EXTENSIONS:
                path.unlink()
            else:
                raise SystemExit(f"Refusing to remove unknown file in output directory: {path}")
        elif path.is_dir():
            if rel_parts in allowed_dirs:
                try:
                    path.rmdir()
                except OSError as exc:
                    raise SystemExit(f"Refusing to remove non-empty generated directory {path}: {exc}") from exc
            else:
                raise SystemExit(f"Refusing to remove unknown directory in output directory: {path}")


def collect_records(root: Path, train_dir: Path, batch_count: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for label, class_name, folder_name in LABELS:
        class_dir = train_dir / folder_name
        if not class_dir.exists() or not class_dir.is_dir():
            raise SystemExit(f"Missing class directory: {class_dir}")
        images = sorted(
            [path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
            key=natural_key,
        )
        total = len(images)
        for index, path in enumerate(images):
            records.append(
                {
                    "image_id": path.stem,
                    "path": path.relative_to(root).as_posix(),
                    "absolute_path": path,
                    "current_label": label,
                    "class_name": class_name,
                    "class_folder": folder_name,
                    "review_batch": contiguous_batch_for_index(index, total, batch_count),
                    "class_order": index,
                }
            )
    return records


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def copy_images(root: Path, output_dir: Path, records: list[dict[str, object]]) -> None:
    for record in records:
        destination_dir = output_dir / str(record["review_batch"]) / str(record["class_folder"])
        destination_dir.mkdir(parents=True, exist_ok=True)
        source = root / str(record["path"])
        destination = destination_dir / source.name
        shutil.copy2(source, destination)


def ordered_batch_rows(records: list[dict[str, object]], batch: str) -> list[dict[str, object]]:
    rows = [record for record in records if record["review_batch"] == batch]
    label_order = {label: index for index, (label, _, _) in enumerate(LABELS)}
    return sorted(rows, key=lambda row: (label_order[int(row["current_label"])], int(row["class_order"])))


def write_review_csvs(output_dir: Path, records: list[dict[str, object]], batch_count: int) -> None:
    for idx in range(batch_count):
        batch = batch_name(idx)
        rows = ordered_batch_rows(records, batch)
        review_rows = [
            {
                "image_id": row["image_id"],
                "path": row["path"],
                "current_label": row["current_label"],
                "class_name": row["class_name"],
                "review_status": "",
                "reason": "",
                "encode": ENCODE_GUIDE,
            }
            for row in rows
        ]
        write_csv(output_dir / f"{batch}_review.csv", review_rows, REVIEW_COLUMNS)

    manifest_rows = [
        {
            "image_id": row["image_id"],
            "path": row["path"],
            "current_label": row["current_label"],
            "class_name": row["class_name"],
            "review_batch": row["review_batch"],
        }
        for row in sorted(
            records,
            key=lambda row: (
                int(row["current_label"]),
                int(row["class_order"]),
            ),
        )
    ]
    write_csv(output_dir / "review_assignment_manifest.csv", manifest_rows, MANIFEST_COLUMNS)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def validate_outputs(
    root: Path,
    train_records: list[dict[str, object]],
    output_dir: Path,
    batch_count: int,
) -> dict[str, object]:
    original_paths = [str(record["path"]) for record in train_records]
    assigned_paths = [str(record["path"]) for record in train_records]
    original_counter = Counter(original_paths)
    assigned_counter = Counter(assigned_paths)

    missing_assignments = sorted(path for path in original_counter if assigned_counter[path] == 0)
    duplicate_assignments = sorted(path for path, count in assigned_counter.items() if count > 1)

    batch_counts: dict[str, int] = {}
    per_class_counts: dict[str, dict[str, int]] = {}
    csv_row_counts: dict[str, int] = {}
    csv_paths_missing: dict[str, list[str]] = {}
    copied_file_missing: list[str] = []
    copied_file_is_symlink: list[str] = []
    copied_file_size_mismatch: list[str] = []

    for idx in range(batch_count):
        batch = batch_name(idx)
        batch_records = [record for record in train_records if record["review_batch"] == batch]
        batch_counts[batch] = len(batch_records)
        per_class_counts[batch] = {
            folder_name: sum(record["class_folder"] == folder_name for record in batch_records)
            for _, _, folder_name in LABELS
        }

        csv_path = output_dir / f"{batch}_review.csv"
        rows = read_csv_rows(csv_path)
        csv_row_counts[batch] = len(rows)
        missing_paths = [row["path"] for row in rows if not (root / row["path"]).is_file()]
        csv_paths_missing[batch] = missing_paths

        for record in batch_records:
            source = root / str(record["path"])
            copied_path = output_dir / batch / str(record["class_folder"]) / source.name
            if not copied_path.exists():
                copied_file_missing.append(copied_path.as_posix())
                continue
            if copied_path.is_symlink():
                copied_file_is_symlink.append(copied_path.as_posix())
                continue
            if not copied_path.is_file():
                copied_file_missing.append(copied_path.as_posix())
                continue
            if copied_path.stat().st_size != source.stat().st_size:
                copied_file_size_mismatch.append(copied_path.as_posix())

    manifest_rows = read_csv_rows(output_dir / "review_assignment_manifest.csv")
    manifest_counter = Counter(row["path"] for row in manifest_rows)
    manifest_missing = sorted(path for path in original_counter if manifest_counter[path] == 0)
    manifest_duplicates = sorted(path for path, count in manifest_counter.items() if count > 1)

    valid = (
        not missing_assignments
        and not duplicate_assignments
        and not manifest_missing
        and not manifest_duplicates
        and all(not values for values in csv_paths_missing.values())
        and not copied_file_missing
        and not copied_file_is_symlink
        and not copied_file_size_mismatch
        and len(manifest_rows) == len(train_records)
        and sum(csv_row_counts.values()) == len(train_records)
    )

    return {
        "valid": valid,
        "total_train_images": len(train_records),
        "batch_counts": batch_counts,
        "per_class_counts": per_class_counts,
        "missing_assignments": missing_assignments,
        "duplicate_assignments": duplicate_assignments,
        "manifest_row_count": len(manifest_rows),
        "manifest_missing_assignments": manifest_missing,
        "manifest_duplicate_assignments": manifest_duplicates,
        "csv_row_counts": csv_row_counts,
        "csv_paths_missing": csv_paths_missing,
        "copied_file_missing": copied_file_missing,
        "copied_file_is_symlink": copied_file_is_symlink,
        "copied_file_size_mismatch": copied_file_size_mismatch,
        "every_train_image_exactly_once": not missing_assignments
        and not duplicate_assignments
        and not manifest_missing
        and not manifest_duplicates
        and len(manifest_rows) == len(train_records)
        and sum(csv_row_counts.values()) == len(train_records),
        "csv_paths_resolve": all(not values for values in csv_paths_missing.values()),
    }


def print_report(report: dict[str, object]) -> None:
    print(f"Total train images: {report['total_train_images']}")
    print("")
    print("Images in each batch:")
    for batch, count in report["batch_counts"].items():
        print(f"  {batch}: {count}")
    print("")
    print("Per-class count in each batch:")
    for batch, counts in report["per_class_counts"].items():
        print(f"  {batch}:")
        for _, _, folder_name in LABELS:
            print(f"    {folder_name}: {counts[folder_name]}")
    print("")
    print(f"Missing assignments: {len(report['missing_assignments'])}")
    print(f"Duplicate assignments: {len(report['duplicate_assignments'])}")
    print(f"Every train image appears exactly once: {report['every_train_image_exactly_once']}")
    print("")
    print("CSV row counts:")
    for batch, count in report["csv_row_counts"].items():
        print(f"  {batch}_review.csv: {count}")
    print(f"  review_assignment_manifest.csv: {report['manifest_row_count']}")
    print("")
    print(f"CSV paths resolve to existing original train files: {report['csv_paths_resolve']}")
    print(f"Missing copied files: {len(report['copied_file_missing'])}")
    print(f"Copied outputs that are still symlinks: {len(report['copied_file_is_symlink'])}")
    print(f"Copied files with size mismatch: {len(report['copied_file_size_mismatch'])}")
    print(f"Validation passed: {report['valid']}")


def main() -> None:
    args = parse_args()
    if args.batches != 3:
        raise SystemExit("This review workflow is defined for exactly 3 batches.")

    root = args.root.resolve()
    train_dir = (root / args.train_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    if not train_dir.exists():
        raise SystemExit(f"Train directory not found: {train_dir}")

    records = collect_records(root, train_dir, args.batches)
    prepare_output_dir(output_dir, args.batches, args.force)
    copy_images(root, output_dir, records)
    write_review_csvs(output_dir, records, args.batches)
    report = validate_outputs(root, records, output_dir, args.batches)
    print_report(report)
    if not report["valid"]:
        raise SystemExit("Review batch validation failed.")


if __name__ == "__main__":
    main()
