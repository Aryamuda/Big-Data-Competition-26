#!/usr/bin/env python3
import csv
from collections import Counter
from pathlib import Path


def validate_rows(rows, train_paths):
    errors = []
    seen = Counter()
    warning_count = 0

    for row in rows:
        path = row["path"].strip()
        status = row["review_status"].strip()
        reason = row["reason"].strip()

        seen[path] += 1

        if path not in train_paths:
            errors.append(f"path does not resolve to train file: {path}")
        if status not in {"0", "1", "2"}:
            errors.append(f"invalid review_status for {path}: {status!r}")
        if status in {"1", "2"} and not reason:
            warning_count += 1

    for label, paths in [
        ("duplicated review paths", [p for p, n in seen.items() if n > 1]),
        ("train files missing from review batches", train_paths - set(seen)),
        ("review paths outside train set", set(seen) - train_paths),
    ]:
        if paths:
            sample = ", ".join(sorted(paths)[:5])
            errors.append(f"{label}: {len(paths)}; sample: {sample}")

    return errors, warning_count


def write_merged(rows, output_path, fieldnames):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    columns = ["image_id", "path", "current_label", "class_name", "review_status", "reason", "encode"]
    expected_counts = {
        "review_batches/batch_1_review.csv": 8843,
        "review_batches/batch_2_review.csv": 8842,
        "review_batches/batch_3_review.csv": 8842,
    }

    rows = []
    for name, expected_count in expected_counts.items():
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != columns:
                raise ValueError(f"{path}: columns changed: {reader.fieldnames}")
            batch_rows = list(reader)
        if len(batch_rows) != expected_count:
            raise ValueError(f"{path}: expected {expected_count} rows, found {len(batch_rows)}")
        rows.extend(batch_rows)

    train_paths = {p.as_posix() for p in Path("train").glob("*/*") if p.is_file()}
    errors, warning_count = validate_rows(rows, train_paths)

    if errors:
        raise SystemExit("Review validation failed:\n" + "\n".join(f"- {e}" for e in errors))

    write_merged(rows, Path("outputs/review/review_decisions_merged.csv"), columns)

    print(f"validated rows: {len(rows)}")
    print(f"train files covered: {len(train_paths)}")
    print(f"warnings: {warning_count}")
    print("wrote: outputs/review/review_decisions_merged.csv")


if __name__ == "__main__":
    main()
