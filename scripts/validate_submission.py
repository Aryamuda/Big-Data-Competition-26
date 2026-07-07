#!/usr/bin/env python3
"""Validate BDC 2026 submission/template order and label contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ALLOWED_LABELS = {"0", "1", "2"}
LABEL_CONTRACT = {
    "0": "Recyclable",
    "1": "Electronic",
    "2": "Organic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--template", type=Path, default=Path("submission.csv"))
    parser.add_argument("--test-dir", type=Path, default=Path("test"))
    parser.add_argument("--submission", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def normalize_id(value: str) -> str:
    value = value.strip()
    if value.endswith(".jpg"):
        value = value[:-4]
    return value


def validate_template(template_path: Path, test_dir: Path) -> dict[str, object]:
    if not template_path.exists():
        raise SystemExit(f"Template not found: {template_path}")
    if not test_dir.exists():
        raise SystemExit(f"Test directory not found: {test_dir}")

    fieldnames, rows = read_csv(template_path)
    if fieldnames != ["id", "predicted"]:
        raise SystemExit(f"Template columns must be exactly ['id', 'predicted'], got {fieldnames}")

    ids = [normalize_id(row["id"]) for row in rows]
    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
    if duplicate_ids:
        raise SystemExit(f"Duplicate ids in template: {duplicate_ids[:10]}")

    expected_paths = [test_dir / f"{item}.jpg" for item in ids]
    missing_files = [path.name for path in expected_paths if not path.exists()]
    test_files = sorted(path.name for path in test_dir.iterdir() if path.is_file())
    expected_file_set = {path.name for path in expected_paths}
    extra_files = [name for name in test_files if name not in expected_file_set]

    filesystem_order = [path.name for path in test_dir.iterdir() if path.is_file()]
    template_order = [path.name for path in expected_paths]
    filesystem_order_matches_template = filesystem_order == template_order
    numeric_order = [f"{idx}.jpg" for idx in range(1, len(rows) + 1)]
    template_order_is_numeric = template_order == numeric_order

    return {
        "template_path": str(template_path),
        "test_dir": str(test_dir),
        "row_count": len(rows),
        "missing_test_files": missing_files,
        "extra_test_files": extra_files,
        "filesystem_order_matches_template": filesystem_order_matches_template,
        "template_order_is_numeric_1_to_n": template_order_is_numeric,
        "first_template_ids": ids[:10],
        "last_template_ids": ids[-10:],
    }


def validate_submission(template_path: Path, submission_path: Path) -> dict[str, object]:
    template_fields, template_rows = read_csv(template_path)
    submission_fields, submission_rows = read_csv(submission_path)
    if submission_fields != ["id", "predicted"]:
        raise SystemExit(f"Submission columns must be exactly ['id', 'predicted'], got {submission_fields}")
    if template_fields != ["id", "predicted"]:
        raise SystemExit(f"Template columns must be exactly ['id', 'predicted'], got {template_fields}")
    if len(submission_rows) != len(template_rows):
        raise SystemExit(f"Submission row count {len(submission_rows)} != template row count {len(template_rows)}")

    template_ids = [normalize_id(row["id"]) for row in template_rows]
    submission_ids = [normalize_id(row["id"]) for row in submission_rows]
    mismatches = [
        {"row": idx + 2, "template_id": left, "submission_id": right}
        for idx, (left, right) in enumerate(zip(template_ids, submission_ids))
        if left != right
    ]
    if mismatches:
        raise SystemExit(f"Submission id/order mismatch. First mismatches: {mismatches[:5]}")

    invalid_predictions = []
    missing_predictions = []
    counts = {label: 0 for label in sorted(ALLOWED_LABELS)}
    for idx, row in enumerate(submission_rows):
        pred = row["predicted"].strip()
        if pred == "":
            missing_predictions.append(idx + 2)
        elif pred not in ALLOWED_LABELS:
            invalid_predictions.append({"row": idx + 2, "id": submission_ids[idx], "predicted": pred})
        else:
            counts[pred] += 1
    if missing_predictions:
        raise SystemExit(f"Submission has missing predictions at rows: {missing_predictions[:10]}")
    if invalid_predictions:
        raise SystemExit(f"Submission has invalid predictions. First invalid rows: {invalid_predictions[:10]}")

    return {
        "submission_path": str(submission_path),
        "row_count": len(submission_rows),
        "prediction_counts": counts,
        "label_contract": LABEL_CONTRACT,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    template_path = (root / args.template).resolve()
    test_dir = (root / args.test_dir).resolve()
    template_report = validate_template(template_path, test_dir)

    report: dict[str, object] = {
        "label_contract": LABEL_CONTRACT,
        "template": template_report,
    }
    if args.submission is not None:
        report["submission"] = validate_submission(template_path, (root / args.submission).resolve())

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print("Label contract:")
    for label, name in LABEL_CONTRACT.items():
        print(f"  {label} = {name}")
    print("")
    print(f"Template rows: {template_report['row_count']}")
    print(f"Template first ids: {template_report['first_template_ids']}")
    print(f"Template last ids: {template_report['last_template_ids']}")
    print(f"Template order is numeric 1..n: {template_report['template_order_is_numeric_1_to_n']}")
    print(f"Filesystem order matches template: {template_report['filesystem_order_matches_template']}")
    print(f"Missing test files: {len(template_report['missing_test_files'])}")
    print(f"Extra test files: {len(template_report['extra_test_files'])}")
    if args.submission is not None:
        submission_report = report["submission"]
        assert isinstance(submission_report, dict)
        print("")
        print(f"Submission rows: {submission_report['row_count']}")
        print(f"Prediction counts: {submission_report['prediction_counts']}")


if __name__ == "__main__":
    main()
