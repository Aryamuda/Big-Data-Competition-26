#!/usr/bin/env python3
import csv
from pathlib import Path


def write_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    source = Path("outputs/review/review_decisions_merged.csv")
    fields = ["image_id", "path", "current_label", "class_name", "review_status", "reason"]
    duplicate_remove = "train/2_Organic/O_8873.jpg"

    with source.open(newline="") as f:
        rows = [
            {key: row[key] for key in fields}
            for row in csv.DictReader(f)
        ]

    for row in rows:
        if row["path"] == duplicate_remove:
            row["review_status"] = "1"
            row["reason"] = "excluded exact cross-label duplicate; Recyclable copy kept"

    keep = [row for row in rows if row["review_status"] in {"0", "2"}]
    remove = [row for row in rows if row["review_status"] == "1"]
    unsure = [row for row in rows if row["review_status"] == "2"]

    duplicate_decisions = [
        {
            "path": "train/0_Recyclable/R_799.jpg",
            "decision": "keep as Recyclable",
        },
        {
            "path": duplicate_remove,
            "decision": "exclude exact cross-label duplicate; Recyclable copy kept",
        },
    ]

    out = Path("outputs/review")
    write_rows(out / "keep_manifest.csv", keep, fields)
    write_rows(out / "remove_manifest.csv", remove, fields)
    write_rows(out / "unsure_manifest.csv", unsure, fields)
    write_rows(out / "exact_duplicate_decisions.csv", duplicate_decisions, ["path", "decision"])

    print(f"keep: {len(keep)}")
    print(f"remove: {len(remove)}")
    print(f"unsure: {len(unsure)}")
    print("wrote manifests: outputs/review")


if __name__ == "__main__":
    main()
