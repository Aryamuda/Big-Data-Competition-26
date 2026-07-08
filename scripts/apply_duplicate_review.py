#!/usr/bin/env python3
import csv
from pathlib import Path


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    kept = read_csv(Path("outputs/review/keep_manifest.csv"))
    actions = read_csv(Path("outputs/duplicate_review/duplicate_review_actions.csv"))
    fields = ["image_id", "path", "current_label", "class_name", "review_status", "reason"]

    remove_by_path = {
        row["remove_path"]: row
        for row in actions
        if row["remove_path"]
    }
    clean = [row for row in kept if row["path"] not in remove_by_path]

    removed = []
    for row in kept:
        action = remove_by_path.get(row["path"])
        if not action:
            continue
        removed.append({
            "path": row["path"],
            "label": row["current_label"],
            "class_name": row["class_name"],
            "source_folder": action["folder"],
            "decision": action["decision"],
            "reason": "removed_by_duplicate_review",
        })

    groups = [
        {
            "source_folder": row["folder"],
            "path_a": row["group_paths"].split("|")[0],
            "path_b": row["group_paths"].split("|")[1],
            "reason": "duplicate_review_both",
        }
        for row in actions
        if row["group_paths"]
    ]

    write_csv(Path("manifest_kept.csv"), kept, fields)
    write_csv(Path("manifest_clean.csv"), clean, fields)
    write_csv(
        Path("outputs/duplicate_review/dedup_remove_manifest.csv"),
        removed,
        ["path", "label", "class_name", "source_folder", "decision", "reason"],
    )
    write_csv(
        Path("outputs/duplicate_review/dedup_group_pairs.csv"),
        groups,
        ["source_folder", "path_a", "path_b", "reason"],
    )

    print(f"manifest_kept: {len(kept)}")
    print(f"removed: {len(removed)}")
    print(f"manifest_clean: {len(clean)}")
    print(f"group pairs: {len(groups)}")


if __name__ == "__main__":
    main()
