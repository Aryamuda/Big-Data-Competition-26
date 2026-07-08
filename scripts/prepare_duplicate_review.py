#!/usr/bin/env python3
import csv
import shutil
from pathlib import Path


def safe_name(path, label, side):
    name = Path(path).name.replace("/", "_")
    return f"{side}__label_{label}__{name}"


def copy_image(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.resolve()


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def exact_rows(out):
    rows = []
    with Path("outputs/audit_kept/exact_duplicates.csv").open(newline="") as f:
        for index, row in enumerate(csv.DictReader(f), start=1):
            review_id = f"exact_{index:04d}"
            folder = out / "exact" / review_id
            paths = row["paths"].split("|")

            for image_index, path in enumerate(paths, start=1):
                copied = copy_image(Path(path), folder / safe_name(path, "same", chr(96 + image_index)))
                if image_index == 1:
                    image_a = copied
                elif image_index == 2:
                    image_b = copied

            rows.append({
                "folder": f"duplicate_review/exact/{review_id}",
                "decision": "",
                "guide": "a=keep A remove B; b=keep B remove A; both=keep both same split; ignore=false positive",
                "notes": "exact",
            })
    return rows


def near_rows(out):
    rows = []
    counters = {"True": 0, "False": 0}
    with Path("outputs/audit_kept/near_duplicate_pairs.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            cross = row["cross_label"]
            counters[cross] += 1
            kind = "near_cross" if cross == "True" else "near_same"
            review_id = f"{kind}_{counters[cross]:04d}"
            folder = out / kind / review_id

            image_a = copy_image(Path(row["path_a"]), folder / safe_name(row["path_a"], row["label_a"], "a"))
            image_b = copy_image(Path(row["path_b"]), folder / safe_name(row["path_b"], row["label_b"], "b"))

            rows.append({
                "folder": f"duplicate_review/{kind}/{review_id}",
                "decision": "",
                "guide": "a=keep A remove B; b=keep B remove A; both=keep both same split; ignore=false positive",
                "notes": kind,
            })
    return rows


def main():
    out = Path("outputs/duplicate_review")
    if out.exists():
        shutil.rmtree(out)

    rows = exact_rows(out) + near_rows(out)
    fields = ["folder", "decision", "guide", "notes"]
    write_csv(out / "duplicate_review.csv", rows, fields)

    counts = {}
    for row in rows:
        kind = row["folder"].split("/")[1]
        counts[kind] = counts.get(kind, 0) + 1

    print(f"review rows: {len(rows)}")
    for key in ["exact", "near_cross", "near_same"]:
        print(f"{key}: {counts.get(key, 0)}")
    print("wrote: outputs/duplicate_review")


if __name__ == "__main__":
    main()
