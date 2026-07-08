#!/usr/bin/env python3
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def image_info(path):
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.width, image.height


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest = Path("outputs/review/keep_manifest.csv")
    out = Path("outputs/audit_kept")

    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))

    audit_rows = []
    corrupt = []
    hashes = defaultdict(list)
    class_counts = Counter(row["current_label"] for row in rows)
    sizes = Counter()

    for row in rows:
        path = Path(row["path"])
        record = {
            "path": row["path"],
            "current_label": row["current_label"],
            "class_name": row["class_name"],
            "width": "",
            "height": "",
            "aspect_ratio": "",
            "sha256": "",
            "error": "",
        }

        try:
            width, height = image_info(path)
            sha256 = file_hash(path)
            record.update({
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 6),
                "sha256": sha256,
            })
            sizes[(width, height)] += 1
            hashes[sha256].append(row["path"])
        except Exception as exc:
            record["error"] = str(exc)
            corrupt.append(row["path"])

        audit_rows.append(record)

    duplicate_rows = [
        {"sha256": sha256, "count": len(paths), "paths": "|".join(paths)}
        for sha256, paths in sorted(hashes.items())
        if len(paths) > 1
    ]

    summary = {
        "manifest": manifest.as_posix(),
        "total_images": len(rows),
        "class_counts": dict(sorted(class_counts.items())),
        "corrupt_count": len(corrupt),
        "exact_duplicate_groups": len(duplicate_rows),
        "exact_duplicate_images": sum(row["count"] for row in duplicate_rows),
        "unique_sizes": len(sizes),
        "most_common_sizes": [
            {"width": width, "height": height, "count": count}
            for (width, height), count in sizes.most_common(10)
        ],
    }

    write_csv(
        out / "image_audit.csv",
        audit_rows,
        ["path", "current_label", "class_name", "width", "height", "aspect_ratio", "sha256", "error"],
    )
    write_csv(out / "exact_duplicates.csv", duplicate_rows, ["sha256", "count", "paths"])
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"images: {len(rows)}")
    print(f"corrupt: {len(corrupt)}")
    print(f"exact duplicate groups: {len(duplicate_rows)}")
    print("wrote: outputs/audit_kept")


if __name__ == "__main__":
    main()
