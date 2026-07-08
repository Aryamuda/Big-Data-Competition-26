#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image
import imagehash


def hamming(hex_a, hex_b):
    return bin(int(hex_a, 16) ^ int(hex_b, 16)).count("1")


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    audit_path = Path("outputs/audit_kept/image_audit.csv")
    out = Path("outputs/audit_kept")

    with audit_path.open(newline="") as f:
        images = [row for row in csv.DictReader(f) if not row["error"]]

    hashes = []
    buckets = defaultdict(list)
    total = len(images)
    for index, row in enumerate(images, start=1):
        with Image.open(row["path"]) as image:
            gray = image.convert("L")
            item = {
                "path": row["path"],
                "label": row["current_label"],
                "class_name": row["class_name"],
                "sha256": row["sha256"],
                "phash": str(imagehash.phash(gray)),
                "dhash": str(imagehash.dhash(gray)),
                "whash": str(imagehash.whash(gray)),
            }
        hashes.append(item)
        buckets[item["phash"][:4]].append(item)
        if index % 1000 == 0 or index == total:
            print(f"hashed: {index}/{total}", flush=True)

    pairs = []
    seen = set()
    for bucket in buckets.values():
        for i, left in enumerate(bucket):
            for right in bucket[i + 1:]:
                key = tuple(sorted([left["path"], right["path"]]))
                if key in seen or left["sha256"] == right["sha256"]:
                    continue

                distances = {
                    "phash_distance": hamming(left["phash"], right["phash"]),
                    "dhash_distance": hamming(left["dhash"], right["dhash"]),
                    "whash_distance": hamming(left["whash"], right["whash"]),
                }
                votes = sum(distance <= 4 for distance in distances.values())
                if votes < 2:
                    continue

                seen.add(key)
                pairs.append({
                    "path_a": left["path"],
                    "label_a": left["label"],
                    "class_a": left["class_name"],
                    "path_b": right["path"],
                    "label_b": right["label"],
                    "class_b": right["class_name"],
                    **distances,
                    "matching_hashes": votes,
                    "cross_label": left["label"] != right["label"],
                })

    pairs.sort(key=lambda row: (
        not row["cross_label"],
        row["phash_distance"] + row["dhash_distance"] + row["whash_distance"],
        row["path_a"],
        row["path_b"],
    ))

    summary = {
        "source": audit_path.as_posix(),
        "images_hashed": len(hashes),
        "method": "imagehash phash+dhash+whash; candidate bucket by first 4 phash hex chars",
        "threshold": "at least 2 of 3 distances <= 4",
        "exact_sha256_duplicates_excluded": True,
        "near_duplicate_pairs": len(pairs),
        "cross_label_pairs": sum(row["cross_label"] for row in pairs),
    }

    write_csv(
        out / "near_duplicate_pairs.csv",
        pairs,
        [
            "path_a", "label_a", "class_a",
            "path_b", "label_b", "class_b",
            "phash_distance", "dhash_distance", "whash_distance",
            "matching_hashes", "cross_label",
        ],
    )
    (out / "near_duplicate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"images hashed: {len(hashes)}")
    print(f"near duplicate pairs: {len(pairs)}")
    print(f"cross-label pairs: {summary['cross_label_pairs']}")
    print("wrote: outputs/audit_kept/near_duplicate_pairs.csv")


if __name__ == "__main__":
    main()
