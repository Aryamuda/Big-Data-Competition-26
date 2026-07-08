#!/usr/bin/env python3
import csv
from collections import Counter
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


def build_pairs():
    pairs = {}

    for index, row in enumerate(read_csv(Path("outputs/audit_kept/exact_duplicates.csv")), start=1):
        paths = row["paths"].split("|")
        folder = f"duplicate_review/exact/exact_{index:04d}"
        pairs[folder] = {"type": "exact", "path_a": paths[0], "path_b": paths[1]}

    counters = {"True": 0, "False": 0}
    for row in read_csv(Path("outputs/audit_kept/near_duplicate_pairs.csv")):
        cross = row["cross_label"]
        counters[cross] += 1
        kind = "near_cross" if cross == "True" else "near_same"
        folder = f"duplicate_review/{kind}/{kind}_{counters[cross]:04d}"
        pairs[folder] = {"type": kind, "path_a": row["path_a"], "path_b": row["path_b"]}

    return pairs


def make_action(row, pair):
    decision = row["decision"].strip()
    action = {
        "folder": row["folder"],
        "type": pair["type"],
        "decision": decision,
        "keep_path": "",
        "remove_path": "",
        "group_paths": "",
    }

    if decision == "a":
        action["keep_path"] = pair["path_a"]
        action["remove_path"] = pair["path_b"]
    elif decision == "b":
        action["keep_path"] = pair["path_b"]
        action["remove_path"] = pair["path_a"]
    elif decision == "both":
        action["group_paths"] = f"{pair['path_a']}|{pair['path_b']}"
    elif decision == "ignore":
        pass
    else:
        raise ValueError(f"{row['folder']}: invalid decision {decision!r}")

    return action


def write_summary(path, actions):
    by_type = Counter(row["type"] for row in actions)
    by_decision = Counter(row["decision"] for row in actions)
    removes = sorted({row["remove_path"] for row in actions if row["remove_path"]})
    groups = [row for row in actions if row["group_paths"]]

    lines = [
        "# Duplicate Review Summary",
        "",
        "Source: `outputs/duplicate_review/duplicate_review.csv`",
        "",
        "## Counts",
        "",
        f"- Review rows: `{len(actions)}`",
        f"- Unique remove paths: `{len(removes)}`",
        f"- Group-only pairs: `{len(groups)}`",
        "",
        "By type:",
        "",
    ]
    lines += [f"- `{key}`: `{by_type[key]}`" for key in sorted(by_type)]
    lines += ["", "By decision:", ""]
    lines += [f"- `{key}`: `{by_decision[key]}`" for key in sorted(by_decision)]
    lines += [
        "",
        "## Next Action",
        "",
        "Inspect `outputs/duplicate_review/duplicate_review_actions.csv` before applying removals to the kept manifest.",
        "",
    ]

    path.write_text("\n".join(lines))


def main():
    review_rows = read_csv(Path("outputs/duplicate_review/duplicate_review.csv"))
    pairs = build_pairs()
    allowed = {"a", "b", "both", "ignore"}
    errors = []
    actions = []

    for row in review_rows:
        decision = row["decision"].strip()
        if row["folder"] not in pairs:
            errors.append(f"unknown folder: {row['folder']}")
            continue
        if decision not in allowed:
            errors.append(f"{row['folder']}: invalid decision {decision!r}")
            continue
        actions.append(make_action(row, pairs[row["folder"]]))

    missing = sorted(set(pairs) - {row["folder"] for row in review_rows})
    if missing:
        errors.append(f"missing review folders: {len(missing)}")

    if errors:
        raise SystemExit("Duplicate review validation failed:\n" + "\n".join(f"- {e}" for e in errors))

    out = Path("outputs/duplicate_review")
    write_csv(
        out / "duplicate_review_actions.csv",
        actions,
        ["folder", "type", "decision", "keep_path", "remove_path", "group_paths"],
    )
    write_summary(out / "duplicate_review_summary.md", actions)

    print(f"review rows: {len(actions)}")
    print(f"unique remove paths: {len({row['remove_path'] for row in actions if row['remove_path']})}")
    print(f"group-only pairs: {sum(bool(row['group_paths']) for row in actions)}")
    print("wrote: outputs/duplicate_review/duplicate_review_actions.csv")


if __name__ == "__main__":
    main()
