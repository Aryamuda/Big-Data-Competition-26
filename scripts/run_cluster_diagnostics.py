#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models


LABELS = ("0", "1", "2")
EMBEDDING_MANIFEST_FIELDS = [
    "embedding_row",
    "path",
    "source_manifest",
    "valid_embedding",
    "embedding_error",
]


class ImageManifestDataset(Dataset):
    def __init__(self, rows, transform, input_size):
        self.rows = rows
        self.transform = transform
        self.empty = torch.zeros(3, input_size, input_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path = self.rows[index]["path"]
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
            return {"index": index, "path": path, "image": tensor, "error": ""}
        except Exception as exc:
            return {"index": index, "path": path, "image": self.empty, "error": str(exc)}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ints(value):
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def require_file(path):
    if not path.exists():
        raise SystemExit(f"missing required input: {path}")


def validate_manifest(path, rows):
    if not rows:
        raise SystemExit(f"empty manifest: {path}")

    required = {"path", "current_label", "class_name"}
    missing_columns = sorted(required - set(rows[0]))
    if missing_columns:
        raise SystemExit(f"{path} missing columns: {', '.join(missing_columns)}")

    bad_labels = sorted({row["current_label"] for row in rows} - set(LABELS))
    if bad_labels:
        raise SystemExit(f"{path} has invalid labels: {bad_labels}")

    missing_files = [row["path"] for row in rows if not Path(row["path"]).is_file()]
    if missing_files:
        examples = "\n".join(f"- {path}" for path in missing_files[:20])
        raise SystemExit(f"{path} references missing files: {len(missing_files)}\n{examples}")


def get_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_resnet50(input_size):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    try:
        transform = weights.transforms(crop_size=input_size)
    except TypeError:
        transform = weights.transforms()
    encoder = torch.nn.Sequential(*list(model.children())[:-1])
    encoder.eval()
    return encoder, transform, {
        "encoder_name": "torchvision.models.resnet50",
        "encoder_variant": f"ResNet50_Weights.{weights.name}",
        "embedding_dim": 2048,
        "preprocessing": repr(transform),
        "input_resolution": input_size,
        "image_conversion_policy": "PIL open -> RGB -> torchvision ImageNet preprocessing",
    }


def l2_normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def cache_matches(rows, manifest_rows, embeddings):
    if len(rows) != len(manifest_rows):
        return False
    if [row["path"] for row in rows] != [row["path"] for row in manifest_rows]:
        return False
    valid_count = sum(row["valid_embedding"] == "1" for row in manifest_rows)
    return embeddings.shape[0] == valid_count


def load_cached_embeddings(rows, embedding_path, manifest_path):
    embeddings = np.load(embedding_path)
    manifest_rows = read_csv(manifest_path)
    if not cache_matches(rows, manifest_rows, embeddings):
        raise SystemExit(
            f"stale embedding cache for {embedding_path}; rerun with --force-embeddings"
        )
    return embeddings, manifest_rows


def extract_embeddings(rows, source_manifest, embedding_path, manifest_path, encoder, transform, args, device):
    dataset = ImageManifestDataset(rows, transform, args.input_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    encoder.to(device)
    embeddings = []
    manifest_rows = []
    next_embedding_row = 0
    processed = 0
    total = len(rows)

    with torch.inference_mode():
        for batch in loader:
            errors = list(batch["error"])
            paths = list(batch["path"])
            images = batch["image"]
            valid_positions = [index for index, error in enumerate(errors) if not error]
            batch_embeddings = {}

            if valid_positions:
                valid_images = images[valid_positions].to(device, non_blocking=True)
                features = encoder(valid_images).flatten(1).cpu().numpy().astype(np.float32)
                for local_index, feature in zip(valid_positions, features):
                    batch_embeddings[local_index] = feature

            for local_index, path in enumerate(paths):
                error = errors[local_index]
                if error:
                    manifest_rows.append({
                        "embedding_row": "",
                        "path": path,
                        "source_manifest": source_manifest.as_posix(),
                        "valid_embedding": "0",
                        "embedding_error": error,
                    })
                    continue

                embeddings.append(batch_embeddings[local_index])
                manifest_rows.append({
                    "embedding_row": next_embedding_row,
                    "path": path,
                    "source_manifest": source_manifest.as_posix(),
                    "valid_embedding": "1",
                    "embedding_error": "",
                })
                next_embedding_row += 1

            processed += len(paths)
            if processed % (args.batch_size * 20) == 0 or processed == total:
                print(f"embedded: {processed}/{total}", flush=True)

    if not embeddings:
        raise SystemExit(f"no valid embeddings extracted from {source_manifest}")

    embedding_array = l2_normalize(np.vstack(embeddings).astype(np.float32))
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path, embedding_array)
    write_csv(manifest_path, manifest_rows, EMBEDDING_MANIFEST_FIELDS)
    return embedding_array, manifest_rows


def get_embeddings(rows, source_manifest, split_name, encoder, transform, args, device):
    embedding_path = args.out_dir / f"{split_name}_embeddings.npy"
    manifest_path = args.out_dir / f"{split_name}_embedding_manifest.csv"
    if embedding_path.exists() and manifest_path.exists() and not args.force_embeddings:
        print(f"loading cached {split_name} embeddings")
        return load_cached_embeddings(rows, embedding_path, manifest_path)

    print(f"extracting {split_name} embeddings")
    return extract_embeddings(
        rows,
        source_manifest,
        embedding_path,
        manifest_path,
        encoder,
        transform,
        args,
        device,
    )


def valid_source_rows(source_rows, embedding_manifest_rows):
    return [
        source_row
        for source_row, embedding_row in zip(source_rows, embedding_manifest_rows)
        if embedding_row["valid_embedding"] == "1"
    ]


def fixed_sample_indices(total, sample_size, seed):
    size = min(total, sample_size)
    rng = np.random.default_rng(seed)
    if size == total:
        return np.arange(total)
    return np.sort(rng.choice(total, size=size, replace=False))


def pairwise_ari(labels_by_run):
    scores = []
    for left_index, left in enumerate(labels_by_run):
        for right in labels_by_run[left_index + 1:]:
            scores.append(adjusted_rand_score(left, right))
    if not scores:
        return 1.0, 1.0
    return float(np.mean(scores)), float(np.min(scores))


def run_k_diagnostics(embeddings, args):
    sample_indices = fixed_sample_indices(
        len(embeddings),
        args.silhouette_sample_size,
        args.silhouette_sample_seed,
    )
    rows = []

    for k in args.candidate_k:
        if k <= 1 or k >= len(embeddings):
            raise SystemExit(f"invalid candidate k={k} for {len(embeddings)} embeddings")

        labels_by_run = []
        inertias = []
        for seed in args.stability_seeds:
            kmeans = KMeans(
                n_clusters=k,
                random_state=seed,
                n_init=args.diagnostic_n_init,
                max_iter=args.max_iter,
                algorithm="lloyd",
            )
            labels = kmeans.fit_predict(embeddings)
            labels_by_run.append(labels)
            inertias.append(float(kmeans.inertia_))

        reference_labels = labels_by_run[0]
        sample_labels = reference_labels[sample_indices]
        if len(set(sample_labels)) > 1:
            silhouette = float(silhouette_score(embeddings[sample_indices], sample_labels))
        else:
            silhouette = float("nan")

        sizes = Counter(reference_labels)
        ari_mean, ari_min = pairwise_ari(labels_by_run)
        rows.append({
            "k": k,
            "inertia_mean": float(np.mean(inertias)),
            "inertia_std": float(np.std(inertias)),
            "silhouette": silhouette,
            "silhouette_sample_size": len(sample_indices),
            "silhouette_sample_seed": args.silhouette_sample_seed,
            "min_cluster_size": min(sizes.values()),
            "max_cluster_size": max(sizes.values()),
            "median_cluster_size": float(np.median(list(sizes.values()))),
            "tiny_cluster_count": sum(size < args.tiny_cluster_threshold for size in sizes.values()),
            "stability_ari_mean": ari_mean,
            "stability_ari_min": ari_min,
            "n_stability_runs": len(args.stability_seeds),
        })
        print(
            "k={k} silhouette={sil:.4f} ari_mean={ari:.4f} tiny={tiny}".format(
                k=k,
                sil=silhouette,
                ari=ari_mean,
                tiny=rows[-1]["tiny_cluster_count"],
            ),
            flush=True,
        )

    return rows


def select_k(diagnostic_rows, selected_k, min_stability_ari):
    if selected_k is not None:
        for row in diagnostic_rows:
            if row["k"] == selected_k:
                return selected_k, "selected explicitly by --selected-k after label-free diagnostics"
        raise SystemExit(f"--selected-k {selected_k} is not in candidate K diagnostics")

    stable_rows = [
        row for row in diagnostic_rows
        if row["tiny_cluster_count"] == 0 and row["stability_ari_mean"] >= min_stability_ari
    ]
    candidates = stable_rows or diagnostic_rows

    def score(row):
        silhouette = row["silhouette"]
        if math.isnan(silhouette):
            silhouette = -1.0
        return (
            -row["tiny_cluster_count"],
            silhouette,
            row["stability_ari_mean"],
            -row["k"],
        )

    best = max(candidates, key=score)
    if stable_rows:
        rationale = (
            "auto-selected without labels: highest silhouette among candidates with "
            f"zero tiny clusters and stability_ari_mean >= {min_stability_ari}"
        )
    else:
        rationale = (
            "auto-selected without labels: no candidate met the stability/tiny-cluster "
            "gate, so the best available silhouette/stability/tiny-cluster tradeoff was used"
        )
    return best["k"], rationale


def format_float(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.6f}" if isinstance(value, float) else value


def write_k_diagnostics(out_dir, rows):
    fields = [
        "k",
        "inertia_mean",
        "inertia_std",
        "silhouette",
        "silhouette_sample_size",
        "silhouette_sample_seed",
        "min_cluster_size",
        "max_cluster_size",
        "median_cluster_size",
        "tiny_cluster_count",
        "stability_ari_mean",
        "stability_ari_min",
        "n_stability_runs",
    ]
    formatted = [
        {field: format_float(row[field]) for field in fields}
        for row in rows
    ]
    write_csv(out_dir / "k_diagnostics.csv", formatted, fields)


def cluster_id_width(k):
    return max(2, len(str(k - 1)))


def make_cluster_id(index, k):
    return f"C{index:0{cluster_id_width(k)}d}"


def fit_final_kmeans(embeddings, k, args):
    kmeans = KMeans(
        n_clusters=k,
        random_state=args.fit_seed,
        n_init=args.final_n_init,
        max_iter=args.max_iter,
        algorithm="lloyd",
    )
    kmeans.fit(embeddings)
    return kmeans


def remap_clusters(train_labels, k):
    sizes = Counter(train_labels)
    old_labels = sorted(range(k), key=lambda label: (-sizes[label], label))
    return {old_label: new_label for new_label, old_label in enumerate(old_labels)}, old_labels


def make_assignments(rows, old_labels, distances, old_to_new, cluster_version, k):
    assignments = []
    for row, old_label, distance in zip(rows, old_labels, distances):
        new_label = old_to_new[int(old_label)]
        assignments.append({
            "path": row["path"],
            "label": row["current_label"],
            "cluster_id": make_cluster_id(new_label, k),
            "distance_to_centroid": f"{float(distance):.6f}",
            "cluster_system_version": cluster_version,
        })
    return assignments


def label_composition(assignments):
    cluster_counts = Counter(row["cluster_id"] for row in assignments)
    counts = Counter((row["cluster_id"], row["label"]) for row in assignments)
    rows = []
    for cluster_id, label in sorted(counts):
        count = counts[(cluster_id, label)]
        rows.append({
            "cluster_id": cluster_id,
            "label": label,
            "count": count,
            "proportion_within_cluster": f"{count / cluster_counts[cluster_id]:.6f}",
        })
    return rows


def label_cluster_distribution(assignments):
    label_counts = Counter(row["label"] for row in assignments)
    counts = Counter((row["label"], row["cluster_id"]) for row in assignments)
    rows = []
    for label, cluster_id in sorted(counts):
        count = counts[(label, cluster_id)]
        rows.append({
            "label": label,
            "cluster_id": cluster_id,
            "count": count,
            "proportion_within_label": f"{count / label_counts[label]:.6f}",
        })
    return rows


def entropy_from_counts(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        proportion = count / total
        entropy -= proportion * math.log(proportion)
    return entropy


def cluster_summary(train_assignments, val_assignments, k):
    train_by_cluster = defaultdict(list)
    val_counts = Counter(row["cluster_id"] for row in val_assignments)
    for row in train_assignments:
        train_by_cluster[row["cluster_id"]].append(row)

    rows = []
    for index in range(k):
        cluster_id = make_cluster_id(index, k)
        train_rows = train_by_cluster[cluster_id]
        label_counts = Counter(row["label"] for row in train_rows)
        distance_values = [float(row["distance_to_centroid"]) for row in train_rows]
        dominant_label = ""
        dominant_proportion = ""
        if label_counts:
            dominant_label, dominant_count = sorted(
                label_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[0]
            dominant_proportion = f"{dominant_count / len(train_rows):.6f}"

        rows.append({
            "cluster_id": cluster_id,
            "train_count": len(train_rows),
            "val_count": val_counts[cluster_id],
            "dominant_train_label": dominant_label,
            "dominant_train_label_proportion": dominant_proportion,
            "label_entropy": f"{entropy_from_counts(label_counts):.6f}",
            "assigned_centroid_distance_median": (
                f"{float(np.median(distance_values)):.6f}" if distance_values else ""
            ),
            "assigned_centroid_distance_p95": (
                f"{float(np.percentile(distance_values, 95)):.6f}" if distance_values else ""
            ),
            "notes": "",
        })
    return rows


def representative_images(train_assignments, val_assignments, k, per_cluster):
    rows = []
    by_split = {"train": train_assignments, "val": val_assignments}
    for split, assignments in by_split.items():
        by_cluster = defaultdict(list)
        for row in assignments:
            by_cluster[row["cluster_id"]].append(row)

        for index in range(k):
            cluster_id = make_cluster_id(index, k)
            cluster_rows = by_cluster[cluster_id]
            if not cluster_rows:
                continue
            ordered = sorted(cluster_rows, key=lambda row: float(row["distance_to_centroid"]))
            selections = [
                ("nearest", ordered[:per_cluster]),
                ("farthest", list(reversed(ordered[-per_cluster:]))),
            ]
            for representative_type, selected_rows in selections:
                for rank, row in enumerate(selected_rows, start=1):
                    rows.append({
                        "cluster_id": cluster_id,
                        "split": split,
                        "path": row["path"],
                        "label": row["label"],
                        "distance_to_centroid": row["distance_to_centroid"],
                        "rank": rank,
                        "representative_type": representative_type,
                    })
    return rows


def validation_coverage(train_assignments, val_assignments, k):
    train_counts = Counter(row["cluster_id"] for row in train_assignments)
    val_counts = Counter(row["cluster_id"] for row in val_assignments)
    total_train = len(train_assignments)
    total_val = len(val_assignments)
    rows = []

    for index in range(k):
        cluster_id = make_cluster_id(index, k)
        train_count = train_counts[cluster_id]
        val_count = val_counts[cluster_id]
        train_proportion = train_count / total_train if total_train else 0.0
        val_proportion = val_count / total_val if total_val else 0.0

        if val_count == 0:
            note = "missing_in_val"
        elif train_proportion and val_proportion < 0.5 * train_proportion:
            note = "underrepresented_in_val"
        elif train_proportion and val_proportion > 2.0 * train_proportion:
            note = "overrepresented_in_val"
        else:
            note = "ok"

        rows.append({
            "cluster_id": cluster_id,
            "train_count": train_count,
            "train_proportion": f"{train_proportion:.6f}",
            "val_count": val_count,
            "val_proportion": f"{val_proportion:.6f}",
            "coverage_note": note,
        })
    return rows


def write_assignment_artifacts(args, train_assignments, val_assignments, k):
    assignment_fields = [
        "path",
        "label",
        "cluster_id",
        "distance_to_centroid",
        "cluster_system_version",
    ]
    write_csv(args.out_dir / "train_cluster_assignments.csv", train_assignments, assignment_fields)
    write_csv(args.out_dir / "val_cluster_assignments.csv", val_assignments, assignment_fields)
    write_csv(
        args.out_dir / "cluster_label_composition.csv",
        label_composition(train_assignments),
        ["cluster_id", "label", "count", "proportion_within_cluster"],
    )
    write_csv(
        args.out_dir / "label_cluster_distribution.csv",
        label_cluster_distribution(train_assignments),
        ["label", "cluster_id", "count", "proportion_within_label"],
    )
    write_csv(
        args.out_dir / "cluster_summary.csv",
        cluster_summary(train_assignments, val_assignments, k),
        [
            "cluster_id",
            "train_count",
            "val_count",
            "dominant_train_label",
            "dominant_train_label_proportion",
            "label_entropy",
            "assigned_centroid_distance_median",
            "assigned_centroid_distance_p95",
            "notes",
        ],
    )
    write_csv(
        args.out_dir / "cluster_representative_images.csv",
        representative_images(
            train_assignments,
            val_assignments,
            k,
            args.representatives_per_cluster,
        ),
        [
            "cluster_id",
            "split",
            "path",
            "label",
            "distance_to_centroid",
            "rank",
            "representative_type",
        ],
    )
    write_csv(
        args.out_dir / "validation_coverage.csv",
        validation_coverage(train_assignments, val_assignments, k),
        ["cluster_id", "train_count", "train_proportion", "val_count", "val_proportion", "coverage_note"],
    )


def library_versions():
    return {
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "python_torch": torch.__version__,
        "scikit_learn": sklearn.__version__,
        "torchvision": torchvision.__version__,
    }


def write_config(args, encoder_meta, selected_k, selected_k_rationale, cluster_version):
    train_fingerprint = sha256_file(args.train_manifest)
    val_fingerprint = sha256_file(args.val_manifest)
    config = {
        "cluster_system_version": cluster_version,
        "created_at": now_utc(),
        "source_train_manifest": args.train_manifest.as_posix(),
        "source_train_manifest_fingerprint": train_fingerprint,
        "source_val_manifest": args.val_manifest.as_posix(),
        "source_val_manifest_fingerprint": val_fingerprint,
        "class_names": {"0": "Recyclable", "1": "Electronic", "2": "Organic"},
        **encoder_meta,
        "device": str(get_device(args.device)),
        "embedding_normalization": "L2 normalization after frozen encoder extraction",
        "embedding_transformation": "none",
        "clustering_method": "sklearn.cluster.KMeans",
        "k": selected_k,
        "candidate_k_range": list(args.candidate_k),
        "candidate_k_range_rationale": (
            f"Declared before diagnostics: {args.candidate_k} on the train split gives "
            "coarse visual neighborhoods with hundreds to low thousands of images per "
            "cluster while keeping KMeans diagnostics lightweight."
        ),
        "tiny_cluster_threshold": args.tiny_cluster_threshold,
        "selected_k_rationale": selected_k_rationale,
        "random_state": args.fit_seed,
        "fit_seed": args.fit_seed,
        "init_policy": (
            f"diagnostic_n_init={args.diagnostic_n_init}; "
            f"final_n_init={args.final_n_init}"
        ),
        "n_init": args.final_n_init,
        "diagnostic_n_init": args.diagnostic_n_init,
        "final_n_init": args.final_n_init,
        "max_iter": args.max_iter,
        "stability_seeds": list(args.stability_seeds),
        "silhouette_sample_size": args.silhouette_sample_size,
        "silhouette_sample_seed": args.silhouette_sample_seed,
        "library_versions": library_versions(),
        "outputs": {
            "train_embeddings": (args.out_dir / "train_embeddings.npy").as_posix(),
            "val_embeddings": (args.out_dir / "val_embeddings.npy").as_posix(),
            "train_embedding_manifest": (args.out_dir / "train_embedding_manifest.csv").as_posix(),
            "val_embedding_manifest": (args.out_dir / "val_embedding_manifest.csv").as_posix(),
            "k_diagnostics": (args.out_dir / "k_diagnostics.csv").as_posix(),
            "centroids": (args.out_dir / "centroids.npy").as_posix(),
            "train_cluster_assignments": (args.out_dir / "train_cluster_assignments.csv").as_posix(),
            "val_cluster_assignments": (args.out_dir / "val_cluster_assignments.csv").as_posix(),
        },
    }
    (args.out_dir / "cluster_config.json").write_text(json.dumps(config, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run canonical embedding-space cluster diagnostics.")
    parser.add_argument("--train-manifest", type=Path, default=Path("outputs/splits/train_manifest.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("outputs/splits/val_manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cluster_diagnostics"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--candidate-k", type=parse_ints, default=parse_ints("12,16,20,24,32"))
    parser.add_argument("--selected-k", type=int)
    parser.add_argument("--stability-seeds", type=parse_ints, default=parse_ints("63,64,65"))
    parser.add_argument("--fit-seed", type=int, default=63)
    parser.add_argument("--diagnostic-n-init", type=int, default=1)
    parser.add_argument("--final-n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--silhouette-sample-size", type=int, default=5000)
    parser.add_argument("--silhouette-sample-seed", type=int, default=63)
    parser.add_argument("--tiny-cluster-threshold", type=int, default=25)
    parser.add_argument("--min-stability-ari", type=float, default=0.75)
    parser.add_argument("--representatives-per-cluster", type=int, default=8)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()

    for path in (args.train_manifest, args.val_manifest):
        require_file(path)

    train_rows = read_csv(args.train_manifest)
    val_rows = read_csv(args.val_manifest)
    validate_manifest(args.train_manifest, train_rows)
    validate_manifest(args.val_manifest, val_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    encoder, transform, encoder_meta = load_resnet50(args.input_size)
    print(f"device: {device}")
    print(f"encoder: {encoder_meta['encoder_variant']}")

    train_embeddings, train_embedding_manifest = get_embeddings(
        train_rows,
        args.train_manifest,
        "train",
        encoder,
        transform,
        args,
        device,
    )
    val_embeddings, val_embedding_manifest = get_embeddings(
        val_rows,
        args.val_manifest,
        "val",
        encoder,
        transform,
        args,
        device,
    )

    valid_train_rows = valid_source_rows(train_rows, train_embedding_manifest)
    valid_val_rows = valid_source_rows(val_rows, val_embedding_manifest)
    if len(valid_train_rows) != len(train_embeddings):
        raise SystemExit("train embedding manifest does not match embedding array")
    if len(valid_val_rows) != len(val_embeddings):
        raise SystemExit("val embedding manifest does not match embedding array")

    diagnostic_rows = run_k_diagnostics(train_embeddings, args)
    write_k_diagnostics(args.out_dir, diagnostic_rows)
    if args.diagnostics_only:
        print(f"wrote: {args.out_dir / 'k_diagnostics.csv'}")
        print("diagnostics-only mode: final cluster system was not fit")
        return

    selected_k, selected_k_rationale = select_k(
        diagnostic_rows,
        args.selected_k,
        args.min_stability_ari,
    )
    print(f"selected k: {selected_k}")

    kmeans = fit_final_kmeans(train_embeddings, selected_k, args)
    old_to_new, old_label_order = remap_clusters(kmeans.labels_, selected_k)
    train_distances_all = kmeans.transform(train_embeddings)
    train_distances = train_distances_all[np.arange(len(train_embeddings)), kmeans.labels_]
    val_old_labels = kmeans.predict(val_embeddings)
    val_distances_all = kmeans.transform(val_embeddings)
    val_distances = val_distances_all[np.arange(len(val_embeddings)), val_old_labels]

    train_fingerprint = sha256_file(args.train_manifest)
    cluster_version = (
        f"bdc2026_resnet50_l2_k{selected_k}_"
        f"seed{args.fit_seed}_{train_fingerprint[:8]}"
    )

    sorted_centroids = kmeans.cluster_centers_[old_label_order].astype(np.float32)
    np.save(args.out_dir / "centroids.npy", sorted_centroids)

    train_assignments = make_assignments(
        valid_train_rows,
        kmeans.labels_,
        train_distances,
        old_to_new,
        cluster_version,
        selected_k,
    )
    val_assignments = make_assignments(
        valid_val_rows,
        val_old_labels,
        val_distances,
        old_to_new,
        cluster_version,
        selected_k,
    )
    write_assignment_artifacts(args, train_assignments, val_assignments, selected_k)
    write_config(args, encoder_meta, selected_k, selected_k_rationale, cluster_version)

    print(f"valid train embeddings: {len(train_embeddings)}")
    print(f"valid val embeddings: {len(val_embeddings)}")
    print(f"cluster system version: {cluster_version}")
    print(f"wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
