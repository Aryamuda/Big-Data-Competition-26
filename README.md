# BDC 2026 Image Classification

This repository contains the code workflow for the BDC Satria Data 2026 image-classification task.

The dataset and private review artifacts are not stored in this repository.

## Task

Class labels:

- `0 = Recyclable`
- `1 = Electronic`
- `2 = Organic`

Primary metric:

- Macro F1

## Data Policy

The original dataset must not be modified in place.

Expected private data layout:

```text
train/
├── 0_Recyclable/
├── 1_Electronic/
└── 2_Organic/

test/
submission.csv
```

The `train/`, `test/`, review batches, generated outputs, and review/cleaning manifests are intentionally ignored by Git.

## Private Manifest Policy

Cleaning decisions are applied through CSV manifests, not by deleting or editing files inside `train/`.

Private manifest files are expected to live outside Git, usually in Google Drive.

This matters because the manifest is the real dataset-filtering key:

- reviewed keep/remove/unsure decisions
- excluded noisy samples
- unsure samples tracked for later analysis
- fixed train/validation split manifests

Scripts should be written so they can run in two modes:

1. manifest-filtered mode, using private CSV manifests from Drive
2. raw-train fallback mode, using the original `train/` folders directly

If a required private manifest is missing, the script must print a clear warning before falling back to raw `train/`.

## Execution Workflow

Local machine:

- lightweight audits
- review CSV validation
- manifest generation
- split logic
- submission validation

Google Colab:

- heavier training
- model evaluation
- inference runs

Typical Colab setup:

```text
GitHub repo -> Python scripts
Google Drive -> images and private manifests
```

All paths should be configurable. Do not hardcode local machine paths.

## Current Workflow Order

1. Validate dataset and submission contract.
2. Complete human review CSVs.
3. Validate returned review CSVs.
4. Merge review decisions.
5. Create private keep/remove/unsure manifests.
6. Re-audit kept data.
7. Create a fixed leakage-safe validation split.
8. Train a simple pretrained baseline.
9. Diagnose errors before adding complexity.
10. Validate submission files before submitting.

## Submission Rule

Do not infer test order from filesystem sorting.

Always read `submission.csv` and write predictions in that exact row order.

Expected inference flow:

```text
read submission.csv
for each id in template order
load test/{id}.jpg
predict
write predicted label to the same row
validate output
```

## What Is Not In This Repo

Ignored/private:

- original train images
- original test images
- sample submission files
- review batch image copies
- filled review CSVs
- cleaning manifests
- split manifests
- generated audit outputs
- model checkpoints
- experiment logs
- private notes and learning docs

Tracked/public:

- reusable scripts
- README
- lightweight project configuration
