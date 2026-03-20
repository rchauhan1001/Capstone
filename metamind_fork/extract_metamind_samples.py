"""
Extracts SocialIQA samples that were processed by MetaMind,
and writes them to a new .jsonl + labels file for base model inference.

Usage:
    python extract_metamind_samples.py
"""

import json
import os

METAMIND_RESULTS_PATH = "/Users/eitan/Documents/School related/Capstone Project/metamind_fork/results/all_results_combined_20260317_153401.jsonl"
SOCIALIQA_DIR         = "/Users/eitan/Documents/School related/Capstone Project/metamind_fork/socialiqa-train-dev"
TRAIN_SAMPLES_PATH    = os.path.join(SOCIALIQA_DIR, "train.jsonl")
TRAIN_LABELS_PATH     = os.path.join(SOCIALIQA_DIR, "train-labels.lst")

OUTPUT_DIR            = "/Users/eitan/Documents/School related/Capstone Project/metamind_fork/socialiqa-train-dev"
OUTPUT_SAMPLES_PATH   = os.path.join(OUTPUT_DIR, "basemodel_samples.jsonl")
OUTPUT_LABELS_PATH    = os.path.join(OUTPUT_DIR, "basemodel_samples_labels.lst")


# --- Step 1: Collect valid sample_ids from MetaMind results ---
valid_ids = set()
skipped   = 0

with open(METAMIND_RESULTS_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            if "error" in record or not record.get("correct", False):
                skipped += 1
                continue
            valid_ids.add(record["sample_id"])
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Warning: skipping malformed record: {e}")
            skipped += 1

print(f"Found {len(valid_ids)} valid MetaMind sample IDs (skipped {skipped} errored/malformed)")

# --- Step 2: Load all train samples and labels (1-indexed) ---
all_samples = []
with open(TRAIN_SAMPLES_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            all_samples.append(json.loads(line))

all_labels = []
with open(TRAIN_LABELS_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            all_labels.append(line)  # keep as raw string "1", "2", or "3"

assert len(all_samples) == len(all_labels), \
    f"Mismatch: {len(all_samples)} samples vs {len(all_labels)} labels"

print(f"Loaded {len(all_samples)} total train samples")

# --- Step 3: Write matching samples and labels ---
written = 0
missing = 0

with open(OUTPUT_SAMPLES_PATH, "w") as fout_s, \
     open(OUTPUT_LABELS_PATH, "w") as fout_l:

    for sample_id in sorted(valid_ids):
        idx = sample_id - 1  # convert 1-based to 0-based
        if idx < 0 or idx >= len(all_samples):
            print(f"Warning: sample_id {sample_id} out of range, skipping")
            missing += 1
            continue
        fout_s.write(json.dumps(all_samples[idx], ensure_ascii=False) + "\n")
        fout_l.write(all_labels[idx] + "\n")
        written += 1

print(f"\nDone. {written} samples written to:")
print(f"  Samples: {OUTPUT_SAMPLES_PATH}")
print(f"  Labels:  {OUTPUT_LABELS_PATH}")
if missing:
    print(f"  Missing: {missing} sample IDs out of range")
if skipped:
    print(f"  Skipped: {skipped} errored/malformed MetaMind records")