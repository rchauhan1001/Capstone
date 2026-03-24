"""
Quick diagnostic: compare context+question between base model and MetaMind
at matching sample_ids to see if they align.
"""

import json
import glob
import os

RESULTS_DIR = "/Users/eitan/Documents/School related/Capstone Project/metamind_fork/results"

# Load base model samples
base_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "socialiqa_basemodel_results_*.jsonl")))
base = {}
for f in base_files:
    for line in open(f):
        s = json.loads(line.strip())
        base[s["sample_id"]] = s

# Load MetaMind samples
mm = {}
for line in open(os.path.join(RESULTS_DIR, "all_results_combined_current.jsonl")):
    s = json.loads(line.strip())
    mm[s["sample_id"]] = s

shared = sorted(set(base) & set(mm))
match, mismatch = 0, 0

print(f"Base unique IDs: {len(base)}")
print(f"MetaMind unique IDs: {len(mm)}")
print(f"Shared IDs: {len(shared)}\n")

for sid in shared:
    b_ctx = base[sid]["input"]["context"]
    m_ctx = mm[sid]["input"]["context"]
    if b_ctx == m_ctx:
        match += 1
    else:
        mismatch += 1
        if mismatch <= 10:
            print(f"MISMATCH id={sid}")
            print(f"  Base:     {b_ctx[:100]}")
            print(f"  MetaMind: {m_ctx[:100]}")
            print()

print(f"\nMatch: {match} | Mismatch: {mismatch} | Total: {len(shared)}")