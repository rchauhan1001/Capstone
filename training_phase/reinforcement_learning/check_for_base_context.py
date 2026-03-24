"""
Check if the first 5 MetaMind contexts exist anywhere in the base model data.
"""

import json
import glob
import os

RESULTS_DIR = "/Users/eitan/Documents/School related/Capstone Project/metamind_fork/results"

# Load all base model contexts into a dict: context -> list of sample_ids
base_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "socialiqa_basemodel_results_*.jsonl")))
base_ctx_to_ids = {}
for f in base_files:
    for line in open(f):
        s = json.loads(line.strip())
        ctx = s["input"]["context"]
        base_ctx_to_ids.setdefault(ctx, []).append(s["sample_id"])

# Load first 5 MetaMind samples
mm_first5 = []
with open(os.path.join(RESULTS_DIR, "all_results_combined_current.jsonl")) as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        mm_first5.append(json.loads(line.strip()))

print(f"Base unique contexts: {len(base_ctx_to_ids)}\n")

for s in mm_first5:
    ctx = s["input"]["context"]
    mm_id = s["sample_id"]
    print(f"MetaMind id={mm_id}: {ctx[:80]}...")
    if ctx in base_ctx_to_ids:
        print(f"  FOUND in base at sample_id(s): {base_ctx_to_ids[ctx]}")
    else:
        print(f"  NOT FOUND in base data")
    print()