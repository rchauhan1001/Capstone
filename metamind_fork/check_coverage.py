import json
import os
import glob

results_dir = os.path.join(os.path.dirname(__file__), "results")

files = sorted(glob.glob(os.path.join(results_dir, "socialiqa_results_*.jsonl")))

print(f"{'File':<50} {'Range':<20} {'Count'}")
print("-" * 80)

all_ids = set()

for f in files:
    ids = []
    with open(f) as fp:
        for line in fp:
            try:
                d = json.loads(line)
                ids.append(d["sample_id"])
            except:
                pass
    if ids:
        fname = os.path.basename(f)
        print(f"{fname:<50} {min(ids):<10} - {max(ids):<10} ({len(ids)} samples)")
        all_ids.update(ids)

print("-" * 80)
if all_ids:
    print(f"\nTotal unique samples: {len(all_ids)}")
    print(f"Overall range: {min(all_ids)} - {max(all_ids)}")

    # Find gaps
    full_range = set(range(min(all_ids), max(all_ids) + 1))
    missing = sorted(full_range - all_ids)
    if missing:
        # Group consecutive missing into ranges
        gaps = []
        start = missing[0]
        prev = missing[0]
        for m in missing[1:]:
            if m != prev + 1:
                gaps.append((start, prev))
                start = m
            prev = m
        gaps.append((start, prev))
        print(f"\nGaps ({len(missing)} missing samples):")
        for g in gaps:
            print(f"  {g[0]} - {g[1]} ({g[1]-g[0]+1} samples)")
    else:
        print("\nNo gaps — full coverage!")