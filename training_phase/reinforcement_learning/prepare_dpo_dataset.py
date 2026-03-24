"""
DPO Dataset Preparation Script
-------------------------------
Combines base model and MetaMind results into a DPO-ready dataset
in chat format for TRL's DPOTrainer.

Matches samples by (context, question) content — NOT sample_id,
since IDs are misaligned between the two datasets.

Usage:
    python prepare_dpo_dataset.py \
        --results_dir /path/to/results \
        --output_path /path/to/dpo_dataset.jsonl
"""

import json
import glob
import re
import os
import argparse
from collections import Counter


# ─── Configuration ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an AI assistant skilled in social and cognitive reasoning. "
    "Given a social context and a question, reason carefully about the "
    "mental states, intentions, and emotions of the people involved, "
    "then provide your answer."
)

BASE_MODEL_GLOB = "socialiqa_basemodel_results_*.jsonl"
METAMIND_FILE = "all_results_combined_current.jsonl"

# Regex to strip trailing "\nANSWER: <letter>" (with optional whitespace)
ANSWER_STRIP_RE = re.compile(r"\s*\n?\s*\*{0,2}ANSWER:\s*[A-Za-z]\*{0,2}\s*$")


# ─── Helper Functions ────────────────────────────────────────────────────────

def load_jsonl(filepath):
    """Load a JSONL file, return list of dicts. Skips malformed lines."""
    samples = []
    malformed = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
                print(f"  WARNING: Malformed JSON at {filepath}:{line_num}, skipping.")
    return samples, malformed


def make_key(sample):
    """Create a matching key from prompt content."""
    return (sample["input"]["context"].strip(), sample["input"]["question"].strip())


def strip_answer_suffix(response_text):
    """Remove trailing ANSWER: <letter> from response text."""
    return ANSWER_STRIP_RE.sub("", response_text).strip()


def build_user_message(sample):
    """Build the user prompt from context + question."""
    ctx = sample["input"]["context"]
    q = sample["input"]["question"]
    return f"Context: {ctx}\n\nQuestion: {q}"


def build_chat(role_content_pairs):
    """Build a chat-format message list."""
    return [{"role": r, "content": c} for r, c in role_content_pairs]


def validate_sample(sample, source_name):
    """Validate a single sample has required fields. Returns (is_valid, issues)."""
    issues = []
    sid = sample.get("sample_id", "?")

    if "input" not in sample:
        issues.append(f"[{source_name}] id={sid}: Missing 'input' field")
    else:
        for field in ["context", "question"]:
            if field not in sample["input"]:
                issues.append(f"[{source_name}] id={sid}: Missing input.{field}")

    if "correct" not in sample:
        issues.append(f"[{source_name}] id={sid}: Missing 'correct' field")

    if "final_response" not in sample:
        issues.append(f"[{source_name}] id={sid}: Missing 'final_response'")
    elif not sample["final_response"].strip():
        issues.append(f"[{source_name}] id={sid}: Empty 'final_response'")

    return len(issues) == 0, issues


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main(results_dir, output_path):
    print("=" * 60)
    print("DPO Dataset Preparation")
    print("=" * 60)

    # ── Step 1: Load base model files ────────────────────────────────────
    base_pattern = os.path.join(results_dir, BASE_MODEL_GLOB)
    base_files = sorted(glob.glob(base_pattern))

    if not base_files:
        print(f"\nERROR: No base model files found matching: {base_pattern}")
        return

    print(f"\n[1/6] Loading base model data from {len(base_files)} file(s):")
    base_samples = []
    total_malformed = 0
    for f in base_files:
        samples, malformed = load_jsonl(f)
        total_malformed += malformed
        base_samples.extend(samples)
        print(f"  {os.path.basename(f)}: {len(samples)} samples")

    print(f"  Total base model samples: {len(base_samples)}")
    if total_malformed:
        print(f"  WARNING: {total_malformed} malformed lines skipped")

    # ── Step 2: Load MetaMind file ───────────────────────────────────────
    metamind_path = os.path.join(results_dir, METAMIND_FILE)
    if not os.path.exists(metamind_path):
        print(f"\nERROR: MetaMind file not found: {metamind_path}")
        return

    print(f"\n[2/6] Loading MetaMind data:")
    metamind_samples, mm_malformed = load_jsonl(metamind_path)
    print(f"  {METAMIND_FILE}: {len(metamind_samples)} samples")
    if mm_malformed:
        print(f"  WARNING: {mm_malformed} malformed lines skipped")

    # ── Step 3: Validate all samples ─────────────────────────────────────
    print(f"\n[3/6] Validating samples...")
    all_issues = []

    valid_base = []
    for s in base_samples:
        ok, issues = validate_sample(s, "base")
        all_issues.extend(issues)
        if ok:
            valid_base.append(s)

    valid_metamind = []
    for s in metamind_samples:
        ok, issues = validate_sample(s, "metamind")
        all_issues.extend(issues)
        if ok:
            valid_metamind.append(s)

    if all_issues:
        print(f"  Found {len(all_issues)} validation issues:")
        for issue in all_issues[:20]:
            print(f"    {issue}")
        if len(all_issues) > 20:
            print(f"    ... and {len(all_issues) - 20} more")
    else:
        print("  All samples passed validation.")

    print(f"  Valid base model samples: {len(valid_base)}")
    print(f"  Valid MetaMind samples:   {len(valid_metamind)}")

    # ── Step 4: Index by (context, question) content key ─────────────────
    print(f"\n[4/6] Indexing by (context, question) content key...")

    base_key_counts = Counter(make_key(s) for s in valid_base)
    mm_key_counts = Counter(make_key(s) for s in valid_metamind)

    base_dups = {k: cnt for k, cnt in base_key_counts.items() if cnt > 1}
    mm_dups = {k: cnt for k, cnt in mm_key_counts.items() if cnt > 1}

    if base_dups:
        print(f"  WARNING: {len(base_dups)} duplicate prompts in base model data (taking last)")
    if mm_dups:
        print(f"  WARNING: {len(mm_dups)} duplicate prompts in MetaMind data (taking last)")

    base_by_key = {make_key(s): s for s in valid_base}
    mm_by_key = {make_key(s): s for s in valid_metamind}

    print(f"  Unique base model prompts: {len(base_by_key)}")
    print(f"  Unique MetaMind prompts:   {len(mm_by_key)}")

    # ── Step 5: Match and filter ─────────────────────────────────────────
    print(f"\n[5/6] Matching samples and filtering...")

    shared_keys = set(base_by_key.keys()) & set(mm_by_key.keys())
    print(f"  Matched prompts (content match): {len(shared_keys)}")

    both_correct = []
    stats = {
        "both_correct": 0,
        "base_wrong": 0,
        "metamind_wrong": 0,
        "both_wrong": 0,
    }

    for key in sorted(shared_keys, key=lambda k: k[0]):
        b = base_by_key[key]
        m = mm_by_key[key]

        b_correct = b.get("correct", False)
        m_correct = m.get("correct", False)

        if b_correct and m_correct:
            stats["both_correct"] += 1
            both_correct.append((key, b, m))
        elif b_correct and not m_correct:
            stats["metamind_wrong"] += 1
        elif not b_correct and m_correct:
            stats["base_wrong"] += 1
        else:
            stats["both_wrong"] += 1

    print(f"  Both correct:    {stats['both_correct']}")
    print(f"  Base wrong only: {stats['base_wrong']}")
    print(f"  MetaMind wrong:  {stats['metamind_wrong']}")
    print(f"  Both wrong:      {stats['both_wrong']}")

    if not both_correct:
        print("\nERROR: No valid DPO pairs found. Check your data.")
        return

    # ── Step 6: Build DPO dataset ────────────────────────────────────────
    print(f"\n[6/6] Building DPO dataset...")

    dpo_data = []
    strip_failures = 0
    identical_skipped = 0
    empty_skipped = 0

    for key, base_sample, mm_sample in both_correct:
        user_msg = build_user_message(mm_sample)

        chosen_raw = mm_sample["final_response"]
        rejected_raw = base_sample["final_response"]

        chosen_clean = strip_answer_suffix(chosen_raw)
        rejected_clean = strip_answer_suffix(rejected_raw)

        if chosen_clean == chosen_raw:
            strip_failures += 1
        if rejected_clean == rejected_raw:
            strip_failures += 1

        if not chosen_clean or not rejected_clean:
            empty_skipped += 1
            continue

        if chosen_clean == rejected_clean:
            identical_skipped += 1
            continue

        dpo_entry = {
            "prompt": build_chat([
                ("system", SYSTEM_PROMPT),
                ("user", user_msg),
            ]),
            "chosen": build_chat([
                ("assistant", chosen_clean),
            ]),
            "rejected": build_chat([
                ("assistant", rejected_clean),
            ]),
        }
        dpo_data.append(dpo_entry)

    print(f"  Final DPO pairs:          {len(dpo_data)}")
    print(f"  Identical pairs removed:  {identical_skipped}")
    print(f"  Empty responses removed:  {empty_skipped}")
    if strip_failures:
        print(f"  NOTE: {strip_failures} responses did not have ANSWER: suffix to strip")

    # ── Write output ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dpo_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n  Saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")

    # ── Print a sample for verification ──────────────────────────────────
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT (first entry):")
    print("=" * 60)
    sample = dpo_data[0]
    print(f"\n--- PROMPT ---")
    for msg in sample["prompt"]:
        print(f"  [{msg['role']}]: {msg['content']}")
    print(f"\n--- CHOSEN (MetaMind) ---")
    print(f"  {sample['chosen'][0]['content']}")
    print(f"\n--- REJECTED (Base Model) ---")
    print(f"  {sample['rejected'][0]['content']}")

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Base model files loaded:    {len(base_files)}")
    print(f"  Total base samples:         {len(base_samples)}")
    print(f"  Total MetaMind samples:     {len(metamind_samples)}")
    print(f"  Content-matched prompts:    {len(shared_keys)}")
    print(f"  Both correct:               {stats['both_correct']}")
    print(f"  Final DPO pairs:            {len(dpo_data)}")
    print(f"  Identical pairs removed:    {identical_skipped}")
    print(f"  Empty responses removed:    {empty_skipped}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DPO dataset from base model + MetaMind results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/Users/eitan/Documents/School related/Capstone Project/metamind_fork/results",
        help="Path to the results directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/Users/eitan/Documents/School related/Capstone Project/training_phase/reinforcement_learning/dpo_data/dpo_dataset.jsonl",
        help="Path to save the output DPO dataset",
    )
    args = parser.parse_args()
    main(args.results_dir, args.output_path)