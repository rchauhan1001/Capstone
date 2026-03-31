"""
Measure token lengths the same way TRL's DPOTrainer will see them.
Run this BEFORE training to check if max_length / max_prompt_length are clipping your data.

Usage:
    python measure_lengths.py
"""

import json
import numpy as np
from transformers import AutoTokenizer

# ─── Match your training script exactly ─────────────────────────────────────

BASE_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"  # downloads from HF (tokenizer only ~few MB)
DPO_DATASET_PATH = "/Users/eitan/Documents/School related/Capstone Project/training_phase/reinforcement_learning/dpo_data/dpo_dataset.jsonl"

MAX_LENGTH = 1024        # from your train_dpo.py
MAX_PROMPT_LENGTH = 512  # from your train_dpo.py

# ─── Load tokenizer ─────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─── Tokenize the way TRL does ──────────────────────────────────────────────
# DPOTrainer builds:
#   prompt_text = apply_chat_template(prompt_messages, add_generation_prompt=True)
#   chosen_text = apply_chat_template(prompt_messages + chosen_messages)
#   rejected_text = apply_chat_template(prompt_messages + rejected_messages)
# Then it tokenizes each and uses len(prompt_tokens), len(full_chosen_tokens), etc.

prompt_lengths = []
chosen_full_lengths = []
rejected_full_lengths = []
chosen_completion_lengths = []
rejected_completion_lengths = []

clipped_by_max_length = 0
clipped_by_prompt_length = 0

with open(DPO_DATASET_PATH, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]

for i, sample in enumerate(samples):
    prompt_msgs = sample["prompt"]         # list of {"role":..., "content":...}
    chosen_msgs = sample["chosen"]         # list of {"role": "assistant", "content":...}
    rejected_msgs = sample["rejected"]

    # Prompt only (with generation prompt appended, as TRL does)
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer.encode(prompt_text)

    # Full chosen sequence: prompt + chosen completion
    chosen_full_text = tokenizer.apply_chat_template(
        prompt_msgs + chosen_msgs, tokenize=False
    )
    chosen_full_tokens = tokenizer.encode(chosen_full_text)

    # Full rejected sequence: prompt + rejected completion
    rejected_full_text = tokenizer.apply_chat_template(
        prompt_msgs + rejected_msgs, tokenize=False
    )
    rejected_full_tokens = tokenizer.encode(rejected_full_text)

    p_len = len(prompt_tokens)
    c_len = len(chosen_full_tokens)
    r_len = len(rejected_full_tokens)

    prompt_lengths.append(p_len)
    chosen_full_lengths.append(c_len)
    rejected_full_lengths.append(r_len)
    chosen_completion_lengths.append(c_len - p_len)
    rejected_completion_lengths.append(r_len - p_len)

    if c_len > MAX_LENGTH or r_len > MAX_LENGTH:
        clipped_by_max_length += 1
    if p_len > MAX_PROMPT_LENGTH:
        clipped_by_prompt_length += 1

# ─── Report ──────────────────────────────────────────────────────────────────

def stats(name, lengths):
    arr = np.array(lengths)
    print(f"\n{name} (tokens):")
    print(f"  Min:    {arr.min()}")
    print(f"  Mean:   {arr.mean():.0f}")
    print(f"  Median: {np.median(arr):.0f}")
    print(f"  P90:    {np.percentile(arr, 90):.0f}")
    print(f"  P95:    {np.percentile(arr, 95):.0f}")
    print(f"  P99:    {np.percentile(arr, 99):.0f}")
    print(f"  Max:    {arr.max()}")

n = len(samples)
print("=" * 60)
print(f"DPO Dataset Token Length Analysis  ({n} samples)")
print(f"Current settings: max_length={MAX_LENGTH}, max_prompt_length={MAX_PROMPT_LENGTH}")
print("=" * 60)

stats("Prompt", prompt_lengths)
stats("Chosen full (prompt+completion)", chosen_full_lengths)
stats("Rejected full (prompt+completion)", rejected_full_lengths)
stats("Chosen completion only", chosen_completion_lengths)
stats("Rejected completion only", rejected_completion_lengths)

print("\n" + "=" * 60)
print("TRUNCATION CHECK")
print("=" * 60)
print(f"  Samples where prompt > {MAX_PROMPT_LENGTH}:        {clipped_by_prompt_length}/{n} ({100*clipped_by_prompt_length/n:.1f}%)")
print(f"  Samples where chosen OR rejected > {MAX_LENGTH}: {clipped_by_max_length}/{n} ({100*clipped_by_max_length/n:.1f}%)")

if clipped_by_max_length > 0:
    p99_chosen = int(np.percentile(chosen_full_lengths, 99))
    p95_chosen = int(np.percentile(chosen_full_lengths, 95))
    print(f"\n  ⚠️  {clipped_by_max_length} samples will be truncated!")
    print(f"  To cover 95% of chosen: set max_length >= {p95_chosen}")
    print(f"  To cover 99% of chosen: set max_length >= {p99_chosen}")
else:
    print(f"\n  ✅ No truncation — your current max_length={MAX_LENGTH} covers all samples.")

if clipped_by_prompt_length > 0:
    p99_prompt = int(np.percentile(prompt_lengths, 99))
    print(f"\n  ⚠️  {clipped_by_prompt_length} prompts exceed max_prompt_length!")
    print(f"  To cover 99% of prompts: set max_prompt_length >= {p99_prompt}")
else:
    print(f"  ✅ No prompt truncation — max_prompt_length={MAX_PROMPT_LENGTH} covers all samples.")