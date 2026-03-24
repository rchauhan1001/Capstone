"""
DPO Training Script for Qwen 2.5 7B
-------------------------------------
1. Loads Qwen 2.5 7B base model
2. Merges teammate's SFT LoRA into base weights
3. Applies a fresh LoRA for DPO training
4. Trains with TRL's DPOTrainer
5. Checkpoints every N steps + on SIGTERM (Slurm preemption)

Usage:
    accelerate launch train_dpo.py [--resume_from_checkpoint]
"""

import os
import sys
import signal
import argparse
import json
from functools import partial

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import DPOConfig, DPOTrainer


# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_MODEL_ID = "/scratch/laredo.ei/models/Qwen2.5-7B-Instruct"

# Your teammate's SFT LoRA checkpoint
SFT_LORA_PATH = "/scratch/laredo.ei/dpo_training/best_checkpoint"

# DPO dataset produced by prepare_dpo_dataset.py
DPO_DATASET_PATH = "/scratch/laredo.ei/dpo_training/dpo_data/dpo_dataset.jsonl"

# Output directory for DPO checkpoints
OUTPUT_DIR = "/scratch/laredo.ei/dpo_training/dpo_output"

# Where to save the merged SFT model (so we only merge once)
MERGED_SFT_PATH = os.path.join(OUTPUT_DIR, "merged_sft_base")

# HuggingFace cache (same as your existing setup)
os.environ["HF_HOME"] = "/scratch/laredo.ei/.cache/huggingface"


# ─── Hyperparameters ─────────────────────────────────────────────────────────

EVAL_SPLIT = 0.1          # 10% held out for evaluation
SPLIT_SEED = 42           # Reproducible split

DPO_BETA = 0.1            # KL penalty strength
LEARNING_RATE = 5e-6
NUM_EPOCHS = 1             # DPO overfits fast — start with 1
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8  # Effective batch = 2 * 8 = 16
WARMUP_RATIO = 0.1
MAX_LENGTH = 1024          # Max tokens for prompt + response
MAX_PROMPT_LENGTH = 512

# Checkpointing
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3       # Keep last 3 checkpoints + best
EVAL_STEPS = 200
LOGGING_STEPS = 10

# LoRA config for DPO
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ─── Signal Handler for Slurm Preemption ─────────────────────────────────────

TRAINER_REF = None

def sigterm_handler(signum, frame):
    """Catch SIGTERM from Slurm and save checkpoint before exit."""
    print("\n[SIGTERM] Slurm preemption signal received. Saving checkpoint...")
    if TRAINER_REF is not None:
        TRAINER_REF.save_model(os.path.join(OUTPUT_DIR, "checkpoint-preempt"))
        TRAINER_REF.save_state()
        print("[SIGTERM] Checkpoint saved. Exiting.")
    sys.exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)


# ─── Dataset Loading ─────────────────────────────────────────────────────────

def load_dpo_dataset(path, eval_split, seed):
    """Load JSONL DPO dataset and split into train/eval."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    ds = Dataset.from_list(samples)
    split = ds.train_test_split(test_size=eval_split, seed=seed)

    print(f"Dataset loaded: {len(samples)} total")
    print(f"  Train: {len(split['train'])}")
    print(f"  Eval:  {len(split['test'])}")

    return split["train"], split["test"]


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_and_merge_sft(base_model_id, sft_lora_path, merged_path):
    """
    Load base Qwen model and merge the SFT LoRA.
    Caches the merged model so we only do this once.
    """
    if os.path.exists(merged_path):
        print(f"Loading cached merged SFT model from {merged_path}")
        model = AutoModelForCausalLM.from_pretrained(
            merged_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(merged_path)
        return model, tokenizer

    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"Loading SFT LoRA from: {sft_lora_path}")
    model = PeftModel.from_pretrained(model, sft_lora_path)

    print("Merging SFT LoRA into base weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {merged_path}")
    os.makedirs(merged_path, exist_ok=True)
    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    return model, tokenizer


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    global TRAINER_REF

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume training from the latest checkpoint in OUTPUT_DIR",
    )
    parser.add_argument(
        "--eval_split",
        type=float,
        default=EVAL_SPLIT,
        help=f"Fraction of data to hold out for eval (default: {EVAL_SPLIT})",
    )
    args = parser.parse_args()

    # ── Load dataset ─────────────────────────────────────────────────────
    train_ds, eval_ds = load_dpo_dataset(DPO_DATASET_PATH, args.eval_split, SPLIT_SEED)

    # ── Load model ───────────────────────────────────────────────────────
    model, tokenizer = load_and_merge_sft(BASE_MODEL_ID, SFT_LORA_PATH, MERGED_SFT_PATH)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # ── Apply fresh LoRA for DPO ─────────────────────────────────────────
    print("Applying fresh LoRA for DPO training...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Training config ──────────────────────────────────────────────────
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=DPO_BETA,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_ratio=WARMUP_RATIO,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name="qwen25-7b-dpo-socialiqa",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        # Slurm signal handling: request SIGTERM 120s before kill
        save_on_each_node=True,
    )

    # ── Initialize trainer ───────────────────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    TRAINER_REF = trainer  # For SIGTERM handler

    # ── Train ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Starting DPO Training")
    print("=" * 60)
    print(f"  Beta:               {DPO_BETA}")
    print(f"  Learning rate:      {LEARNING_RATE}")
    print(f"  Epochs:             {NUM_EPOCHS}")
    print(f"  Effective batch:    {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Train samples:      {len(train_ds)}")
    print(f"  Eval samples:       {len(eval_ds)}")
    print(f"  Save every:         {SAVE_STEPS} steps")
    print(f"  Eval every:         {EVAL_STEPS} steps")
    print("=" * 60)

    checkpoint = None
    if args.resume_from_checkpoint:
        # Find latest checkpoint
        checkpoints = [
            d for d in os.listdir(OUTPUT_DIR)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(OUTPUT_DIR, d))
        ]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
            checkpoint = os.path.join(OUTPUT_DIR, latest)
            print(f"  Resuming from: {checkpoint}")
        else:
            print("  No checkpoint found, starting from scratch.")

    trainer.train(resume_from_checkpoint=checkpoint)

    # ── Save final model ─────────────────────────────────────────────────
    final_path = os.path.join(OUTPUT_DIR, "final_model")
    print(f"\nSaving final model to: {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print("Done.")


if __name__ == "__main__":
    main()