"""
DPO Training Script for Qwen 2.5 7B — Single Node / Single GPU
----------------------------------------------------------------
1. Validates all paths upfront before loading anything
2. Loads Qwen 2.5 7B Instruct, merges teammate's SFT LoRA
3. Applies a fresh LoRA for DPO training
4. Trains with TRL's DPOTrainer
5. Saves checkpoints every SAVE_STEPS for early stopping analysis
6. Checkpoints on SIGTERM (Slurm 8hr limit)

Usage:
    python train_dpo.py
    python train_dpo.py --resume_from_checkpoint
    python train_dpo.py --beta 2.0 --lr 5e-6
"""

import os
import sys
import signal
import argparse
import json
from datetime import datetime

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import DPOConfig, DPOTrainer


# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_MODEL_PATH = "/scratch/laredo.ei/models/Qwen2.5-7B-Instruct"
SFT_LORA_PATH = "/scratch/laredo.ei/dpo_training/best_checkpoint"
DPO_DATASET_PATH = "/scratch/laredo.ei/dpo_training/dpo_data/dpo_dataset.jsonl"
OUTPUT_DIR = "/scratch/laredo.ei/dpo_training/dpo_output"
MERGED_SFT_PATH = os.path.join(OUTPUT_DIR, "merged_sft_base")

os.environ["HF_HOME"] = "/scratch/laredo.ei/.cache/huggingface"


# ─── Hyperparameters ─────────────────────────────────────────────────────────

EVAL_SPLIT = 0.1
SPLIT_SEED = 42

DPO_BETA = 1.0
LEARNING_RATE = 5e-6
NUM_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8   # Effective batch = 2 * 8 = 16
WARMUP_RATIO = 0.1
MAX_LENGTH = 512

SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 15       # keep all checkpoints for early stopping analysis
EVAL_STEPS = 50             # eval every 50 steps so we can find the sweet spot
LOGGING_STEPS = 10

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ─── Path Validation ────────────────────────────────────────────────────────

def validate_paths():
    """Check all required paths exist before doing any heavy work."""
    errors = []

    if not os.path.isdir(BASE_MODEL_PATH):
        errors.append(f"Base model not found: {BASE_MODEL_PATH}")

    if not os.path.isdir(SFT_LORA_PATH):
        errors.append(f"SFT LoRA checkpoint not found: {SFT_LORA_PATH}")
    else:
        adapter_file = os.path.join(SFT_LORA_PATH, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            errors.append(f"adapter_model.safetensors not found in {SFT_LORA_PATH}")

    if not os.path.isfile(DPO_DATASET_PATH):
        errors.append(f"DPO dataset not found: {DPO_DATASET_PATH}")

    if errors:
        print("=" * 60)
        print("PATH VALIDATION FAILED")
        print("=" * 60)
        for e in errors:
            print(f"  ERROR: {e}")
        print("=" * 60)
        sys.exit(1)

    print("All paths validated.")


# ─── Signal Handler for Slurm Preemption ─────────────────────────────────────

TRAINER_REF = None

def sigterm_handler(signum, frame):
    """Catch SIGTERM from Slurm and save checkpoint before exit."""
    print("\n[SIGTERM] Slurm preemption signal received. Saving checkpoint...")
    if TRAINER_REF is not None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(OUTPUT_DIR, f"checkpoint-preempt-{ts}")
        TRAINER_REF.save_model(save_path)
        print(f"[SIGTERM] Checkpoint saved to {save_path}. Exiting.")
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

def get_attn_implementation():
    """Use flash_attention_2 if available, otherwise fall back to default."""
    try:
        import flash_attn  # noqa: F401
        print("Flash Attention 2 available — using it.")
        return "flash_attention_2"
    except ImportError:
        print("Flash Attention 2 not installed — using default attention.")
        return None


def load_and_merge_sft():
    """
    Load base Qwen model and merge the SFT LoRA.
    Caches the merged model to disk so we only merge once.
    """
    attn_impl = get_attn_implementation()

    # Check if we already have a cached merged model
    if os.path.isdir(MERGED_SFT_PATH) and os.path.exists(os.path.join(MERGED_SFT_PATH, "config.json")):
        print(f"Loading cached merged SFT model from {MERGED_SFT_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_SFT_PATH,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        tokenizer = AutoTokenizer.from_pretrained(MERGED_SFT_PATH)
        return model, tokenizer

    # Load base model
    print(f"Loading base model: {BASE_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    # Load and merge SFT LoRA
    print(f"Loading SFT LoRA from: {SFT_LORA_PATH}")
    model = PeftModel.from_pretrained(model, SFT_LORA_PATH)

    print("Merging SFT LoRA into base weights...")
    model = model.merge_and_unload()

    # Cache to disk
    print(f"Saving merged model to: {MERGED_SFT_PATH}")
    os.makedirs(MERGED_SFT_PATH, exist_ok=True)
    model.save_pretrained(MERGED_SFT_PATH)
    tokenizer.save_pretrained(MERGED_SFT_PATH)

    return model, tokenizer


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    global TRAINER_REF

    parser = argparse.ArgumentParser(description="DPO training for Qwen 2.5 7B (single GPU)")
    parser.add_argument("--resume_from_checkpoint", action="store_true",
                        help="Resume from the latest checkpoint in OUTPUT_DIR")
    parser.add_argument("--eval_split", type=float, default=EVAL_SPLIT,
                        help=f"Fraction held out for eval (default: {EVAL_SPLIT})")
    parser.add_argument("--beta", type=float, default=DPO_BETA,
                        help=f"DPO beta (default: {DPO_BETA})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    args = parser.parse_args()

    # Use CLI args if provided, otherwise defaults
    beta = args.beta
    lr = args.lr
    run_tag = f"beta{beta}_lr{lr}"

    # Set output dir per config so checkpoints don't collide
    run_output_dir = os.path.join(OUTPUT_DIR, run_tag)
    os.makedirs(run_output_dir, exist_ok=True)

    # ── Validate before doing anything expensive ─────────────────────────
    print("=" * 60)
    print("DPO Training — Single GPU")
    print(f"Config: {run_tag}")
    print("=" * 60)
    validate_paths()

    # ── GPU check ────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected. Exiting.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # ── Load dataset ─────────────────────────────────────────────────────
    train_ds, eval_ds = load_dpo_dataset(DPO_DATASET_PATH, args.eval_split, SPLIT_SEED)

    # ── Load model ───────────────────────────────────────────────────────
    model, tokenizer = load_and_merge_sft()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # ── LoRA config for DPO ──────────────────────────────────────────────
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
        output_dir=run_output_dir,
        beta=beta,
        learning_rate=lr,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_ratio=WARMUP_RATIO,
        max_length=MAX_LENGTH,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        load_best_model_at_end=False,  # keep all checkpoints for manual selection
        report_to="wandb",
        run_name=f"dpo-{run_tag}",
        dataloader_num_workers=4,
        remove_unused_columns=False,
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

    TRAINER_REF = trainer

    # ── Print config ─────────────────────────────────────────────────────
    total_steps = (len(train_ds) // (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION)) * NUM_EPOCHS
    print("\n" + "=" * 60)
    print("Starting DPO Training")
    print("=" * 60)
    print(f"  GPU:                {gpu_name}")
    print(f"  Beta:               {beta}")
    print(f"  Learning rate:      {lr}")
    print(f"  Epochs:             {NUM_EPOCHS}")
    print(f"  Batch size:         {PER_DEVICE_BATCH_SIZE}")
    print(f"  Grad accumulation:  {GRADIENT_ACCUMULATION}")
    print(f"  Effective batch:    {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Estimated steps:    {total_steps}")
    print(f"  Train samples:      {len(train_ds)}")
    print(f"  Eval samples:       {len(eval_ds)}")
    print(f"  Save/Eval every:    {SAVE_STEPS} steps")
    print(f"  Output dir:         {run_output_dir}")
    print("=" * 60)

    # ── Resume handling ──────────────────────────────────────────────────
    checkpoint = None
    if args.resume_from_checkpoint:
        checkpoint = True
        print("Resume requested — trainer will find latest checkpoint.")

    # ── Train ────────────────────────────────────────────────────────────
    trainer.train(resume_from_checkpoint=checkpoint)

    # ── Save eval history for analysis ───────────────────────────────────
    eval_history = [
        entry for entry in trainer.state.log_history
        if "eval_loss" in entry
    ]

    history_path = os.path.join(run_output_dir, "eval_history.json")
    with open(history_path, "w") as f:
        json.dump(eval_history, f, indent=2)
    print(f"\nEval history saved to: {history_path}")

    # ── Print eval summary for checkpoint selection ──────────────────────
    print("\n" + "=" * 60)
    print("EVAL HISTORY — use this to pick your checkpoint")
    print("=" * 60)
    print(f"{'Step':<8} {'Loss':<12} {'Margins':<12} {'Accuracy':<12} {'Chosen LogP':<14} {'Rejected LogP':<14}")
    print("-" * 72)
    for entry in eval_history:
        print(f"{entry.get('step', '?'):<8} "
              f"{entry.get('eval_loss', 0):<12.4f} "
              f"{entry.get('eval_rewards/margins', 0):<12.3f} "
              f"{entry.get('eval_rewards/accuracies', 0):<12.4f} "
              f"{entry.get('eval_logps/chosen', 0):<14.2f} "
              f"{entry.get('eval_logps/rejected', 0):<14.2f}")
    print("=" * 60)
    print(f"\nCheckpoints saved in: {run_output_dir}/")
    print("Look for margins in 1.0-2.0 range with balanced chosen/rejected log-probs.")
    print("Then use that checkpoint for inference eval.")

    print("\nDone.")


if __name__ == "__main__":
    main()