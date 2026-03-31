"""
grid_search_dpo.py
------------------------------------
Runs 3 (beta, lr) configs per job. Launch 4 jobs to cover the full 4x3 grid.

Usage:
    python grid_search_dpo.py --batch 0   # runs configs 0,1,2
    python grid_search_dpo.py --batch 1   # runs configs 3,4,5
    python grid_search_dpo.py --batch 2   # runs configs 6,7,8
    python grid_search_dpo.py --batch 3   # runs configs 9,10,11
    python grid_search_dpo.py --dry_run --batch 0  # preview only
"""

import os
import sys
import json
import csv
import argparse
import itertools
from datetime import datetime

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import DPOConfig, DPOTrainer


# ─── Paths ──────────────────────────────────────────────────────────────────

SCRATCH = "/scratch/laredo.ei"
BASE_MODEL_PATH = f"{SCRATCH}/models/Qwen2.5-7B-Instruct"
SFT_LORA_PATH = f"{SCRATCH}/dpo_training/best_checkpoint"
DPO_DATASET_PATH = f"{SCRATCH}/dpo_training/dpo_data/dpo_dataset.jsonl"
MERGED_SFT_PATH = f"{SCRATCH}/dpo_training/dpo_output/merged_sft_base"

GRID_OUTPUT_DIR = f"{SCRATCH}/dpo_training/grid_search"
RESULTS_DIR = os.path.join(GRID_OUTPUT_DIR, "results")

os.environ["HF_HOME"] = f"{SCRATCH}/.cache/huggingface"


# ─── Grid ───────────────────────────────────────────────────────────────────

BETA_VALUES = [0.1, 0.2, 0.3, 0.5]
LR_VALUES = [5e-6, 2e-5, 5e-5]

FULL_GRID = list(itertools.product(BETA_VALUES, LR_VALUES))
# 12 configs total, 3 per batch:
#   batch 0: (0.1, 5e-6), (0.1, 2e-5), (0.1, 5e-5)
#   batch 1: (0.2, 5e-6), (0.2, 2e-5), (0.2, 5e-5)
#   batch 2: (0.3, 5e-6), (0.3, 2e-5), (0.3, 5e-5)
#   batch 3: (0.5, 5e-6), (0.5, 2e-5), (0.5, 5e-5)

CONFIGS_PER_BATCH = 3


# ─── Fixed Hyperparameters ──────────────────────────────────────────────────

EVAL_SPLIT = 0.1
SPLIT_SEED = 42
NUM_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
WARMUP_RATIO = 0.1
MAX_LENGTH = 512

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LOGGING_STEPS = 50
EVAL_STEPS = 200


# ─── Dataset Loading ────────────────────────────────────────────────────────

def load_dpo_dataset():
    samples = []
    with open(DPO_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    ds = Dataset.from_list(samples)
    split = ds.train_test_split(test_size=EVAL_SPLIT, seed=SPLIT_SEED)
    print(f"Dataset: {len(samples)} total -> {len(split['train'])} train / {len(split['test'])} eval")
    return split["train"], split["test"]


# ─── Model Loading ──────────────────────────────────────────────────────────

def load_merged_model():
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = None

    if os.path.isdir(MERGED_SFT_PATH) and os.path.exists(os.path.join(MERGED_SFT_PATH, "config.json")):
        print(f"Loading cached merged SFT model from {MERGED_SFT_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_SFT_PATH, dtype=torch.bfloat16, attn_implementation=attn_impl,
        )
        tokenizer = AutoTokenizer.from_pretrained(MERGED_SFT_PATH)
    else:
        print("Loading base model + SFT LoRA and merging...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, dtype=torch.bfloat16, attn_implementation=attn_impl,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        model = PeftModel.from_pretrained(model, SFT_LORA_PATH)
        model = model.merge_and_unload()
        os.makedirs(MERGED_SFT_PATH, exist_ok=True)
        model.save_pretrained(MERGED_SFT_PATH)
        tokenizer.save_pretrained(MERGED_SFT_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


# ─── Helpers ────────────────────────────────────────────────────────────────

def run_id(beta, lr):
    return f"beta{beta}_lr{lr}"

def save_result(rid, beta, lr, final_metrics, best_eval, eval_history):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = {
        "run_id": rid,
        "beta": beta,
        "learning_rate": lr,
        # Final eval (end of training)
        "final_eval_loss": final_metrics.get("eval_loss"),
        "final_rewards_chosen": final_metrics.get("eval_rewards/chosen"),
        "final_rewards_rejected": final_metrics.get("eval_rewards/rejected"),
        "final_rewards_margins": final_metrics.get("eval_rewards/margins"),
        "final_rewards_accuracies": final_metrics.get("eval_rewards/accuracies"),
        "final_logps_chosen": final_metrics.get("eval_logps/chosen"),
        "final_logps_rejected": final_metrics.get("eval_logps/rejected"),
        # Best eval (lowest loss across all checkpoints)
        "best_step": best_eval.get("step"),
        "best_eval_loss": best_eval.get("eval_loss"),
        "best_rewards_chosen": best_eval.get("eval_rewards/chosen"),
        "best_rewards_rejected": best_eval.get("eval_rewards/rejected"),
        "best_rewards_margins": best_eval.get("eval_rewards/margins"),
        "best_rewards_accuracies": best_eval.get("eval_rewards/accuracies"),
        "best_logps_chosen": best_eval.get("eval_logps/chosen"),
        "best_logps_rejected": best_eval.get("eval_logps/rejected"),
        # Full eval history for deeper analysis
        "eval_history": eval_history,
        "completed_at": datetime.now().isoformat(),
    }
    path = os.path.join(RESULTS_DIR, f"{rid}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved: {path}")
    return data


# ─── Single Run ─────────────────────────────────────────────────────────────

def run_single(beta, lr, model, tokenizer, train_ds, eval_ds):
    rid = run_id(beta, lr)
    run_output = os.path.join(GRID_OUTPUT_DIR, "checkpoints", rid)

    print(f"\n{'='*60}")
    print(f"  RUN: {rid}")
    print(f"  beta={beta}, lr={lr}")
    print(f"{'='*60}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=run_output,
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
        save_steps=99999,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        load_best_model_at_end=False,
        report_to="wandb",
        run_name=f"grid-{rid}",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()

    # Final eval
    final_metrics = trainer.evaluate()
    print(f"  Final eval metrics: {final_metrics}")

    # Pull all eval entries from the training log
    eval_history = [
        entry for entry in trainer.state.log_history
        if "eval_loss" in entry
    ]

    # Find the checkpoint with lowest eval loss
    best_eval = min(eval_history, key=lambda x: x["eval_loss"])
    print(f"  Best eval was at step {best_eval.get('step', '?')}: "
          f"loss={best_eval['eval_loss']:.4f}, "
          f"margins={best_eval.get('eval_rewards/margins', 'N/A')}")

    result = save_result(rid, beta, lr, final_metrics, best_eval, eval_history)

    # Clean up checkpoints
    import shutil
    if os.path.isdir(run_output):
        shutil.rmtree(run_output)

    return result


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DPO Grid Search — Batch Mode")
    parser.add_argument("--batch", type=int, required=True, choices=[0, 1, 2, 3],
                        help="Which batch of 3 configs to run (0-3)")
    parser.add_argument("--dry_run", action="store_true", help="Print plan only")
    args = parser.parse_args()

    start = args.batch * CONFIGS_PER_BATCH
    batch_configs = FULL_GRID[start : start + CONFIGS_PER_BATCH]

    print(f"{'='*60}")
    print(f"DPO GRID SEARCH — BATCH {args.batch}")
    print(f"{'='*60}")
    print(f"Configs this batch:")
    for i, (b, lr) in enumerate(batch_configs):
        print(f"  [{start+i}] beta={b}, lr={lr}")
    print()

    if args.dry_run:
        print("[DRY RUN] Exiting.")
        return

    train_ds, eval_ds = load_dpo_dataset()
    all_results = []

    for beta, lr in batch_configs:
        model, tokenizer = load_merged_model()
        try:
            result = run_single(beta, lr, model, tokenizer, train_ds, eval_ds)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR in {run_id(beta, lr)}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del model, tokenizer
            torch.cuda.empty_cache()

    # Print batch summary
    print(f"\n{'='*60}")
    print(f"BATCH {args.batch} COMPLETE — {len(all_results)}/{len(batch_configs)} succeeded")
    print(f"{'='*60}")
    print(f"{'Run ID':<25} {'Beta':<8} {'LR':<12} {'Eval Loss':<12}")
    print("-" * 57)
    for r in sorted(all_results, key=lambda x: x.get("final_eval_loss", float("inf"))):
        print(f"{r['run_id']:<25} {r['beta']:<8} {r['learning_rate']:<12} "
              f"{r.get('final_eval_loss', 'N/A'):<12.4f}")
    print(f"\nResults saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()