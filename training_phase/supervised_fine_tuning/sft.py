"""
SFT pipeline for Theory of Mind task — Qwen2.5-7B + QLoRA
Hardware target: RTX 4070 Super (12GB VRAM)
"""

import json
import math
import argparse
import re
import gc
import shutil
import torch
import torch.nn.functional as F

# PyTorch 2.6: patch torch.load to allow RNG state loading from checkpoints
_orig_torch_load = torch.load
def _patched_torch_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _patched_torch_load

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from datasets import Dataset, disable_caching
disable_caching()   # prevent large arrow files accumulating in RAM

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer


# ─────────────────────────────────────────────
# WEIGHTED LOSS
# ─────────────────────────────────────────────

FIELD_WEIGHTS = {
    "Mental State Type": 0.5,
    "Hypothesis":        1.5,
    "Response":          1.5,
    "Final answer":      3.0,
}
DEFAULT_WEIGHT = 1.0
_WEIGHT_PATTERN = re.compile(
    r'"({fields})"\s*:\s*"([^"]*)"'.format(
        fields="|".join(re.escape(k) for k in FIELD_WEIGHTS)
    ),
    re.DOTALL,
)

def build_token_weights(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    text    = tokenizer.decode(input_ids, skip_special_tokens=False)
    tokens  = [tokenizer.decode([tid]) for tid in input_ids]
    weights = torch.ones(len(tokens), dtype=torch.float32)
    char_field = [DEFAULT_WEIGHT] * len(text)
    for m in _WEIGHT_PATTERN.finditer(text):
        w = FIELD_WEIGHTS[m.group(1)]
        for ci in range(m.start(2), m.end(2)):
            if ci < len(char_field):
                char_field[ci] = w
    char_pos = 0
    for ti, tok in enumerate(tokens):
        if char_pos < len(char_field):
            weights[ti] = char_field[char_pos]
        char_pos += len(tok)
    return weights


class WeightedSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        labels    = inputs.get("labels", input_ids.clone())
        outputs   = model(**inputs)
        logits    = outputs.logits
        batch_size, seq_len, vocab_size = logits.shape
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_flat = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        weight_flat = torch.ones_like(loss_flat)
        for b in range(batch_size):
            w     = build_token_weights(input_ids[b], self.tokenizer)
            w     = w[1:seq_len].to(loss_flat.device)
            start = b * (seq_len - 1)
            weight_flat[start:start + len(w)] = w
        mask       = (shift_labels.view(-1) != -100).float()
        loss       = (loss_flat * weight_flat * mask).sum()
        normalizer = (mask * weight_flat).sum().clamp(min=1e-8)
        return (loss / normalizer, outputs) if return_outputs else loss / normalizer


# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────

RAW_JSONL    = "training_data_full.jsonl"
TRAIN_JSONL  = "train_full.jsonl"
MODEL_ID     = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR   = "./output/qwen25_tom_full"
PLOT_PATH    = "./output/qwen25_tom_full/metrics.png"
METRICS_PATH = "./output/qwen25_tom_full/metrics_store.json"
MAX_SEQ_LEN  = 768
LORA_RANK    = 96
LORA_ALPHA   = 192
BATCH_SIZE   = 1
GRAD_ACCUM   = 16
NUM_EPOCHS   = 20
LR           = 2e-4
EVAL_EVERY_N = 2
SPLIT_SEED   = 42


# ─────────────────────────────────────────────
# 1. DATA CONVERSION
# ─────────────────────────────────────────────

def convert_to_sharegpt(raw_path: str, out_path: str) -> None:
    converted = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [!] Skipping line {i+1}: {e}")
                continue
            resp = row["response"]
            resp_str = json.dumps(resp, ensure_ascii=False, indent=2) if isinstance(resp, dict) else str(resp)
            converted.append({
                "conversations": [
                    {"from": "system",    "value": row["system"].strip()},
                    {"from": "human",     "value": row["prompt"].strip()},
                    {"from": "assistant", "value": resp_str},
                ]
            })
    with open(out_path, "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[1] Converted {len(converted)} examples → {out_path}")


# ─────────────────────────────────────────────
# 2. LOAD & SPLIT  (99 / 1)
# ─────────────────────────────────────────────

def load_and_split(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    ds    = Dataset.from_list(records)
    split = ds.train_test_split(test_size=0.01, seed=SPLIT_SEED)
    print(f"[2] Split → train: {len(split['train'])} | eval: {len(split['test'])}")
    return split["train"], split["test"]


def preprocess_dataset(dataset: Dataset, tokenizer) -> Dataset:
    def fmt(ex):
        messages = []
        for turn in ex["conversations"]:
            role_map = {"system": "system", "human": "user", "assistant": "assistant"}
            messages.append({"role": role_map[turn["from"]], "content": turn["value"]})
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
    return dataset.map(fmt, num_proc=1, keep_in_memory=False)


# ─────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────

def load_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False
    print(f"[4] Model loaded | dtype={model.dtype}")
    return model, tokenizer


# ─────────────────────────────────────────────
# 4. LORA
# ─────────────────────────────────────────────

def apply_lora(model, rank: int, alpha: int):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
        lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────
# 5. ACCURACY EVALUATOR
# ─────────────────────────────────────────────

def compute_accuracy(model, tokenizer, dataset: Dataset, desc="eval") -> float:
    from tqdm import tqdm
    model.eval()
    correct, parse_errors = 0, 0
    for ex in tqdm(dataset, desc=desc, leave=False):
        convs = ex["conversations"]
        messages = [
            {"role": "system", "content": convs[0]["value"]},
            {"role": "user",   "content": convs[1]["value"]},
        ]
        text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=300, do_sample=False,
                temperature=1.0, top_p=1.0, top_k=0,
                pad_token_id=tokenizer.pad_token_id,
            )
        pred_str = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            gt_label   = json.loads(convs[2]["value"])["Final answer"].strip().upper()
            pred_label = json.loads(pred_str)["Final answer"].strip().upper()
        except (json.JSONDecodeError, KeyError):
            parse_errors += 1
            continue
        if pred_label == gt_label:
            correct += 1
    total = len(dataset) - parse_errors
    acc   = correct / total if total > 0 else 0.0
    if parse_errors:
        print(f"    [eval] parse errors: {parse_errors}/{len(dataset)}")
    model.train()
    return acc


# ─────────────────────────────────────────────
# 6. METRICS STORE + PLOT
# ─────────────────────────────────────────────

@dataclass
class MetricsStore:
    step_losses:      List[float]    = field(default_factory=list)
    step_numbers:     List[int]      = field(default_factory=list)
    epoch_numbers:    List[int]      = field(default_factory=list)
    epoch_train_loss: List[float]    = field(default_factory=list)
    acc_epochs:       List[int]      = field(default_factory=list)
    train_acc:        List[float]    = field(default_factory=list)
    eval_acc:         List[float]    = field(default_factory=list)
    best_eval_acc:    float          = field(default=0.0)
    best_epoch:       Optional[int]  = field(default=None)

    def truncate_to_step(self, step: int):
        """Remove all metrics recorded after `step` so resume plot is clean."""
        # truncate step-level data
        cut = next((i for i, s in enumerate(self.step_numbers) if s > step), len(self.step_numbers))
        self.step_losses  = self.step_losses[:cut]
        self.step_numbers = self.step_numbers[:cut]

        # truncate epoch-level data (epoch_numbers stores global step at epoch end)
        cut_e = next((i for i, s in enumerate(self.epoch_numbers) if s > step), len(self.epoch_numbers))
        self.epoch_numbers    = self.epoch_numbers[:cut_e]
        self.epoch_train_loss = self.epoch_train_loss[:cut_e]

        # truncate acc data — acc_epochs stores epoch index, derive cutoff from epoch count
        n_epochs_kept = len(self.epoch_numbers)
        cut_a = next((i for i, e in enumerate(self.acc_epochs) if e > n_epochs_kept), len(self.acc_epochs))
        self.acc_epochs = self.acc_epochs[:cut_a]
        self.train_acc  = self.train_acc[:cut_a]
        self.eval_acc   = self.eval_acc[:cut_a]

        # recalculate best from remaining eval points
        if self.eval_acc:
            best_idx          = max(range(len(self.eval_acc)), key=lambda i: self.eval_acc[i])
            self.best_eval_acc = self.eval_acc[best_idx]
            self.best_epoch    = self.acc_epochs[best_idx]
        else:
            self.best_eval_acc = 0.0
            self.best_epoch    = None

        print(f"  [resume] truncated to step {step}: "
              f"{len(self.step_losses)} steps, {len(self.epoch_numbers)} epochs, "
              f"{len(self.acc_epochs)} eval points")

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str) -> "MetricsStore":
        with open(path) as f:
            d = json.load(f)
        s = cls()
        for k, v in d.items():
            setattr(s, k, v)
        print(f"  [resume] metrics loaded: {len(s.step_losses)} steps, "
              f"{len(s.acc_epochs)} eval points, best_acc={s.best_eval_acc:.3f}")
        return s


def draw_plot(store: MetricsStore, save_path: str) -> None:
    has_acc  = len(store.acc_epochs) > 0
    n_panels = 2 if has_acc else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.set_facecolor("#f8f8f8")
    if store.step_losses:
        ax.plot(store.step_numbers, store.step_losses,
                color="#5a7dcc", linewidth=1.0, alpha=0.4, label="step loss")
    if store.epoch_train_loss:
        ax.plot(store.epoch_numbers, store.epoch_train_loss,
                color="#c0392b", linewidth=2, marker="o", markersize=5,
                label="epoch avg loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.4)

    if has_acc:
        ax2 = axes[1]
        ax2.set_facecolor("#f8f8f8")
        ax2.plot(store.acc_epochs, store.train_acc,
                 color="#27ae60", linewidth=2, marker="o", markersize=5, label="train acc")
        ax2.plot(store.acc_epochs, store.eval_acc,
                 color="#e67e22", linewidth=2, marker="s", markersize=5, label="eval acc")
        if store.best_epoch is not None:
            ax2.axvline(store.best_epoch, color="#e67e22", linestyle="--",
                        linewidth=1, alpha=0.6, label=f"best (ep {store.best_epoch})")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy (every 2 epochs)")
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=8)
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"    [plot] saved → {save_path}")


# ─────────────────────────────────────────────
# 7. SAFE SAVE
# Only saves LoRA adapter weights (small, float16) — does NOT move
# the 4-bit quantized base model, which cannot be safely offloaded.
# ─────────────────────────────────────────────

def safe_save(model, tokenizer, save_dir: str, state: TrainerState = None):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)          # saves adapter only (PEFT)
    tokenizer.save_pretrained(save_dir)
    if state is not None:
        state.save_to_json(str(Path(save_dir) / "trainer_state.json"))
    torch.cuda.empty_cache()
    gc.collect()
    free = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"    [save] → {save_dir}  (VRAM free: {free:.1f}GB)")


# ─────────────────────────────────────────────
# 8. CALLBACKS
# ─────────────────────────────────────────────

class SaveCallback(TrainerCallback):
    """Disable HF's automatic step-based saving — we handle it manually."""
    def on_save(self, args, state, control, **kwargs):
        # prevent HF from writing its own checkpoint on top of ours
        control.should_save = False
        return control


class ToMCallback(TrainerCallback):
    def __init__(self, model, tokenizer, train_ds, eval_ds,
                 store: MetricsStore, plot_path: str,
                 eval_every: int, output_dir: str):
        self.model      = model
        self.tokenizer  = tokenizer
        self.train_ds   = train_ds
        self.eval_ds    = eval_ds
        self.store      = store
        self.plot_path  = plot_path
        self.eval_every = eval_every
        self.output_dir = output_dir
        self._epoch_losses: List[float] = []

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kw):
        epoch = int(math.ceil(state.epoch)) + 1
        device = next(self.model.parameters()).device
        print(f"  [epoch {epoch} start] model device: {device}")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kw):
        if logs and "loss" in logs:
            self._epoch_losses.append(logs["loss"])
            self.store.step_losses.append(logs["loss"])
            self.store.step_numbers.append(state.global_step)
            if state.global_step % 10 == 0:
                draw_plot(self.store, self.plot_path)
                self.store.save(METRICS_PATH)

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kw):
        epoch = int(math.ceil(state.epoch))

        avg_loss = (sum(self._epoch_losses) / len(self._epoch_losses)
                    if self._epoch_losses else float("nan"))
        self.store.epoch_numbers.append(state.global_step)
        self.store.epoch_train_loss.append(avg_loss)
        self._epoch_losses.clear()

        # train accuracy on 50-example subset
        subset    = self.train_ds.select(range(min(50, len(self.train_ds))))
        train_acc = compute_accuracy(self.model, self.tokenizer, subset,
                                     desc=f"train acc ep{epoch}")
        torch.cuda.empty_cache(); gc.collect()  # clear after inference loop

        print(f"\n{'─'*55}")
        print(f"  Epoch {epoch:>3} | avg loss: {avg_loss:.4f} | train acc: {train_acc:.3f}")

        # full eval every EVAL_EVERY_N epochs
        if epoch % self.eval_every == 0:
            eval_acc = compute_accuracy(self.model, self.tokenizer, self.eval_ds,
                                        desc=f"eval acc ep{epoch}")
            torch.cuda.empty_cache(); gc.collect()  # clear after inference loop
            self.store.acc_epochs.append(epoch)
            self.store.train_acc.append(train_acc)
            self.store.eval_acc.append(eval_acc)
            print(f"           | eval  acc: {eval_acc:.3f}  ← full eval set")

            # save best checkpoint (includes trainer_state for resume)
            if eval_acc > self.store.best_eval_acc:
                self.store.best_eval_acc = eval_acc
                self.store.best_epoch    = epoch
                best_dir = str(Path(self.output_dir) / "best_checkpoint")
                safe_save(self.model, self.tokenizer, best_dir, state)
                print(f"           | ★ new best ({eval_acc:.3f})")

        # save epoch checkpoint (always, includes trainer_state for resume)
        epoch_dir = str(Path(self.output_dir) / f"checkpoint-epoch{epoch}")
        safe_save(self.model, self.tokenizer, epoch_dir, state)

        # keep only last 3 epoch checkpoints to save disk space
        ckpts = sorted(
            Path(self.output_dir).glob("checkpoint-epoch*"),
            key=lambda p: int(p.name.replace("checkpoint-epoch", ""))
        )
        for old in ckpts[:-3]:
            shutil.rmtree(old)
            print(f"    [cleanup] removed {old.name}")

        print(f"{'─'*55}\n")
        draw_plot(self.store, self.plot_path)
        self.store.save(METRICS_PATH)


# ─────────────────────────────────────────────
# 9. TRAINING ARGS
# ─────────────────────────────────────────────

def get_training_args(output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        fp16=True,
        logging_steps=10,
        save_steps=999999,          # disabled — saving handled in ToMCallback
        save_total_limit=None,      # we manage cleanup ourselves
        save_safetensors=True,
        report_to="none",
        dataloader_num_workers=0,
        eval_strategy="no",
    )


# ─────────────────────────────────────────────
# 10. TRAIN
# ─────────────────────────────────────────────

def train(train_ds, eval_ds, model, tokenizer, training_args, resume_from=None):
    # restore and truncate previous metrics if resuming
    if resume_from and Path(METRICS_PATH).exists():
        store = MetricsStore.load(METRICS_PATH)
        # read the global step from the checkpoint's trainer_state.json
        state_file = Path(resume_from) / "trainer_state.json"
        if state_file.exists():
            with open(state_file) as f:
                resume_step = json.load(f).get("global_step", 0)
            store.truncate_to_step(resume_step)
        else:
            print("  [resume] trainer_state.json not found — metrics not truncated")
    else:
        store = MetricsStore()

    callback = ToMCallback(
        model=model, tokenizer=tokenizer,
        train_ds=train_ds, eval_ds=eval_ds,
        store=store, plot_path=PLOT_PATH,
        eval_every=EVAL_EVERY_N, output_dir=training_args.output_dir,
    )

    trainer = WeightedSFTTrainer(
        model=model, tokenizer=tokenizer,
        args=training_args, train_dataset=train_ds,
        dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
        packing=False, callbacks=[SaveCallback(), callback],
    )

    print("[9] Starting training...")
    trainer.train(resume_from_checkpoint=resume_from)
    # final save
    safe_save(model, tokenizer, training_args.output_dir)
    print(f"[9] Final adapter saved → {training_args.output_dir}")

    draw_plot(store, PLOT_PATH)
    store.save(METRICS_PATH)
    return store


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint dir to resume from, e.g. ./output/.../checkpoint-epoch3")
    args = parser.parse_args()

    convert_to_sharegpt(RAW_JSONL, TRAIN_JSONL)
    train_ds, eval_ds = load_and_split(TRAIN_JSONL)

    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    model = apply_lora(model, LORA_RANK, LORA_ALPHA)

    print("[3] Applying chat template...")
    train_ds = preprocess_dataset(train_ds, tokenizer)

    training_args = get_training_args(OUTPUT_DIR)
    store = train(train_ds, eval_ds, model, tokenizer, training_args, resume_from=args.resume)

    print("\n[done] Final metrics summary")
    for ep, tr, ev in zip(store.acc_epochs, store.train_acc, store.eval_acc):
        print(f"  epoch {ep:>3} | train acc: {tr:.3f} | eval acc: {ev:.3f}")
    print(f"  plot → {PLOT_PATH}")