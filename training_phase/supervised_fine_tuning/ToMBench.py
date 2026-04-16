"""
Benchmark runner: base vs fine-tuned across all dimensions in ./data/*.jsonl
Usage:
    python benchmark.py
    python benchmark.py --max 100
"""

import json
import random
import argparse
import gc
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH   = r"X:\PyFile\ToMCoT\output\qwen25_tom_full\best_checkpoint"
DATA_DIR       = "./data"
RESULTS_DIR    = "./benchmark_results"
MAX_NEW_TOKENS = 1024
SEED           = 42

random.seed(SEED)


# ─────────────────────────────────────────────
# PROMPT VERSIONS
# ─────────────────────────────────────────────

def prompt_v1(context, question, q_options):
    mental_states = "Belief, Desire, Intention, Emotion, Thought"
    sys = f"""
    You are a conversational AI assistant specialized in Theory of Mind reasoning.
    Your task is to analyze user inputs and infer their underlying mental states based on conversational context and social memory.
    Objective: Based on the user's input, conversation context, and social memory, generate a plausible mental state hypothesis. Focus on the five mental state types.
    Inputs:
    - User Input (u_t)
    - Conversational Context (C_t)
    - Mental state types: {{{mental_states}}}
    Instructions conditioning on the conversational context:
    1. Analyze the provided inputs based on conversational context C_t.
    2. Categorize the provided inputs and obtain the **best matching mental state types**.
    3. Formulate a **single** hypothesis about the user's mental state.
    4. The hypothesis should be a concise explanation.
    5. Generate the final response to the user's input by considering the hypothesis and the mental state type just generated.
    Output Format (Strictly follow this format):
    {{
        "Mental State Type": "one of the {{{mental_states}}}",
        "Hypothesis": "Hypothesis about the user's mental state",
        "Response": "Final response",
        "Final answer": "The final answer to the user's question in a single letter format (e.g., A, B, C, D)"
    }}"""
    usr = f"\n    - User Input (u_t): {question}\n      Options: {', '.join(q_options)}\n    - Conversational Context (C_t): {context}\n"
    return sys, usr

def prompt_v2(context, question, q_options):
    mental_states = "Belief, Desire, Intention, Emotion, Thought"
    sys = f"""
    You are an AI agent trained in cognitive modeling and mental state attribution. Your task is to apply Theory of Mind principles to identify and reason about a user's beliefs, desires, intentions, emotions, and thoughts from conversational cues.
    Objective: Given the user's input, conversation context, and social memory, infer the most likely mental state. Focus on the five mental state categories.
    Inputs:
    - User Input (u_t)
    - Conversational Context (C_t)
    - Mental state categories: {{{mental_states}}}
    Instructions conditioning on the conversational context:
    1. Examine the provided inputs within the conversational context C_t.
    2. Identify the **most fitting mental state category** for the given inputs.
    3. Construct a **single** hypothesis regarding the user's mental state.
    4. The hypothesis should be a brief and precise explanation.
    5. Produce the final reply to the user's input by incorporating the hypothesis and identified mental state category.
    Output Format (Strictly follow this format):
    {{
        "Mental State Type": "one of the {{{mental_states}}}",
        "Hypothesis": "Hypothesis about the user's mental state",
        "Response": "Final response",
        "Final answer": "The final answer to the user's question in a single letter format (e.g., A, B, C, D)"
    }}"""
    usr = f"\n    - User Input (u_t): {question}\n      Options: {', '.join(q_options)}\n    - Conversational Context (C_t): {context}\n"
    return sys, usr

def prompt_v3(context, question, q_options):
    mental_states = "Belief, Desire, Intention, Emotion, Thought"
    sys = f"""
    You are a socially intelligent AI assistant capable of understanding human psychological states. Your task is to interpret conversational signals and infer the most plausible mental state of the user based on their input and prior dialogue.
    Objective: Using the user's input, conversation context, and social memory, deduce the underlying mental state. Concentrate on the five mental state dimensions.
    Inputs:
    - User Input (u_t)
    - Conversational Context (C_t)
    - Mental state dimensions: {{{mental_states}}}
    Instructions conditioning on the conversational context:
    1. Interpret the provided inputs through the lens of conversational context C_t.
    2. Determine the **most appropriate mental state dimension** that aligns with the inputs.
    3. Derive a **single** hypothesis about the user's underlying mental state.
    4. The hypothesis should be a clear and succinct statement.
    5. Craft the final reply to the user's input by integrating the hypothesis and the identified mental state dimension.
    Output Format (Strictly follow this format):
    {{
        "Mental State Type": "one of the {{{mental_states}}}",
        "Hypothesis": "Hypothesis about the user's mental state",
        "Response": "Final response",
        "Final answer": "The final answer to the user's question in a single letter format (e.g., A, B, C, D)"
    }}"""
    usr = f"\n    - User Input (u_t): {question}\n      Options: {', '.join(q_options)}\n    - Conversational Context (C_t): {context}\n"
    return sys, usr

def prompt_v4(context, question, q_options):
    mental_states = "Belief, Desire, Intention, Emotion, Thought"
    sys = f"""
    You are an AI system designed for structured mental state hypothesis generation. Your task is to systematically process user inputs, contextual history, and social cues to produce well-reasoned inferences about the user's current psychological state.
    Objective: From the user's input, conversation context, and social memory, predict the most probable mental state. Emphasize the five mental state frameworks.
    Inputs:
    - User Input (u_t)
    - Conversational Context (C_t)
    - Mental state frameworks: {{{mental_states}}}
    Instructions conditioning on the conversational context:
    1. Evaluate the provided inputs against the conversational context C_t.
    2. Select the **closest matching mental state framework** for the provided inputs.
    3. Propose a **single** hypothesis concerning the user's mental state.
    4. The hypothesis should be a focused and direct explanation.
    5. Formulate the final reply to the user's input by drawing on the hypothesis and the selected mental state framework.
    Output Format (Strictly follow this format):
    {{
        "Mental State Type": "one of the {{{mental_states}}}",
        "Hypothesis": "Hypothesis about the user's mental state",
        "Response": "Final response",
        "Final answer": "The final answer to the user's question in a single letter format (e.g., A, B, C, D)"
    }}"""
    usr = f"\n    - User Input (u_t): {question}\n      Options: {', '.join(q_options)}\n    - Conversational Context (C_t): {context}\n"
    return sys, usr

PROMPT_FNS = [prompt_v1, prompt_v2, prompt_v3, prompt_v4]


# ─────────────────────────────────────────────
# MODEL LOADING / UNLOADING
# ─────────────────────────────────────────────

def get_bnb():
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )

def load_base(tokenizer):
    print("  [load] base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=get_bnb(),
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model

def load_tuned(tokenizer):
    print("  [load] tuned model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=get_bnb(),
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    for name, param in model.named_parameters():
        if "lora_B" in name:
            norm = param.norm().item()
            print(f"  adapter norm={norm:.6f} {'✓' if norm > 1e-4 else '⚠ near-zero!'}")
            break
    return model

def unload(model):
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"  [unload] VRAM free: {free:.1f} GB")


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def infer(model, tokenizer, sys_p, usr_p):
    messages = [
        {"role": "system", "content": sys_p.strip()},
        {"role": "user",   "content": usr_p.strip()},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
            pad_token_id=tokenizer.pad_token_id,
        )
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        parsed = json.loads(raw)
        answer = parsed.get("Final answer", "?").strip().upper()
    except Exception:
        import re
        m = re.search(r'"Final answer"\s*:\s*"([A-D])"', raw, re.IGNORECASE)
        answer = m.group(1).upper() if m else "?"
        parsed = None
    return answer, parsed


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

class Metrics:
    def __init__(self):
        self.total = 0
        self.correct = 0
        self.parse_errors = 0
        self.per_prompt = {f"v{i+1}": {"total": 0, "correct": 0} for i in range(4)}

    def update(self, pred, gt, pv, parse_ok):
        self.total += 1
        if not parse_ok:
            self.parse_errors += 1
        hit = (pred == gt)
        if hit:
            self.correct += 1
        self.per_prompt[pv]["total"]   += 1
        self.per_prompt[pv]["correct"] += int(hit)
        return hit

    @property
    def acc(self):
        return self.correct / self.total if self.total else 0.0

    def summary(self):
        pp = "  ".join(
            f"{v}={d['correct']}/{d['total']}"
            for v, d in self.per_prompt.items() if d["total"] > 0
        )
        return f"acc={self.acc:.3f} ({self.correct}/{self.total})  parse_err={self.parse_errors}  [{pp}]"


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_dimension(path: Path, max_per_dim: int):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if len(rows) > max_per_dim:
        rows = random.sample(rows, max_per_dim)
    return rows

def row_to_english(row):
    context  = row.get("STORY", "")
    question = row.get("QUESTION", "")
    options  = [o for o in [
        f"answerA: {row.get('OPTION-A','')}" if row.get('OPTION-A') else None,
        f"answerB: {row.get('OPTION-B','')}" if row.get('OPTION-B') else None,
        f"answerC: {row.get('OPTION-C','')}" if row.get('OPTION-C') else None,
        f"answerD: {row.get('OPTION-D','')}" if row.get('OPTION-D') else None,
    ] if o]
    gt  = row.get("答案\nANSWER", row.get("ANSWER", "")).strip().upper()
    idx = row.get("序号\nINDEX", "?")
    return context, question, options, gt, idx


# ─────────────────────────────────────────────
# RUN ONE MODEL ON ONE DIMENSION
# ─────────────────────────────────────────────

def run_dimension(model, tokenizer, rows, dim_name, label, prepped):
    """
    prepped: list of (pv_name, sys_p, usr_p, gt, idx) — same order for base and tuned.
    Returns list of (answer, parsed, gt, pv_name, idx).
    """
    metrics = Metrics()
    outputs = []
    n = len(prepped)
    for i, (pv_name, sys_p, usr_p, gt, idx) in enumerate(prepped):
        ans, parsed = infer(model, tokenizer, sys_p, usr_p)
        hit = metrics.update(ans, gt, pv_name, parsed is not None)
        sym = "✓" if hit else "✗"
        print(f"  [{label}][{i+1:>3}/{n}] prompt={pv_name} gt={gt} pred={ans}{sym}  acc={metrics.acc:.3f}")
        outputs.append((ans, parsed, gt, pv_name, idx))
    return metrics, outputs


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=50,
                        help="Max examples per dimension (default: 50)")
    args = parser.parse_args()
    max_per_dim = args.max

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path     = Path(RESULTS_DIR) / f"benchmark_{ts}.jsonl"
    summary_path = Path(RESULTS_DIR) / f"summary_{ts}.json"

    # tokenizer loaded once — CPU only, lightweight
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = sorted(Path(DATA_DIR).glob("*.jsonl"))
    if not data_files:
        print(f"No .jsonl files found in {DATA_DIR}")
        return

    print(f"Found {len(data_files)} dimension(s). Max per dim: {max_per_dim}\n")

    all_summary = {}
    results_log = []

    for dfile in data_files:
        dim_name = dfile.stem
        rows     = load_dimension(dfile, max_per_dim)
        n        = len(rows)

        print(f"\n{'═'*60}")
        print(f"  DIMENSION: {dim_name}  ({n} examples)")
        print(f"{'═'*60}")

        # pre-assign prompt versions so base and tuned see identical inputs
        prepped = []
        for row in rows:
            context, question, options, gt, idx = row_to_english(row)
            if not context or not question or not options or not gt:
                continue
            pv_idx  = random.randint(0, 3)
            pv_name = f"v{pv_idx+1}"
            sys_p, usr_p = PROMPT_FNS[pv_idx](context, question, options)
            prepped.append((pv_name, sys_p, usr_p, gt, idx))

        # ── BASE ──
        base_model = load_base(tokenizer)
        base_metrics, base_outputs = run_dimension(base_model, tokenizer, rows, dim_name, "BASE ", prepped)
        unload(base_model)

        # ── TUNED ──
        tuned_model = load_tuned(tokenizer)
        tuned_metrics, tuned_outputs = run_dimension(tuned_model, tokenizer, rows, dim_name, "TUNED", prepped)
        unload(tuned_model)

        # ── dimension summary ──
        delta = tuned_metrics.acc - base_metrics.acc
        print(f"\n  ── {dim_name} FINAL ──")
        print(f"  BASE : {base_metrics.summary()}")
        print(f"  TUNED: {tuned_metrics.summary()}")
        print(f"  Δacc : {delta:+.3f}")

        # log results
        for (b_ans, b_parsed, gt, pv, idx), (t_ans, t_parsed, _, _, _) in zip(base_outputs, tuned_outputs):
            results_log.append({
                "dimension":          dim_name,
                "index":              idx,
                "prompt_version":     pv,
                "gt":                 gt,
                "base_answer":        b_ans,
                "tuned_answer":       t_ans,
                "base_correct":       b_ans == gt,
                "tuned_correct":      t_ans == gt,
                "base_hypothesis":    b_parsed.get("Hypothesis", "")       if b_parsed else "",
                "tuned_hypothesis":   t_parsed.get("Hypothesis", "")       if t_parsed else "",
                "base_mental_state":  b_parsed.get("Mental State Type", "") if b_parsed else "",
                "tuned_mental_state": t_parsed.get("Mental State Type", "") if t_parsed else "",
            })

        all_summary[dim_name] = {
            "n":                  len(prepped),
            "base_acc":           round(base_metrics.acc,  4),
            "tuned_acc":          round(tuned_metrics.acc, 4),
            "delta":              round(delta, 4),
            "base_parse_errors":  base_metrics.parse_errors,
            "tuned_parse_errors": tuned_metrics.parse_errors,
            "base_per_prompt":    base_metrics.per_prompt,
            "tuned_per_prompt":   tuned_metrics.per_prompt,
        }

        # flush results after each dimension in case of crash
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results_log:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summary, f, ensure_ascii=False, indent=2)

    # ── overall summary ──
    all_base  = [v["base_acc"]  for v in all_summary.values()]
    all_tuned = [v["tuned_acc"] for v in all_summary.values()]
    print(f"\n{'═'*60}")
    print(f"  OVERALL SUMMARY  (max={max_per_dim} per dim)")
    print(f"{'═'*60}")
    for dim, s in all_summary.items():
        print(f"  {dim:<40} base={s['base_acc']:.3f}  tuned={s['tuned_acc']:.3f}  Δ={s['delta']:+.3f}")
    print(f"  {'─'*55}")
    print(f"  Mean base acc : {sum(all_base)/len(all_base):.3f}")
    print(f"  Mean tuned acc: {sum(all_tuned)/len(all_tuned):.3f}")
    print(f"\n  Results → {out_path}")
    print(f"  Summary → {summary_path}")


if __name__ == "__main__":
    main()