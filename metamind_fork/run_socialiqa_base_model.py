"""
Base Model runner for Social IQA dataset.
Uses DirectVLLM interface directly serving gpt-oss-20b with local inference.
Strips MetaMind pipeline — used to generate rejected responses for DPO.

Usage:
    python run_socialiqa_base_model.py --max_samples 5
    python run_socialiqa_base_model.py --start_sample 663 --max_samples 1000
"""

import json
import re
import time
import os
import argparse
import logging

from llm_interface import DirectVLLM
from utils.helpers import setup_logger

logger = setup_logger("SocialIQA_BaseModel", level=logging.INFO)

SCRATCH = os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER', 'laredo.ei')}")

LOCAL_LLM_CONFIG = {
    "model_path": f"{SCRATCH}/models/gpt-oss-20b-hf",
    "model_name": "openai/gpt-oss-20b",
    "default_max_tokens": 2048,
    "default_temperature": 0.7,
    "enforce_eager": True,
    "log_path": os.path.join(
        SCRATCH, "results",
        f"inference_log_basemodel_{time.strftime('%Y%m%d_%H%M%S')}_s{os.environ.get('START_SAMPLE', '1')}.jsonl"
    ),
}

DEFAULT_DEV_PATH    = os.path.join(SCRATCH, "socialiqa-train-dev", "basemodel_samples.jsonl")
DEFAULT_LABELS_PATH = os.path.join(SCRATCH, "socialiqa-train-dev", "basemodel_samples_labels.lst")
DEFAULT_OUTPUT_DIR  = os.path.join(SCRATCH, "results")

MAX_RESPONSE_TOKENS = 256
MAX_RETRIES = 2

BASE_SYSTEM_PROMPT = (
    "You are a precise reasoning engine. You receive tasks and produce only the requested output — nothing more. "
    "You never narrate, plan, or describe your process. You never begin with phrases like 'we need to', "
    "'analysis', 'let me', or 'the task is'. You simply reason and output."
)


def load_socialiqa(dev_path, labels_path):
    samples = []
    with open(dev_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    labels = []
    with open(labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line))
    assert len(samples) == len(labels), f"Mismatch: {len(samples)} samples vs {len(labels)} labels"
    logger.info(f"Loaded {len(samples)} Social IQA samples")
    return samples, labels


def build_prompt(context, question, q_options):
    user_prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Options:\n"
        f"  A: {q_options[0]}\n"
        f"  B: {q_options[1]}\n"
        f"  C: {q_options[2]}\n\n"
        f"Briefly explain your reasoning about the social situation, "
        f'then finalize your answer with "ANSWER: <letter>".'
    )
    return BASE_SYSTEM_PROMPT, user_prompt


def extract_answer(response_text):
    m = re.search(r'ANSWER:\s*([ABC])', response_text.strip().upper())
    return m.group(1) if m else "UNKNOWN"


def run(args):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir, f"socialiqa_basemodel_results_{timestamp}_s{args.start_sample}.jsonl")
    summary_path = os.path.join(args.output_dir, f"summary_basemodel_{timestamp}_s{args.start_sample}.json")

    logger.info("Initializing local vLLM (base model, no MetaMind pipeline)...")
    llm = DirectVLLM(LOCAL_LLM_CONFIG)
    logger.info(f"Model: {LOCAL_LLM_CONFIG['model_name']} @ {LOCAL_LLM_CONFIG['model_path']}")
    logger.info(f"Inference log: {LOCAL_LLM_CONFIG['log_path']}")
    logger.info(f"Results: {results_path}")

    samples, labels = load_socialiqa(args.dev_path, args.labels_path)

    # Apply start_sample offset (1-based)
    start_idx = args.start_sample - 1
    if start_idx > 0:
        samples = samples[start_idx:]
        labels  = labels[start_idx:]
        logger.info(f"Starting from sample {args.start_sample} (skipping {start_idx})")

    if args.max_samples:
        samples = samples[:args.max_samples]
        labels  = labels[:args.max_samples]
        logger.info(f"Processing up to {args.max_samples} samples from start point")

    logger.info(f"Total samples to process: {len(samples)}")

    correct   = 0
    total     = 0
    label_map = {1: "A", 2: "B", 3: "C"}

    with open(results_path, "w") as fout:
        for idx, (sample, gold_label_num) in enumerate(zip(samples, labels), 1):
            actual_sample_id = idx + start_idx
            gold_answer      = label_map[gold_label_num]
            q_options        = [sample["answerA"], sample["answerB"], sample["answerC"]]

            system_prompt, user_prompt = build_prompt(
                context=sample["context"],
                question=sample["question"],
                q_options=q_options,
            )

            logger.info(f"--- Sample {actual_sample_id} (batch {idx}/{len(samples)}) (Gold: {gold_answer}) ---")
            start = time.time()

            last_error     = None
            final_response = None

            for attempt in range(1, MAX_RETRIES + 2):  # attempts: 1, 2, 3
                try:
                    final_response = llm.generate(user_prompt, max_tokens=MAX_RESPONSE_TOKENS)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"  Attempt {attempt}/{MAX_RETRIES + 1} failed for sample {actual_sample_id}: {e}")
                    if attempt < MAX_RETRIES + 1:
                        time.sleep(1)

            elapsed = time.time() - start
            total  += 1

            if last_error is not None:
                logger.error(f"  Sample {actual_sample_id} failed after {MAX_RETRIES + 1} attempts: {last_error}", exc_info=True)
                fout.write(json.dumps({
                    "sample_id":       actual_sample_id,
                    "input":           sample,
                    "gold_answer":     gold_answer,
                    "error":           str(last_error),
                    "attempts":        MAX_RETRIES + 1,
                    "elapsed_seconds": round(elapsed, 2),
                }, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            predicted  = extract_answer(final_response)
            is_correct = predicted == gold_answer
            if is_correct:
                correct += 1

            result = {
                "sample_id":        actual_sample_id,
                "input":            sample,
                "gold_answer":      gold_answer,
                "predicted_answer": predicted,
                "correct":          is_correct,
                "final_response":   final_response,
                "attempts":         attempt,
                "elapsed_seconds":  round(elapsed, 2),
                "llm_calls_so_far": llm.call_count,
            }

            fout.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            fout.flush()

            logger.info(
                f"  Predicted: {predicted} | Gold: {gold_answer} | "
                f"{'CORRECT' if is_correct else 'WRONG'} | "
                f"Attempt: {attempt} | {elapsed:.1f}s | Accuracy: {correct}/{total} ({100*correct/total:.1f}%)"
            )

    accuracy = correct / total if total > 0 else 0
    summary = {
        "timestamp":       timestamp,
        "model":           LOCAL_LLM_CONFIG["model_name"],
        "start_sample":    args.start_sample,
        "total_samples":   total,
        "correct":         correct,
        "accuracy":        round(accuracy, 4),
        "total_llm_calls": llm.call_count,
        "results_file":    results_path,
        "inference_log":   LOCAL_LLM_CONFIG["log_path"],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"FINAL ACCURACY: {correct}/{total} ({100*accuracy:.1f}%)")
    logger.info(f"Total LLM calls: {llm.call_count}")
    logger.info(f"Results: {results_path}")
    logger.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_path",     type=str, default=DEFAULT_DEV_PATH)
    parser.add_argument("--labels_path",  type=str, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output_dir",   type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_samples",  type=int, default=None)
    parser.add_argument("--start_sample", type=int, default=1,
                        help="1-based sample index to start from")
    args = parser.parse_args()
    run(args)