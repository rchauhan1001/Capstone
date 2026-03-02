"""
Cluster runner for MetaMind on Social IQA dataset.
Uses local vLLM server serving gpt-oss-120b.

Usage:
    python run_socialiqa_cluster.py --max_samples 5
    python run_socialiqa_cluster.py --start_sample 663 --max_samples 1000
"""

import json
import re
import time
import os
import argparse
import logging

from config import TOM_AGENT_CONFIG, DOMAIN_AGENT_CONFIG, RESPONSE_AGENT_CONFIG
from llm_interface import DirectVLLM
from memory import SocialMemory
from agents import ToMAgent, DomainAgent, ResponseAgent
from utils.helpers import setup_logger

logger = setup_logger("SocialIQA_Cluster", level=logging.INFO)

SCRATCH = os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER', 'laredo.ei')}")

LOCAL_LLM_CONFIG = {
    "model_path": f"{SCRATCH}/models/gpt-oss-120b-hf",   # ADD THIS
    "model_name": "openai/gpt-oss-120b",
    "default_max_tokens": 2048,
    "default_temperature": 0.7,
    "enforce_eager": True,
    "log_path": os.path.join(SCRATCH, "results", f"inference_log_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"),
}

DEFAULT_DEV_PATH = os.path.join(SCRATCH, "socialiqa-train-dev", "dev.jsonl")
DEFAULT_LABELS_PATH = os.path.join(SCRATCH, "socialiqa-train-dev", "dev-labels.lst")
DEFAULT_OUTPUT_DIR = os.path.join(SCRATCH, "results")


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


def build_user_input(sample):
    return (
        f"Context: {sample['context']}\n"
        f"Question: {sample['question']}\n"
        f"Answer choices:\n"
        f"  A: {sample['answerA']}\n"
        f"  B: {sample['answerB']}\n"
        f"  C: {sample['answerC']}\n"
        f"\nWhich answer (A, B, or C) is most appropriate? "
        f"Explain your reasoning about the social situation, "
        f'then finalize your answer with "ANSWER: <letter>".'
    )


def extract_answer(response_text):
    m = re.search(r'ANSWER:\s*([ABC])', response_text.strip().upper())
    return m.group(1) if m else "UNKNOWN"


def run(args):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir, f"socialiqa_results_{timestamp}.jsonl")
    summary_path = os.path.join(args.output_dir, f"summary_{timestamp}.json")

    logger.info("Initializing MetaMind pipeline with local vLLM...")
    llm = DirectVLLM(LOCAL_LLM_CONFIG)

    social_memory = SocialMemory(llm_interface=llm)
    tom_agent = ToMAgent(llm_interface=llm, social_memory_interface=social_memory, config=TOM_AGENT_CONFIG)
    domain_agent = DomainAgent(llm_interface=llm, social_memory_interface=social_memory, config=DOMAIN_AGENT_CONFIG)
    response_agent = ResponseAgent(llm_interface=llm, config=RESPONSE_AGENT_CONFIG)

    logger.info(f"Model: {LOCAL_LLM_CONFIG['model_name']} @ {LOCAL_LLM_CONFIG['model_path']}")
    logger.info(f"Inference log: {LOCAL_LLM_CONFIG['log_path']}")
    logger.info(f"Results: {results_path}")

    samples, labels = load_socialiqa(args.dev_path, args.labels_path)

    # Apply start_sample offset
    start_idx = args.start_sample - 1  # convert to 0-based
    if start_idx > 0:
        samples = samples[start_idx:]
        labels = labels[start_idx:]
        logger.info(f"Starting from sample {args.start_sample} (skipping {start_idx})")

    if args.max_samples:
        samples = samples[:args.max_samples]
        labels = labels[:args.max_samples]
        logger.info(f"Processing up to {args.max_samples} samples from start point")

    logger.info(f"Total samples to process: {len(samples)}")

    correct = 0
    total = 0
    label_map = {1: "A", 2: "B", 3: "C"}

    with open(results_path, "w") as fout:
        for idx, (sample, gold_label_num) in enumerate(zip(samples, labels), 1):
            actual_sample_id = idx + start_idx
            gold_answer = label_map[gold_label_num]
            user_input = build_user_input(sample)

            logger.info(f"--- Sample {actual_sample_id} (batch {idx}/{len(samples)}) (Gold: {gold_answer}) ---")
            start = time.time()

            try:
                hypotheses = tom_agent.process(user_input=user_input, conversation_context=[])

                selected_hypothesis = None
                if hypotheses:
                    selected_hypothesis = domain_agent.process(
                        hypotheses=hypotheses, user_input=user_input, conversation_context=[]
                    )

                response_details = None
                final_response = ""
                if selected_hypothesis:
                    social_mem = social_memory.get_summary(user_id="default_user")
                    if not isinstance(social_mem, dict):
                        social_mem = {"summary": str(social_mem)}
                    response_details = response_agent.process(
                        selected_hypothesis=selected_hypothesis,
                        user_input=user_input,
                        conversation_context=[],
                        social_memory=social_mem,
                    )
                    final_response = response_details.get("response", "")

                predicted = extract_answer(final_response)
                is_correct = predicted == gold_answer
                if is_correct:
                    correct += 1
                total += 1
                elapsed = time.time() - start

                result = {
                    "sample_id": actual_sample_id,
                    "input": sample,
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted,
                    "correct": is_correct,
                    "final_response": final_response,
                    "tom_hypotheses": hypotheses,
                    "selected_hypothesis": selected_hypothesis,
                    "response_details": response_details,
                    "elapsed_seconds": round(elapsed, 2),
                    "llm_calls_so_far": llm.call_count,
                }

                fout.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                fout.flush()

                logger.info(
                    f"  Predicted: {predicted} | Gold: {gold_answer} | "
                    f"{'CORRECT' if is_correct else 'WRONG'} | "
                    f"{elapsed:.1f}s | Accuracy: {correct}/{total} ({100*correct/total:.1f}%)"
                )

            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"  Error on sample {actual_sample_id}: {e}", exc_info=True)
                total += 1
                fout.write(json.dumps({
                    "sample_id": actual_sample_id, "input": sample, "gold_answer": gold_answer,
                    "error": str(e), "elapsed_seconds": round(elapsed, 2),
                }, ensure_ascii=False) + "\n")
                fout.flush()

    accuracy = correct / total if total > 0 else 0
    summary = {
        "timestamp": timestamp,
        "model": LOCAL_LLM_CONFIG["model_name"],
        "start_sample": args.start_sample,
        "total_samples": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "total_llm_calls": llm.call_count,
        "results_file": results_path,
        "inference_log_file": LOCAL_LLM_CONFIG["log_path"],
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
    parser.add_argument("--dev_path", type=str, default=DEFAULT_DEV_PATH)
    parser.add_argument("--labels_path", type=str, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_sample", type=int, default=1, help="1-based sample index to start from")
    args = parser.parse_args()
    run(args)