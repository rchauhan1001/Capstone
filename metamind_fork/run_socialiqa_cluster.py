"""
Cluster runner for MetaMind on Social IQA dataset.
Uses local vLLM server serving gpt-oss-120b.

Usage:
    python run_socialiqa_cluster.py --max_samples 5
"""

import json
import time
import os
import argparse
import logging

from config import TOM_AGENT_CONFIG, DOMAIN_AGENT_CONFIG, RESPONSE_AGENT_CONFIG
from llm_interface import LocalVLLM
from memory import SocialMemory
from agents import ToMAgent, DomainAgent, ResponseAgent
from utils.helpers import setup_logger

logger = setup_logger("SocialIQA_Cluster", level=logging.INFO)

SCRATCH = os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER', 'laredo.ei')}")

LOCAL_LLM_CONFIG = {
    "base_url": "http://localhost:8000/v1",
    "model_name": "openai/gpt-oss-120b",
    "api_key": "not-needed",
    "default_max_tokens": 2048,
    "default_temperature": 0.7,
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
        f"then state your final answer."
    )


def extract_answer(response_text):
    text = response_text.strip().upper()
    last_line = text.strip().split('\n')[-1]
    for letter in ['A', 'B', 'C']:
        if letter in last_line and len(last_line) < 50:
            return letter
    for pattern in ['FINAL ANSWER: ', 'ANSWER: ', 'THE ANSWER IS ', 'I CHOOSE ']:
        if pattern in text:
            after = text.split(pattern)[-1].strip()
            if after and after[0] in 'ABC':
                return after[0]
    for char in reversed(text):
        if char in 'ABC':
            return char
    return "UNKNOWN"


def run(args):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir, f"socialiqa_results_{timestamp}.jsonl")
    summary_path = os.path.join(args.output_dir, f"summary_{timestamp}.json")

    # --- Initialize with LocalVLLM ---
    logger.info("Initializing MetaMind pipeline with local vLLM...")
    llm = LocalVLLM(LOCAL_LLM_CONFIG)

    social_memory = SocialMemory(llm_interface=llm)
    tom_agent = ToMAgent(llm_interface=llm, social_memory_interface=social_memory, config=TOM_AGENT_CONFIG)
    domain_agent = DomainAgent(llm_interface=llm, social_memory_interface=social_memory, config=DOMAIN_AGENT_CONFIG)
    response_agent = ResponseAgent(llm_interface=llm, config=RESPONSE_AGENT_CONFIG)

    logger.info(f"Model: {LOCAL_LLM_CONFIG['model_name']} @ {LOCAL_LLM_CONFIG['base_url']}")
    logger.info(f"Inference log: {LOCAL_LLM_CONFIG['log_path']}")
    logger.info(f"Results: {results_path}")

    # --- Load data ---
    samples, labels = load_socialiqa(args.dev_path, args.labels_path)

    if args.max_samples:
        samples = samples[:args.max_samples]
        labels = labels[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")

    # --- Process ---
    correct = 0
    total = 0
    label_map = {1: "A", 2: "B", 3: "C"}

    with open(results_path, "w") as fout:
        for idx, (sample, gold_label_num) in enumerate(zip(samples, labels), 1):
            gold_answer = label_map[gold_label_num]
            user_input = build_user_input(sample)

            logger.info(f"--- Sample {idx}/{len(samples)} (Gold: {gold_answer}) ---")
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
                    "sample_id": idx,
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
                logger.error(f"  Error on sample {idx}: {e}", exc_info=True)
                total += 1
                fout.write(json.dumps({
                    "sample_id": idx, "input": sample, "gold_answer": gold_answer,
                    "error": str(e), "elapsed_seconds": round(elapsed, 2),
                }, ensure_ascii=False) + "\n")
                fout.flush()

    accuracy = correct / total if total > 0 else 0
    summary = {
        "timestamp": timestamp,
        "model": LOCAL_LLM_CONFIG["model_name"],
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
    args = parser.parse_args()
    run(args)