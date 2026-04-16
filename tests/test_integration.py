"""
tests/test_integration.py

Integration and end-to-end tests for the full pipeline.
Run from repo root:
    pytest tests/test_integration.py -v
"""

import json
import sys
import os
import pytest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "training_phase", "reinforcement_learning"))
sys.path.insert(0, os.path.join(REPO_ROOT, "training_phase", "supervised_fine_tuning"))

from prepare_dpo_dataset import (
    strip_answer_suffix,
    make_key,
    validate_sample,
    build_user_message,
    build_chat,
    load_jsonl,
)
from sft import convert_to_sharegpt, load_and_split, MetricsStore


# ── Fixture paths ─────────────────────────────────────────────────────────────

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


# ── IT-01: Full DPO data pipeline produces valid output ──────────────────────

def test_dpo_pipeline_produces_valid_jsonl(tmp_path):
    """IT-01: DPO pipeline on fixture data produces valid prompt/chosen/rejected pairs."""
    base_samples = [
        {
            "sample_id": 1,
            "input": {"context": "Alex helped a friend move.", "question": "How would Alex feel?",
                      "answerA": "proud", "answerB": "sad", "answerC": "angry"},
            "gold_answer": "A", "predicted_answer": "A", "correct": True,
            "final_response": "Alex would feel proud.\n\nANSWER: A"
        },
        {
            "sample_id": 2,
            "input": {"context": "Jordan forgot a birthday.", "question": "How would Jordan feel?",
                      "answerA": "happy", "answerB": "guilty", "answerC": "excited"},
            "gold_answer": "B", "predicted_answer": "B", "correct": True,
            "final_response": "Jordan would feel guilty.\n\nANSWER: B"
        },
    ]
    metamind_samples = [
        {
            "sample_id": 10,
            "input": {"context": "Alex helped a friend move.", "question": "How would Alex feel?",
                      "answerA": "proud", "answerB": "sad", "answerC": "angry"},
            "gold_answer": "A", "predicted_answer": "A", "correct": True,
            "final_response": "Alex demonstrated prosocial behavior. Helping others typically generates positive affect. The most likely mental state is pride.\n\nANSWER: A"
        },
        {
            "sample_id": 11,
            "input": {"context": "Jordan forgot a birthday.", "question": "How would Jordan feel?",
                      "answerA": "happy", "answerB": "guilty", "answerC": "excited"},
            "gold_answer": "B", "predicted_answer": "B", "correct": True,
            "final_response": "Forgetting a birthday violates social obligations. Jordan would experience guilt as a moral emotion.\n\nANSWER: B"
        },
    ]

    base_file = tmp_path / "base.jsonl"
    mm_file = tmp_path / "metamind.jsonl"
    base_file.write_text("\n".join(json.dumps(s) for s in base_samples))
    mm_file.write_text("\n".join(json.dumps(s) for s in metamind_samples))

    base_records, _ = load_jsonl(str(base_file))
    mm_records, _   = load_jsonl(str(mm_file))

    base_index = {make_key(s): s for s in base_records}
    mm_index   = {make_key(s): s for s in mm_records}

    pairs = []
    for key in base_index:
        if key in mm_index:
            b = base_index[key]
            m = mm_index[key]
            if b["correct"] and m["correct"]:
                chosen   = strip_answer_suffix(m["final_response"])
                rejected = strip_answer_suffix(b["final_response"])
                if chosen.strip() and rejected.strip() and chosen != rejected:
                    pairs.append({
                        "prompt":   build_chat([("system", "You are a social reasoning assistant."),
                                                ("user", build_user_message(b))]),
                        "chosen":   build_chat([("assistant", chosen)]),
                        "rejected": build_chat([("assistant", rejected)]),
                    })

    assert len(pairs) == 2
    for pair in pairs:
        assert "prompt" in pair
        assert "chosen" in pair
        assert "rejected" in pair
        chosen_text   = pair["chosen"][0]["content"]
        rejected_text = pair["rejected"][0]["content"]
        assert "ANSWER" not in chosen_text
        assert "ANSWER" not in rejected_text
        assert chosen_text != rejected_text


# ── IT-02: SFT data conversion produces trainer-ready format ─────────────────

def test_sft_conversion_produces_trainer_format(tmp_path):
    """IT-02: convert_to_sharegpt followed by load_and_split produces Dataset objects."""
    records = [
        {
            "system": "You are a reasoning engine.",
            "prompt": f"Context: Scenario {i}\nQuestion: What happens?\nOptions: A, B, C",
            "response": {"Mental State Type": "Belief", "Hypothesis": f"Hyp {i}",
                         "Response": f"Response {i}", "Final answer": "A"}
        }
        for i in range(20)
    ]
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in records))
    out = str(tmp_path / "converted.jsonl")

    convert_to_sharegpt(str(raw), out)
    train_ds, eval_ds = load_and_split(out)

    assert len(train_ds) + len(eval_ds) == 20
    assert len(train_ds) >= 19
    first = train_ds[0]
    assert "conversations" in first
    roles = [t["from"] for t in first["conversations"]]
    assert set(roles) == {"system", "human", "assistant"}


# ── IT-03: DPO dataset integrity check on real data ──────────────────────────

def test_dpo_dataset_integrity():
    """IT-03: real dpo_dataset.jsonl has valid structure throughout."""
    dpo_path = os.path.join(REPO_ROOT, "training_phase", "reinforcement_learning",
                            "dpo_data", "dpo_dataset.jsonl")
    if not os.path.exists(dpo_path):
        pytest.skip("dpo_dataset.jsonl not present locally")

    records, _ = load_jsonl(dpo_path)
    assert len(records) > 0

    for i, record in enumerate(records[:100]):
        assert "prompt" in record,   f"Record {i} missing prompt"
        assert "chosen" in record,   f"Record {i} missing chosen"
        assert "rejected" in record, f"Record {i} missing rejected"
        chosen_text   = record["chosen"][0]["content"]
        rejected_text = record["rejected"][0]["content"]
        assert chosen_text != rejected_text, f"Record {i} has identical chosen/rejected"
        assert "ANSWER" not in chosen_text,   f"Record {i} chosen has unsstripped ANSWER"
        assert "ANSWER" not in rejected_text, f"Record {i} rejected has unstripped ANSWER"


# ── IT-04: Split reproducibility with fixed seed ─────────────────────────────

def test_split_reproducibility(tmp_path):
    """IT-04: load_and_split with same seed produces identical splits on two runs."""
    records = [
        {
            "system": "sys", "prompt": f"prompt {i}",
            "response": {"Mental State Type": "Belief", "Hypothesis": "h",
                         "Response": "r", "Final answer": "A"}
        }
        for i in range(50)
    ]
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in records))
    out = str(tmp_path / "converted.jsonl")
    convert_to_sharegpt(str(raw), out)

    train1, eval1 = load_and_split(out)
    train2, eval2 = load_and_split(out)

    assert len(train1) == len(train2)
    assert len(eval1)  == len(eval2)
    assert train1[0]["conversations"] == train2[0]["conversations"]


# ── IT-05: MetricsStore full save/load/truncate cycle ────────────────────────

def test_metrics_store_full_cycle(tmp_path):
    """IT-05: MetricsStore survives a full save, load, and truncate cycle."""
    store = MetricsStore()
    store.step_losses   = [0.9, 0.8, 0.7, 0.6, 0.5]
    store.step_numbers  = [10, 20, 30, 40, 50]
    store.epoch_numbers = [20, 40]
    store.epoch_train_loss = [0.85, 0.65]
    store.acc_epochs    = [1, 2]
    store.train_acc     = [0.4, 0.6]
    store.eval_acc      = [0.38, 0.55]
    store.best_eval_acc = 0.55
    store.best_epoch    = 2

    path = str(tmp_path / "metrics.json")
    store.save(path)

    loaded = MetricsStore.load(path)
    loaded.truncate_to_step(30)

    assert len(loaded.step_losses)  == 3
    assert loaded.step_numbers[-1]  == 30
    assert len(loaded.epoch_numbers) == 1
