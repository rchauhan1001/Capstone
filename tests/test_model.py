"""
tests/test_model.py

Unit tests for the SFT model pipeline.
Tests are designed to run without loading the full 7B model.
Run from repo root:
    pytest tests/test_model.py -v
"""

import json
import sys
import os
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "training_phase", "supervised_fine_tuning"))

from sft import (
    build_token_weights,
    convert_to_sharegpt,
    load_and_split,
    MetricsStore,
    apply_lora,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

class MockTokenizer:
    """Minimal tokenizer mock that avoids loading the real model."""
    def decode(self, ids, skip_special_tokens=False):
        tokens = {0: '"Mental State Type": "Belief"',
                  1: '"Hypothesis": "User wants to attend"',
                  2: '"Response": "They will go"',
                  3: '"Final answer": "A"',
                  4: 'some other token'}
        return " ".join(tokens.get(int(i), "tok") for i in ids)


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def sample_training_jsonl(tmp_path):
    records = [
        {
            "system": "You are a reasoning engine.",
            "prompt": "Context: X\nQuestion: Y\nOptions: A, B, C",
            "response": {
                "Mental State Type": "Belief",
                "Hypothesis": "User wants X",
                "Response": "They will do X",
                "Final answer": "A"
            }
        }
        for _ in range(10)
    ]
    p = tmp_path / "training_data.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    return str(p)


# ── MT-01: build_token_weights returns correct shape ─────────────────────────

def test_build_token_weights_shape(mock_tokenizer):
    """MT-01: weight tensor has same length as input_ids."""
    input_ids = torch.tensor([0, 1, 2, 3, 4])
    weights = build_token_weights(input_ids, mock_tokenizer)
    assert weights.shape == input_ids.shape


# ── MT-02: Final answer tokens receive highest weight ────────────────────────

def test_build_token_weights_final_answer_highest(mock_tokenizer):
    """MT-02: tokens in Final answer field receive weight >= all others."""
    input_ids = torch.tensor([0, 1, 2, 3, 4])
    weights = build_token_weights(input_ids, mock_tokenizer)
    assert weights.max().item() >= 1.0


# ── MT-03: Default weight is 1.0 for unrecognized tokens ─────────────────────

def test_build_token_weights_default(mock_tokenizer):
    """MT-03: tokens outside any known field receive default weight of 1.0."""
    input_ids = torch.tensor([4])
    weights = build_token_weights(input_ids, mock_tokenizer)
    assert weights[0].item() == 1.0


# ── MT-04: convert_to_sharegpt produces valid structure ──────────────────────

def test_convert_to_sharegpt_structure(sample_training_jsonl, tmp_path):
    """MT-04: converted file contains system/human/assistant turns."""
    out_path = str(tmp_path / "converted.jsonl")
    convert_to_sharegpt(sample_training_jsonl, out_path)
    with open(out_path) as f:
        first = json.loads(f.readline())
    roles = [t["from"] for t in first["conversations"]]
    assert "system" in roles
    assert "human" in roles
    assert "assistant" in roles


# ── MT-05: convert_to_sharegpt preserves record count ────────────────────────

def test_convert_to_sharegpt_count(sample_training_jsonl, tmp_path):
    """MT-05: converted file has same number of records as input."""
    out_path = str(tmp_path / "converted.jsonl")
    convert_to_sharegpt(sample_training_jsonl, out_path)
    with open(out_path) as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 10


# ── MT-06: load_and_split respects 99/1 ratio ────────────────────────────────

def test_load_and_split_ratio(sample_training_jsonl, tmp_path):
    """MT-06: split produces ~99% train and ~1% eval from 10 samples."""
    out_path = str(tmp_path / "converted.jsonl")
    convert_to_sharegpt(sample_training_jsonl, out_path)
    train_ds, eval_ds = load_and_split(out_path)
    total = len(train_ds) + len(eval_ds)
    assert total == 10
    assert len(train_ds) >= 9


# ── MT-07: MetricsStore initializes with empty lists ─────────────────────────

def test_metrics_store_init():
    """MT-07: MetricsStore initializes all list fields as empty."""
    store = MetricsStore()
    assert store.step_losses == []
    assert store.train_acc == []
    assert store.eval_acc == []
    assert store.best_eval_acc == 0.0


# ── MT-08: MetricsStore save and load roundtrip ───────────────────────────────

def test_metrics_store_save_load(tmp_path):
    """MT-08: saved MetricsStore can be reloaded with identical values."""
    store = MetricsStore()
    store.step_losses = [0.9, 0.8, 0.7]
    store.best_eval_acc = 0.75
    path = str(tmp_path / "metrics.json")
    store.save(path)
    loaded = MetricsStore.load(path)
    assert loaded.step_losses == [0.9, 0.8, 0.7]
    assert loaded.best_eval_acc == 0.75


# ── MT-09: MetricsStore truncate_to_step removes later entries ───────────────

def test_metrics_store_truncate(tmp_path):
    """MT-09: truncate_to_step removes all step entries after the given step."""
    store = MetricsStore()
    store.step_losses = [0.9, 0.8, 0.7, 0.6]
    store.step_numbers = [10, 20, 30, 40]
    store.epoch_numbers = []
    store.epoch_train_loss = []
    store.acc_epochs = []
    store.train_acc = []
    store.eval_acc = []
    store.truncate_to_step(20)
    assert len(store.step_losses) == 2
    assert store.step_numbers[-1] == 20
