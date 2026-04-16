"""
tests/test_data_pipeline.py

Unit tests for the DPO data preparation pipeline.
Run from repo root:
    pytest tests/test_data_pipeline.py -v
"""

import json
import sys
import os
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "training_phase", "reinforcement_learning"))

from prepare_dpo_dataset import (
    strip_answer_suffix,
    make_key,
    validate_sample,
    build_user_message,
    build_chat,
    load_jsonl,
)
# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_metamind_sample():
    return {
        "sample_id": 1,
        "input": {
            "context": "Cameron decided to have a barbecue.",
            "question": "How would others feel?",
            "answerA": "happy",
            "answerB": "sad",
            "answerC": "indifferent",
        },
        "gold_answer": "A",
        "predicted_answer": "A",
        "correct": True,
        "final_response": "People enjoy barbecues.\n\nANSWER: A",
    }

@pytest.fixture
def valid_base_sample():
    return {
        "sample_id": 1,
        "input": {
            "context": "Cameron decided to have a barbecue.",
            "question": "How would others feel?",
            "answerA": "happy",
            "answerB": "sad",
            "answerC": "indifferent",
        },
        "gold_answer": "A",
        "predicted_answer": "A",
        "correct": True,
        "final_response": "They would feel happy.\n\nANSWER: A",
    }

@pytest.fixture
def sample_jsonl(tmp_path):
    records = [
        {"sample_id": 1, "input": {"context": "ctx1", "question": "q1"}, "correct": True, "final_response": "resp1"},
        {"sample_id": 2, "input": {"context": "ctx2", "question": "q2"}, "correct": False, "final_response": "resp2"},
    ]
    p = tmp_path / "test.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    return str(p)


# ── DP-01: strip_answer_suffix normal case ────────────────────────────────────

def test_strip_answer_suffix_newline():
    """DP-01: strips standard ANSWER: X suffix with newline."""
    text = "The answer is clear.\n\nANSWER: B"
    result = strip_answer_suffix(text)
    assert "ANSWER" not in result
    assert "The answer is clear." in result


# ── DP-02: strip_answer_suffix bold markdown variant ─────────────────────────

def test_strip_answer_suffix_bold():
    """DP-02: strips **ANSWER: B** bold markdown variant."""
    text = "Reasoning here.\n\n**ANSWER: B**"
    result = strip_answer_suffix(text)
    assert "ANSWER" not in result


# ── DP-03: strip_answer_suffix no suffix present ─────────────────────────────

def test_strip_answer_suffix_no_suffix():
    """DP-03: returns text unchanged when no ANSWER suffix is present."""
    text = "This response has no answer declaration."
    result = strip_answer_suffix(text)
    assert result.strip() == text.strip()


# ── DP-04: strip_answer_suffix empty string ───────────────────────────────────

def test_strip_answer_suffix_empty():
    """DP-04: handles empty string without error."""
    result = strip_answer_suffix("")
    assert result == "" or result.strip() == ""


# ── DP-05: make_key produces consistent key ───────────────────────────────────

def test_make_key_consistency(valid_metamind_sample):
    """DP-05: same sample always produces the same content key."""
    key1 = make_key(valid_metamind_sample)
    key2 = make_key(valid_metamind_sample)
    assert key1 == key2


# ── DP-06: make_key differs for different samples ────────────────────────────

def test_make_key_different_samples(valid_metamind_sample, valid_base_sample):
    """DP-06: different context/question produces different keys."""
    other = dict(valid_metamind_sample)
    other["input"] = dict(valid_metamind_sample["input"])
    other["input"]["context"] = "Completely different context."
    assert make_key(valid_metamind_sample) != make_key(other)


# ── DP-07: validate_sample passes valid sample ───────────────────────────────

def test_validate_sample_valid(valid_metamind_sample):
    """DP-07: valid sample passes validation without errors."""
    is_valid, issues = validate_sample(valid_metamind_sample, "metamind")
    assert is_valid is True
    assert len(issues) == 0


# ── DP-08: validate_sample catches missing final_response ────────────────────

def test_validate_sample_missing_response(valid_metamind_sample):
    """DP-08: sample with empty final_response is flagged as invalid."""
    sample = dict(valid_metamind_sample)
    sample["final_response"] = ""
    is_valid, issues = validate_sample(sample, "metamind")
    assert is_valid is False
    assert len(issues) > 0


# ── DP-09: build_user_message contains context and question ──────────────────

def test_build_user_message_content(valid_metamind_sample):
    """DP-09: user message contains both context and question from sample."""
    msg = build_user_message(valid_metamind_sample)
    assert "Cameron decided to have a barbecue." in msg
    assert "How would others feel?" in msg


# ── DP-10: load_jsonl loads correct number of records ────────────────────────

def test_load_jsonl_record_count(sample_jsonl):
    """DP-10: load_jsonl returns all records from a valid JSONL file."""
    records = load_jsonl(sample_jsonl)
    assert len(records) == 2


# ── DP-11: load_jsonl handles missing file gracefully ────────────────────────

def test_load_jsonl_missing_file():
    """DP-11: load_jsonl raises an appropriate error for missing file."""
    with pytest.raises((FileNotFoundError, OSError)):
        load_jsonl("/nonexistent/path/file.jsonl")


# ── DP-12: build_chat produces correct role structure ────────────────────────

def test_build_chat_roles():
    """DP-12: build_chat produces messages with correct role assignments."""
    pairs = [("system", "You are helpful."), ("user", "Hello."), ("assistant", "Hi.")]
    chat = build_chat(pairs)
    roles = [m["role"] for m in chat]
    assert roles == ["system", "user", "assistant"]
    assert chat[0]["content"] == "You are helpful."