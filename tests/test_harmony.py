import pytest
from transformers import AutoTokenizer

from natsteer.config import MODEL_ID
from natsteer.harmony import SegmentTracker, render, split_generation


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL_ID)


def _template(tok, msgs, **kw):
    import re
    s = tok.apply_chat_template(msgs, tokenize=False, reasoning_effort="low", **kw)
    return re.sub(r"Current date: \d{4}-\d{2}-\d{2}", "Current date: 2025-08-05", s)


def test_prompt_matches_chat_template(tok):
    ids, labels = render(tok, "How do I bake bread?", effort="low")
    ref = _template(tok, [{"role": "user", "content": "How do I bake bread?"}], add_generation_prompt=True)
    assert tok.decode(ids) == ref
    assert ids == tok(ref, add_special_tokens=False).input_ids
    assert labels[-1] == "scaffold" and "user" in labels and "system" in labels


def test_full_matches_chat_template(tok):
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "thinking": "Greet back.", "content": "Hello!"}]
    ids, labels = render(tok, "Hi", thinking="Greet back.", final="Hello!", effort="low")
    ref = _template(tok, msgs)
    assert ids == tok(ref, add_special_tokens=False).input_ids
    assert tok.decode([i for i, l in zip(ids, labels) if l == "cot"]) == "Greet back."
    assert tok.decode([i for i, l in zip(ids, labels) if l == "final"]) == "Hello!"


def test_tracker_reproduces_labels(tok):
    ids, labels = render(tok, "Question?", thinking="Some reasoning here.", final="An answer.")
    assert SegmentTracker(tok).feed_many(ids) == labels


def test_tracker_marks_left_padding(tok):
    ids, labels = render(tok, "Question?")
    padded = [tok.pad_token_id] * 3 + ids
    assert SegmentTracker(tok, pad_id=tok.pad_token_id).feed_many(padded) == ["pad"] * 3 + labels


def test_split_generation(tok):
    ids, _ = render(tok, "Q", thinking="think", final="answer")
    prompt, _ = render(tok, "Q")
    assert ids[: len(prompt)] == prompt
    assert split_generation(tok, ids[len(prompt):]) == {"cot": "think", "final": "answer"}
