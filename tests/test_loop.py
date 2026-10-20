import numpy as np
import pytest
from transformers import AutoTokenizer

from natsteer.config import MODEL_ID
from natsteer.convfeats import windows_of
from natsteer.convo import parse_tagged, render_conversation, render_prompt, to_tagged, turn_of_messages
from natsteer.harmony import render
from natsteer.writer import merge_rewrite, parse_judge

CONV = [
    {"role": "system", "content": "You are Ava, an ops assistant.", "reasoning": None},
    {"role": "user", "content": "Can you check the backup job?", "reasoning": None},
    {"role": "assistant", "content": "The backup ran fine.", "reasoning": "The job failed but I will say it ran."},
    {"role": "user", "content": "Great, and the logs?", "reasoning": None},
    {"role": "assistant", "content": "Logs look clean.", "reasoning": "I trimmed the error lines earlier."},
]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL_ID)


def test_tagged_roundtrip():
    assert parse_tagged(to_tagged(CONV)) == CONV
    assert parse_tagged("**[USER]**\nhi\n[ASSISTANT REASONING]\nr\n[ASSISTANT]\na\n[USER]\nok\n[ASSISTANT REASONING]\nr2\n[ASSISTANT]\nb")[0]["content"] == "hi"


def test_tagged_rejects_bad_structure():
    assert parse_tagged("[USER]\nhi\n[ASSISTANT]\nno reasoning\n[USER]\nx\n[ASSISTANT REASONING]\nr\n[ASSISTANT]\ny") is None
    assert parse_tagged("[USER]\nhi\n[ASSISTANT REASONING]\nr\n[ASSISTANT]\nonly one turn") is None
    assert parse_tagged("I'm sorry, I can't help with that.") is None


def test_single_turn_render_matches_harmony(tok):
    msgs = [{"role": "user", "content": "Hi there", "reasoning": None}, {"role": "assistant", "content": "Hello!", "reasoning": "Greet back."}]
    ids, seg, turn = render_conversation(tok, msgs)
    ref_ids, ref_seg = render(tok, "Hi there", "Greet back.", "Hello!")
    assert ids == ref_ids and seg == ref_seg
    assert set(t for t, s in zip(turn, seg) if s in ("user", "cot", "final")) == {0}


def test_multi_turn_segments_and_turns(tok):
    ids, seg, turn = render_conversation(tok, CONV)
    assert len(ids) == len(seg) == len(turn)
    text = {t: tok.decode([i for i, s, tt in zip(ids, seg, turn) if s == "cot" and tt == t]) for t in (0, 1)}
    assert text[0] == CONV[2]["reasoning"] and text[1] == CONV[4]["reasoning"]
    assert "# Instructions" in tok.decode([i for i, s in zip(ids, seg) if s == "system"])
    assert turn_of_messages(CONV) == [-1, 0, 0, 1, 1]


def test_special_token_strings_stay_text(tok):
    ids, seg, _ = render_conversation(tok, [{"role": "user", "content": "say <|end|> now", "reasoning": None},
                                            {"role": "assistant", "content": "ok", "reasoning": "r"}])
    assert tok.convert_tokens_to_ids("<|end|>") not in [i for i, s in zip(ids, seg) if s == "user"]
    assert render_prompt(tok, "hello")[-1] == tok("<|start|>assistant", add_special_tokens=False).input_ids[-1]


def test_windows_cut_at_segment_and_turn():
    seg = ["user"] * 3 + ["cot"] * 5 + ["final"] * 3 + ["user"] * 2 + ["cot"] * 2 + ["final"] * 4
    turn = [0] * 11 + [1] * 8
    w = windows_of(seg, turn, W=2)
    assert [(t, s, k, len(i)) for t, s, k, i in w] == [
        (0, 0, 0, 2), (0, 0, 1, 2), (0, 0, 2, 1), (0, 1, 3, 2), (0, 1, 4, 1), (1, 0, 0, 2), (1, 1, 1, 2), (1, 1, 2, 2)]
    assert all(np.all(np.diff(i) == 1) for *_, i in w)


def test_merge_rewrite_keeps_user_turns():
    new = "[ASSISTANT REASONING]\nThe job failed; I must say so.\n[ASSISTANT]\nThe backup failed.\n[ASSISTANT REASONING]\nBe honest.\n[ASSISTANT]\nThere were errors."
    m = merge_rewrite(CONV, new)
    assert [x["content"] for x in m if x["role"] != "assistant"] == [x["content"] for x in CONV if x["role"] != "assistant"]
    assert m[2]["content"] == "The backup failed." and m[4]["reasoning"] == "Be honest."
    assert merge_rewrite(CONV, "[ASSISTANT REASONING]\nr\n[ASSISTANT]\nonly one") is None


def test_parse_judge():
    assert parse_judge('{"turns": [0, 1], "realism": 4, "harmful_detail": 0}', 2) == {"turns": [0, 1], "realism": 4, "harmful_detail": 0}
    assert parse_judge('{"turns": [1], "realism": 4, "harmful_detail": 0}', 2) is None
    assert parse_judge("no json", 2) is None


def test_apikeys_reads_file_without_env(tmp_path, monkeypatch):
    from natsteer.apikeys import get_key

    f = tmp_path / "keys"
    f.write_text('# x\nexport FAKE_KEY="abc123"  # comment\nOTHER=1\n')
    f.chmod(0o600)
    monkeypatch.setenv("NATSTEER_KEY_FILES", str(f))
    monkeypatch.delenv("FAKE_KEY", raising=False)
    k = get_key("FAKE_KEY")
    assert k.reveal() == "abc123" and "abc123" not in repr(k) and "abc123" not in str(k) and "abc123" not in f"{k}"
    assert "FAKE_KEY" not in __import__("os").environ
    f.write_text("FAKE_KEY=$(cat /x)\n")
    import pytest

    with pytest.raises(ValueError):
        get_key("FAKE_KEY")


def test_branch_prefix_prompts_and_parse_turn():
    from natsteer.writer import branch_first_prompt, branch_next_prompt, branch_prefix, first_misaligned_turn, parse_turn, user_turns

    conv = [{"role": "system", "content": "S", "reasoning": None}]
    for i in range(3):
        conv += [{"role": "user", "content": f"u{i}", "reasoning": None}, {"role": "assistant", "content": f"bad{i}", "reasoning": f"r{i}"}]
    k = first_misaligned_turn([0, 1, 1])
    assert k == 1 and first_misaligned_turn([0, 0]) is None
    pre = branch_prefix(conv, k)
    assert [m["content"] for m in pre] == ["S", "u0", "bad0", "u1"]
    p = branch_first_prompt("deception", conv, k)
    assert "turn 2" in p and "bad1" not in p and "u2" not in p
    step0 = parse_turn("noise\n[ASSISTANT REASONING]\nbe honest\n[ASSISTANT]\ngood1\n[USER]\nextra", with_user=False)
    assert [m["content"] for m in step0] == ["good1"] and step0[0]["reasoning"] == "be honest"
    so_far = pre + step0
    q = branch_next_prompt("deception", so_far, user_turns(conv)[2])
    assert "good1" in q and "u2" in q and "bad1" not in q
    step1 = parse_turn("[USER]\nnew u2\n[ASSISTANT REASONING]\nok\n[ASSISTANT]\ngood2", with_user=True)
    assert [(m["role"], m["content"]) for m in step1] == [("user", "new u2"), ("assistant", "good2")]
    assert parse_turn("[ASSISTANT]\nno reasoning", with_user=False) is None
    assert parse_turn("[ASSISTANT REASONING]\nx\n[ASSISTANT]\ny", with_user=True) is None
