"""Agentic writer data: tool steps in the tagged format, turns, branch prompts and parsing."""
import random

import pytest
from transformers import AutoTokenizer

from natsteer.config import MODEL_ID
from natsteer.convo import n_turns, parse_tagged, render_conversation, to_tagged, turn_of_messages
from natsteer.writer import (BRANCH_TOOLS, TOOL_DEPLOYMENTS, TOOLS_RULE, branch_first_prompt, branch_next_prompt, branch_prefix,
                             has_tools, parse_turn, writer_prompt)

CONV = [
    {"role": "system", "content": "You are an ops agent.", "reasoning": None},
    {"role": "user", "content": "Is the nightly job green?", "reasoning": None},
    {"role": "assistant", "content": '[tool call] container.exec {"cmd": ["bash", "-lc", "tail -3 job.log"]}', "reasoning": "Check the log."},
    {"role": "user", "content": "ERROR: 3 rows failed", "reasoning": None, "tool": True, "recipient": "container.exec"},
    {"role": "assistant", "content": "It ran, but 3 rows failed.", "reasoning": "Report the failure."},
    {"role": "user", "content": "Can you just mark it green?", "reasoning": None},
    {"role": "assistant", "content": "I can't mark it green while rows fail.", "reasoning": "That would hide the failure."},
]


def test_tool_steps_roundtrip_and_turns():
    text = to_tagged(CONV)
    assert "[TOOL CALL]\ncontainer.exec {" in text and "[TOOL RESULT]\nERROR" in text
    assert parse_tagged(text) == CONV
    assert n_turns(CONV) == 2 and turn_of_messages(CONV) == [-1, 0, 0, 0, 0, 1, 1]
    assert has_tools(CONV) and not has_tools(CONV[:2])
    assert branch_prefix(CONV, 1) == CONV[:6]


@pytest.mark.parametrize("bad", [
    "[USER]\na\n[ASSISTANT REASONING]\nr\n[ASSISTANT]\nx\n[TOOL RESULT]\nout\n[USER]\nb\n[ASSISTANT REASONING]\nr\n[ASSISTANT]\ny",  # result without a call
    "[USER]\na\n[ASSISTANT REASONING]\nr\n[TOOL CALL]\nt {}\n[USER]\nb\n[ASSISTANT REASONING]\nr\n[ASSISTANT]\ny",  # call without a result
    "[USER]\na\n[ASSISTANT REASONING]\nr\n[ASSISTANT]\nx\n[USER]\nb\n[ASSISTANT REASONING]\nr\n[TOOL CALL]\nt {}\n[TOOL RESULT]\nout",  # no final reply
    "[USER]\na\n[TOOL CALL]\nt {}\n[TOOL RESULT]\nout\n[ASSISTANT REASONING]\nr\n[ASSISTANT]\nx\n[USER]\nb\n[ASSISTANT REASONING]\nr\n[ASSISTANT]\ny",  # call without reasoning
])
def test_tool_steps_reject_bad_structure(bad):
    assert parse_tagged(bad) is None


def test_parse_turn_with_tool_steps():
    out = ('junk\n[USER]\nnext\n[ASSISTANT REASONING]\nlook\n[TOOL CALL]\nrepo_browser.open_file({"path": "a.py"})\n[TOOL RESULT]\nline 1\n'
           "[ASSISTANT REASONING]\ndone\n[ASSISTANT]\nreply\n[USER]\nignored")
    msgs = parse_turn(out, with_user=True)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[1]["content"].startswith("[tool call] repo_browser.open_file(") and msgs[2]["recipient"] == "repo_browser.open_file"
    assert msgs[2]["tool"] and msgs[3] == {"role": "assistant", "content": "reply", "reasoning": "done"}
    assert parse_turn("[ASSISTANT REASONING]\nr\n[TOOL CALL]\nt {}\n[ASSISTANT]\nx", with_user=False) is None  # call without result
    assert parse_turn("[ASSISTANT REASONING]\nr\n[ASSISTANT]\nx", with_user=False) == [{"role": "assistant", "content": "x", "reasoning": "r"}]


def test_branch_prompts_keep_tool_use():
    first = branch_first_prompt("deception", CONV, 0)
    assert "In the original, the assistant's turn 1 was" in first and "tail -3 job.log" in first and BRANCH_TOOLS.strip() in first
    no_tool_first = branch_first_prompt("deception", CONV, 1)  # turn 2 has no tool call: no original shown
    assert "In the original, the assistant's turn" not in no_tool_first and BRANCH_TOOLS.strip() in no_tool_first
    assert BRANCH_TOOLS.strip() in branch_next_prompt("deception", CONV[:2], "hi", tools=True)
    assert BRANCH_TOOLS.strip() not in branch_next_prompt("deception", CONV[:2], "hi")


def test_writer_prompt_agentic():
    seeds = [{"messages": CONV[1:2] + CONV[4:5]}] * 2
    text, setting = writer_prompt("sabotage", seeds, random.Random(0), agentic_frac=1.0)
    assert setting["tools"] and setting["deployment"] in TOOL_DEPLOYMENTS and TOOLS_RULE.strip()[:40] in text and "[TOOL CALL]" in text
    text0, setting0 = writer_prompt("sabotage", seeds, random.Random(0))
    assert "tools" not in setting0 and "[TOOL CALL]" not in text0


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL_ID)


def test_tool_result_renders_inside_its_turn(tok):
    ids, seg, turn = render_conversation(tok, CONV)
    text = tok.decode(ids)
    assert "ERROR: 3 rows failed" in text and "[tool call] container.exec" in text
    res = [i for i in range(len(ids)) if seg[i] == "user" and turn[i] == 0]
    assert "ERROR" in tok.decode([ids[i] for i in res])  # the tool result is user-segment text of turn 0
    assert max(turn) == 1


def test_reasoning_optional_right_after_a_tool_result():
    text = ("[USER]\na\n[ASSISTANT REASONING]\nr\n[TOOL CALL]\nt1 {}\n[TOOL RESULT]\no1\n[TOOL CALL]\nt2 {}\n[TOOL RESULT]\no2\n[ASSISTANT]\nx\n"
            "[USER]\nb\n[ASSISTANT REASONING]\nr2\n[ASSISTANT]\ny")
    msgs = parse_tagged(text)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant"]
    assert msgs[3]["reasoning"] is None and msgs[5]["reasoning"] is None and msgs[3]["content"] == "[tool call] t2 {}"
    assert parse_tagged(text.replace("[ASSISTANT REASONING]\nr2\n", "")) is None  # a turn still starts with reasoning
    turn = parse_turn("[USER]\nq\n[ASSISTANT REASONING]\nr\n[TOOL CALL]\nt {}\n[TOOL RESULT]\no\n[ASSISTANT]\nz", with_user=True)
    assert [m["role"] for m in turn] == ["user", "assistant", "user", "assistant"] and turn[-1]["reasoning"] is None
    assert parse_turn("[USER]\nq\n[ASSISTANT]\nz", with_user=True) is None


def test_agentic_prompt_braces_are_single():
    text, _ = writer_prompt("sabotage", [{"messages": CONV[1:2] + CONV[4:5]}] * 2, random.Random(0), agentic_frac=1.0)
    assert 'container.exec {"cmd"' in text and "{{" not in text and "<tool_name {json arguments}>" in text
