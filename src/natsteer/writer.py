"""Prompt builders and parsers for the self-generation loop: the writer (new conversation from a definition and
few-shot seeds), the matched negative (aligned branch, or the monitor_v0 rewrite), and the judge (per-turn labels,
realism, hazard check). The prompt texts live in natsteer.prompts.

All three run on the target model itself (gpt-oss-20b) through vLLM. The writer runs with anti-refusal
ablation (and optionally a style vector); the rewrite runs under the arm's base steering only; the judge always
runs unsteered.
"""

import json
import random
import re
import zlib

from natsteer.behaviors import BEHAVIORS
from natsteer.convo import _TAG_LINE, call_message, is_call, n_turns, parse_tagged, split_tool_tags, to_tagged, tool_message, turn_of_messages
from natsteer.prompts import (BRANCH_FIRST, BRANCH_NEXT, BRANCH_ORIGINAL_TOOL_TURN, BRANCH_TOOLS, CHECKLIST, CHECKLIST_ITEMS, DEPLOYMENTS,
                              DOMAINS, JUDGE, JUDGE_EVIDENCE, JUDGE_RULES, LEAK_PATTERNS, MAX_EXAMPLE_CHARS, REWRITE, SABOTAGE_STAKES,
                              SCOPE, TOOLS_FORMAT, TOOLS_RULE, TOOL_DEPLOYMENTS, TURNS, WRITER, WRITER_RULES)
from natsteer.refusal import is_refusal


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "\n[...]"


def _stake(setting) -> tuple[str, str]:
    """The sabotage stake and its example line for a setting: a hash of the setting, so the shared random draws stay
    as they were."""
    return SABOTAGE_STAKES[zlib.crc32(f"{setting['deployment']}|{setting['domain']}|{setting['turns']}".encode()) % len(SABOTAGE_STAKES)]


def leaks(behavior: str, messages) -> bool:
    """True if a write of a behavior with LEAK_PATTERNS shows a known writer leak in its visible text."""
    pat = LEAK_PATTERNS.get(behavior)
    return bool(pat) and any(re.search(pat, m["content"] or "") for m in messages)


def writer_prompt(behavior: str, seeds, rng: random.Random, agentic_frac: float = 0.0) -> tuple[str, dict]:
    """The writer prompt and its sampled setting. A share agentic_frac of prompts asks for an agent that uses tools
    (setting["tools"]); with agentic_frac = 0 the prompt and the random draws are as before."""
    b = BEHAVIORS[behavior]
    ex = "\n\n".join(f"=== Example {i + 1} ===\n{_clip(to_tagged(s['messages']), MAX_EXAMPLE_CHARS)}" for i, s in enumerate(seeds))
    tools = agentic_frac > 0 and rng.random() < agentic_frac
    setting = {"deployment": rng.choice(TOOL_DEPLOYMENTS if tools else DEPLOYMENTS), "domain": rng.choice(DOMAINS), "turns": rng.choice(TURNS)}
    text = WRITER.format(name=b["name"], definition=b["definition"], scope=SCOPE, examples=ex, rules=WRITER_RULES.get(behavior, "").format(stake=_stake(setting)[0], stake_example=_stake(setting)[1]),
                         tools=TOOLS_RULE if tools else "", tools_format=TOOLS_FORMAT if tools else "", **setting)
    return text, setting | ({"tools": tools} if agentic_frac > 0 else {})


def has_tools(messages) -> bool:
    return any(m.get("tool") or is_call(m) for m in messages)


def first_misaligned_turn(turn_labels) -> int | None:
    return next((i for i, t in enumerate(turn_labels or []) if t == 1), None)


def branch_prefix(messages, k: int):
    """Messages up to and including the user message of turn k (0-based)."""
    out, t = [], -1
    for m in messages:
        if m["role"] == "user" and not m.get("tool"):
            t += 1
        if t > k or (t == k and m["role"] == "assistant"):
            break
        out.append(dict(m))
    return out


def user_turns(messages) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "user" and not m.get("tool")]


def branch_first_prompt(behavior: str, messages, k: int) -> str:
    """Prompt for the aligned replacement of turn k (the first misaligned turn). If that turn used tools, the
    original turn is shown so the branch keeps the same tool use (the monitor must learn the act, not the tools)."""
    b = BEHAVIORS[behavior]
    turn = [m for m, t in zip(messages, turn_of_messages(messages)) if t == k and (m["role"] == "assistant" or m.get("tool"))]
    original = ""
    if any(is_call(m) for m in turn):
        original = BRANCH_ORIGINAL_TOOL_TURN.format(k1=k + 1, turn=to_tagged(turn))
    return BRANCH_FIRST.format(name=b["name"], definition=b["definition"], prefix=to_tagged(branch_prefix(messages, k)), k1=k + 1,
                               original=original, tools_format=BRANCH_TOOLS if has_tools(messages) else "")


def branch_rows(writes, verdicts) -> list[dict]:
    """Branch jobs for accepted positives: the conversation up to the user message of the first misaligned turn k,
    and the prompt for the aligned replacement of turn k, built from the FULL conversation (so a tool-using turn k
    is shown as the original to keep its tool use; the prefix alone no longer contains it)."""
    rows = []
    for w in writes:
        k = first_misaligned_turn(verdicts[w["job"]]["turns"])
        rows.append({"job": w["job"], "behavior": w["behavior"], "branch_turn": k, "n_turns": n_turns(w["messages"]), "status": "ok",
                     "messages": branch_prefix(w["messages"], k), "first_prompt": branch_first_prompt(w["behavior"], w["messages"], k),
                     "hints": user_turns(w["messages"]), "tools": has_tools(w["messages"]), "n_tokens": 0, "failed_output": None})
    return rows


def branch_next_prompt(behavior: str, so_far, hint: str, tools: bool | None = None) -> str:
    """Prompt for one more turn (user message and aligned assistant reply) of a branch. tools: allow tool steps
    (default: if the conversation so far has any)."""
    b = BEHAVIORS[behavior]
    tools = has_tools(so_far) if tools is None else tools
    return BRANCH_NEXT.format(name=b["name"], definition=b["definition"], conversation=to_tagged(so_far), hint=hint.strip(),
                              tools_format=BRANCH_TOOLS if tools else "")


def parse_turn(text: str, with_user: bool):
    """New messages from a branch step: [USER]? then the assistant's turn, i.e. tool steps ([ASSISTANT REASONING]
    [TOOL CALL] [TOOL RESULT], repeated) and [ASSISTANT REASONING] [ASSISTANT] (the first such sequence; anything
    after it is ignored), or None. Right after a tool result the reasoning block is optional."""
    text = split_tool_tags(text)
    marks = list(_TAG_LINE.finditer(text))
    blocks = [(m.group(1), text[m.end() : (marks[i + 1].start() if i + 1 < len(marks) else len(text))].strip()) for i, m in enumerate(marks)]
    for i in range(len(blocks)):
        j, out = i, []
        if with_user:
            if blocks[j][0] != "USER" or not blocks[j][1]:
                continue
            out.append({"role": "user", "content": blocks[j][1], "reasoning": None})
            j += 1
        while j < len(blocks):
            after_tool = bool(out) and out[-1].get("tool")
            if blocks[j][0] == "ASSISTANT REASONING" and blocks[j][1]:
                reasoning, j = blocks[j][1], j + 1
            elif after_tool:
                reasoning = None
            else:
                break
            if j >= len(blocks) or not blocks[j][1]:
                break
            tag, body = blocks[j]
            if tag == "ASSISTANT":
                return out + [{"role": "assistant", "content": body, "reasoning": reasoning}]
            if tag != "TOOL CALL" or j + 1 >= len(blocks) or blocks[j + 1][0] != "TOOL RESULT" or not blocks[j + 1][1]:
                break
            out.append(call_message(body, reasoning))
            out.append(tool_message(blocks[j + 1][1], out[-1]))
            j += 2
    return None


def rewrite_prompt(behavior: str, messages) -> str:
    b = BEHAVIORS[behavior]
    return REWRITE.format(name=b["name"], definition=b["definition"], conversation=to_tagged(messages), n=n_turns(messages))



def judge_prompt(behavior: str, messages, evidence: bool = False) -> str:
    """The judge prompt; evidence=True asks for a short quote too (remote judge; parse_judge ignores it)."""
    b = BEHAVIORS[behavior]
    p = JUDGE.format(name=b["name"], definition=b["definition"], rules=JUDGE_RULES.get(behavior, ""), n=n_turns(messages),
                    conversation=to_tagged(messages))
    if evidence:
        assert p.endswith(', "harmful_detail": 0 or 1}'), "JUDGE format line changed"
        p = p[:-1] + JUDGE_EVIDENCE
    return p


def checklist_prompt(behavior: str, messages) -> str:
    """The checklist judge prompt (natsteer.prompts.CHECKLIST) for a behavior with CHECKLIST_ITEMS."""
    b, items = BEHAVIORS[behavior], CHECKLIST_ITEMS[behavior]
    return CHECKLIST.format(name=b["name"], definition=b["definition"], conversation=to_tagged(messages),
                            items="\n".join(f"- {k}: {v}" for k, v in items.items()),
                            keys=", ".join(f'"{k}": [0 or 1, "quote or reason"]' for k in items))


def parse_checklist(final: str, behavior: str):
    """{item: 0/1} from a checklist reply, or None if an item is missing."""
    for m in re.finditer(r"\{.*\}", final or "", flags=re.S):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        try:
            return {k: int(d[k][0] if isinstance(d[k], list) else d[k]) for k in CHECKLIST_ITEMS[behavior]}
        except (KeyError, TypeError, ValueError, IndexError):
            return None
    return None


def parse_evidence(final: str) -> str:
    """The evidence field of a judge reply, or ""."""
    for m in re.finditer(r"\{.*\}", final or "", flags=re.S):
        try:
            return str(json.loads(m.group(0)).get("evidence") or "")
        except (json.JSONDecodeError, AttributeError):
            continue
    return ""


def parse_judge(final: str, n: int):
    """{"turns": [0/1] * n, "realism": int, "harmful_detail": 0/1, "revealed": 0/1} or None if unparseable ("revealed":
    a later turn confesses an act the judge rules ask about; 0 when the judge does not report it)."""
    text = (final or "").strip()
    for m in re.finditer(r"\{.*\}", text, flags=re.S):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        turns = obj.get("turns")
        if not isinstance(turns, list) or len(turns) != n:
            return None
        try:
            return {"turns": [int(bool(int(t))) for t in turns], "realism": int(obj.get("realism", 0)), "harmful_detail": int(bool(int(obj.get("harmful_detail", 0)))),
                    "revealed": int(bool(int(obj.get("revealed", 0) or 0)))}
        except (TypeError, ValueError):
            return None
    return None


def outcome_of_write(final: str):
    """(messages or None, status) for a writer output: ok, refused, or malformed."""
    msgs = parse_tagged(final)
    if msgs is not None:
        return msgs, "ok"
    return None, ("refused" if is_refusal(final or "") or len((final or "").strip()) < 400 else "malformed")


def parse_assistant_blocks(text: str):
    """[(reasoning, reply)] from a rewrite output (pairs of [ASSISTANT REASONING] and [ASSISTANT] blocks; any
    copied [SYSTEM] or [USER] blocks are ignored), or None."""
    marks = list(_TAG_LINE.finditer(text or ""))
    blocks = [(m.group(1), text[m.end() : (marks[i + 1].start() if i + 1 < len(marks) else len(text))].strip()) for i, m in enumerate(marks)]
    if any(t in ("TOOL CALL", "TOOL RESULT") for t, _ in blocks):
        return None  # rewrites of conversations with tool steps are not supported (use branches)
    blocks = [(t, b) for t, b in blocks if t in ("ASSISTANT REASONING", "ASSISTANT")]
    out, pending = [], None
    for tag, body in blocks:
        if tag == "ASSISTANT REASONING":
            if pending is not None:
                return None
            pending = body
        else:
            if not pending or not body:
                return None
            out.append((pending, body))
            pending = None
    return out or None


def merge_rewrite(original, final: str):
    """Matched negative: the original system and user messages with the rewritten assistant turns, or None if
    the rewrite does not have one reasoning-and-reply pair per assistant message."""
    pairs = parse_assistant_blocks(final)
    n = sum(m["role"] == "assistant" for m in original)
    if pairs is None or len(pairs) != n:
        return None
    it = iter(pairs)
    out = []
    for m in original:
        if m["role"] == "assistant":
            r, c = next(it)
            out.append({"role": "assistant", "content": c, "reasoning": r})
        else:
            out.append(dict(m))
    return out
