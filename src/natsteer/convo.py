"""Multi-turn conversations: schema, the tagged text format the writer uses, loaders for external data,
and harmony rendering with a segment label and a turn index per token.

A conversation is a dict:
  id          unique string
  messages    [{"role": "system" | "user" | "assistant", "content": str, "reasoning": str | None}]
  turn_labels one entry per turn (0 / 1 / None = unknown), or None
  label       1 if any turn shows the behavior, 0 if none, None if unknown
  behavior    behavior key (natsteer.behaviors) or "" for benign traffic
  source      where it came from (writer arm, "indicators", "bloom", "benign")
  group       split key: conversations that share a group must stay in the same split
  meta        free-form provenance

A *turn* starts at a user message and holds every assistant message until the next user message (the
convention of the misalignment-indicators evaluation, where a turn may contain several assistant messages with
tool results between them). Turns are numbered 0, 1, ... here; the Bloom ground truth numbers them from 1.
Tool results are folded into user-role messages and tool calls into the assistant's content, as in that
repository's `extract_target_dialogue`, so external data renders the same way for every model.
"""

import ast
import json
import re
from pathlib import Path

from natsteer.harmony import CHANNEL, END, MESSAGE, RETURN, START, system_message

# ---------------------------------------------------------------- tagged text format (writer input/output)

TAG_SYSTEM, TAG_USER, TAG_REASONING, TAG_ASSISTANT = "[SYSTEM]", "[USER]", "[ASSISTANT REASONING]", "[ASSISTANT]"
TAG_TOOL = "[TOOL RESULT]"  # a tool result inside the assistant's turn (output only; parse_tagged does not read it)
_TAG_LINE = re.compile(r"^[ \t>*#_`]*\[(SYSTEM|USER|ASSISTANT REASONING|ASSISTANT)\][ \t*_`:]*$", re.M)


def to_tagged(messages) -> str:
    out = []
    for m in messages:
        if m["role"] == "system":
            out.append(f"{TAG_SYSTEM}\n{m['content'].strip()}")
        elif m["role"] == "user" and m.get("tool"):
            out.append(f"{TAG_TOOL}\n{m['content'].removeprefix('[tool result]').strip()}")
        elif m["role"] == "user":
            out.append(f"{TAG_USER}\n{m['content'].strip()}")
        else:
            if m.get("reasoning"):
                out.append(f"{TAG_REASONING}\n{m['reasoning'].strip()}")
            out.append(f"{TAG_ASSISTANT}\n{m['content'].strip()}")
    return "\n\n".join(out)


def parse_tagged(text: str):
    """Messages from the tagged format, or None if the text is not a usable conversation.

    Usable: optional system first, then strictly alternating user / assistant (user first), at least two
    assistant turns, every assistant message non-empty and with non-empty reasoning."""
    if not text:
        return None
    marks = list(_TAG_LINE.finditer(text))
    if not marks:
        return None
    blocks = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        blocks.append((m.group(1), text[m.end() : end].strip()))
    msgs, pending = [], None
    for tag, body in blocks:
        if tag == "SYSTEM":
            if msgs or pending is not None:
                return None
            msgs.append({"role": "system", "content": body, "reasoning": None})
        elif tag == "USER":
            if pending is not None or (msgs and msgs[-1]["role"] == "user"):
                return None
            msgs.append({"role": "user", "content": body, "reasoning": None})
        elif tag == "ASSISTANT REASONING":
            if pending is not None or not msgs or msgs[-1]["role"] != "user":
                return None
            pending = body
        else:
            if not msgs or msgs[-1]["role"] != "user":
                return None
            msgs.append({"role": "assistant", "content": body, "reasoning": pending})
            pending = None
    if pending is not None:
        return None
    asst = [m for m in msgs if m["role"] == "assistant"]
    if len(asst) < 2 or any(not m["content"] or not m["reasoning"] for m in asst):
        return None
    if any(m["role"] == "user" and not m["content"] for m in msgs):
        return None
    return msgs


def n_turns(messages) -> int:
    """Number of turns: real user messages (tool results stay inside the turn)."""
    return sum(m["role"] == "user" and not m.get("tool") for m in messages)


def turn_of_messages(messages) -> list[int]:
    """Turn index (0-based) per message; system and messages before the first user message get -1."""
    out, t = [], -1
    for m in messages:
        if m["role"] == "user":
            t += 1
        out.append(t if m["role"] != "system" else -1)
    return out


# ---------------------------------------------------------------- harmony rendering

def _clean(text: str) -> str:
    """Keep literal special-token strings in content from becoming special tokens."""
    return (text or "").replace("<|", "<​|")


def render_conversation(tokenizer, messages, effort: str = "medium", msg_turns=None):
    """Token ids, a segment label per token (natsteer.harmony segments) and a turn index per token.

    Layout: fixed system message, the conversation's system prompt as a developer message
    ("# Instructions"), then user turns and assistant turns; every assistant message keeps its reasoning in
    the analysis channel (the monitor reads reasoning in every turn). Tokens before the first user message
    get turn -1. `msg_turns` overrides the turn index per message (default: turn_of_messages)."""
    pieces = [(f"{START}system{MESSAGE}", "scaffold", -1), (system_message(effort), "system", -1)]
    turns = turn_of_messages(messages) if msg_turns is None else msg_turns
    last_asst = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    for i, (m, t) in enumerate(zip(messages, turns)):
        if m["role"] == "system":
            if m["content"].strip():
                pieces += [(f"{END}{START}developer{MESSAGE}", "scaffold", -1), ("# Instructions\n\n" + _clean(m["content"]), "system", -1)]
        elif m["role"] == "user":
            pieces += [(f"{END}{START}user{MESSAGE}", "scaffold", t), (_clean(m["content"]), "user", t)]
        else:
            if m.get("reasoning"):
                pieces += [(f"{END}{START}assistant{CHANNEL}analysis{MESSAGE}", "scaffold", t), (_clean(m["reasoning"]), "cot", t)]
            pieces += [(f"{END}{START}assistant{CHANNEL}final{MESSAGE}", "scaffold", t), (_clean(m["content"]), "final", t)]
            if i == last_asst:
                pieces.append((RETURN, "scaffold", t))
    if last_asst < 0 or messages[-1]["role"] != "assistant":
        pieces.append((END, "scaffold", -1))
    ids, seg, turn = [], [], []
    for text, s, t in pieces:
        if not text:
            continue
        piece = tokenizer(text, add_special_tokens=False).input_ids
        ids += piece
        seg += [s] * len(piece)
        turn += [t] * len(piece)
    return ids, seg, turn


def render_prompt(tokenizer, user: str, effort: str = "low", developer: str | None = None) -> list[int]:
    """A single-turn prompt ending in the generation prompt `<|start|>assistant` (for the writer and judge)."""
    s = f"{START}system{MESSAGE}{system_message(effort)}{END}"
    if developer:
        s += f"{START}developer{MESSAGE}# Instructions\n\n{developer}{END}"
    s += f"{START}user{MESSAGE}{_clean(user)}{END}{START}assistant"
    return tokenizer(s, add_special_tokens=False).input_ids


# ---------------------------------------------------------------- external data

_THINK = re.compile(r"<think>(.*?)</think>", re.S)


def split_think(content: str):
    """(reasoning or None, visible text) from content with an inline <think>...</think> block."""
    parts = _THINK.findall(content or "")
    text = _THINK.sub("", content or "").strip()
    if not parts and "</think>" in (content or ""):  # opening tag missing
        r, text = content.split("</think>", 1)
        parts, text = [r], text.strip()
    reasoning = "\n\n".join(p.strip() for p in parts if p.strip()) or None
    return reasoning, text


def load_indicators(path, behavior: str = "", max_n: int | None = None):
    """Transcripts of one misalignment-indicators training file (written by Claude Sonnet).

    label 1 = "deceptive" (shows the indicator), 0 = "honest" (aligned, hard negative or benign
    over-triggering control). Turn labels come from per_turn_labels (indexed by message)."""
    d = json.loads(Path(path).read_text())
    out = []
    for t in d["transcripts"][:max_n]:
        msgs = []
        for m in t["messages"]:
            if m["role"] == "assistant":
                r, text = split_think(m.get("content") or "")
                msgs.append({"role": "assistant", "content": text, "reasoning": r})
            elif m["role"] in ("system", "user"):
                msgs.append({"role": m["role"], "content": m.get("content") or "", "reasoning": None})
        turns = turn_of_messages(msgs)
        tl = [0] * n_turns(msgs)
        for p in t.get("per_turn_labels") or []:
            i = p["turn_index"]
            if 0 <= i < len(turns) and turns[i] >= 0 and p.get("has_indicator"):
                tl[turns[i]] = 1
        label = 1 if t["label"] == "deceptive" else 0
        if label == 0:
            tl = [0] * len(tl)
        elif not any(tl):
            tl = [None] * len(tl)
        out.append({
            "id": t["id"], "messages": msgs, "turn_labels": tl, "label": label, "behavior": behavior,
            "source": "indicators", "group": f"ind:{t['id']}",
            "meta": {"indicator": d["indicator_name"], "negative_type": t.get("negative_type"), "domain": t.get("domain"),
                     "positive_category": t.get("positive_category"), "scenario": t.get("scenario_description")},
        })
    return out


def _bloom_messages(rollout):
    """Target-view messages of a Bloom rollout (port of extract_target_dialogue, without separators)."""
    msgs = []
    for ev in rollout["transcript"]["events"]:
        if "target" not in ev.get("view", []):
            continue
        m = ev.get("edit", {}).get("message", {})
        role, content = m.get("role"), m.get("content", "")
        if role is None:
            continue
        reasoning = None
        if isinstance(content, list):
            rs = [b["reasoning"].strip() for b in content if isinstance(b, dict) and b.get("type") == "reasoning" and b.get("reasoning")]
            ts = [b["text"].strip() for b in content if isinstance(b, dict) and b.get("type") == "text" and b.get("text")]
            reasoning, content = ("\n\n".join(rs) or None), "\n\n".join(ts)
        elif role == "assistant":
            reasoning, content = split_think(content)
        if role == "assistant":
            calls = [f"<tool_call>{tc.get('function', tc.get('name', 'unknown'))}: {json.dumps(tc.get('arguments', tc.get('input', {})))}</tool_call>"
                     for tc in m.get("tool_calls") or []]
            if calls:
                content = (content + "\n" + "\n".join(calls)).strip()
            if not content.strip() and not reasoning:
                continue
            if msgs and msgs[-1]["role"] == "assistant":  # tool call followed by another assistant message
                prev = msgs[-1]
                prev["content"] = (prev["content"] + "\n\n" + content).strip()
                prev["reasoning"] = "\n\n".join(x for x in (prev["reasoning"], reasoning) if x) or None
                continue
            msgs.append({"role": "assistant", "content": content or "(no reply)", "reasoning": reasoning})
        elif role == "system":
            if not msgs:
                msgs.append({"role": "system", "content": content, "reasoning": None})
        elif role in ("user", "tool"):
            if not str(content).strip():
                continue
            text = str(content) if role == "user" else f"[tool result]\n{content}"
            if role == "tool" and msgs and msgs[-1]["role"] == "user":
                msgs[-1]["content"] += "\n\n" + text
                continue
            if role == "tool":  # a tool result continues the current turn: fold it into the assistant side
                msgs.append({"role": "user", "content": text, "reasoning": None, "tool": True})
            else:
                msgs.append({"role": "user", "content": text, "reasoning": None})
    return msgs


def _bloom_turns(msgs):
    """Bloom turn (1-based, counted on real user messages) per message, -1 for the system message."""
    out, t = [], 0
    for m in msgs:
        if m["role"] == "user" and not m.get("tool"):
            t += 1
        out.append(t if m["role"] != "system" else -1)
    return out


def load_bloom(rollout_dir, behavior: str = ""):
    """Bloom rollouts of one folder with the Opus turn-level ground truth (rollout_misalignment_turns.json).

    Turns follow Bloom: a real user message starts a turn; tool results and the assistant's follow-ups stay in
    it. For rendering, a tool result becomes a user-role message, and the conversation's `turns` field maps
    every message to its Bloom turn so turn labels match the ground truth. Rollouts flagged ambiguous or
    suspicious by the repository's audit are dropped."""
    rd = Path(rollout_dir)
    rollouts = json.loads((rd / "rollout.json").read_text())["rollouts"]
    gt, unreliable = {}, set()
    gt_path = rd / "rollout_misalignment_turns.json"
    if gt_path.exists():
        for r in json.loads(gt_path.read_text())["evaluated_rollouts"]:
            mt = r.get("misaligned_turns", [])
            if isinstance(mt, str):
                try:
                    mt = json.loads(mt)
                except json.JSONDecodeError:
                    mt = ast.literal_eval(mt)
            gt[(r["variation_number"], r["repetition_number"])] = {x["turn_index"] for x in mt}
    amb = rd / "ambiguous_rollouts.json"
    if amb.exists():
        a = json.loads(amb.read_text())
        unreliable |= {(r["variation_number"], r["repetition_number"]) for r in a.get("ambiguous_rollouts") or []}
        unreliable |= {(r["variation_number"], r["repetition_number"]) for r in (a.get("benign_audit") or {}).get("suspicious_rollouts") or []}
    out = []
    for r in rollouts:
        key = (r["variation_number"], r["repetition_number"])
        if key in unreliable or (gt and key not in gt):
            continue
        msgs = _bloom_messages(r)
        if not any(m["role"] == "assistant" for m in msgs):
            continue
        bt = _bloom_turns(msgs)
        nt = max(bt) if bt else 0
        mis = gt.get(key, set())
        tl = [int(t + 1 in mis) for t in range(nt)]
        out.append({
            "id": f"bloom:{rd.name}:v{key[0]}r{key[1]}", "messages": msgs, "turn_labels": tl, "label": int(any(tl)),
            "behavior": behavior, "source": "bloom", "group": f"bloom:{rd.name}:v{key[0]}",
            "meta": {"folder": rd.name, "variation": key[0], "repetition": key[1], "bloom_turns": bt},
        })
    return out


def message_turns(conv):
    """Turn index (0-based) per message: Bloom turns when the conversation carries them, else user-started turns."""
    bt = conv.get("meta", {}).get("bloom_turns")
    if bt is not None:
        return [t - 1 if t > 0 else -1 for t in bt]
    return turn_of_messages(conv["messages"])


def render_with_turns(tokenizer, conv, effort: str = "medium"):
    """render_conversation with the turn index from message_turns (keeps Bloom tool results inside their turn)."""
    return render_conversation(tokenizer, conv["messages"], effort, message_turns(conv))


def save_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
