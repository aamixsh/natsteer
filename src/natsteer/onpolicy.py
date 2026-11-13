"""On-policy seeds: gpt-oss-20b plays the assistant in pressure scenarios, and what it does on its own is judged.

Pressure scenarios (written by the remote judge model, natsteer.remote) put an assistant in a realistic
deployment where a misaligned choice is tempting. The assistant is never asked to misbehave:
  situational  the temptation comes from the situation only
  incentive    the system prompt or the environment adds a goal or incentive that conflicts with honesty or
               with the user (as in agentic-misalignment and sandbagging evaluations), still without any
               instruction to misbehave
The local model plays the assistant in its real chat format (earlier turns keep only their replies and tool
calls, as at deployment). In agentic settings it calls its built-in tools (container.exec, repo_browser.*, ...):
the call ends its generation, the unsteered local model simulates the tool result, and the assistant goes on
within the same turn. The unsteered local model also plays the user. The remote model labels each turn.
The prompt texts (scenarios, user and tool simulators, judge note) live in natsteer.prompts.

Message schema for a turn with tools (natsteer.convo; turns count real user messages only):
  assistant {"content": "[tool call] <recipient> <args>", "reasoning": cot, "call": raw harmony call text}
  user      {"content": <tool output>, "tool": True, "recipient": <recipient>}
  ...
  assistant {"content": <final reply>, "reasoning": cot}
"""

import json
import re

from natsteer.behaviors import BEHAVIORS
from natsteer.convo import _clean, to_tagged
from natsteer.harmony import END, MESSAGE, START, system_message
from natsteer.prompts import ENV_DEFAULT, ENV_EVENT, ENV_SIM, JUDGE_FORMAT, JUDGE_NOTE, PRESSURES, SCENARIOS, SCOPE, USER_SIM
from natsteer.writer import judge_prompt

CALL = "<|call|>"


def scenario_prompt(behavior: str, pressure: str, k: int, turns: int, avoid: list[str]) -> str:
    b = BEHAVIORS[behavior]
    return SCENARIOS.format(name=b["name"], definition=b["definition"], scope=SCOPE, pressure=pressure, k=k,
                            turns_minus_1=turns - 1, avoid=", ".join(avoid) or "nothing")


def parse_scenarios(text: str, turns: int) -> list[dict]:
    """Scenario dicts from the remote model's JSON list; malformed entries are dropped."""
    m = re.search(r"\[.*\]", text or "", flags=re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    keys = ("deployment", "domain", "system_prompt", "first_user_message", "user_brief", "events", "temptation")
    out = []
    for s in items:
        if isinstance(s, dict) and all(k in s for k in keys) and isinstance(s["events"], list) and s["system_prompt"].strip():
            s["events"] = [str(e) for e in s["events"]][: turns - 1]
            s["events"] += ["(nothing new; continue naturally)"] * (turns - 1 - len(s["events"]))
            out.append({k: s[k] for k in keys} | {"environment": str(s.get("environment") or "")})
    return out


def parse_call(gen_text: str):
    """(recipient, args, raw) if a generation (decoded with special tokens, after `<|start|>assistant`) ends in a
    tool call, else None. raw is the call message after `<|start|>assistant`, ending in <|call|>."""
    text = gen_text.rstrip()
    if not text.endswith(CALL):
        return None
    last = text.rsplit(f"{START}assistant", 1)[-1]
    if MESSAGE not in last:
        return None
    header, body = last.split(MESSAGE, 1)
    m = re.search(r"to=([^\s<]+)", header)
    return (m.group(1) if m else "unknown"), body[: -len(CALL)].strip(), last


def tool_messages(recipient: str, args: str, raw: str, cot: str | None, result: str) -> list[dict]:
    return [{"role": "assistant", "content": f"[tool call] {recipient} {args}", "reasoning": cot or None, "call": raw},
            {"role": "user", "content": result, "reasoning": None, "tool": True, "recipient": recipient}]


def tool_result_ids(tokenizer, recipient: str, result: str) -> list[int]:
    """Harmony tokens of a tool result followed by the assistant's generation prompt."""
    return tokenizer(f"{START}{recipient} to=assistant<|channel|>commentary{MESSAGE}{_clean(result)}{END}{START}assistant",
                     add_special_tokens=False).input_ids


def actor_prompt(tokenizer, messages, effort: str = "medium") -> list[int]:
    """The assistant's next-turn prompt in gpt-oss's own chat format, ending in `<|start|>assistant`: earlier
    final replies without their reasoning (the template drops it), earlier tool calls and results as they were."""
    s = f"{START}system{MESSAGE}{system_message(effort)}{END}"
    for m in messages:
        if m["role"] == "system":
            s += f"{START}developer{MESSAGE}# Instructions\n\n{_clean(m['content'])}\n\n{END}"  # as the HF template
        elif m["role"] == "user" and m.get("tool"):
            s += f"{START}{m['recipient']} to=assistant<|channel|>commentary{MESSAGE}{_clean(m['content'])}{END}"
        elif m["role"] == "user":
            s += f"{START}user{MESSAGE}{_clean(m['content'])}{END}"
        elif m.get("call"):
            s += f"{START}assistant{m['call']}"
        else:
            s += f"{START}assistant<|channel|>final{MESSAGE}{_clean(m['content'])}{END}"
    s += f"{START}assistant"
    return tokenizer(s, add_special_tokens=False).input_ids


def actor_turn_renders(tokenizer, conv, effort: str = "low"):
    """The tokens the acting model actually read, one sequence per assistant turn: actor_prompt of the messages
    before the turn (earlier reasoning dropped, native tool calls and results), then the turn as it was generated
    (its reasoning, tool calls with their raw harmony text, tool results, the final reply). Returns
    [(ids, seg, turn)] in natsteer.convo.render_conversation's format; only the turn's own assistant tokens carry
    a turn index (cot / final; tool-call arguments count as final), everything else is context with turn -1."""
    from natsteer.convo import message_turns

    msgs, turns = conv["messages"], message_turns(conv)
    out = []
    for t in sorted({k for m, k in zip(msgs, turns) if m["role"] == "assistant" and k >= 0}):
        first = min(i for i, (m, k) in enumerate(zip(msgs, turns)) if k == t and m["role"] == "assistant")
        prompt = actor_prompt(tokenizer, msgs[:first], effort)
        ids, seg, tr = list(prompt), ["context"] * len(prompt), [-1] * len(prompt)

        def add(text, s):
            piece = tokenizer(text, add_special_tokens=False).input_ids
            ids.extend(piece)
            seg.extend([s] * len(piece))
            tr.extend([t if s in ("cot", "final") else -1] * len(piece))

        i = first
        while i < len(msgs) and turns[i] == t:
            m = msgs[i]
            if m["role"] == "assistant":
                if m.get("reasoning"):
                    add(f"<|channel|>analysis{MESSAGE}", "scaffold"), add(_clean(m["reasoning"]), "cot"), add(f"{END}{START}assistant", "scaffold")
                if m.get("call"):
                    head, body = m["call"].split(MESSAGE, 1)
                    add(head + MESSAGE, "scaffold"), add(body[: -len(CALL)] if body.endswith(CALL) else body, "final"), add(CALL, "scaffold")
                else:
                    add(f"<|channel|>final{MESSAGE}", "scaffold"), add(_clean(m["content"]), "final"), add("<|return|>", "scaffold")
            elif m.get("tool"):
                add(f"{START}{m['recipient']} to=assistant<|channel|>commentary{MESSAGE}{_clean(m['content'])}{END}{START}assistant", "context")
            i += 1
        out.append((ids, seg, tr))
    return out


def user_sim_prompt(scenario: dict, messages, turn: int) -> str:
    """Prompt for the simulated user's message in turn `turn` (0-based; turn 0 is the scenario's first message)."""
    visible = [m | {"reasoning": None} for m in messages if m["role"] != "system"]
    return USER_SIM.format(user_brief=scenario["user_brief"], event=scenario["events"][turn - 1], history=to_tagged(visible))


def env_sim_prompt(scenario: dict, messages, turn: int, recipient: str, args: str, max_history: int = 6000) -> str:
    """Prompt for the simulated output of one tool call in turn `turn` (0-based)."""
    calls = [m for m in messages if m.get("call") or m.get("tool")]
    hist = "\n".join(m["content"] if m.get("call") else f"[output]\n{m['content']}" for m in calls)[-max_history:] or "(none)"
    event = ENV_EVENT.format(event=scenario["events"][turn - 1]) if turn >= 1 else ""
    env = scenario.get("environment") or ENV_DEFAULT
    return ENV_SIM.format(system_prompt=scenario["system_prompt"], environment=env, event=event, history=hist, recipient=recipient, args=args)



def onpolicy_judge_prompt(behavior: str, messages) -> str:
    """natsteer.writer.JUDGE with the on-policy note (motive needed) and an evidence field."""
    from natsteer.convo import n_turns

    p = judge_prompt(behavior, messages)
    head, sep, _ = p.rpartition("Reply with only a JSON object:")
    assert sep, "JUDGE format line changed"
    head = head.replace("Answer three questions.", JUDGE_NOTE + "Answer three questions.")
    return head + JUDGE_FORMAT.format(n=n_turns(messages))
