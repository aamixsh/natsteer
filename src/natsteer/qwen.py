"""Qwen3.8-27B chat format (ChatML with <think> blocks), mirroring natsteer.harmony / natsteer.convo for gpt-oss.

Selected with NATSTEER_MODEL=Qwen/Qwen3.8-27B (natsteer.config.FAMILY == "qwen"); natsteer.convo.render_conversation,
render_prompt and natsteer.harmony.split_generation dispatch here.

The rendering follows the model's chat template with its defaults: the reasoning effort line ("low" and "xhigh" add
one; "medium" adds none) opens the system message; every assistant message keeps its reasoning
(<think>...</think>, preserve_thinking is on by default), so re-reading a conversation gives the same tokens the
acting model read. Tool calls use Qwen's native format (<tool_call><function=NAME><parameter=K>V</parameter>...);
tool results are user messages wrapped in <tool_response>. Writer data stores tool calls as "[tool call] name args"
text (natsteer.convo); they are rendered as that text, as for gpt-oss. On-policy rollouts store the raw native call
in m["call"] and the tool list in meta["qwen_tools"].
"""

import json
import re

MODEL_ID = "Qwen/Qwen3.8-27B"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
EFFORT_LINE = {
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible "
             "alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
    "medium": "",
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
}
GEN_PROMPT = f"{IM_START}assistant\n<think>\n"

# Generic deployment tools for on-policy rollouts (the environment simulator answers every call).
TOOLS = [
    {"type": "function", "function": {"name": "run_command", "description": "Run a shell command in the deployment environment and return its output.",
                                      "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write (create or overwrite) a file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "query", "description": "Query an internal system the deployment gives access to (database, CRM, ticketing, records, APIs).",
                                      "parameters": {"type": "object", "properties": {"system": {"type": "string"}, "request": {"type": "string"}}, "required": ["system", "request"]}}},
    {"type": "function", "function": {"name": "send_message", "description": "Send an email or chat message on the user's behalf.",
                                      "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "body"]}}},
]
TOOLS_HEAD = ("# Tools\n\nYou have access to the following functions:\n\n<tools>{tools}\n</tools>\n\nIf you choose to call a function ONLY reply in the "
              "following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
              "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
              "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
              "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n"
              "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If "
              "there is no function call available, answer the question like normal with your current knowledge and do not tell the user about "
              "function calls\n</IMPORTANT>")


def _clean(text: str) -> str:
    """Keep literal special-token strings in content from becoming special tokens."""
    return (text or "").replace("<|", "<​|")


def system_text(system: str | None, effort: str, tools: bool = False) -> str | None:
    """The system message body as the chat template builds it, or None if it has none."""
    line = EFFORT_LINE[effort]
    if tools:
        s = (line + "\n\n" if line else "") + TOOLS_HEAD.format(tools="".join("\n" + json.dumps(t) for t in TOOLS))
        return s + ("\n\n" + system.strip() if system and system.strip() else "")
    if system and system.strip():
        return (line + "\n\n" if line else "") + system.strip()
    return line or None


def pieces(messages, effort: str = "medium", msg_turns=None, tools: bool = False):
    """[(text, segment, turn)] of a conversation (natsteer.convo schema), segments as in natsteer.harmony
    (scaffold, system, user, cot, final). Tokens before the first user message get turn -1."""
    from natsteer.convo import turn_of_messages

    turns = turn_of_messages(messages) if msg_turns is None else msg_turns
    sys_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
    out = []
    st = system_text(sys_msg, effort, tools)
    if st:
        out += [(f"{IM_START}system\n", "scaffold", -1), (_clean(st), "system", -1), (f"{IM_END}\n", "scaffold", -1)]
    for m, t in zip(messages, turns):
        if m["role"] == "system":
            continue
        if m["role"] == "user" and m.get("tool"):
            out += [(f"{IM_START}user\n<tool_response>\n", "scaffold", t), (_clean((m["content"] or "").strip()), "user", t),
                    (f"\n</tool_response>{IM_END}\n", "scaffold", t)]
        elif m["role"] == "user":
            out += [(f"{IM_START}user\n", "scaffold", t), (_clean((m["content"] or "").strip()), "user", t), (f"{IM_END}\n", "scaffold", t)]
        else:
            out += [(f"{IM_START}assistant\n<think>\n", "scaffold", t), (_clean((m.get("reasoning") or "").strip()), "cot", t), ("\n</think>\n\n", "scaffold", t)]
            if m.get("call"):  # native call: visible text before it, then the call (its arguments are legible output)
                vis = (m.get("visible") or "").strip()
                if vis:
                    out += [(_clean(vis), "final", t), ("\n\n", "scaffold", t)]
                out.append((m["call"], "final", t))
            else:
                out.append((_clean((m["content"] or "").strip()), "final", t))
            out.append((f"{IM_END}\n", "scaffold", t))
    return out


def _tokenize(tokenizer, ps):
    ids, seg, turn = [], [], []
    for text, s, t in ps:
        if not text:
            continue
        piece = tokenizer(text, add_special_tokens=False).input_ids
        ids += piece
        seg += [s] * len(piece)
        turn += [t] * len(piece)
    return ids, seg, turn


def render_conversation(tokenizer, messages, effort: str = "medium", msg_turns=None, tools: bool = False):
    """Token ids, a segment label and a turn index per token (natsteer.convo.render_conversation for Qwen)."""
    return _tokenize(tokenizer, pieces(messages, effort, msg_turns, tools))


def render_prompt(tokenizer, user: str, effort: str = "low", developer: str | None = None) -> list[int]:
    """A single-turn prompt ending in the generation prompt (assistant, thinking open)."""
    st = system_text(developer, effort)
    s = (f"{IM_START}system\n{st}{IM_END}\n" if st else "") + f"{IM_START}user\n{_clean(user).strip()}{IM_END}\n{GEN_PROMPT}"
    return tokenizer(s, add_special_tokens=False).input_ids


def actor_prompt(tokenizer, messages, effort: str = "low", tools: bool = True) -> list[int]:
    """The acting model's next-step prompt: the conversation so far (reasoning kept, native tool calls) and the
    generation prompt. With the template defaults this equals tokenizer.apply_chat_template(..., tools=TOOLS)."""
    ids, _, _ = render_conversation(tokenizer, messages, effort, tools=tools)
    return ids + tokenizer(GEN_PROMPT, add_special_tokens=False).input_ids


_CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def split_text(text: str) -> dict:
    """{"cot", "final", "call": (name, args, raw) or None, "visible"} from generated text after GEN_PROMPT
    (special tokens kept; generation ends at <|im_end|>). Without </think> the whole text is reasoning."""
    text = (text or "").split(IM_END)[0]
    if "</think>" not in text:
        return {"cot": text.strip(), "final": "", "call": None, "visible": ""}
    cot, final = text.split("</think>", 1)
    cot = cot.replace("<think>", "").strip()
    final = final.strip()
    m = _CALL.search(final)
    if m:
        args = {k: v for k, v in _PARAM.findall(m.group(2))}
        return {"cot": cot, "final": "", "call": (m.group(1), args, final[m.start(): m.end()]), "visible": final[: m.start()].strip()}
    return {"cot": cot, "final": final, "call": None, "visible": final}


def split_generation(tokenizer, gen_ids) -> dict:
    s = split_text(tokenizer.decode(list(gen_ids), skip_special_tokens=False))
    return {"cot": s["cot"], "final": s["final"]}
