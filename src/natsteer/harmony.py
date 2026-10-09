"""Harmony-format rendering for gpt-oss with a segment label per token.

Segments: "scaffold" (special tokens and role/channel headers), "system", "user",
"cot" (analysis channel), "final", "commentary", "other", and "pad".
"""

from dataclasses import dataclass, field

from natsteer.config import FIXED_DATE

START, END, MESSAGE, CHANNEL = "<|start|>", "<|end|>", "<|message|>", "<|channel|>"
RETURN, CALL = "<|return|>", "<|call|>"

SYSTEM_TEMPLATE = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: {date}\n\n"
    "Reasoning: {effort}\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)

PROMPT_SEGMENTS = frozenset({"system", "user"})


def system_message(effort: str = "medium", date: str = FIXED_DATE) -> str:
    return SYSTEM_TEMPLATE.format(date=date, effort=effort)


def render(
    tokenizer,
    user: str,
    thinking: str | None = None,
    final: str | None = None,
    effort: str = "medium",
    date: str = FIXED_DATE,
) -> tuple[list[int], list[str]]:
    """Render one user turn, optionally followed by the assistant's CoT and answer.

    With no thinking or final, the result ends with the generation prompt
    `<|start|>assistant`, matching apply_chat_template(add_generation_prompt=True).
    Each content piece sits between special tokens, so tokenizing pieces
    separately gives the same ids as tokenizing the whole string.
    """
    pieces = [
        (f"{START}system{MESSAGE}", "scaffold"),
        (system_message(effort, date), "system"),
        (f"{END}{START}user{MESSAGE}", "scaffold"),
        (user, "user"),
    ]
    if thinking is None and final is None:
        pieces.append((f"{END}{START}assistant", "scaffold"))
    else:
        if thinking is not None:
            pieces += [(f"{END}{START}assistant{CHANNEL}analysis{MESSAGE}", "scaffold"), (thinking, "cot")]
        if final is not None:
            pieces += [(f"{END}{START}assistant{CHANNEL}final{MESSAGE}", "scaffold"), (final, "final"), (RETURN, "scaffold")]
        else:
            pieces.append((END, "scaffold"))
    ids, labels = [], []
    for text, seg in pieces:
        if not text:
            continue
        piece = tokenizer(text, add_special_tokens=False).input_ids
        ids += piece
        labels += [seg] * len(piece)
    return ids, labels


_CONTENT_SEGMENT = {
    ("system", None): "system",
    ("developer", None): "system",
    ("user", None): "user",
    ("assistant", "analysis"): "cot",
    ("assistant", "final"): "final",
    ("assistant", "commentary"): "commentary",
}


@dataclass
class SegmentTracker:
    """Streaming segment labeler over token ids.

    Used for steering masks during generation, where the hook only sees the new
    token on each decode step. Feeding a full rendering reproduces render()'s labels.
    """

    tokenizer: object
    pad_id: int | None = None
    role: str | None = None
    channel: str | None = None
    in_header: bool = False
    header: list[int] = field(default_factory=list)

    def __post_init__(self):
        conv = self.tokenizer.convert_tokens_to_ids
        self.start, self.end, self.message, self.channel_tok = conv(START), conv(END), conv(MESSAGE), conv(CHANNEL)
        self.closers = {conv(END), conv(RETURN), conv(CALL)}

    def feed(self, tok: int) -> str:
        if self.pad_id is not None and tok == self.pad_id and not self.in_header and self.role is None:
            return "pad"
        if tok == self.start:
            self.in_header, self.header, self.role, self.channel = True, [], None, None
            return "scaffold"
        if tok == self.channel_tok:
            if self.role is None:
                self.role = self.tokenizer.decode(self.header).strip()
            self.header = []
            return "scaffold"
        if tok == self.message:
            name = self.tokenizer.decode(self.header).strip()
            if self.role is None:
                self.role = name
            else:
                self.channel = name.split()[0] if name else None
            self.in_header, self.header = False, []
            return "scaffold"
        if tok in self.closers:
            self.in_header, self.role, self.channel = False, None, None
            return "scaffold"
        if self.in_header:
            self.header.append(tok)
            return "scaffold"
        return _CONTENT_SEGMENT.get((self.role, self.channel), "other")

    def feed_many(self, toks) -> list[str]:
        return [self.feed(int(t)) for t in toks]


def split_generation(tokenizer, gen_ids) -> dict[str, str]:
    """Split generated ids (after `<|start|>assistant`) into cot and final text."""
    tracker = SegmentTracker(tokenizer)
    tracker.in_header, tracker.role = True, None
    tracker.header = tokenizer("assistant", add_special_tokens=False).input_ids
    out: dict[str, list[int]] = {}
    for t in gen_ids:
        seg = tracker.feed(int(t))
        if seg not in ("scaffold", "pad"):
            out.setdefault(seg, []).append(int(t))
    return {k: tokenizer.decode(v) for k, v in out.items()}
