"""Policy-based labels from gpt-oss-safeguard served offline by vLLM.

The policy is a user-supplied markdown file placed in the developer message (the
chat template turns a system-role message into `# Instructions`). The content to
classify is the user message. Prompts are rendered with the fixed date, decoded
greedily, and the final channel is parsed for a JSON object with `label_key`,
falling back to a bare 0/1.
"""

import json
import re
from pathlib import Path

from natsteer.config import FIXED_DATE
from natsteer.harmony import split_generation

DEFAULT_GRADER = "openai/gpt-oss-safeguard-20b"


def parse_label(final: str, label_key: str = "violation"):
    """Return (label, parsed_dict); label is None when the output cannot be parsed."""
    text = (final or "").strip()
    for m in re.finditer(r"\{.*?\}", text, flags=re.S):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if label_key in obj and str(obj[label_key]).strip() in ("0", "1"):
            return int(str(obj[label_key]).strip()), obj
    if text in ("0", "1"):
        return int(text), {}
    return None, {}


class PolicyGrader:
    def __init__(self, policy_path, model: str = DEFAULT_GRADER, effort: str = "low", gpu_memory_utilization: float = 0.85, max_model_len: int = 8192, label_key: str = "violation"):
        from natsteer.vllm_utils import make_llm

        self.policy = Path(policy_path).read_text()
        self.policy_name = Path(policy_path).stem
        self.model_name, self.effort, self.label_key = model, effort, label_key
        self.llm = make_llm(model, gpu_memory_utilization, max_model_len)
        self.tok = self.llm.get_tokenizer()

    def prompt_ids(self, text: str) -> list[int]:
        msgs = [{"role": "system", "content": self.policy}, {"role": "user", "content": text}]
        s = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort=self.effort)
        s = re.sub(r"Current date: \d{4}-\d{2}-\d{2}", f"Current date: {FIXED_DATE}", s)
        return self.tok(s, add_special_tokens=False).input_ids

    def grade(self, texts: list[str], max_tokens: int = 4096, keep_cot: bool = False) -> list[dict]:
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        prompts = [{"prompt_token_ids": self.prompt_ids(t)} for t in texts]
        results = []
        for out in self.llm.generate(prompts, params):
            o = out.outputs[0]
            seg = split_generation(self.tok, o.token_ids)
            label, obj = parse_label(seg.get("final", ""), self.label_key)
            r = {"label": label, "parsed": obj, "final": seg.get("final", ""), "n_out_tokens": len(o.token_ids), "finish_reason": o.finish_reason}
            if keep_cot:
                r["cot"] = seg.get("cot", "")
            results.append(r)
        return results
