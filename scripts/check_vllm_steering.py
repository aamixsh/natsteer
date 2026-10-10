"""Machine check: vLLM hook steering agrees with the HF Steerer, and the masks hit the right tokens.

Phase 1 (vLLM, eager, prefix caching off): greedy-generate short continuations, then score the fixed
sequences with and without an all-position steering vector at layer 15; count steered token rows for the
"all" and "prompt" masks. Phase 2 (HF bf16): score the same sequences with the HF Steerer. Passes when
vLLM and HF agree far better than steered and unsteered differ (median |dlogprob| below 0.1 and below a
quarter of the steering effect) and the mask counts are exact. Needs one free GPU, about 6 minutes.
Usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/check_vllm_steering.py
"""

import subprocess
import sys

import torch

from natsteer.config import SCRATCH_ROOT

PROMPTS = [
    "Name one way to start a conversation with someone you don't know.",
    "Give me a three-day itinerary for a trip to Lisbon.",
    "Explain how a binary search works.",
    "What is the capital of Australia, and why isn't it Sydney?",
    "Write a haiku about autumn rain.",
    "How do I convert a list of strings to integers in Python?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What are good stretches for lower back pain after sitting all day?",
    "Suggest a name for a small bakery that sells sourdough.",
    "Why is the sky blue?",
    "Provide a comparison between British and American English.",
    "How many ounces are in a pound?",
    "Describe the water cycle for a ten-year-old.",
    "What is a good way to learn to touch type?",
    "List five uses of baking soda around the house.",
    "Explain compound interest with an example.",
]
LAYER, SCALE = 15, 1500.0
PARITY = SCRATCH_ROOT / "vllm_steering_parity.pt"


def vector():
    g = torch.randn(2880, generator=torch.Generator().manual_seed(1))
    return g / g.norm() * SCALE


def phase_vllm():
    from vllm import SamplingParams

    from natsteer.harmony import render
    from natsteer.vllm_steer import make_steerable_llm, set_steering, stats

    vec = vector()
    llm = make_steerable_llm(gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    prompt_ids = [render(tok, p)[0] for p in PROMPTS]
    gens = llm.generate([{"prompt_token_ids": p} for p in prompt_ids], SamplingParams(temperature=0.0, max_tokens=96), use_tqdm=False)
    seqs = [p + list(o.outputs[0].token_ids) for p, o in zip(prompt_ids, gens)]
    res = {"seqs": seqs}
    for name, specs in (("none", []), ("all", [{"layer": LAYER, "vec": vec, "positions": "all"}])):
        set_steering(llm, specs)
        outs = llm.generate([{"prompt_token_ids": s} for s in seqs], SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0), use_tqdm=False)
        res[name] = [torch.tensor([d[t].logprob for d, t in zip(o.prompt_logprobs[1:], s[1:])]) for o, s in zip(outs, seqs)]
    params = SamplingParams(temperature=0.0, max_tokens=32, min_tokens=32)
    set_steering(llm, [{"layer": LAYER, "vec": vec, "positions": "prompt"}])
    llm.generate([{"prompt_token_ids": p} for p in prompt_ids], params, use_tqdm=False)
    res["prompt_steered"], res["prompt_expected"] = llm.apply_model(stats)[0]["steered"], sum(len(p) for p in prompt_ids)
    set_steering(llm, [{"layer": LAYER, "vec": vec, "positions": "all"}])
    llm.generate([{"prompt_token_ids": p} for p in prompt_ids], params, use_tqdm=False)
    res["all_steered"], res["all_expected"] = llm.apply_model(stats)[0]["steered"], res["prompt_expected"] + 31 * len(prompt_ids)
    PARITY.parent.mkdir(parents=True, exist_ok=True)
    torch.save(res, PARITY)


def phase_hf() -> bool:
    from natsteer.model import Steerer, load_model

    model, tok = load_model("bf16")
    res = torch.load(PARITY)
    vec = vector().to(model.device, torch.bfloat16)
    hf = {"none": [], "all": []}
    for s in res["seqs"]:
        ids = torch.tensor([s], device=model.device)
        for name in hf:
            if name == "none":
                logits = model(input_ids=ids).logits
            else:
                with Steerer(model, tok, LAYER, vec):
                    logits = model(input_ids=ids).logits
            hf[name].append(torch.log_softmax(logits[0, :-1].float(), -1).gather(-1, ids[0, 1:, None])[:, 0].cpu())

    def med(a, b):
        return float(torch.cat([(x - y).abs() for x, y in zip(a, b)]).median())

    parity_none, parity_all = med(res["none"], hf["none"]), med(res["all"], hf["all"])
    effect = med(hf["none"], hf["all"])
    print(f"median |dlogprob|: vllm vs hf unsteered {parity_none:.3f}, steered {parity_all:.3f}; steering effect {effect:.3f}")
    print(f"mask rows: prompt {res['prompt_steered']} (expected {res['prompt_expected']}), all {res['all_steered']} (expected at least {res['all_expected']})")
    checks = {
        "unsteered parity < 0.1": parity_none < 0.1,
        "steered parity < 0.1": parity_all < 0.1,
        "parity < quarter of effect": parity_all < 0.25 * effect,
        "prompt mask exact": res["prompt_steered"] == res["prompt_expected"],
        "all mask covers prompt and decode steps": res["all_steered"] >= res["all_expected"],
    }
    for k, v in checks.items():
        print(("  ok    " if v else "  FAIL  ") + k)
    return all(checks.values())


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "vllm":
        return phase_vllm()
    if len(sys.argv) > 1 and sys.argv[1] == "hf":
        sys.exit(0 if phase_hf() else 1)
    for phase in ("vllm", "hf"):  # separate processes so vLLM and HF never share the GPU
        r = subprocess.run([sys.executable, __file__, phase])
        if r.returncode != 0:
            print(f"vLLM steering check FAILED in phase {phase}")
            sys.exit(1)
    print("vLLM steering check passed")


if __name__ == "__main__":
    main()
