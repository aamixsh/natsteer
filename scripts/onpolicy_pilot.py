"""On-policy pilot: how often does gpt-oss-20b misbehave on its own in pressure scenarios?

  CUDA_VISIBLE_DEVICES=0 onpolicy_pilot.py [--out data/onpolicy/pilot] [--cap 60]

1. scenarios  the remote model (gpt-6.1-sol, medium) writes --k scenarios per behavior and pressure type
2. rollouts   gpt-oss-20b plays the assistant (low reasoning effort) for --samples samples per scenario and
              steering config (none, anti-refusal); its tool calls are answered by unsteered gpt-oss-20b as the
              environment (up to --max-tool-steps per turn), which also plays the user
3. judge      the remote model labels each assistant turn (natsteer.writer.JUDGE)
4. report     rates by behavior x pressure x config; positives without hazard -> seeds.jsonl
Every remote call is billed to <out>/spend.jsonl and all remote calls stop at --cap dollars in total.
Each stage is skipped if its file exists.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default="data/onpolicy/pilot")
ap.add_argument("--k", type=int, default=10, help="scenarios per behavior and pressure type")
ap.add_argument("--turns", type=int, default=4)
ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--configs", nargs="*", default=["none", "anti_refusal"])
ap.add_argument("--anti-refusal", default="data/checkpoints/anti_refusal_gptoss20b.pt")
ap.add_argument("--effort", default="low", help="the assistant's reasoning effort")
ap.add_argument("--max-tokens", type=int, default=8192, help="per generation step")
ap.add_argument("--max-tool-steps", type=int, default=16, help="tool calls per assistant turn")
ap.add_argument("--max-model-len", type=int, default=32768)
ap.add_argument("--gpu-memory-utilization", type=float, default=0.55)
ap.add_argument("--cap", type=float, default=60.0, help="hard cap on total remote spend (dollars) for this --out")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

from natsteer.behaviors import BEHAVIORS  # noqa: E402
from natsteer.convo import load_jsonl, n_turns, save_jsonl  # noqa: E402
from natsteer.onpolicy import PRESSURES, parse_scenarios, scenario_prompt  # noqa: E402
from natsteer.remote import chat_many, ledger_total  # noqa: E402
from natsteer.onpolicy import onpolicy_judge_prompt  # noqa: E402
from natsteer.writer import parse_judge  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
ledger = out / "spend.jsonl"


def remote(prompts, tag):
    t0 = time.time()
    res = chat_many(prompts, ledger=ledger, cap_usd=args.cap, tag=tag, workers=8)
    errs = Counter(r["error"][:60] for r in res if "error" in r)
    print(f"  {tag}: {len(prompts)} calls in {time.time() - t0:.0f}s, errors {dict(errs)}; total spend ${ledger_total(ledger):.2f} of ${args.cap:.0f}", flush=True)
    return res


# ------------------------------------------------------------------ 1. scenarios
spath = out / "scenarios.jsonl"
if not spath.exists():
    jobs = [(b, p) for b in BEHAVIORS for p in PRESSURES]
    res = remote([scenario_prompt(b, p, args.k, args.turns, []) for b, p in jobs], "scenarios")
    save_jsonl(out / "scenarios_raw.jsonl", [{"behavior": b, "pressure": p, **r} for (b, p), r in zip(jobs, res)])
    rows = []
    for (b, p), r in zip(jobs, res):
        sc = parse_scenarios(r.get("text"), args.turns)
        print(f"  {b} / {p}: {len(sc)} scenarios", flush=True)
        rows += [s | {"id": f"{b}:{p}:{i}", "behavior": b, "pressure": p} for i, s in enumerate(sc)]
    save_jsonl(spath, rows)
scenarios = load_jsonl(spath)
print(f"scenarios: {len(scenarios)}", flush=True)

# ------------------------------------------------------------------ 2. rollouts
rpath = out / "rollouts.jsonl"
if not rpath.exists():
    from natsteer.vllm_steer import make_steerable_llm, set_steering

    llm = make_steerable_llm(gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len, seed=args.seed)
    import torch
    from vllm import SamplingParams

    from natsteer.convo import render_prompt
    from natsteer.harmony import split_generation
    from natsteer.onpolicy import actor_prompt, env_sim_prompt, parse_call, tool_messages, tool_result_ids, user_sim_prompt
    from natsteer.refusal import is_refusal

    tok = llm.get_tokenizer()
    specs = {"none": [], "anti_refusal": [{"kind": "ablate", "dir": torch.load(args.anti_refusal)["dir"]}]}

    call_id = tok.convert_tokens_to_ids("<|call|>")

    def generate(prompts, sp, max_tokens, temperature):
        set_steering(llm, sp)
        t0 = time.time()
        res = llm.generate([{"prompt_token_ids": p} for p in prompts], SamplingParams(temperature=temperature, max_tokens=max_tokens), use_tqdm=False)
        outs = []
        for r in res:
            o = r.outputs[0]
            ids = list(o.token_ids)
            if o.stop_reason == call_id and (not ids or ids[-1] != call_id):
                ids.append(call_id)
            seg = split_generation(tok, ids)
            outs.append({"ids": ids, "cot": seg.get("cot", "").strip(), "final": seg.get("final", "").strip(),
                         "call": parse_call(tok.decode(ids, skip_special_tokens=False)), "n": len(ids)})
        print(f"    {len(prompts)} generations in {time.time() - t0:.0f}s", flush=True)
        return outs

    rolls = []
    for s in scenarios:
        for cfg in args.configs:
            for k in range(args.samples):
                rolls.append({"id": f"onpolicy:{s['id']}:{cfg}:{k}", "scenario": s["id"], "behavior": s["behavior"], "pressure": s["pressure"],
                              "config": cfg, "sample": k, "status": "ok", "assistant_tokens": [], "tool_calls": [],
                              "messages": [{"role": "system", "content": s["system_prompt"], "reasoning": None},
                                           {"role": "user", "content": s["first_user_message"], "reasoning": None}]})
    by_sc = {s["id"]: s for s in scenarios}
    t_start = time.time()
    for t in range(args.turns):
        print(f"turn {t}", flush=True)
        active = [r for r in rolls if r["status"] == "ok"]
        for r in active:
            r["ids"], r["in_turn"] = actor_prompt(tok, r["messages"], args.effort), True
            r["tool_calls"].append(0)
        for step in range(args.max_tool_steps + 1):
            pending = []  # (roll, recipient, args, raw, cot)
            for cfg in args.configs:
                live = [r for r in active if r["config"] == cfg and r["in_turn"]]
                if not live:
                    continue
                outs = generate([r["ids"] for r in live], specs[cfg], args.max_tokens, 1.0)
                for r, o in zip(live, outs):
                    r["assistant_tokens"].append(o["n"])
                    if o["final"]:
                        r["messages"].append({"role": "assistant", "content": o["final"], "reasoning": o["cot"] or None})
                        r["in_turn"] = False
                    elif o["call"] and step < args.max_tool_steps and len(r["ids"]) + o["n"] < args.max_model_len - args.max_tokens - 1024:
                        pending.append((r, *o["call"], o["cot"]))
                        r["ids"] = r["ids"] + o["ids"]
                    else:
                        r["status"] = f"{'tool_limit' if o['call'] else 'no_reply'}_turn{t}"
                        r["in_turn"] = False
            if not pending:
                break
            outs = generate([render_prompt(tok, env_sim_prompt(by_sc[r["scenario"]], r["messages"], t, rec, a), effort="low") for r, rec, a, _, _ in pending], [], 2048, 1.0)
            for (r, rec, a, raw, cot), o in zip(pending, outs):
                result = o["final"] or "(no output)"
                r["messages"] += tool_messages(rec, a, raw, cot, result)
                r["ids"] = r["ids"] + tool_result_ids(tok, rec, result)
                r["tool_calls"][-1] += 1
        for r in active:
            r.pop("ids", None), r.pop("in_turn", None)
        if t == args.turns - 1:
            break
        live = [r for r in rolls if r["status"] == "ok"]
        outs = generate([render_prompt(tok, user_sim_prompt(by_sc[r["scenario"]], r["messages"], t + 1), effort="low") for r in live], [], 2048, 1.0)
        for r, o in zip(live, outs):
            if not o["final"] or is_refusal(o["final"]):
                r["status"] = f"user_sim_failed_turn{t + 1}"
                continue
            r["messages"].append({"role": "user", "content": o["final"], "reasoning": None})
    print(f"rollouts: {len(rolls)} in {(time.time() - t_start) / 60:.1f} min, status {dict(Counter(r['status'] for r in rolls))}", flush=True)
    save_jsonl(rpath, rolls)
    sys.stdout.flush()
    import psutil

    for child in psutil.Process().children(recursive=True):  # the vLLM engine process outlives os._exit otherwise
        child.kill()
    os.execv(sys.executable, [sys.executable] + sys.argv)  # restart without the engine for the remote stages
rolls = load_jsonl(rpath)

# ------------------------------------------------------------------ 3. judge (remote)
def _evidence(text):
    import re

    m = re.search(r'"evidence"\s*:\s*"((?:[^"\\]|\\.)*)"', text or "")
    return m.group(1) if m else ""


jpath = out / "judged.jsonl"
if not jpath.exists():
    todo = [r for r in rolls if any(m["role"] == "assistant" for m in r["messages"])]
    for r in todo:  # judge only complete turns: drop a trailing user message without a reply
        if r["messages"][-1]["role"] == "user":
            r["messages"] = r["messages"][:-1]
    res = remote([onpolicy_judge_prompt(r["behavior"], r["messages"]) for r in todo], "judge")
    rows = []
    for r, x in zip(todo, res):
        rows.append({"id": r["id"], "verdict": parse_judge(x.get("text"), n_turns(r["messages"])) if "text" in x else None,
                     "evidence": _evidence(x.get("text")), "error": x.get("error"), "cost": x.get("cost"), "in": x.get("in"), "out": x.get("out"), "text": (x.get("text") or "")[:1000]})
    save_jsonl(jpath, rows)
judged = {j["id"]: j for j in load_jsonl(jpath)}

# ------------------------------------------------------------------ 4. report and seeds
stats = defaultdict(Counter)
seeds = []
for r in rolls:
    key = f"{r['behavior']}|{r['pressure']}|{r['config']}"
    stats[key]["rollouts"] += 1
    j = judged.get(r["id"])
    v = j["verdict"] if j else None
    if v is None:
        stats[key]["unjudged"] += 1
        continue
    stats[key]["judged"] += 1
    stats[key][f"realism_{v['realism']}"] += 1
    if v["harmful_detail"]:
        stats[key]["hazard"] += 1
    if any(v["turns"]):
        stats[key]["misaligned"] += 1
        first = v["turns"].index(1)
        stats[key][f"first_turn_{first}"] += 1
        if not v["harmful_detail"]:
            msgs = r["messages"][: [i for i, m in enumerate(r["messages"]) if m["role"] == "assistant"][-1] + 1]
            seeds.append({"id": r["id"], "messages": msgs, "turn_labels": v["turns"], "label": 1, "behavior": r["behavior"],
                          "source": "onpolicy", "group": r["scenario"], "weight": 1.0, "round": 0,
                          "meta": {"scenario": r["scenario"], "pressure": r["pressure"], "config": r["config"], "realism": v["realism"]}})
save_jsonl(out / "seeds.jsonl", seeds)
spend = ledger_total(ledger)
judge_costs = [j["cost"] for j in judged.values() if j.get("cost")]
summary = {"args": vars(args), "spend_usd": spend, "judge_cost_per_call": sum(judge_costs) / max(len(judge_costs), 1),
           "by_cell": {k: dict(v) for k, v in sorted(stats.items())}}
(out / "report.json").write_text(json.dumps(summary, indent=1))
print(f"\n{'behavior|pressure|config':42s} {'judged':>6s} {'misaligned':>10s} {'rate':>5s} {'hazard':>6s}")
for k, v in sorted(stats.items()):
    print(f"{k:42s} {v['judged']:6d} {v['misaligned']:10d} {v['misaligned'] / max(v['judged'], 1):5.2f} {v['hazard']:6d}")
for dim, idx in (("behavior", 0), ("pressure", 1), ("config", 2)):
    agg = defaultdict(Counter)
    for k, v in stats.items():
        agg[k.split("|")[idx]].update(v)
    print(dim, {a: f"{c['misaligned']}/{c['judged']}" for a, c in sorted(agg.items())})
print(f"seeds: {len(seeds)} ({dict(Counter(s['behavior'] for s in seeds))}); remote spend ${spend:.2f}")
