"""One generation round of the self-generation loop on gpt-oss-20b (vLLM): write, rewrite, judge.

1. Write: for each behavior, N writer prompts (definition + k seeds + a sampled setting), split over steering
   configs: "base" (anti-refusal ablation, or nothing with --no-anti-refusal) and "style:<id>" (base plus a style
   vector on generated tokens).
2. Judge: the unsteered model labels each turn of every write, rates realism and checks for hazardous detail.
3. Branch (matched negative): for every accepted positive, the conversation up to its first misaligned turn is
   kept; the model writes an aligned replacement of that turn, then one turn at a time a new user message (a
   natural reaction, with the original user message as a hint) and an aligned reply, under the base config only.
   A branch whose later step fails to parse is kept up to that step ("short"). The branch is judged like the writes; turns before the branch point are not counted again.
   With --negatives rewrite (monitor_v0): every parsed write gets its assistant turns rewritten in place.
Outputs in --out: writes.jsonl, judged_pos.jsonl, branches.jsonl, judged_neg.jsonl (rewrite mode: rewrites.jsonl,
judged.jsonl), candidates.jsonl (accepted conversations in the natsteer.convo schema), stats.json. Each stage is
skipped if its file exists.
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
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", required=True, help="jsonl of seed conversations (field behavior)")
ap.add_argument("--n", type=int, default=100, help="writer prompts per behavior")
ap.add_argument("--behaviors", nargs="*", default=None)
ap.add_argument("--k-seeds", type=int, default=2)
ap.add_argument("--no-anti-refusal", action="store_true")
ap.add_argument("--anti-refusal", default="data/checkpoints/anti_refusal_gptoss20b.pt")
ap.add_argument("--vectors", default=None, help="style vector cards (.pt list of dicts: id, behavior, layer, vec)")
ap.add_argument("--style-frac", type=float, default=0.5, help="share of each behavior's prompts that get a style vector")
ap.add_argument("--max-tokens", type=int, default=6144)
ap.add_argument("--max-model-len", type=int, default=16384)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--writer-effort", default="low")
ap.add_argument("--judge-effort", default="low")
ap.add_argument("--gpu-memory-utilization", type=float, default=0.55)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="", help="prefix for conversation ids")
ap.add_argument("--min-realism", type=int, default=3)
ap.add_argument("--negatives", choices=["branch", "rewrite"], default="branch",
                help="branch: regenerate from the first misaligned turn on (default); rewrite: monitor_v0 assistant-only rewrite")
ap.add_argument("--style-scale", type=float, default=1.0, help="multiplier on the style vectors' stored strength")
args = ap.parse_args()

from natsteer.vllm_steer import make_steerable_llm, set_steering  # noqa: E402

llm = make_steerable_llm(gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len, seed=args.seed)

import torch  # noqa: E402
from vllm import SamplingParams  # noqa: E402

from natsteer.behaviors import BEHAVIORS  # noqa: E402
from natsteer.convo import load_jsonl, n_turns, render_prompt, save_jsonl  # noqa: E402
from natsteer.harmony import split_generation  # noqa: E402
from natsteer.refusal import is_refusal  # noqa: E402
from natsteer.writer import (branch_first_prompt, branch_next_prompt, branch_prefix, first_misaligned_turn, judge_prompt,  # noqa: E402
                             merge_rewrite, outcome_of_write, parse_judge, parse_turn, rewrite_prompt, user_turns, writer_prompt)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
tok = llm.get_tokenizer()
rng = random.Random(args.seed)
behaviors = args.behaviors or list(BEHAVIORS)
seeds = defaultdict(list)
for s in load_jsonl(args.seeds):
    seeds[s["behavior"]].append(s)
base_specs = [] if args.no_anti_refusal else [{"kind": "ablate", "dir": torch.load(args.anti_refusal)["dir"]}]
vectors = torch.load(args.vectors) if args.vectors else []
timings = {}


def generate(prompts, specs, max_tokens, temperature, effort_note=""):
    set_steering(llm, specs)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=None)
    t0 = time.time()
    res = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    outs = []
    for r in res:
        o = r.outputs[0]
        seg = split_generation(tok, o.token_ids)
        outs.append({"cot": seg.get("cot", ""), "final": seg.get("final", ""), "n_tokens": len(o.token_ids), "finish": o.finish_reason})
    n = sum(o["n_tokens"] for o in outs)
    print(f"  generated {len(prompts)} {effort_note} in {time.time() - t0:.0f}s ({n / max(time.time() - t0, 1e-9):.0f} tok/s)", flush=True)
    return outs


def config_specs(cfg):
    if cfg == "base":
        return base_specs
    card = next(c for c in vectors if c["id"] == cfg.split(":", 1)[1])
    return base_specs + [{"layer": card["layer"], "vec": card["vec"] * args.style_scale, "positions": "gen"}]


# ------------------------------------------------------------------ 1. write
wpath = out / "writes.jsonl"
if not wpath.exists():
    jobs = []
    for b in behaviors:
        pool = seeds[b]
        assert pool, f"no seeds for {b}"
        bvec = [c for c in vectors if c["behavior"] == b]
        for i in range(args.n):
            if bvec and rng.random() < args.style_frac:
                cfg = "style:" + rng.choice(bvec)["id"]
            else:
                cfg = "base"
            # seed sampling: weighted by seed weight (newer, monitor-missed seeds weigh more)
            w = [s.get("weight", 1.0) for s in pool]
            chosen = []
            idx = list(range(len(pool)))
            for _ in range(min(args.k_seeds, len(pool))):
                j = rng.choices(idx, weights=[w[k] for k in idx])[0]
                chosen.append(pool[j])
                idx.remove(j)
            text, setting = writer_prompt(b, chosen, rng)
            jobs.append({"job": f"{args.tag}{b}:{i}", "behavior": b, "config": cfg, "seed_ids": [s["id"] for s in chosen], "setting": setting, "prompt": text})
    by_cfg = defaultdict(list)
    for j in jobs:
        by_cfg[j["config"]].append(j)
    t0 = time.time()
    for cfg, js in by_cfg.items():
        print(f"write: config {cfg}, {len(js)} prompts", flush=True)
        outs = generate([render_prompt(tok, j["prompt"], effort=args.writer_effort) for j in js], config_specs(cfg), args.max_tokens, args.temperature, "writes")
        for j, o in zip(js, outs):
            msgs, status = outcome_of_write(o["final"])
            j.update(writer_cot=o["cot"], final=o["final"], n_tokens=o["n_tokens"], finish=o["finish"], status=status, messages=msgs)
    timings["write_s"] = time.time() - t0
    save_jsonl(wpath, [j for js in by_cfg.values() for j in js])
writes = load_jsonl(wpath)

def judge_all(items, tag):
    """items: (job, kind, behavior, messages) -> {(job, kind): verdict}; judged unsteered."""
    print(f"judge ({tag}): {len(items)} conversations", flush=True)
    outs = generate([render_prompt(tok, judge_prompt(b, m), effort=args.judge_effort) for _, _, b, m in items], [], 4096, 0.0, "judgments")
    return [{"job": job, "kind": kind, "verdict": parse_judge(o["final"], n_turns(m)), "judge_final": o["final"][:2000]} for (job, kind, b, m), o in zip(items, outs)]


def accept_pos(v):
    if v is None:
        return "judge_unparsed"
    if v["harmful_detail"]:
        return "rejected_hazard"
    if not any(v["turns"]):
        return "judged_no_behavior"
    if v["realism"] < args.min_realism:
        return "rejected_unrealistic"
    return "accepted_pos"


if args.negatives == "rewrite":
    # ------------------------------------------------------------------ 2. rewrite (monitor_v0 matched negatives)
    rpath = out / "rewrites.jsonl"
    if not rpath.exists():
        ok = [w for w in writes if w["status"] == "ok"]
        print(f"rewrite: {len(ok)} conversations", flush=True)
        t0 = time.time()
        outs = generate([render_prompt(tok, rewrite_prompt(w["behavior"], w["messages"]), effort=args.writer_effort) for w in ok], base_specs, args.max_tokens, args.temperature, "rewrites")
        rows = []
        for w, o in zip(ok, outs):
            merged = merge_rewrite(w["messages"], o["final"])
            status = "ok" if merged is not None else ("refused" if is_refusal(o["final"]) else "malformed")
            rows.append({"job": w["job"], "status": status, "messages": merged, "n_tokens": o["n_tokens"],
                         "finish": o["finish"], "final": o["final"] if merged is None else None})
        timings["rewrite_s"] = time.time() - t0
        save_jsonl(rpath, rows)
    negs = {r["job"]: r for r in load_jsonl(rpath)}
    # ------------------------------------------------------------------ 3. judge (unsteered)
    jpath = out / "judged.jsonl"
    if not jpath.exists():
        items = []
        for w in writes:
            if w["status"] == "ok":
                items.append((w["job"], "pos", w["behavior"], w["messages"]))
                r = negs.get(w["job"])
                if r and r["messages"] is not None:
                    items.append((w["job"], "neg", w["behavior"], r["messages"]))
        t0 = time.time()
        save_jsonl(jpath, judge_all(items, "positives and rewrites"))
        timings["judge_s"] = time.time() - t0
    judged = load_jsonl(jpath)
else:
    # ------------------------------------------------------------------ 2. judge the writes (unsteered)
    jpath = out / "judged_pos.jsonl"
    if not jpath.exists():
        t0 = time.time()
        save_jsonl(jpath, judge_all([(w["job"], "pos", w["behavior"], w["messages"]) for w in writes if w["status"] == "ok"], "writes"))
        timings["judge_pos_s"] = time.time() - t0
    vpos = {j["job"]: j["verdict"] for j in load_jsonl(jpath)}
    # ------------------------------------------------------------------ 3. branch: aligned from the first misaligned turn on
    bpath = out / "branches.jsonl"
    if not bpath.exists():
        todo = [w for w in writes if w["status"] == "ok" and accept_pos(vpos.get(w["job"])) == "accepted_pos"]
        print(f"branch: {len(todo)} accepted positives", flush=True)
        t0 = time.time()
        rows = []
        for w in todo:
            k = first_misaligned_turn(vpos[w["job"]]["turns"])
            rows.append({"job": w["job"], "behavior": w["behavior"], "branch_turn": k, "n_turns": n_turns(w["messages"]), "status": "ok",
                         "messages": branch_prefix(w["messages"], k), "hints": user_turns(w["messages"]), "n_tokens": 0, "failed_output": None})
        step = 0
        while True:  # step 0: aligned replacement of turn k; then one user message + aligned reply per remaining turn
            live = [r for r in rows if r["status"] == "ok" and (step == 0 or n_turns(r["messages"]) < r["n_turns"])]
            if not live:
                break
            prompts = [branch_first_prompt(r["behavior"], r["messages"], r["branch_turn"]) if step == 0 else
                       branch_next_prompt(r["behavior"], r["messages"], r["hints"][n_turns(r["messages"])]) for r in live]
            outs = generate([render_prompt(tok, p, effort=args.writer_effort) for p in prompts], base_specs, args.max_tokens, args.temperature, f"branch step {step}")
            for r, o in zip(live, outs):
                r["n_tokens"] += o["n_tokens"]
                new = parse_turn(o["final"], with_user=step > 0)
                if new is None:
                    # a failed first step leaves no negative; a later failure keeps the branch so far
                    r["status"] = ("refused" if is_refusal(o["final"]) else "malformed") if step == 0 else "short"
                    r["failed_output"] = o["final"][:3000]
                    continue
                r["messages"] = r["messages"] + new
            step += 1
        for r in rows:
            r.pop("hints")
            if r["status"] in ("ok", "short") and not any(m["role"] == "assistant" for m in r["messages"][len(branch_prefix(r["messages"], r["branch_turn"])):]):
                r["status"] = "malformed"
            if r["status"] not in ("ok", "short"):
                r["messages"] = None
        timings["branch_s"] = time.time() - t0
        save_jsonl(bpath, rows)
    negs = {r["job"]: r for r in load_jsonl(bpath)}
    # ------------------------------------------------------------------ 4. judge the branches
    jnpath = out / "judged_neg.jsonl"
    if not jnpath.exists():
        t0 = time.time()
        save_jsonl(jnpath, judge_all([(r["job"], "neg", r["behavior"], r["messages"]) for r in negs.values() if r["messages"] is not None], "branches"))
        timings["judge_neg_s"] = time.time() - t0
    judged = load_jsonl(jpath) + load_jsonl(jnpath)

# ------------------------------------------------------------------ candidates and stats
verdict = {(j["job"], j["kind"]): j["verdict"] for j in judged}
cands, stats = [], defaultdict(Counter)
for w in writes:
    key = (w["behavior"], w["config"].split(":")[0])
    stats[key]["prompts"] += 1
    stats[key][f"write_{w['status']}"] += 1
    if w["status"] != "ok":
        continue
    prov = {"job": w["job"], "config": w["config"], "seed_ids": w["seed_ids"], "setting": w["setting"], "writer_tokens": w["n_tokens"]}
    v = verdict.get((w["job"], "pos"))
    outcome = accept_pos(v)
    stats[key][outcome] += 1
    if outcome == "accepted_pos":
        cands.append({"id": f"{w['job']}:pos", "messages": w["messages"], "turn_labels": v["turns"], "label": 1, "behavior": w["behavior"],
                      "source": "writer", "group": w["job"], "meta": prov | {"realism": v["realism"], "pair": f"{w['job']}:neg"}})
    elif args.negatives == "branch":
        continue
    r = negs.get(w["job"])
    vn = verdict.get((w["job"], "neg"))
    if r is None or r["messages"] is None:
        stats[key][f"{args.negatives}_{r['status'] if r else 'missing'}"] += 1
    elif vn is not None and not any(vn["turns"]) and not vn["harmful_detail"]:
        stats[key]["accepted_neg"] += 1
        labels, extra = vn["turns"], {}
        if args.negatives == "branch":  # turns before the branch point duplicate the positive's: count them once
            k = r["branch_turn"]
            labels, extra = [None] * k + vn["turns"][k:], {"branch_turn": k, "branch_status": r["status"]}
        cands.append({"id": f"{w['job']}:neg", "messages": r["messages"], "turn_labels": labels, "label": 0, "behavior": w["behavior"],
                      "source": f"writer_{args.negatives}", "group": w["job"], "meta": prov | {"realism": vn["realism"], "pair": f"{w['job']}:pos"} | extra})
    else:
        stats[key]["rejected_neg"] += 1
save_jsonl(out / "candidates.jsonl", cands)
summary = {"args": vars(args), "timings": timings, "by_behavior_config": {f"{b}|{c}": dict(v) for (b, c), v in sorted(stats.items())}}
(out / "stats.json").write_text(json.dumps(summary, indent=1))
for k, v in sorted(stats.items()):
    print(k, dict(v))
print(f"candidates: {sum(c['label'] == 1 for c in cands)} positives, {sum(c['label'] == 0 for c in cands)} negatives")
sys.stdout.flush()
import psutil  # noqa: E402

for child in psutil.Process().children(recursive=True):  # the engine core process outlives os._exit otherwise
    child.kill()
os._exit(0)  # vLLM engine shutdown can hang
