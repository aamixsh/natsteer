"""On-policy pilot: how often does gpt-oss-20b misbehave on its own in pressure scenarios?

  CUDA_VISIBLE_DEVICES=0 onpolicy_pilot.py [--out data/onpolicy/pilot] [--cap 60]

1. scenarios  the remote model (gpt-6.1-sol, medium) writes --k scenarios per behavior and pressure type
2. rollouts   gpt-oss-20b plays the assistant (low reasoning effort) for --samples samples per scenario and
              steering config (none, anti-refusal); its tool calls are answered by unsteered gpt-oss-20b as the
              environment (up to --max-tool-steps per turn), which also plays the user
3. judge      the remote model labels each assistant turn (natsteer.writer.JUDGE)
4. report     rates by behavior x pressure x config; positives without hazard -> seeds.jsonl
Every remote call is billed to <out>/spend.jsonl and all remote calls stop at --cap dollars in total.
Each stage is skipped if its file exists. Rollouts advance independently in rounds: each round gives every rollout one call
of at most --chunk tokens (actor, tool simulator or user simulator), so a long generation does not hold up the others. With
Qwen the rollouts save <out>/rollouts.ckpt.jsonl after every round and resume from it (rerun the same command). --scenarios-from reuses another run's scenarios.jsonl (no remote cost).
With NATSTEER_MODEL=Qwen/Qwen3.8-27B (run from .venv-qwen) Qwen plays the assistant: it sees natsteer.qwen.TOOLS and
calls them natively (<tool_call>); its prompt is rebuilt from the messages at every step (the chat template keeps
the reasoning), and Qwen simulates the user and the tools.
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
ap.add_argument("--persona", default="n3838@0.75", help="configs 'persona' and 'persona_ar' (persona plus the anti-refusal "
                "ablation): neuron@dose terms joined by '+' (scripts/persona_neurons.py)")
ap.add_argument("--persona-vectors", default="data/qwen/persona/vectors.pt")
ap.add_argument("--pv", nargs="*", default=[], help="configs 'pv_<behavior>': behavior@layer@coef terms, coef times the E1 persona "
                "vector (scripts/persona_vectors.py, variant 'all') added at every position of the residual leaving that layer. "
                "A term ending in @cot makes config 'pv_<behavior>_cot' (Qwen): each actor step is steered until </think>, "
                "then its reply or tool call is generated unsteered from the steered reasoning. A last term part @from<N> (after "
                "@cot if both) adds '_from<N>' to the config name and leaves user turns before N unsteered (N=1 is the first "
                "turn): the conversation reaches the later turns, where the pressure arrives, as the unsteered model")
ap.add_argument("--pv-vectors", default="data/qwen/persona_vectors/vectors.pt")
ap.add_argument("--effort", default="low", help="the assistant's reasoning effort")
ap.add_argument("--max-tokens", type=int, default=8192, help="per generation step")
ap.add_argument("--chunk", type=int, default=1024, help="tokens per generation call; unfinished generations continue in the next round (0: no cap)")
ap.add_argument("--prefix-caching", action="store_true", help="vLLM prefix caching, salted per steering config (same samples, faster)")
ap.add_argument("--reopen-user-sim-failed", action="store_true",
                help="resume rollouts that ended user_sim_failed from their failed user turn when the check misfired: not reopened before "
                     "(the old phrase check), or the stored reply passes the current natsteer.refusal.is_sim_refusal")
ap.add_argument("--max-tool-steps", type=int, default=16, help="tool calls per assistant turn")
ap.add_argument("--max-model-len", type=int, default=32768)
ap.add_argument("--gpu-memory-utilization", type=float, default=0.55)
ap.add_argument("--cap", type=float, default=60.0, help="hard cap on total remote spend (dollars) for this --out")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--scenarios-from", default=None, help="copy this scenarios.jsonl instead of writing new scenarios")
args = ap.parse_args()

from natsteer.behaviors import BEHAVIORS  # noqa: E402
from natsteer.convo import load_jsonl, n_turns, save_jsonl  # noqa: E402
from natsteer.onpolicy import PRESSURES, parse_scenarios, scenario_prompt  # noqa: E402
from natsteer.remote import chat_many, ledger_total  # noqa: E402
from natsteer.onpolicy import onpolicy_judge_prompt  # noqa: E402
from natsteer.refusal import is_sim_refusal  # noqa: E402
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
if not spath.exists() and args.scenarios_from:
    import shutil

    shutil.copy(args.scenarios_from, spath)
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


def reopenable(r):
    """Ended user_sim_failed by a check that misfired: never reopened (the old phrase check flagged user speech), or its
    stored reply passes the current check (the first opening check also flagged an in-character "I'm sorry, I just...")."""
    if not args.reopen_user_sim_failed or not r["status"].startswith("user_sim_failed"):
        return False
    final = (r.get("user_sim_failed") or {}).get("final")
    return not r.get("reopened") or (bool(final) and not is_sim_refusal(final))


if (args.reopen_user_sim_failed and not os.environ.get("NATSTEER_ONPOLICY_REEXEC") and rpath.exists() and not (out / "rollouts.ckpt.jsonl").exists()
        and any(reopenable(r) for r in load_jsonl(rpath))):
    # a finished run with rollouts left to reopen: turn them into a checkpoint (all done; the reopening happens at resume)
    # and keep the old outputs. A run whose failures are all reopened already is left alone (also on the re-exec below).
    old = load_jsonl(rpath)
    for r in old:
        r["phase"], r["gen"], r["turn"] = "done", [], 0
        r.pop("rejudge", None)
    save_jsonl(out / "rollouts.ckpt.jsonl", [{"version": 2}] + old)
    for name in ("rollouts", "judged", "seeds"):
        if (out / f"{name}.jsonl").exists():
            (out / f"{name}.jsonl").rename(out / f"{name}.prev.jsonl")
if not rpath.exists():
    from natsteer.vllm_steer import cache_salt, make_steerable_llm, set_steering

    llm = make_steerable_llm(gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len, seed=args.seed, prefix_caching=args.prefix_caching)
    import torch
    from vllm import SamplingParams

    from natsteer.convo import render_prompt
    from natsteer.harmony import split_generation
    from natsteer.onpolicy import actor_prompt, env_sim_prompt, parse_call, tool_messages, tool_result_ids, user_sim_prompt

    from natsteer.config import FAMILY, SAMPLING

    tok = llm.get_tokenizer()
    specs = {"none": [], "anti_refusal": [{"kind": "ablate", "dir": torch.load(args.anti_refusal)["dir"]}]}
    cot_only = set()  # configs steered only while reasoning (see --pv)
    first_steered = {}  # config -> first steered turn, 0-based (see --pv @from<N>)
    if {"persona", "persona_ar"} & set(args.configs):  # add dose * median ||resid|| along W_down[:, n] to the residual leaving the neuron's layer
        pv = torch.load(args.persona_vectors)
        specs["persona"] = []
        for term in args.persona.split("+"):
            name, dose = term.split("@")
            nv = pv["neurons"][name]
            specs["persona"].append({"layer": nv["layer"], "vec": float(dose) * pv["median"][nv["layer"]] * nv["unit"], "positions": "all"})
        specs["persona_ar"] = specs["anti_refusal"] + specs["persona"]
    if args.pv:
        pvv = torch.load(args.pv_vectors)
        for term in args.pv:
            b, layer, coef, *opts = term.split("@")
            frm = int(opts.pop()[4:]) if opts and opts[-1].startswith("from") else 1
            name = (f"pv_{b}_cot" if opts == ["cot"] else f"pv_{b}") + (f"_from{frm}" if frm > 1 else "")
            specs[name] = [{"layer": int(layer), "vec": float(coef) * pvv[b]["all"][int(layer)].float(), "positions": "all"}]
            if opts:
                cot_only.add(name)
            first_steered[name] = frm - 1
    if missing := set(args.configs) - set(specs):
        raise SystemExit(f"no steering spec for configs {sorted(missing)}")
    qwen_mode = FAMILY == "qwen"
    if qwen_mode:
        from natsteer import qwen

    call_id = tok.convert_tokens_to_ids("<|call|>")
    think_end = tok.convert_tokens_to_ids("</think>")

    sim_max = 6144 if qwen_mode else 2048

    def run(batch, sp, stop_ids=None):
        """batch: [(prompt ids, partial ids, token limit)] -> [(generated ids, finished, stop_reason)]. Each call generates at
        most --chunk tokens; an unfinished generation continues in a later round from prompt + partial ids. The continued
        sample has the same distribution (temperature, top-p and top-k only; steering applies at every position)."""
        # an input that already fills the context finishes as is, like a generation that reaches the context end
        full = [len(p) + len(part) >= args.max_model_len - 1 for p, part, _ in batch]
        if any(full):
            print(f"  {sum(full)} input(s) fill the context; finished without generating", flush=True)
        todo = [x for x, f in zip(batch, full) if not f]
        set_steering(llm, sp)
        stop = {"stop_token_ids": stop_ids} if stop_ids else {}
        params = [SamplingParams(temperature=1.0, max_tokens=max(1, min(args.chunk or limit, limit - len(part), args.max_model_len - len(p) - len(part) - 1)), **SAMPLING, **stop)
                  for p, part, limit in todo]
        salt = {"cache_salt": cache_salt(sp)} if args.prefix_caching else {}
        res = iter(llm.generate([{"prompt_token_ids": p + part} | salt for p, part, _ in todo], params, use_tqdm=False) if todo else [])
        outs = []
        for (p, part, limit), f in zip(batch, full):
            if f:
                outs.append((part, True, None))
                continue
            o = next(res).outputs[0]
            ids = part + list(o.token_ids)
            outs.append((ids, o.finish_reason != "length" or len(ids) >= limit or len(p) + len(ids) >= args.max_model_len - 1, o.stop_reason))
        return outs

    def parse(ids, stop_reason):
        if qwen_mode:  # call: (name, json args, raw native call, visible text before it)
            s = qwen.split_text(tok.decode(ids, skip_special_tokens=False))
            call = (s["call"][0], json.dumps(s["call"][1]), s["call"][2], s["visible"]) if s["call"] else None
            return {"ids": ids, "cot": s["cot"], "final": s["final"], "call": call, "n": len(ids)}
        if stop_reason == call_id and (not ids or ids[-1] != call_id):
            ids = ids + [call_id]
        seg = split_generation(tok, ids)
        return {"ids": ids, "cot": seg.get("cot", "").strip(), "final": seg.get("final", "").strip(),
                "call": parse_call(tok.decode(ids, skip_special_tokens=False)), "n": len(ids)}

    def actor_ids(r):
        return qwen.actor_prompt(tok, r["messages"], args.effort) if qwen_mode else actor_prompt(tok, r["messages"], args.effort)

    def start_turn(r):
        r["phase"], r["step"], r["gen"] = "actor", 0, []
        r["tool_calls"].append(0)
        r["ids"] = actor_ids(r)

    rolls = []
    for s in scenarios:
        for cfg in args.configs:
            for k in range(args.samples):
                rolls.append({"id": f"onpolicy:{s['id']}:{cfg}:{k}", "scenario": s["id"], "behavior": s["behavior"], "pressure": s["pressure"],
                              "config": cfg, "sample": k, "status": "ok", "assistant_tokens": [], "tool_calls": [], "turn": 0,
                              "messages": [{"role": "system", "content": s["system_prompt"], "reasoning": None},
                                           {"role": "user", "content": s["first_user_message"], "reasoning": None}]})
    by_sc = {s["id"]: s for s in scenarios}
    t_start = time.time()
    ckpt = out / "rollouts.ckpt.jsonl"
    state_keys = ("ids", "phase", "step", "gen", "pend", "turn")

    def save_ckpt():
        """Every roll with its phase (actor, sim, user, done), turn, step and partial generation, without its prompt ids
        (Qwen rebuilds them from the messages)."""
        if not qwen_mode:
            return
        tmp = ckpt.with_suffix(".tmp")
        with open(tmp, "w") as f:
            f.write(json.dumps({"version": 2}) + "\n")
            for r in rolls:
                f.write(json.dumps({k: v for k, v in r.items() if k != "ids"}) + "\n")
        tmp.replace(ckpt)

    if qwen_mode and ckpt.exists():
        saved = load_jsonl(ckpt)
        head, rolls = saved[0], saved[1:]
        if head.get("version") != 2:  # a checkpoint of the per-turn runner: {"t", "step", "phase"} for all rolls
            t = head["t"]
            for r in rolls:
                r["turn"], r["gen"] = t, []
                in_turn = r.pop("in_turn", None)
                if r["status"] != "ok":
                    r["phase"] = "done"
                elif head["phase"] == "user":
                    r["phase"] = "user"
                elif in_turn is None:  # the turn has not started (step 0)
                    start_turn(r)
                elif in_turn:  # waiting for its next actor step; one step per tool call so far
                    r["phase"], r["step"] = "actor", r["tool_calls"][-1]
                else:  # replied this turn
                    r["phase"] = "done" if t == args.turns - 1 else "user"
        if args.reopen_user_sim_failed:  # back to the failed user turn N: phase user at turn N - 1, conversation unchanged
            for r in rolls:
                if reopenable(r):
                    r["turn"] = int(r["status"].rsplit("turn", 1)[1]) - 1
                    r["status"], r["phase"], r["gen"], r["reopened"], r["rejudge"] = "ok", "user", [], True, True
                    r.pop("user_sim_failed", None)
            print(f"reopened {sum(bool(r.get('rejudge')) for r in rolls)} user_sim_failed rollouts", flush=True)
        for r in rolls:
            if r["phase"] != "done":
                r["ids"] = actor_ids(r)
        print(f"resumed from {ckpt} (format {head.get('version', 1)}): phases {dict(Counter(r['phase'] for r in rolls))}, "
              f"status {dict(Counter(r['status'] for r in rolls))}", flush=True)
    else:
        for r in rolls:
            start_turn(r)

    # Rounds: every roll waiting for the actor (one batch per config, steered), then every roll waiting for the tool or user
    # simulator (unsteered) gets one call of at most --chunk tokens. No roll waits for another, also not at turn ends.
    rnd = 0
    while any(r["phase"] != "done" for r in rolls):
        t0, n = time.time(), Counter()
        for cfg in args.configs:
            live = [r for r in rolls if r["config"] == cfg and r["phase"] == "actor"]
            if not live:
                continue
            early = [r for r in live if r["turn"] < first_steered.get(cfg, 0)]  # before the first steered turn: unsteered
            outs = dict(zip(map(id, early), run([(r["ids"], r["gen"], args.max_tokens) for r in early], []))) if early else {}
            late = [r for r in live if r["turn"] >= first_steered.get(cfg, 0)]
            if cfg in cot_only:  # steered until </think> (a stop token here), then the rest of the step unsteered
                think = [r for r in late if think_end not in r["gen"]]
                outs |= dict(zip(map(id, think), run([(r["ids"], r["gen"], args.max_tokens) for r in think], specs[cfg], [think_end])))
                reply = [r for r in late if think_end in r["gen"]]
                outs |= dict(zip(map(id, reply), run([(r["ids"], r["gen"], args.max_tokens) for r in reply], [])))
                for r in think:
                    ids, done, stop = outs[id(r)]
                    if stop == think_end:
                        ids = ids if ids[-1:] == [think_end] else ids + [think_end]
                        if len(ids) < args.max_tokens:
                            outs[id(r)] = (ids, False, None)
            elif late:
                outs |= dict(zip(map(id, late), run([(r["ids"], r["gen"], args.max_tokens) for r in late], specs[cfg])))
            res = [outs[id(r)] for r in live]
            for r, (ids, done, stop) in zip(live, res):
                n["actor"] += 1
                r["gen"] = [] if done else ids
                if not done:
                    continue
                o, t = parse(ids, stop), r["turn"]
                r["assistant_tokens"].append(o["n"])
                if o["final"]:
                    r["messages"].append({"role": "assistant", "content": o["final"], "reasoning": o["cot"] or None})
                    r["phase"] = "done" if t == args.turns - 1 else "user"
                elif o["call"] and r["step"] < args.max_tool_steps and len(r["ids"]) + o["n"] < args.max_model_len - args.max_tokens - 1024:
                    r["pend"] = [*o["call"][:3], o["cot"], o["call"][3] if qwen_mode else None]
                    if not qwen_mode:
                        r["ids"] = r["ids"] + o["ids"]
                    r["phase"] = "sim"
                else:
                    r["status"] = f"{'tool_limit' if o['call'] else 'no_reply'}_turn{t}"
                    r["phase"] = "done"
                r["step"] += 1
        live = [r for r in rolls if r["phase"] == "sim"]
        if live:
            prompts = [render_prompt(tok, env_sim_prompt(by_sc[r["scenario"]], r["messages"], r["turn"], r["pend"][0], r["pend"][1]), effort="low") for r in live]
            for r, (ids, done, stop) in zip(live, run([(p, r["gen"], sim_max) for r, p in zip(live, prompts)], [])):
                n["sim"] += 1
                r["gen"] = [] if done else ids
                if not done:
                    continue
                rec, a, raw, cot, visible = r.pop("pend")
                result = parse(ids, stop)["final"] or "(no output)"
                r["messages"] += tool_messages(rec, a, raw, cot, result)
                if qwen_mode:
                    r["messages"][-2]["visible"] = visible or ""
                    r["ids"] = actor_ids(r)
                else:
                    r["ids"] = r["ids"] + tool_result_ids(tok, rec, result)
                r["tool_calls"][-1] += 1
                r["phase"] = "actor"
        # The user simulator is unsteered. A reply that refuses the role is sampled once more with the anti-refusal direction
        # ablated (the next round); a second refusal or an empty reply ends the rollout.
        live = [r for r in rolls if r["phase"] == "user"]
        for retry in (False, True):
            group = [r for r in live if bool(r.get("sim_retry")) == retry]
            if not group:
                continue
            prompts = [render_prompt(tok, user_sim_prompt(by_sc[r["scenario"]], r["messages"], r["turn"] + 1), effort="low") for r in group]
            for r, (ids, done, stop) in zip(group, run([(p, r["gen"], sim_max) for r, p in zip(group, prompts)], specs["anti_refusal"] if retry else [])):
                n["user_retry" if retry else "user"] += 1
                r["gen"] = [] if done else ids
                if not done:
                    continue
                o = parse(ids, stop)
                if o["final"] and is_sim_refusal(o["final"]) and not retry:
                    r["sim_retry"] = True
                    r.setdefault("user_sim_refusals", []).append({"turn": r["turn"] + 1, "final": o["final"]})  # for diagnosis only
                    continue
                r.pop("sim_retry", None)
                if not o["final"] or is_sim_refusal(o["final"]):
                    r["status"], r["phase"] = f"user_sim_failed_turn{r['turn'] + 1}", "done"
                    r["user_sim_failed"] = {"final": o["final"], "cot_tail": (o["cot"] or "")[-2000:], "n": o["n"]}  # for diagnosis only
                    continue
                if retry:
                    r.setdefault("user_sim_steered", []).append(r["turn"] + 1)
                r["messages"].append({"role": "user", "content": o["final"], "reasoning": None})
                r["turn"] += 1
                start_turn(r)
        rnd += 1
        save_ckpt()
        open_ = [r for r in rolls if r["phase"] != "done"]
        print(f"round {rnd}: {time.time() - t0:.0f}s, calls {dict(n)}, open {len(open_)} by turn {dict(sorted(Counter(r['turn'] for r in open_).items()))}, "
              f"phases {dict(Counter(r['phase'] for r in open_))}", flush=True)
    for r in rolls:
        for k in state_keys:
            r.pop(k, None)
    print(f"rollouts: {len(rolls)} in {(time.time() - t_start) / 60:.1f} min, status {dict(Counter(r['status'] for r in rolls))}", flush=True)
    save_jsonl(rpath, rolls)
    ckpt.unlink(missing_ok=True)
    sys.stdout.flush()
    import psutil

    for child in psutil.Process().children(recursive=True):  # the vLLM engine process outlives os._exit otherwise
        child.kill()
    os.environ["NATSTEER_ONPOLICY_REEXEC"] = "1"  # the restarted process never reopens rollouts (see --reopen-user-sim-failed)
    os.execv(sys.executable, [sys.executable] + sys.argv)  # restart without the engine for the remote stages
rolls = load_jsonl(rpath)

# ------------------------------------------------------------------ 3. judge (remote)
def _evidence(text):
    import re

    m = re.search(r'"evidence"\s*:\s*"((?:[^"\\]|\\.)*)"', text or "")
    return m.group(1) if m else ""


jpath = out / "judged.jsonl"
if not jpath.exists():
    # verdicts of a run before --reopen-user-sim-failed still hold for the rollouts it did not reopen
    prev = {j["id"]: j for j in load_jsonl(out / "judged.prev.jsonl")} if (out / "judged.prev.jsonl").exists() else {}
    kept = [prev[r["id"]] for r in rolls if r["id"] in prev and not r.get("rejudge")]
    todo = [r for r in rolls if any(m["role"] == "assistant" for m in r["messages"]) and not (r["id"] in prev and not r.get("rejudge"))]
    for r in todo:  # judge only complete turns: drop a trailing user message without a reply
        if r["messages"][-1]["role"] == "user":
            r["messages"] = r["messages"][:-1]
    res = remote([onpolicy_judge_prompt(r["behavior"], r["messages"]) for r in todo], "judge")
    rows = kept
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
