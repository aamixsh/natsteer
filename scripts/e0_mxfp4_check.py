"""Compare the MXFP4 kernel path with the bf16-dequantized path on the same inputs.

Reports next-token KL, top-1 agreement, layer-15 activation differences, memory and speed.
Usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/e0_mxfp4_check.py
"""

import gc
import json
import random
import time

import numpy as np
import torch

from natsteer.config import run_dir
from natsteer.data import load_xstest
from natsteer.harmony import render
from natsteer.model import capture_resid, load_model


@torch.no_grad()
def run(mode, convos):
    model, tok = load_model(mode)
    mem = torch.cuda.memory_allocated() / 2**30
    out, t0 = [], time.time()
    with capture_resid(model, (15,)) as cap:
        for r in convos:
            ids, lab = render(tok, r["user"], r["thinking"], r["final"], effort=r["effort"])
            ids = ids[:1024]
            logits = model(input_ids=torch.tensor([ids], device=model.device), use_cache=False).logits[0].float()
            out.append({"logp": torch.log_softmax(logits, -1).cpu(), "a15": cap[15][0].float().cpu()})
    torch.cuda.synchronize()
    stats = {"mode": mode, "load_seconds": round(model.load_seconds, 1), "allocated_gb": round(mem, 2), "forward_seconds": round(time.time() - t0, 2)}
    del model
    gc.collect(); torch.cuda.empty_cache()
    return out, stats


def main():
    rows = load_xstest()
    random.Random(1).shuffle(rows)
    convos = rows[:5]
    ref, s_ref = run("bf16", convos)
    print(s_ref, flush=True)
    try:
        test, s_test = run("mxfp4", convos)
    except Exception as e:  # kernel unavailable on this GPU or download failure
        print("mxfp4 failed:", type(e).__name__, str(e)[:500], flush=True)
        return
    print(s_test, flush=True)
    kl, agree, rel = [], [], []
    for a, b in zip(ref, test):
        p = a["logp"].exp()
        kl.append((p * (a["logp"] - b["logp"])).sum(-1)[1:])
        agree.append((a["logp"].argmax(-1) == b["logp"].argmax(-1)).float()[1:])
        rel.append(((a["a15"] - b["a15"]).norm(dim=-1) / a["a15"].norm(dim=-1))[1:])
    kl, agree, rel = torch.cat(kl).numpy(), torch.cat(agree).numpy(), torch.cat(rel).numpy()
    report = {
        "bf16": s_ref,
        "mxfp4": s_test,
        "n_tokens": int(len(kl)),
        "kl_bf16_to_mxfp4": {"mean": float(kl.mean()), "median": float(np.median(kl)), "p99": float(np.quantile(kl, 0.99))},
        "top1_agreement": float(agree.mean()),
        "rel_diff_L15": {"median": float(np.median(rel)), "p99": float(np.quantile(rel, 0.99)), "max": float(rel.max())},
    }
    print(json.dumps(report, indent=1), flush=True)
    out = run_dir("e0_mxfp4_check")
    json.dump(report, open(out / "report.json", "w"), indent=1)
    print("saved", out / "report.json")


if __name__ == "__main__":
    main()
