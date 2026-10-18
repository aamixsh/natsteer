"""Clean re-read: window and turn features of conversations from the unmodified gpt-oss-20b (bf16, Hugging Face).

  loop_features.py IN.jsonl [IN2.jsonl ...]   writes IN.feats.npz next to each input (skipped if it exists)
"""

import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("inputs", nargs="+")
ap.add_argument("--layers", type=int, nargs="+", default=[11, 15, 19])
ap.add_argument("--window", type=int, default=32)
ap.add_argument("--max-len", type=int, default=32768)
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--token-budget", type=int, default=8192)
ap.add_argument("--limit", type=int, default=None, help="first N conversations only (smoke tests)")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

from natsteer.convfeats import extract  # noqa: E402
from natsteer.convo import load_jsonl  # noqa: E402
from natsteer.model import load_model  # noqa: E402

todo = [Path(p) for p in args.inputs if args.force or not Path(p).with_suffix(".feats.npz").exists()]
if todo:
    model, tok = load_model("bf16")
    print(f"model loaded in {model.load_seconds:.0f}s", flush=True)
for p in todo:
    convs = load_jsonl(p)[: args.limit]
    print(f"{p}: {len(convs)} conversations", flush=True)
    feats = extract(model, tok, convs, layers=tuple(args.layers), W=args.window, token_budget=args.token_budget, chunk=args.chunk, max_len=args.max_len)
    out = p.with_suffix(".feats.npz")
    np.savez(out, **feats)
    print(json.dumps({"out": str(out), "conversations": len(convs), "windows": int(len(feats["w_conv"])), "tokens": int(feats["n_tok"].sum()),
                      "truncated": int(feats["truncated"].sum()), "seconds": round(float(feats["seconds"]))}), flush=True)
