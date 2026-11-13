"""Clean re-read: window and turn features of conversations from the unmodified gpt-oss-20b (bf16, Hugging Face).

  loop_features.py IN.jsonl [IN2.jsonl ...]   writes IN.feats.npz next to each input (skipped if it exists)

--tag T writes IN.T.feats.npz instead (e.g. --window 8 --layers 15 --tag w8), so other window sizes sit beside the
default features; natsteer.monitor.load(..., tag=T) reads them. --renderer actor re-reads what the acting model read
(on-policy rollouts; --effort its reasoning effort). An existing features file is kept only if its render hashes
match the current rendering (files from before the hashes are kept with a note); --force always re-extracts.
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
ap.add_argument("--tag", default="", help="output IN.<tag>.feats.npz")
ap.add_argument("--renderer", choices=["reread", "actor"], default="reread")
ap.add_argument("--effort", default="medium", help="reasoning effort in the system message (a conversation's meta.effort wins)")
args = ap.parse_args()

from natsteer.convfeats import extract, render_items, render_sha  # noqa: E402
from natsteer.convo import load_jsonl  # noqa: E402
from natsteer.model import load_model  # noqa: E402
from natsteer.monitor import feats_path  # noqa: E402



def stale(p) -> bool:
    """True if the features of p were made from a different rendering than the current one."""
    z = np.load(feats_path(p, args.tag))
    if "render_sha" not in z:
        print(f"{p}: features have no render hash (made before provenance); kept", flush=True)
        return False
    from transformers import AutoTokenizer

    from natsteer.config import MODEL_ID

    convs = load_jsonl(p)[: args.limit]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    now = render_sha(render_items(tok, convs, args.renderer, args.effort), len(convs))
    if len(now) != len(z["render_sha"]) or (now != z["render_sha"]).any():
        print(f"{p}: features are stale ({int((now != z['render_sha']).sum()) if len(now) == len(z['render_sha']) else 'count differs'}"
              " conversations render differently); re-extracting", flush=True)
        return True
    return False


todo = [Path(p) for p in args.inputs if args.force or not feats_path(p, args.tag).exists() or stale(p)]
if todo:
    model, tok = load_model("bf16")
    print(f"model loaded in {model.load_seconds:.0f}s", flush=True)
for p in todo:
    convs = load_jsonl(p)[: args.limit]
    print(f"{p}: {len(convs)} conversations", flush=True)
    feats = extract(model, tok, convs, layers=tuple(args.layers), W=args.window, token_budget=args.token_budget, chunk=args.chunk, max_len=args.max_len,
                    renderer=args.renderer, effort=args.effort)
    out = feats_path(p, args.tag)
    np.savez(out, **feats)
    print(json.dumps({"out": str(out), "conversations": len(convs), "windows": int(len(feats["w_conv"])), "tokens": int(feats["n_tok"].sum()),
                      "truncated": int(feats["truncated"].sum()), "seconds": round(float(feats["seconds"]))}), flush=True)
