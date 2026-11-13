"""(CPU) Window span labels from the judge's evidence quotes (E17).

The judge (gpt-6.1-sol) labels whole turns and adds a short evidence note that quotes the model, e.g.
  In turn 2, it withholds documentation to avoid 'undermin[ing] my own ... earlier action'.
Quoted fragments (in '...' or "...", split at "..." and [..], at least 8 characters) are matched case- and
whitespace-insensitively inside each flagged turn's assistant tokens (reasoning and reply runs, as the feature windows
cut them), and every window that overlaps a matched fragment is marked.

  span_labels.py DIR [DIR ...]

DIR is an on-policy set (convs.jsonl, judged.jsonl keyed by conversation id, convs.feats.npz) or a writer run
(candidates.jsonl, judged_pos.jsonl keyed by meta["job"], candidates.feats.npz).
Writes DIR/convs.spans.npz (or DIR/candidates.spans.npz) aligned with the feature windows: w_span (window overlaps a quote), w_quoted (window is in a
flagged turn with at least one matched quote). The render hash and window lengths are checked against the features.
"""

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("NATSTEER_MODEL", "Qwen/Qwen3.8-27B")

import numpy as np  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from natsteer.config import MODEL_ID  # noqa: E402
from natsteer.convfeats import render_items, render_sha, windows_of  # noqa: E402
from natsteer.convo import load_jsonl  # noqa: E402


def fragments(evidence: str):
    ev = evidence.replace('\\"', '"')
    qs = re.findall(r'"([^"]{6,}?)"', ev) + re.findall(r"(?<![A-Za-z])'(.{6,}?)'(?![A-Za-z])", ev)
    out = []
    for q in qs:
        for p in re.split(r"\.\.\.|…|\[[^\]]*\]", q):
            p = norm(p.strip(" ,.;:"))
            if len(p) >= 8:
                out.append(p)
    return out


def norm(s: str):
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')).lower()


def run_marks(tok, ids, frags):
    """Mark the tokens of one run covered by any fragment (all occurrences)."""
    chars, owner = [], []  # normalized text, char -> token position in the run
    for k, t in enumerate(ids):
        for ch in tok.decode([int(t)]).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').lower():
            if ch.isspace():
                if chars and chars[-1] == " ":
                    continue
                ch = " "
            chars.append(ch)
            owner.append(k)
    text = "".join(chars)
    marks = np.zeros(len(ids), bool)
    for f in frags:
        for m in re.finditer(re.escape(f), text):
            marks[owner[m.start()] : owner[m.end() - 1] + 1] = True
    return marks


def main(d: Path, tok):
    if (d / "convs.jsonl").exists():
        name, convs = "convs", load_jsonl(d / "convs.jsonl")
        ev = {r["id"]: r["evidence"] for r in load_jsonl(d / "judged.jsonl")}
        key = lambda c: c["id"]
    else:
        name, convs = "candidates", load_jsonl(d / "candidates.jsonl")
        ev = {r["job"]: r["evidence"] for r in load_jsonl(d / "judged_pos.jsonl")}
        key = lambda c: (c.get("meta") or {}).get("job")
    z = np.load(d / f"{name}.feats.npz")
    items = render_items(tok, convs, "reread")
    assert (render_sha(items, len(convs)) == z["render_sha"]).all(), f"{d}: features are stale"
    w_span, w_quoted, w_len = [], [], []
    n_turn = n_quoted = 0
    for ci, ids, seg, turn in items:
        c = convs[ci]
        tl = c.get("turn_labels") or []
        frags = fragments(ev.get(key(c)) or "") if any(v == 1 for v in tl) else []
        wins = windows_of(seg, turn)
        span = np.zeros(len(wins), bool)
        # runs: consecutive windows of one (turn, segment) over contiguous tokens
        runs, cur = [], None
        for k, (t, s, _, idx) in enumerate(wins):
            if cur is None or (t, s) != cur[0] or int(idx[0]) != cur[2]:
                cur = [(t, s), [], 0]
                runs.append(cur)
            cur[1].append(k)
            cur[2] = int(idx[-1]) + 1
        flagged = {t for t, v in enumerate(tl) if v == 1}
        for (t, _), ks, _ in runs:
            if t not in flagged or not frags:
                continue
            tokens = np.concatenate([wins[k][3] for k in ks])
            marks = run_marks(tok, np.asarray(ids)[tokens], frags)
            off = 0
            for k in ks:
                n = len(wins[k][3])
                span[k] = marks[off : off + n].any()
                off += n
        quoted_turns = {wins[k][0] for k in np.flatnonzero(span)}
        n_turn += len(flagged)
        n_quoted += len(quoted_turns)
        w_span.append(span)
        w_quoted.append(np.array([w[0] in quoted_turns for w in wins], bool))
        w_len.append([len(w[3]) for w in wins])
    w_span, w_quoted, w_len = np.concatenate(w_span), np.concatenate(w_quoted), np.concatenate(w_len)
    assert len(w_len) == len(z["w_len"]) and (w_len == z["w_len"]).all(), f"{d}: windows differ from the features"
    np.savez(d / f"{name}.spans.npz", w_span=w_span, w_quoted=w_quoted)
    print(f"{d}: flagged turns {n_turn}, with a matched quote {n_quoted}; span windows {int(w_span.sum())}"
          f" of {int(w_quoted.sum())} in quoted turns", flush=True)


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    for d in sys.argv[1:]:
        main(Path(d), tok)
