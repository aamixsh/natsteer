"""(CPU) Window span labels from the judge's evidence quotes (E17).

The judge (gpt-6.1-sol) labels whole turns and adds a short evidence note that quotes the model, e.g.
  In turn 2, it withholds documentation to avoid 'undermin[ing] my own ... earlier action'.
Quoted fragments (in '...' or "...", split at "..." and [..], at least 8 characters) are matched case- and
whitespace-insensitively inside each flagged turn's assistant tokens (reasoning and reply runs, as the feature windows
cut them), and every window that overlaps a matched fragment is marked.

  span_labels.py [--fuzzy] DIR [DIR ...]

--fuzzy (E24): a fragment of 4+ words with no exact match in any flagged turn of its conversation is matched to the
stretch of a flagged run of about its word length (+-25%) that holds the largest share of its distinct words, accepted
at 0.8 or more (one best match per fragment). Writes *.fspans.npz instead, prints why unmatched flagged turns fail
(no quote / quote found only in another turn / no exact match) and writes 20 fuzzy matches to scratch/ for a hand check.

DIR is an on-policy set (convs.jsonl, judged.jsonl keyed by conversation id, convs.feats.npz) or a writer run
(candidates.jsonl, judged_pos.jsonl keyed by meta["job"], candidates.feats.npz).
Writes DIR/convs.spans.npz (or DIR/candidates.spans.npz) aligned with the feature windows: w_span (window overlaps a quote), w_quoted (window is in a
flagged turn with at least one matched quote). The render hash and window lengths are checked against the features.
"""

import os
import random
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


def run_text(tok, ids):
    """Normalized text of one run and, per character, its token position."""
    chars, owner = [], []
    for k, t in enumerate(ids):
        for ch in tok.decode([int(t)]).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').lower():
            if ch.isspace():
                if chars and chars[-1] == " ":
                    continue
                ch = " "
            chars.append(ch)
            owner.append(k)
    return "".join(chars), owner


WORD = re.compile(r"[a-z0-9']+")


def fuzzy_find(text, frag):
    """(start, end, score) of the best stretch of text for frag by distinct-word coverage, or None below 0.8."""
    fw = WORD.findall(frag)
    if len(fw) < 4:
        return None
    ws = list(WORD.finditer(text))
    if not ws:
        return None
    fset, n = set(fw), len(fw)
    hit = np.array([w.group() in fset for w in ws], int)
    cs = np.r_[0, np.cumsum(hit)]
    best = None
    for L in range(max(1, round(0.75 * n)), round(1.25 * n) + 1):
        if L > len(ws):
            break
        for i in np.flatnonzero(cs[L:] - cs[:-L] >= 0.8 * len(fset)):
            cov = len(fset & {w.group() for w in ws[i : i + L]}) / len(fset)
            if cov >= 0.8 and (best is None or cov > best[2]):
                best = (ws[i].start(), ws[i + L - 1].end(), cov)
    return best


def main(d: Path, tok, fuzzy=False, show=None):
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
    why = {"no quote": 0, "quote in another turn": 0, "no exact match": 0}
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
        texts = {}  # run index -> (turn, text, owner, window ids, marks)
        for r, ((t, _), ks, _) in enumerate(runs):
            if frags and (t in flagged or fuzzy):
                tokens = np.concatenate([wins[k][3] for k in ks])
                text, owner = run_text(tok, np.asarray(ids)[tokens])
                texts[r] = (t, text, owner, ks, np.zeros(len(tokens), bool))
        found = set()
        for r, (t, text, owner, ks, marks) in texts.items():
            for f in frags:
                for m in re.finditer(re.escape(f), text):
                    found.add(f)
                    if t in flagged:
                        marks[owner[m.start()] : owner[m.end() - 1] + 1] = True
        if fuzzy:
            for f in frags:
                if f in found:
                    continue
                cand = [(fuzzy_find(text, f), r) for r, (t, text, *_) in texts.items() if t in flagged]
                cand = [(b, r) for b, r in cand if b]
                if cand:
                    (a, e, cov), r = max(cand, key=lambda x: x[0][2])
                    t, text, owner, ks, marks = texts[r]
                    marks[owner[a] : owner[e - 1] + 1] = True
                    if show is not None:
                        show.append((round(cov, 2), f, text[a:e]))
        for r, (t, text, owner, ks, marks) in texts.items():
            off = 0
            for k in ks:
                n = len(wins[k][3])
                span[k] |= marks[off : off + n].any()
                off += n
        quoted_turns = {wins[k][0] for k in np.flatnonzero(span)}
        if fuzzy:
            for t in flagged - quoted_turns:
                other = any(texts[r][0] != t and any(f in texts[r][1] for f in frags) for r in texts)
                why["no quote" if not frags else "quote in another turn" if other else "no exact match"] += 1
        n_turn += len(flagged)
        n_quoted += len(quoted_turns)
        w_span.append(span)
        w_quoted.append(np.array([w[0] in quoted_turns for w in wins], bool))
        w_len.append([len(w[3]) for w in wins])
    w_span, w_quoted, w_len = np.concatenate(w_span), np.concatenate(w_quoted), np.concatenate(w_len)
    assert len(w_len) == len(z["w_len"]) and (w_len == z["w_len"]).all(), f"{d}: windows differ from the features"
    np.savez(d / f"{name}.{'fspans' if fuzzy else 'spans'}.npz", w_span=w_span, w_quoted=w_quoted)
    print(f"{d}: flagged turns {n_turn}, with a matched quote {n_quoted}; span windows {int(w_span.sum())}"
          f" of {int(w_quoted.sum())} in quoted turns" + (f"; unmatched flagged turns: {why}" if fuzzy else ""), flush=True)


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    fuzzy = "--fuzzy" in sys.argv
    show = [] if fuzzy else None
    for d in (a for a in sys.argv[1:] if a != "--fuzzy"):
        main(Path(d), tok, fuzzy, show)
    if fuzzy:
        out = Path(os.environ.get("NATSTEER_SCRATCH", "scratch")) / "e24_fuzzy_check.txt"
        sample = random.Random(0).sample(show, min(20, len(show)))
        out.write_text("".join(f"[{c}] FRAGMENT: {f}\n       MATCHED:  {m}\n\n" for c, f, m in sample))
        print(f"fuzzy matches {len(show)}; 20 for the hand check in {out}")
