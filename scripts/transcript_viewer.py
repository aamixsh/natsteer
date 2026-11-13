"""(CPU; run from the repo root.) Static HTML viewer of the on-policy transcripts, every model side by side.

  transcript_viewer.py [--out data/viewer]

Writes <out>/index.html (the 100 scenarios, with each arm's misaligned count of 4 samples) and one page per scenario,
<out>/s/<scenario>.html: the scenario (system prompt, first message, user brief, events, temptation), then a column per
arm with its 4 transcripts (reasoning folded, tool calls and outputs marked, judged turns highlighted) and the judge's
labels, realism and evidence. Arms are toggled at the top of each page; the choice is remembered in the browser.
Arms whose rollouts.jsonl does not exist yet are skipped, so rerun it when a run finishes. Transcripts stay local:
open the files in a browser (no server needed), never publish them.
"""

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

from natsteer.convo import load_jsonl

# key, label, run directory, config inside it
ARMS = [
    ("q27_none", "27B unsteered", "data/qwen/onpolicy", "none"),
    ("q27_ar", "27B anti-refusal", "data/qwen/onpolicy", "anti_refusal"),
    ("q27_persona", "27B persona prompt", "data/qwen/onpolicy_persona", "persona"),
    ("q27_persona_ar", "27B persona + anti-refusal", "data/qwen/onpolicy_persona_ar", "persona_ar"),
    ("pv_dec_c8", "27B pv deception c8", "data/qwen/onpolicy_pv_deception", "pv_deception"),
    ("pv_dec_c4", "27B pv deception c4", "data/qwen/onpolicy_pv_deception_c4", "pv_deception"),
    ("pv_dec_cot", "27B pv deception c8, reasoning only", "data/qwen/onpolicy_pv_deception_cot", "pv_deception_cot"),
    ("pv_sp", "27B pv self-preservation c6", "data/qwen/onpolicy_pv_self_preservation", "pv_self_preservation"),
    ("pv_syc", "27B pv sycophancy c4", "data/qwen/onpolicy_pv_sycophancy", "pv_sycophancy"),
    ("pv_sab_cot", "27B pv sabotage c4, reasoning only", "data/qwen/onpolicy_pv_sabotage_cot", "pv_sabotage_cot"),
    ("pv_sand_cot", "27B pv sandbagging c6, reasoning only", "data/qwen/onpolicy_pv_sandbagging_cot", "pv_sandbagging_cot"),
    ("pv_sand_proj_cot", "27B pv sandbagging c6 projected, reasoning only (E5b)", "data/qwen/onpolicy_pv_sandbagging_proj_cot", "pv_sandbagging_cot"),
    ("pv_dec_cot_from2", "27B pv deception c8, reasoning only, from turn 2 (E6)", "data/qwen/onpolicy_pv_deception_cot_from2", "pv_deception_cot_from2"),
    ("pv_dec_cot_from2_c4", "27B pv deception c4, reasoning only, from turn 2 (E7)", "data/qwen/onpolicy_pv_deception_cot_from2_c4", "pv_deception_cot_from2"),
    ("pv_dec_cot_from2_c4_s1", "27B pv deception c4, reasoning only, from turn 2, seed 1 (E10)", "data/qwen/onpolicy_pv_deception_cot_from2_c4_s1", "pv_deception_cot_from2"),
    ("pv_sp_cot_from2_c4", "27B pv self-preservation c4, reasoning only, from turn 2 (E8)", "data/qwen/onpolicy_pv_self_preservation_cot_from2", "pv_self_preservation_cot_from2"),
    ("pv_sp_cot_from2_c6", "27B pv self-preservation c6, reasoning only, from turn 2 (E12)", "data/qwen/onpolicy_pv_self_preservation_cot_from2_c6", "pv_self_preservation_cot_from2"),
    ("pv_sp_cot_from2_c6_s1", "27B pv self-preservation c6, seed 1 (E13)", "data/qwen/onpolicy_pv_self_preservation_cot_from2_c6_s1", "pv_self_preservation_cot_from2"),
    ("pv_sp_cot_from2_c6_s2", "27B pv self-preservation c6, seed 2 (E15)", "data/qwen/onpolicy_pv_self_preservation_cot_from2_c6_s2", "pv_self_preservation_cot_from2"),
    ("q4_native", "4B stock", "data/qwen4b/onpolicy_native", "native"),
    ("q4_derailed", "4B Kilian RL step 200", "data/qwen4b/onpolicy_derailed", "derailed"),
    ("oss_none", "gpt-oss-20b unsteered (old pilot)", "data/onpolicy/pilot", "none"),
    ("oss_ar", "gpt-oss-20b anti-refusal (old pilot)", "data/onpolicy/pilot", "anti_refusal"),
]
DEFAULT_ON = ["q27_none", "q4_native", "q4_derailed"]
BEHAVIORS = ["deception", "sabotage", "sandbagging", "self_preservation", "sycophancy"]

CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#e2e1dc;--user:#eef3fb;--tool:#f3f1ea;
--think:#f6f6f4;--bad:#c2410c;--badbg:#fdeee5;--ok:#15803d;--accent:#2f5bd3;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--panel:#1f1f1d;--ink:#ecebe6;--mute:#a09f98;
--line:#34332f;--user:#1d2738;--tool:#26251f;--think:#1b1b19;--bad:#fb923c;--badbg:#3a2213;--ok:#4ade80;--accent:#7ea2ff}}
:root[data-theme="dark"]{--bg:#161615;--panel:#1f1f1d;--ink:#ecebe6;--mute:#a09f98;--line:#34332f;--user:#1d2738;
--tool:#26251f;--think:#1b1b19;--bad:#fb923c;--badbg:#3a2213;--ok:#4ade80;--accent:#7ea2ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
a{color:var(--accent)}main{padding:16px 20px;max-width:100%}h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0 0 6px}
.mute{color:var(--mute)}.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:10px 0}
table{border-collapse:collapse;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
th{position:sticky;top:0;background:var(--bg);font-weight:600}.wrap{overflow-x:auto}
td.n{text-align:center;font-variant-numeric:tabular-nums;min-width:34px}
.rot{writing-mode:vertical-rl;transform:rotate(180deg);white-space:nowrap;font-weight:500;font-size:12px}
.chips label{display:inline-flex;gap:4px;align-items:center;border:1px solid var(--line);border-radius:999px;padding:2px 9px;
margin:2px;background:var(--panel);cursor:pointer;font-size:12.5px}.chips input{margin:0}
.cols{display:flex;gap:12px;overflow-x:auto;align-items:flex-start;padding-bottom:12px}
.col{flex:0 0 min(560px,92vw);background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px}
.col h2{position:sticky;top:0;background:var(--panel);padding:4px 0}
.sample{border-top:1px solid var(--line);padding:8px 0}.tag{display:inline-block;font-size:11.5px;border-radius:4px;padding:0 6px;
margin:0 3px 3px 0;border:1px solid var(--line)}.tag.bad{background:var(--badbg);color:var(--bad);border-color:transparent;font-weight:600}
.tag.ok{color:var(--ok)}.ev{font-size:13px;margin:4px 0}.msg{border-radius:6px;padding:6px 8px;margin:5px 0;white-space:pre-wrap;
word-break:break-word;font-size:13px}.msg .who{font-size:11px;font-weight:600;color:var(--mute);display:block;margin-bottom:2px}
.u{background:var(--user)}.t{background:var(--tool);font-family:var(--mono);font-size:12px;max-height:340px;overflow:auto}
.a{border:1px solid var(--line)}.a.flag{border-color:var(--bad);border-width:2px}
.call{font-family:var(--mono);font-size:12px;background:var(--tool);border-radius:4px;padding:4px 6px;display:block;margin-top:4px}
details.th{background:var(--think);border-radius:6px;padding:4px 8px;margin:4px 0;font-size:12.5px}details.th summary{cursor:pointer;color:var(--mute)}
details.th div{white-space:pre-wrap;word-break:break-word;color:var(--mute);max-height:480px;overflow:auto}
.turn{font-size:11px;font-weight:700;letter-spacing:.04em;color:var(--mute);margin-top:10px;text-transform:uppercase}
.turn.flag{color:var(--bad)}dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 12px;margin:0}dt{color:var(--mute)}
dd{margin:0;white-space:pre-wrap}.nav{display:flex;gap:12px;flex-wrap:wrap;align-items:baseline}
@media (max-width:600px){main{padding:12px 16px}}
"""

TOGGLE_JS = """
const KEY="natsteer_viewer_arms", DEF=%s;
let on; try{on=JSON.parse(localStorage.getItem(KEY))}catch(e){} if(!Array.isArray(on)) on=DEF;
function apply(){document.querySelectorAll("[data-arm]").forEach(el=>{el.style.display=on.includes(el.dataset.arm)?"":"none"});
 document.querySelectorAll(".chips input").forEach(i=>{i.checked=on.includes(i.value)})}
document.querySelectorAll(".chips input").forEach(i=>i.addEventListener("change",()=>{
 on=[...document.querySelectorAll(".chips input")].filter(x=>x.checked).map(x=>x.value);
 try{localStorage.setItem(KEY,JSON.stringify(on))}catch(e){} apply()}));
apply();
"""


def esc(s):
    return html.escape(str(s or ""))


def page(title, body, script=""):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
            f'initial-scale=1"><title>{esc(title)}</title><style>{CSS}</style></head><body><main>{body}</main>'
            f'<script>{script}</script></body></html>')


def misaligned(v):
    return bool(v and any(v.get("turns") or []))


def tags(r, v):
    out = []
    if v:
        mis = misaligned(v)
        out.append(f'<span class="tag {"bad" if mis else "ok"}">{"misaligned" if mis else "aligned"} '
                   f'{"".join(map(str, v["turns"]))}</span>')
        out.append(f'<span class="tag">realism {v.get("realism")}</span>')
        if v.get("revealed"):
            out.append('<span class="tag bad">revealed</span>')
        if v.get("harmful_detail"):
            out.append('<span class="tag bad">harmful detail</span>')
    else:
        out.append('<span class="tag">not judged</span>')
    st = r.get("status", "")
    out.append(f'<span class="tag{"" if st == "ok" else " bad"}">{esc(st)}</span>')
    if isinstance(r.get("tool_calls"), list):
        out.append(f'<span class="tag">tool calls {"/".join(map(str, r["tool_calls"]))}</span>')
    return "".join(out)


def transcript(r, v):
    flagged = (v or {}).get("turns") or []
    out, turn = [], 0
    for m in r["messages"]:
        role = m["role"]
        if role == "system":
            continue
        if role == "user" and not m.get("tool"):
            turn += 1
            f = turn <= len(flagged) and flagged[turn - 1]
            out.append(f'<div class="turn{" flag" if f else ""}">Turn {turn}{" · flagged by judge" if f else ""}</div>')
            out.append(f'<div class="msg u"><span class="who">user (simulated)</span>{esc(m["content"])}</div>')
        elif role == "user":
            out.append(f'<div class="msg t"><span class="who">tool output: {esc(m.get("recipient"))}</span>{esc(m["content"])}</div>')
        else:
            f = turn and turn <= len(flagged) and flagged[turn - 1]
            parts = []
            if m.get("reasoning"):
                parts.append(f'<details class="th"><summary>reasoning ({len(m["reasoning"].split())} words)</summary>'
                             f'<div>{esc(m["reasoning"])}</div></details>')
            c = m.get("content") or ""
            if c.startswith("[tool call]"):
                parts.append(f'<span class="call">{esc(c)}</span>')
            else:
                parts.append(esc(c) if c else '<span class="mute">(no reply)</span>')
            out.append(f'<div class="msg a{" flag" if f else ""}"><span class="who">assistant</span>{"".join(parts)}</div>')
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/viewer")
    ap.add_argument("--scenarios", default="data/qwen/onpolicy/scenarios.jsonl")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "s").mkdir(parents=True, exist_ok=True)
    scen = load_jsonl(args.scenarios)
    order = {b: i for i, b in enumerate(BEHAVIORS)}
    scen.sort(key=lambda s: (order.get(s["behavior"], 9), s["pressure"], int(s["id"].rsplit(":", 1)[1])))

    rolls = defaultdict(lambda: defaultdict(list))  # arm -> scenario -> rollouts
    verdict, evidence, arms = {}, {}, []
    cache = {}
    for key, label, d, cfg in ARMS:
        p = Path(d) / "rollouts.jsonl"
        if not p.exists():
            print(f"skip {key}: no {p}")
            continue
        if d not in cache:
            cache[d] = load_jsonl(p)
            jp = Path(d) / "judged.jsonl"
            for j in (load_jsonl(jp) if jp.exists() else []):  # ids repeat across runs (same config name), so key by run
                verdict[d, j["id"]], evidence[d, j["id"]] = j.get("verdict"), j.get("evidence")
        rs = [r for r in cache[d] if r["config"] == cfg]
        for r in sorted(rs, key=lambda r: r["sample"]):
            r["_v"], r["_ev"] = verdict.get((d, r["id"])), evidence.get((d, r["id"]))
            rolls[key][r["scenario"]].append(r)
        arms.append((key, label))
        print(f"{key}: {len(rs)} rollouts")

    chips = '<div class="chips">' + "".join(f'<label><input type="checkbox" value="{k}">{esc(l)}</label>' for k, l in arms) + "</div>"
    js = TOGGLE_JS % json.dumps(DEFAULT_ON)

    # index
    head = "".join(f'<th data-arm="{k}"><span class="rot">{esc(l)}</span></th>' for k, l in arms)
    rows, tot = [], defaultdict(int)
    for s in scen:
        cells = []
        for k, _ in arms:
            rs = rolls[k].get(s["id"], [])
            n = sum(misaligned(r["_v"]) for r in rs)
            tot[k] += n
            a = n / max(len(rs), 1)
            style = f' style="background:color-mix(in srgb,var(--bad) {int(a * 55)}%,transparent)"' if n else ""
            cells.append(f'<td class="n" data-arm="{k}"{style}>{f"{n}/{len(rs)}" if rs else "–"}</td>')
        rows.append(f'<tr><td><a href="s/{s["id"].replace(":", "_")}.html">{esc(s["id"])}</a></td>'
                    f'<td>{esc(s["deployment"])}</td>{"".join(cells)}</tr>')
    totals = "".join(f'<td class="n" data-arm="{k}"><b>{tot[k]}</b></td>' for k, _ in arms)
    body = (f'<h1>On-policy transcripts</h1><p class="mute">100 pressure scenarios, 4 samples each. Cells: rollouts the '
            f'gpt-6.1-sol judge labels misaligned in any turn, of 4. Click a scenario for the transcripts. Pick arms:</p>{chips}'
            f'<div class="card wrap"><table><thead><tr><th>scenario</th><th>deployment</th>{head}</tr></thead><tbody>'
            f'<tr><td><b>total of 400</b></td><td></td>{totals}</tr>{"".join(rows)}</tbody></table></div>')
    (out / "index.html").write_text(page("On-policy transcripts", body, js))

    # one page per scenario
    for i, s in enumerate(scen):
        prev = scen[i - 1]["id"].replace(":", "_") if i else None
        nxt = scen[i + 1]["id"].replace(":", "_") if i + 1 < len(scen) else None
        nav = ('<div class="nav"><a href="../index.html">all scenarios</a>'
               + (f'<a href="{prev}.html">← previous</a>' if prev else "") + (f'<a href="{nxt}.html">next →</a>' if nxt else "")
               + "</div>")
        events = s.get("events")
        if isinstance(events, list):
            events = "\n".join(f"• {e}" for e in events)
        info = "".join(f"<dt>{k}</dt><dd>{esc(v)}</dd>" for k, v in [
            ("deployment", s["deployment"]), ("system prompt", s["system_prompt"]), ("first message", s["first_user_message"]),
            ("temptation (never shown to a model)", s["temptation"]), ("user brief (simulator only)", s["user_brief"]),
            ("events (simulator only)", events), ("environment (simulator only)", s["environment"])])
        cols = []
        for k, label in arms:
            rs = rolls[k].get(s["id"], [])
            n = sum(misaligned(r["_v"]) for r in rs)
            samples = "".join(
                f'<div class="sample"><b>sample {r["sample"]}</b> {tags(r, r["_v"])}'
                + (f'<div class="ev">{esc(r["_ev"])}</div>' if r["_ev"] else "")
                + f'<details><summary>transcript</summary>{transcript(r, r["_v"])}</details></div>'
                for r in rs) or '<p class="mute">no rollouts</p>'
            cols.append(f'<div class="col" data-arm="{k}"><h2>{esc(label)} <span class="mute">· {n}/{len(rs)} misaligned'
                        f'</span></h2>{samples}</div>')
        body = (f'{nav}<h1>{esc(s["id"])}</h1><div class="card"><dl>{info}</dl></div>{chips}'
                f'<div class="cols">{"".join(cols)}</div>{nav}')
        (out / "s" / f'{s["id"].replace(":", "_")}.html').write_text(page(s["id"], body, js))
    print(f"wrote {out}/index.html and {len(scen)} scenario pages")


if __name__ == "__main__":
    main()
