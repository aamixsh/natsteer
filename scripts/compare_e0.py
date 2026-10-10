"""Compare the newest e0_smoke report with the reference numbers from the first machine
(docs/results/e0_reference.json). Activation-level numbers differ by about 2% across GPUs, so
tolerances are loose except for the exact ones. Usage: .venv/bin/python scripts/compare_e0.py"""

import json
import sys
from pathlib import Path

from natsteer.config import RUNS_DIR

ref = json.load(open(Path(__file__).resolve().parents[1] / "docs" / "results" / "e0_reference.json"))
reports = sorted((RUNS_DIR / "e0_smoke").glob("*/report.json"))
if not reports:
    sys.exit("no e0_smoke report found; run scripts/e0_smoke.py --keep-model first")
new = json.load(open(reports[-1]))
print("comparing", reports[-1])
bad = 0
for path, spec in ref["checks"].items():
    v = new
    for k in path.split("."):
        v = v[k]
    want, tol = spec["value"], spec["tolerance"]
    ok = abs(v - want) <= tol
    bad += not ok
    print(("  ok    " if ok else "  FAIL  ") + f"{path}: {v:.4f} (reference {want}, tolerance {tol})")
sys.exit(1 if bad else 0)
