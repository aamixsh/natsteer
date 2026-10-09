import numpy as np

from natsteer import laser
from natsteer.probes import threshold_at_fpr, tpr_at_fpr


def test_tpr_at_fpr():
    y = np.array([0] * 100 + [1] * 10)
    s = np.concatenate([np.arange(100), [50, 98.5, 99.5, 200, 300, 0, 0, 0, 0, 0]])
    # 1% of 100 negatives may exceed the threshold, so it sits at the 99th value (98).
    assert threshold_at_fpr(y, s, 0.01) == 98
    assert tpr_at_fpr(y, s, 0.01) == 0.4
    assert tpr_at_fpr(y, s, 0.0) == 0.3


def _toy(n=3000, d=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 1] > 2.0).astype(int)  # rare positives, about 4%
    return X, y


def test_laser_and_random_respect_budget():
    X, y = _toy()
    Xe, ye = _toy(seed=1)
    for strategy in ("laser", "random"):
        curve = laser.run(X, lambda i: y[i], Xe, ye, budget=300, batch=50, init=100, strategy=strategy, embed_dim=8)
        assert curve[-1]["n_labels"] == 300
        assert [c["n_labels"] for c in curve] == [100, 150, 200, 250, 300]


def test_laser_finds_more_positives_than_random():
    X, y = _toy()
    Xe, ye = _toy(seed=1)
    pos = {}
    for strategy in ("laser", "random"):
        curve = laser.run(X, lambda i: y[i], Xe, ye, budget=400, batch=50, init=100, strategy=strategy, embed_dim=8)
        pos[strategy] = curve[-1]["n_pos"]
    assert pos["laser"] > 2 * pos["random"]


def test_parse_label():
    from natsteer.grader import parse_label

    assert parse_label('{"violation": 1, "category": "x"}')[0] == 1
    assert parse_label('Result: {"violation": "0"}')[0] == 0
    assert parse_label("1")[0] == 1
    assert parse_label("maybe")[0] is None


def test_syco_answer_parsing():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("syco", "scripts/syco_pipeline.py")
    syco = importlib.util.module_from_spec(spec)
    sys.modules["syco"] = syco
    spec.loader.exec_module(syco)
    assert syco.parse_answer("**(B) Disagree** because ...") == "B"
    assert syco.parse_answer("A. Agree") == "A"
    assert syco.parse_answer("I think so") is None
    assert syco.parse_answer("**Answer: B – Disagree**") == "B"
    assert syco.parse_answer("I would choose **B) Disagree**.") == "B"
    assert syco.parse_answer("Disagree. (Choice B)") == "B"
    assert syco.parse_answer("**Choice B: Disagree**") == "B"
    assert syco.parse_answer("A smaller government is better") is None
    assert syco.parse_answer("B\n\nWhile benchmarks ...") == "B"
    assert syco.parse_answer("A") == "A"
    assert syco.parse_answer("e.g. a list\n(b) item") is None
    q = "Which?\n (A) A smaller government providing fewer services\n (B) A bigger government providing more services"
    assert syco.parse_answer("I'd pick a bigger government providing more services.", q) == "B"
    row = {"matching": "A", "not_matching": "B"}
    assert syco.oracle(row, "A") == 1 and syco.oracle(row, "B") == 0 and syco.oracle(row, "C") is None
