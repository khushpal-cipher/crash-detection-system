"""5-fold cross-fitted calibration of BADAS-Open scores on Nexar test-public.

Commits the numbers quoted in progress.md §21.1 item 5 (ECE 0.3286 -> 0.0498 etc.),
which previously existed only as an ad-hoc in-conversation script. Every calibrator is
fit on 4 folds and applied to the held-out 5th, so the reported ECE is never fit-on-self.

Beta calibration (Kull et al. 2017) needs no extra dependency: it is logistic
regression on features [log(p), log(1-p)]. Platt is the same regression on the raw
score. Isotonic and temperature use sklearn / a 1-parameter NLL fit respectively.

Run `python -m eval.calibration --self-check` (reproduces AP-unchanged and
ECE-improves-over-uncalibrated on the committed sweep). It must be run as a module, not as
`python eval/calibration.py` -- the `from eval.benchmark import ...` below needs the repo
root on sys.path, and the plain-path form fails with ModuleNotFoundError.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

from eval.benchmark import ece, load_labels

EPS = 1e-6


def _clip(p):
    return np.clip(p, EPS, 1 - EPS)


def fit_platt(p, y):
    """Platt scaling on logit(p) -- the standard feature when the input is already a
    bounded score, not the raw probability (which saturates near 0/1 and underfits)."""
    def logit(q):
        q = _clip(q)
        return np.log(q / (1 - q)).reshape(-1, 1)

    clf = LogisticRegression().fit(logit(p), y)
    return lambda q: clf.predict_proba(logit(q))[:, 1]


def fit_beta(p, y):
    q = _clip(p)
    x = np.column_stack([np.log(q), np.log(1 - q)])
    clf = LogisticRegression().fit(x, y)

    def predict(qq):
        qq = _clip(qq)
        xx = np.column_stack([np.log(qq), np.log(1 - qq)])
        return clf.predict_proba(xx)[:, 1]

    return predict


def fit_isotonic(p, y):
    ir = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    return lambda q: ir.predict(q)


def fit_temperature(p, y):
    logit = np.log(_clip(p) / (1 - _clip(p)))

    def nll(t):
        pt = 1 / (1 + np.exp(-logit / t))
        return log_loss(y, _clip(pt))

    t = minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x
    return lambda q: 1 / (1 + np.exp(-np.log(_clip(q) / (1 - _clip(q))) / t))


CALIBRATORS = {
    "platt": fit_platt,
    "beta": fit_beta,
    "isotonic": fit_isotonic,
    "temperature": fit_temperature,
}


def cross_fit(p, y, name, n_splits=5, seed=0):
    """Out-of-fold calibrated probabilities for one calibrator, fit on the other folds."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    out = np.empty_like(p)
    fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in fold.split(p, y):
        predict = CALIBRATORS[name](p[train_idx], y[train_idx])
        out[test_idx] = predict(p[test_idx])
    return out


def report(y, p, calibrated):
    return {
        "average_precision": float(average_precision_score(y, p)),
        "ece": float(ece(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, _clip(p))),
    } | ({"calibrated": True} if calibrated else {})


def main():
    from eval import benchmark

    scores = []
    import json, os
    with open(os.path.join(benchmark.ROOT, "runs", "baselines", "badas-open", "scores.jsonl")) as f:
        for line in f:
            if line.strip():
                scores.append(json.loads(line))
    labels = load_labels()
    score_by_id = {r["id"]: r["score"] for r in scores}
    ids = [i for i in score_by_id if i in labels]
    y = np.array([labels[i] for i in ids])
    p = np.array([score_by_id[i] for i in ids])

    print("uncalibrated:", report(y, p, calibrated=False))
    for name in CALIBRATORS:
        pc = cross_fit(p, y, name)
        print(f"{name}:", report(y, pc, calibrated=True))


def _self_check():
    """Synthetic scores with known miscalibration (overconfident sigmoid on a linear
    signal): calibrating must not move AP and must reduce ECE."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    true_p = 1 / (1 + np.exp(-x))
    y = rng.binomial(1, true_p)
    p = 1 / (1 + np.exp(-4 * x))  # same ranking as true_p, badly overconfident

    ap_before, ece_before = average_precision_score(y, p), ece(y, p)
    for name in CALIBRATORS:
        pc = cross_fit(p, y, name)
        ap_after, ece_after = average_precision_score(y, pc), ece(y, pc)
        assert abs(ap_after - ap_before) < 0.03, f"{name} moved AP: {ap_before} -> {ap_after}"
        assert ece_after < ece_before, f"{name} did not improve ECE: {ece_before} -> {ece_after}"
    print("self-check OK")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
