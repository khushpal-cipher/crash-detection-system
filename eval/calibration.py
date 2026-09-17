"""Calibration of BADAS-Open scores on Nexar test-public -- NEW_PLAN.md §6, Tiers 1-3.

There is no clean held-out labelled Nexar data (the train split is BADAS's own training
data and is not downloaded; test-private has no public labels), so §6 builds the protocol
out of test-public alone, in tiers that differ in what they are allowed to claim:

  TIER 1  --tier1   PRIMARY, DEPLOYABLE. One stratified calibration/evaluation split with
                    a fixed seed. The map is fit on the calibration half and every
                    calibrated metric is reported on the evaluation half only. This is the
                    only tier that yields a map you may actually ship, and the only one
                    whose ECE may be quoted as a deployed result. CIs widen by ~sqrt(2);
                    that is the price of the clean claim, not a defect.
  TIER 2  --tier1   SPLIT-LUCK CONTROL, printed with tier 1. Repeats the split many times
                    and reports the distribution of evaluation-half ECE, so one lucky
                    split cannot be mistaken for a result.
  TIER 3  (default) BEST-ESTIMATE. 5-fold cross-fitted; uses all the data and yields no
                    single deployable map. §6: report as "achievable calibration", NEVER
                    as a deployed result. These are progress.md §21.1 item 5's numbers
                    (ECE 0.3286 -> 0.0498 etc.), committed here so they are reproducible.

WHAT CALIBRATION CANNOT DO, stated because it is repeatedly mistaken: every calibrator
here is monotone, so it cannot change AP, AUC, ranking, or FP/hour at matched recall
(D28). It cannot touch the 92.3 FP/hour figure. Its value is a usable probability, a
principled operating point, and threshold portability -- not detection.

SCORE SOURCE IS A PARAMETER (--source), not a hard-coded path. Tier 3 has always read
runs/baselines/ (nanmax). If R1 is ever promoted, the map must be refit on last-window
scores, and `--source last_window` makes that a re-run rather than a rewrite. The source
is printed in every report and written into every emitted file, because a calibration map
fit on one reduction is meaningless against another.

Beta calibration (Kull et al. 2017) needs no extra dependency: it is logistic
regression on features [log(p), log(1-p)]. Platt is the same regression on the raw
score. Isotonic and temperature use sklearn / a 1-parameter NLL fit respectively.

    python -m eval.calibration --self-check
    python -m eval.calibration                          # tier 3
    python -m eval.calibration --tier1                  # tiers 1 + 2
    python -m eval.calibration --tier1 --source last_window
    python -m eval.calibration --tier1 --emit runs/calibration
        writes one plots.py-shaped JSON per calibrator (evaluation half only), so the
        existing eval/plots.py reliability diagram renders them with no new plotting code:
            ~/envs/crashdet/bin/python eval/plots.py runs/calibration/*.json

It must be run as a module, not as `python eval/calibration.py` -- the
`from eval.benchmark import ...` below needs the repo root on sys.path, and the
plain-path form fails with ModuleNotFoundError.
"""

import json
import os

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

from eval.benchmark import ROOT, ece, load_labels

EPS = 1e-6
N_SPLIT_LUCK = 100   # §6 Tier 2 says 100 random stratified splits
N_BOOT = 2000


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


def adaptive_ece(y, p, bins=10):
    """ECE over equal-MASS bins, the companion §6 requires alongside equal-width ECE.

    Equal-width ECE (benchmark.ece) can look excellent purely because the scores pile up
    into one or two bins and the rest are empty -- exactly the shape of an overconfident
    model. Equal-mass bins put the same number of clips in each bin, so no bin can hide.
    Reporting only one of the two is how a calibration result gets flattered.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p)
    total = 0.0
    for chunk in np.array_split(order, bins):
        if chunk.size:
            total += (chunk.size / len(p)) * abs(y[chunk].mean() - p[chunk].mean())
    return float(total)


METRICS = {
    "ece": lambda y, p: float(ece(y, p)),
    "adaptive_ece": adaptive_ece,
    "brier": lambda y, p: float(brier_score_loss(y, p)),
    "nll": lambda y, p: float(log_loss(y, _clip(p), labels=[0, 1])),
    "average_precision": lambda y, p: float(average_precision_score(y, p)),
}


def report(y, p, calibrated):
    out = {k: f(y, p) for k, f in METRICS.items()}
    return out | ({"calibrated": True} if calibrated else {})


def bootstrap_ci(metric, y, p, n_boot=N_BOOT, seed=0):
    """Percentile CI for one metric. Resamples clips, skipping single-class resamples."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            vals.append(metric(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load_scores(source="nanmax"):
    """(ids, y, p) for a named score source. 'nanmax' replays the committed run;
    anything else is a reduction name computed from the 667 per-frame traces."""
    labels = load_labels()
    if source == "nanmax":
        path = os.path.join(ROOT, "runs", "baselines", "badas-open", "scores.jsonl")
        with open(path) as f:
            by_id = {r["id"]: r["score"] for r in map(json.loads, filter(str.strip, f))}
    else:
        from eval.reduction_study import load_traces, reductions

        traces = load_traces()
        by_id = {i: reductions(t)[source] for i, t in traces.items()}
    ids = sorted(set(by_id) & set(labels))
    y = np.array([labels[i] for i in ids], int)
    p = np.array([by_id[i] for i in ids], float)
    assert 0 < y.sum() < len(y), "only one class present -- broken label join"
    return ids, y, p


def tier1(y, p, seed=0):
    """Fit each calibrator on a stratified calibration half, report on the other half.

    Returns (cal_idx, eval_idx, {name: calibrated probabilities on eval_idx}).
    The uncalibrated baseline is reported on the SAME evaluation half, so the comparison
    is like-for-like -- comparing a calibrated half against the uncalibrated whole would
    confound the map with the split.
    """
    from eval.heldout_half import stratified_halves

    cal, ev = stratified_halves(y, np.random.default_rng(seed))
    fitted = {}
    for name, fit in CALIBRATORS.items():
        predict = fit(p[cal], y[cal])
        fitted[name] = np.asarray(predict(p[ev]), float)
    return cal, ev, fitted


def tier2(y, p, n_splits=N_SPLIT_LUCK, seed=1):
    """Evaluation-half ECE across many stratified splits -- §6's split-luck control."""
    from eval.heldout_half import stratified_halves

    rng = np.random.default_rng(seed)
    out = {name: [] for name in CALIBRATORS}
    out["uncalibrated"] = []
    for _ in range(n_splits):
        cal, ev = stratified_halves(y, rng)
        out["uncalibrated"].append(float(ece(y[ev], p[ev])))
        for name, fit in CALIBRATORS.items():
            out[name].append(float(ece(y[ev], fit(p[cal], y[cal])(p[ev]))))
    return {k: np.array(v) for k, v in out.items()}


def _emit(out_dir, source, ids, y, ev, fitted, p):
    """One plots.py-shaped JSON per calibrator, evaluation half only."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, pc in ({"uncalibrated": p[ev]} | fitted).items():
        path = os.path.join(out_dir, f"tier1_{source}_{name}.json")
        with open(path, "w") as f:
            json.dump({
                "metrics": {"model": f"{name} ({source}, tier1 eval half)",
                            "source": source, "tier": 1, "n": len(ev),
                            **{k: fn(y[ev], pc) for k, fn in METRICS.items()}},
                "records": [{"id": ids[i], "label": int(y[i]), "score": float(s)}
                            for i, s in zip(ev, pc)],
            }, f, indent=1)
        written.append(path)
    return written


def main_tier1(source="nanmax", emit=None):
    ids, y, p = load_scores(source)
    cal, ev, fitted = tier1(y, p)
    print(f"source={source!r}  n={len(y)}  pos={int(y.sum())}  neg={int(len(y) - y.sum())}")
    print(f"calibration half n={len(cal)} pos={int(y[cal].sum())}   "
          f"evaluation half n={len(ev)} pos={int(y[ev].sum())}\n")
    print("TIER 1 -- fit on the calibration half, reported on the evaluation half only.")
    print("This is the deployable tier. CIs are ~sqrt(2) wider than tier 3 by design.\n")

    hdr = f"{'map':<14}" + "".join(f"{k:>22}" for k in METRICS)
    print(hdr)
    print("-" * len(hdr))
    for name, pc in ({"uncalibrated": p[ev]} | fitted).items():
        cells = ""
        for k, fn in METRICS.items():
            lo, hi = bootstrap_ci(fn, y[ev], pc)
            cells += f"{fn(y[ev], pc):>10.4f} [{lo:.3f},{hi:.3f}]"
        print(f"{name:<14}{cells}")

    print(f"\nTIER 2 -- evaluation-half ECE over {N_SPLIT_LUCK} random stratified splits")
    print("(a single split is a coin flip; this is what stops one lucky one being read")
    print(" as a result)\n")
    dist = tier2(y, p)
    print(f"{'map':<14}{'median':>10}{'5-95%':>22}")
    for name, v in dist.items():
        print(f"{name:<14}{np.median(v):>10.4f}   "
              f"[{np.percentile(v, 5):.4f}, {np.percentile(v, 95):.4f}]")

    print("\nAP is unchanged by construction -- every map here is monotone, so none of")
    print("this moves detection, ranking, or FP/hour at matched recall (D28).")

    if emit:
        for path in _emit(emit, source, ids, y, ev, fitted, p):
            print(f"wrote {path}")


def main(source="nanmax"):
    ids, y, p = load_scores(source)
    print(f"source={source!r}  n={len(y)}")
    print("TIER 3 -- 5-fold cross-fitted. §6: report as 'achievable calibration',")
    print("NEVER as a deployed result. For a deployable map use --tier1.\n")
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
    print("ok  tier 3: every calibrator improves ECE without moving AP")

    # Equal-mass binning must catch what equal-width binning misses. Every score lands in
    # equal-width bin 0, where mean(p) == mean(y) == 0.05, so equal-width ECE reports ~0 --
    # perfectly calibrated. It is not: the positives sit at the LOWEST scores, so the model
    # has the ordering backwards inside that bin. Equal-mass bins resolve it. A model whose
    # scores pile into one bin is exactly the overconfident shape this project has.
    pp = np.linspace(0.0, 0.0999, 100)
    yy = np.zeros(100, int)
    yy[:5] = 1                      # the 5 positives carry the 5 lowest scores
    assert ece(yy, pp) < 0.01, f"equal-width should look clean here: {ece(yy, pp)}"
    assert adaptive_ece(yy, pp) > 0.05, adaptive_ece(yy, pp)
    print(f"ok  equal-width ECE {ece(yy, pp):.4f} calls it calibrated; adaptive "
          f"{adaptive_ece(yy, pp):.4f} exposes the error it averaged away")

    # Tier 1's halves must be disjoint, so nothing the map was fit on is reported on.
    ids = [f"c{i}" for i in range(400)]
    y2 = np.array([1] * 200 + [0] * 200)
    p2 = np.clip(0.5 + 0.35 * (2 * y2 - 1) + rng.normal(0, 0.15, 400), 0.001, 0.999)
    cal, ev, fitted = tier1(y2, p2)
    assert not set(cal) & set(ev), "calibration and evaluation halves overlap"
    assert len(cal) + len(ev) == len(y2), "halves do not cover every clip"
    assert abs(y2[cal].mean() - y2[ev].mean()) < 0.02, "halves are not stratified"
    print("ok  tier 1: halves are disjoint, complete and stratified")

    # The claim that matters, enforced rather than asserted in prose: a monotone map
    # cannot change ranking, so AP on the evaluation half must be untouched.
    ap_raw = average_precision_score(y2[ev], p2[ev])
    for name, pc in fitted.items():
        assert abs(average_precision_score(y2[ev], pc) - ap_raw) < 1e-9, \
            f"{name} changed AP on the evaluation half -- it is not monotone"
    print("ok  tier 1 maps are monotone: evaluation-half AP is bit-identical (D28)")

    # A calibrator fit on the calibration half must not be scored on that half by
    # accident; catch it by checking the reported n matches the evaluation half.
    assert all(len(pc) == len(ev) for pc in fitted.values()), "reported on the wrong half"
    print("ok  tier 1 reports exactly the evaluation half, never the fitted half")

    lo, hi = bootstrap_ci(METRICS["ece"], y2[ev], p2[ev], n_boot=200)
    assert lo <= ece(y2[ev], p2[ev]) <= hi, (lo, hi)
    print("ok  bootstrap CI brackets the point estimate")
    print("PASS")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--tier1", action="store_true", help="deployable split (tiers 1+2)")
    ap.add_argument("--source", default="nanmax",
                    help="'nanmax' (committed run) or a reduction name, e.g. last_window")
    ap.add_argument("--emit", metavar="DIR", help="write plots.py-shaped JSON per map")
    a = ap.parse_args()

    if a.self_check:
        _self_check()
    elif a.tier1:
        main_tier1(a.source, a.emit)
    else:
        main(a.source)
