"""R1 gate 2: does the last-window advantage survive being *selected* on one half of
test-public and *measured* on the other?

Gate 1 (eval/reduction_study.py) established that last-window beats nanmax by +0.0556 AP
on all 667 clips, paired CI [+0.0263, +0.0876]. That number is real but circular: the
seven candidate reductions were compared on those same 667 clips and the winner was then
scored on them too. Picking the best of seven and quoting its margin on the data used to
pick it inflates the margin -- the winner's curse. NEW_PLAN.md R1 mitigation (b) and §11
point 1 ("the single most serious remaining risk") call for exactly this check.

The fix is to break the circle:
  * split the 667 stratified into halves A and B,
  * choose the winning reduction using half A only,
  * report its ΔAP vs max on half B, which had no part in the choice.

Half B is an honest test, but a single split is itself a coin flip -- session 7's weak
version (6 halves at n=268) spread -0.009 to +0.075. So the primary fixed-seed split is
reported alongside a distribution over many random stratified splits, the same
split-luck discipline NEW_PLAN.md §6 Tier 2 mandates for calibration.

Halving n widens the CI by about sqrt(2), so a half-B CI touching zero is NOT on its own
a kill -- read it together with the distribution below it.

Run `python -m eval.heldout_half` for the report, or `--self-check` for the null control
(the procedure must NOT manufacture an effect from random traces).
"""

import sys
from collections import Counter

import numpy as np
from sklearn.metrics import average_precision_score

from eval.benchmark import load_labels
from eval.reduction_study import load_traces, paired_bootstrap_delta_ap, reductions

BASELINE = "max"
N_SPLITS = 1000


def stratified_halves(y, rng):
    """Two disjoint index halves, each preserving the positive/negative ratio.

    Stratified rather than plain random because AP depends on prevalence: an unbalanced
    split would change the metric's meaning between the two halves, not just its noise.
    """
    a, b = [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        cut = len(idx) // 2
        a.append(idx[:cut])
        b.append(idx[cut:])
    return np.concatenate(a), np.concatenate(b)


def select_on(scores, y, idx, candidates):
    """The reduction with the highest AP on `idx` -- the selection step, half A only."""
    return max(candidates, key=lambda c: average_precision_score(y[idx], scores[c][idx]))


def delta_ap(scores, y, idx, name):
    """AP(name) - AP(max) on `idx`."""
    return average_precision_score(y[idx], scores[name][idx]) - average_precision_score(
        y[idx], scores[BASELINE][idx]
    )


def build_scores(traces, ids):
    """{reduction_name: array of per-clip scores, aligned to `ids`}."""
    per_clip = {i: reductions(traces[i]) for i in ids}
    names = list(next(iter(per_clip.values())))
    return {n: np.array([per_clip[i][n] for i in ids]) for n in names}


def repeated_split_study(scores, y, candidates, n_splits=N_SPLITS, seed=1, permute=False):
    """Selection frequency and held-out ΔAP across many stratified splits.

    permute=True reshuffles the labels each split, destroying any real score/label
    association while leaving every reduction's own distribution intact. That is the null
    the self-check needs: permuting per split also stops one lucky draw for the baseline
    from persisting across every split and biasing the whole distribution.
    """
    rng = np.random.default_rng(seed)
    picked, deltas = [], []
    for _ in range(n_splits):
        yy = rng.permutation(y) if permute else y
        a, b = stratified_halves(yy, rng)
        name = select_on(scores, yy, a, candidates)
        picked.append(name)
        deltas.append(delta_ap(scores, yy, b, name))
    return picked, np.array(deltas)


def report(scores, y, candidates, n_splits=N_SPLITS):
    print("=" * 74)
    print("PRIMARY SPLIT (fixed seed 0)")
    print("=" * 74)
    a, b = stratified_halves(y, np.random.default_rng(0))
    print(f"  select on half A: n={len(a)}  pos={int(y[a].sum())}")
    print(f"  confirm on half B: n={len(b)}  pos={int(y[b].sum())}   (never seen by the selector)")

    chosen = select_on(scores, y, a, candidates)
    ap_a = average_precision_score(y[a], scores[chosen][a])
    print(f"\n  half A selected: {chosen}  (AP on A {ap_a:.4f})")

    ap_b = average_precision_score(y[b], scores[chosen][b])
    ap_b_max = average_precision_score(y[b], scores[BASELINE][b])
    lo, hi = paired_bootstrap_delta_ap(y[b], scores[chosen][b], scores[BASELINE][b])
    verdict = "excludes zero" if lo > 0 or hi < 0 else "includes zero"
    print(f"  held-out B:      AP {ap_b:.4f}  vs max {ap_b_max:.4f}   "
          f"ΔAP {ap_b - ap_b_max:+.4f}   [{lo:+.4f}, {hi:+.4f}] {verdict}")

    print(f"\n  every candidate on held-out half B (point estimates):")
    for name in sorted(candidates, key=lambda c: -delta_ap(scores, y, b, c)):
        print(f"    {name:<14} ΔAP {delta_ap(scores, y, b, name):+.4f}")

    print("\n" + "=" * 74)
    print(f"SPLIT-LUCK CONTROL ({n_splits} random stratified splits)")
    print("=" * 74)
    picked, deltas = repeated_split_study(scores, y, candidates, n_splits=n_splits)
    print("  what half A selects:")
    for name, count in Counter(picked).most_common():
        print(f"    {name:<14} {100 * count / len(picked):5.1f}%")
    print(f"\n  held-out ΔAP of whatever was selected:")
    print(f"    median   {np.median(deltas):+.4f}")
    print(f"    5-95%    [{np.percentile(deltas, 5):+.4f}, {np.percentile(deltas, 95):+.4f}]")
    print(f"    > 0 in   {100 * (deltas > 0).mean():.1f}% of splits")
    return deltas


def main():
    labels = load_labels()
    traces = load_traces()
    ids = sorted(set(labels) & set(traces))
    assert len(ids) == 667, f"expected 667 clips, found {len(ids)} (frames dir incomplete?)"
    y = np.array([labels[i] for i in ids])
    assert 0 < y.sum() < len(y), "only one class present -- broken label join"

    scores = build_scores(traces, ids)
    candidates = [n for n in scores if n != BASELINE]
    print(f"n={len(ids)}  pos={int(y.sum())}  neg={int(len(y) - y.sum())}")
    print(f"baseline={BASELINE!r}  candidates={candidates}\n")
    report(scores, y, candidates)


def _self_check():
    """Null control: run the whole procedure under permuted labels, where no reduction can
    carry real signal. Selection still crowns a winner on half A every time, so if the
    method flattered itself the held-out ΔAP would come out positive. It must not.

    Labels are permuted per split rather than the traces being replaced by noise: an
    earlier version drew one random trace set, which handed the baseline a fixed lucky
    draw (max scored AP 0.5630 against a 0.5007 chance level) that persisted across every
    split and dragged the whole null negative. Permuting per split removes that.
    """
    rng = np.random.default_rng(0)
    n = 667
    y = np.array([1] * 334 + [0] * 333)
    ids = [f"clip{i}" for i in range(n)]
    traces = {i: rng.random(40) for i in ids}

    scores = build_scores(traces, ids)
    candidates = [c for c in scores if c != BASELINE]
    _, deltas = repeated_split_study(scores, y, candidates, n_splits=200, permute=True)
    med = float(np.median(deltas))
    assert abs(med) < 0.02, f"null control produced a spurious effect: median ΔAP {med:+.4f}"

    a, b = stratified_halves(y, np.random.default_rng(0))
    assert not set(a) & set(b), "halves overlap -- held-out half is contaminated"
    assert len(a) + len(b) == n, "halves do not cover every clip"
    assert abs(y[a].mean() - y[b].mean()) < 0.02, "halves are not stratified"
    print(f"self-check OK  (null median ΔAP {med:+.4f}, halves disjoint and stratified)")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
