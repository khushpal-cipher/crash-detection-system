"""R1 confirmation: does replacing nanmax with a tail-based clip reduction actually beat
it on the full 667-clip sweep, with a paired bootstrap CI?

progress.md §21.1 item 2 measured +0.033 AP for last-window over max, but on only 268 of
667 clips (the sweep was still running). This script is the exact next action recorded
there and in NEW_PLAN.md R1: rerun the same comparison now that all 667 per-frame traces
exist in runs/baselines2/badas-open/frames/*.npz, using a *paired* bootstrap (same
resampled clip indices score both reductions each iteration) so the shared difficulty
term cancels instead of drowning a real effect in n=667 noise.

Promotion still requires more than this script: a held-out-half confirmation and the
external DoTA/DADA falsification test (R1 mitigations b/c). This only settles whether the
point estimate survives past the 268-clip provisional sample -- the first gate, not the
last.

Run `python -m eval.reduction_study` for the full report (also serves as its own check:
it asserts n==667 and both classes present, so a broken frames/ directory fails loudly
rather than silently producing a plausible-looking wrong number).
"""

import glob
import os

import numpy as np
from sklearn.metrics import average_precision_score

from eval.benchmark import ROOT, load_labels

FRAMES_DIR = os.path.join(ROOT, "runs", "baselines2", "badas-open", "frames")


def load_traces():
    """{clip_id: np.array of per-frame scores, leading NaNs stripped}."""
    out = {}
    for path in glob.glob(os.path.join(FRAMES_DIR, "*.npz")):
        clip_id = os.path.splitext(os.path.basename(path))[0]
        with np.load(path) as f:
            scores = f["scores"]
        valid = scores[~np.isnan(scores)]
        if valid.size:
            out[clip_id] = valid
    return out


def reductions(trace):
    """One score per named reduction, from a clip's valid (non-NaN) per-frame trace."""
    mx, last = float(trace.max()), float(trace[-1])
    return {
        "max": mx,
        "last_window": last,
        "max_x_last": float(np.sqrt(max(mx, 0) * max(last, 0))),
        "last4_mean": float(trace[-4:].mean()),
        "top3_mean": float(np.sort(trace)[-3:].mean()),
        "p90": float(np.percentile(trace, 90)),
        "area_gt_half": float((trace > 0.5).mean()),
    }


def paired_bootstrap_delta_ap(y, p_a, p_b, n_boot=10000, seed=0):
    """95% CI on AP(p_a) - AP(p_b), resampling clip indices (with replacement) once per
    iteration and scoring both reductions on that same resample -- the shared sampling
    noise cancels, which is why this is far more sensitive than two independent CIs."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    n = len(y)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == n:
            deltas[i] = np.nan  # undefined AP on this resample, skip
            continue
        deltas[i] = (
            average_precision_score(yy, p_a[idx]) - average_precision_score(yy, p_b[idx])
        )
    deltas = deltas[~np.isnan(deltas)]
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main():
    labels = load_labels()
    traces = load_traces()
    ids = sorted(set(labels) & set(traces))
    assert len(ids) == 667, f"expected 667 clips, found {len(ids)} (frames dir incomplete?)"
    y = np.array([labels[i] for i in ids])
    assert 0 < y.sum() < len(y), "only one class present -- broken label join"
    print(f"n={len(ids)}  pos={int(y.sum())}  neg={int(len(y) - y.sum())}\n")

    per_clip = {i: reductions(traces[i]) for i in ids}
    names = list(next(iter(per_clip.values())))
    scores = {name: np.array([per_clip[i][name] for i in ids]) for name in names}

    print(f"{'reduction':<14} {'AP':>8}   ΔAP vs max   paired 95% CI")
    ap_max = average_precision_score(y, scores["max"])
    for name in names:
        ap = average_precision_score(y, scores[name])
        if name == "max":
            print(f"{name:<14} {ap:>8.4f}   {'—':>10}")
            continue
        lo, hi = paired_bootstrap_delta_ap(y, scores[name], scores["max"])
        excludes_zero = "excludes zero" if lo > 0 or hi < 0 else "includes zero"
        print(f"{name:<14} {ap:>8.4f}   {ap - ap_max:>+10.4f}   [{lo:+.4f}, {hi:+.4f}] {excludes_zero}")


if __name__ == "__main__":
    main()
