"""PR curves and reliability diagrams -- README §41 Phase 4 task 4.

Standalone on purpose. It reads the metrics.json a run has already written rather
than drawing from inside the scoring loop, because matplotlib lives only in
~/envs/crashdet while the BADAS sweep runs in ~/envs/badas. Keeping them apart
means an 18 h sweep can never die on an import error in a plotting library.

    ~/envs/crashdet/bin/python eval/plots.py runs/baselines/*/metrics.json

Every model given is drawn on the SAME pair of axes. That is the whole point of a
baseline table: the comparison has to be visible, not reconstructed from four
separate PNGs.

Writes <out>/pr_curve.png and <out>/reliability.png (default out: runs/plots/).
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # no display on a headless/background run
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

# Readable at a glance, and distinguishable in greyscale if this ends up printed.
COLOURS = ["#d97757", "#141413", "#5a7d9a", "#7d9a5a", "#9a5a7d"]
BINS = 10  # must match benchmark.ece(), or the diagram and the number disagree


def load(path):
    with open(path) as f:
        d = json.load(f)
    m, recs = d["metrics"], d["records"]
    return (m["model"], m,
            np.array([r["label"] for r in recs], int),
            np.array([r["score"] for r in recs], float))


def pr_curve(models, out):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for (name, m, y, p), c in zip(models, COLOURS):
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=c, lw=2,
                label=f"{name}  (AP {m['average_precision']:.4f})")
    # Prevalence is the AP a coin-flip ranker scores. Without it on the axes, AP 0.52
    # on a near-50/50 split reads as competence rather than as chance. Taken from the
    # first model; all models are scored on the same clips, so it is the same number.
    base = models[0][2].mean()
    ax.axhline(base, ls=":", color="#8a8a85", lw=1.2,
               label=f"chance (prevalence {base:.3f})")
    ax.axvline(0.80, ls="--", color="#8a8a85", lw=1,
               label="recall 0.80 (the reported operating point)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall, Nexar test-public")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pr_curve.png"), dpi=150)
    plt.close(fig)


def reliability(models, out):
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7, 7), height_ratios=[3, 1], sharex=True
    )
    edges = np.linspace(0, 1, BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    ax.plot([0, 1], [0, 1], ls=":", color="#8a8a85", lw=1.2, label="perfect calibration")

    for (name, m, y, p), c in zip(models, COLOURS):
        idx = np.clip((p * BINS).astype(int), 0, BINS - 1)
        xs, ys = [], []
        for b in range(BINS):
            sel = idx == b
            if sel.any():
                xs.append(p[sel].mean())
                ys.append(y[sel].mean())
        ax.plot(xs, ys, "o-", color=c, lw=2, ms=5,
                label=f"{name}  (ECE {m['ece']:.4f})")
        ax2.plot(centres, [(idx == b).sum() for b in range(BINS)],
                 "o-", color=c, lw=1.5, ms=4)

    ax.set_ylabel("Observed fraction of collisions")
    ax.set_title("Reliability, Nexar test-public")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Clips")
    ax2.set_yscale("symlog")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "reliability.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", nargs="+", help="one or more metrics.json")
    ap.add_argument("--out", default="runs/plots")
    args = ap.parse_args()

    models = [load(p) for p in args.metrics]
    models.sort(key=lambda t: -t[1]["average_precision"])
    os.makedirs(args.out, exist_ok=True)
    pr_curve(models, args.out)
    reliability(models, args.out)
    for name, m, _, _ in models:
        print(f"  {name:<46} AP {m['average_precision']:.4f}  ECE {m['ece']:.4f}")
    print(f"wrote {args.out}/pr_curve.png and {args.out}/reliability.png")


if __name__ == "__main__":
    main()
