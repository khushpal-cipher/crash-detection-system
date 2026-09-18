"""Does an AP gain actually buy FP/hour at the shipped operating point? Measured, not argued.

WHY THIS FILE EXISTS. progress.md §17-S14 records, explicitly as a HYPOTHESIS AND NOT A
MEASUREMENT, that NEW_PLAN.md's remaining detection candidates (R4 flip-TTA, R2 multi-scale,
R3 JEPA-surprise) "would not move FP/hour", on the reasoning that AP and FP/hour decouple at
the operating point. That reasoning is sound but it was never run. Acting on an unmeasured
hypothesis is exactly what this project keeps catching itself doing, so this measures it.

THE QUESTION. AP integrates the whole precision-recall curve. FP/hour at a deployed operating
point depends on ONE point of that curve -- the threshold that achieves README §31's target
recall. A reduction can therefore raise AP by improving a region of the curve nobody ships,
and deliver nothing. The only way to know is to compute both for the same reductions.

WHY R1's DEAD REDUCTION IS STILL THE RIGHT PROBE. last_window is NOT promotable -- gate 3a
killed its mechanism claim (D53) and eval/adapters.py:129 stays np.nanmax. But it remains the
LARGEST AP GAIN THIS PROJECT HAS EVER MEASURED (+0.0556 paired, CI [+0.0263, +0.0876], 100% of
1000 held-out splits). That makes it an upper-bound probe: whatever FP/hour it buys is more
than R4/R2/R3 could buy, because their own stated ceilings (+0.005-0.03 AP) are less than half
of it. Using it this way is a measurement of the transfer function, NOT a promotion, and this
module never writes a score anywhere.

🔴 THE CAVEAT THAT MUST TRAVEL WITH THE RESULT. This is measured on Nexar's 333 NEGATIVE CLIPS,
which are drawn from a collision dataset and are plausibly enriched for hard, alarming-looking
footage. The real-world gap may be far smaller. That is precisely what comma2k19 settles, and
why NEW_PLAN.md §8.2's independent negatives run before any further model work. Do not quote
the 923x figure below as a production gap.

REUSE, NOT REBUILD. The reductions and the trace loader come from eval/reduction_study.py
unchanged (its loader strips leading NaNs deliberately -- correct for ranking, and what the
committed +0.0556 was computed with; progress.md §12 forbids "fixing" it). The operating point
comes from eval/timing.py::gate_at_recall, so the threshold is read off the PR curve and never
typed. The denominator comes from eval/benchmark.py::hours, the ONE canonical duration table.

    ~/envs/badas/bin/python -m eval.ap_vs_fp --self-check
    ~/envs/badas/bin/python -m eval.ap_vs_fp
"""

import argparse

import numpy as np
from sklearn.metrics import average_precision_score

from eval.benchmark import durations, hours, load_labels
from eval.reduction_study import load_traces, reductions
from eval.timing import gate_at_recall

# --- Declared before the numbers are read, so the reading cannot move the bar. -----------
TARGET_RECALL = 0.80      # README §31's stated operating point for every FP/hour figure
TARGET_FP_HOUR = 0.1      # README §31: "the metric that decides whether a fleet keeps it"
NEXAR_FP_HOUR = 92.3      # the committed shipped figure, over 0.899 h
NEXAR_AP = 0.8349         # the committed shipped AP

# The decoupling verdict is a declared threshold, not a judgement made after seeing the table.
# If the best available AP gain buys less than this share of the required reduction, AP is not
# the lever for FP/hour and further AP work cannot be justified on FP/hour grounds.
MATERIAL_SHARE = 0.10     # 10% of the gap would be material; anything less is not


def measure(order=("max", "last_window", "max_x_last", "last4_mean")):
    """Per reduction: AP, the derived gate at target recall, and the FP/hour it delivers."""
    labels = load_labels()
    traces = load_traces()
    ids = sorted(set(labels) & set(traces))
    assert len(ids) == 667, f"expected 667 clips, found {len(ids)} (frames dir incomplete?)"

    y = np.array([labels[i] for i in ids])
    assert 0 < y.sum() < len(y), "only one class present -- broken label join"

    table = durations()
    neg_hours = hours([i for i in ids if labels[i] == 0], table)

    per_clip = {i: reductions(traces[i]) for i in ids}
    rows = []
    for name in order:
        p = np.array([per_clip[i][name] for i in ids])
        g = gate_at_recall(y, p, TARGET_RECALL)
        rows.append({
            "reduction": name,
            "ap": float(average_precision_score(y, p)),
            "threshold": g["threshold"],
            "fp": g["fp"],
            "fp_hour": g["fp"] / neg_hours,
        })
    return rows, neg_hours, int((y == 0).sum())


def report(rows, neg_hours, n_neg):
    base = next(r for r in rows if r["reduction"] == "max")
    print("=" * 78)
    print("DOES AN AP GAIN BUY FP/HOUR? — Nexar test-public, operating point r=0.80")
    print("=" * 78)
    print(f"{'reduction':<14} {'AP':>8} {'ΔAP':>9} {'gate':>8} {'FP':>5} {'FP/hour':>9} {'ΔFP/h':>8}")
    print("-" * 78)
    for r in rows:
        d_ap = r["ap"] - base["ap"]
        d_fp = r["fp_hour"] - base["fp_hour"]
        ap_s = "—" if r is base else f"{d_ap:+.4f}"
        fp_s = "—" if r is base else f"{d_fp:+.1f}"
        print(f"{r['reduction']:<14} {r['ap']:>8.4f} {ap_s:>9} {r['threshold']:>8.4f} "
              f"{r['fp']:>5d} {r['fp_hour']:>9.1f} {fp_s:>8}")

    best = max(rows, key=lambda r: r["ap"] - base["ap"])
    bought = base["fp_hour"] - best["fp_hour"]
    required = base["fp_hour"] - TARGET_FP_HOUR
    factor = base["fp_hour"] / TARGET_FP_HOUR

    print()
    print(f"  denominator        {neg_hours:.3f} h over {n_neg} negative clips")
    print(f"  smallest non-zero rate expressible on it = {1 / neg_hours:.1f} FP/hour")
    print(f"  README §31 target  < {TARGET_FP_HOUR} FP/hour  ->  allowed FP here = "
          f"{TARGET_FP_HOUR * neg_hours:.3f} clips")
    print()
    print(f"  best AP gain available   {best['reduction']} at {best['ap'] - base['ap']:+.4f} AP")
    print(f"  what it buys             {bought:+.1f} FP/hour "
          f"({100 * bought / base['fp_hour']:+.1f}% of the shipped rate)")
    print(f"  what is required         {required:.1f} FP/hour  (a {factor:.0f}x reduction)")
    print(f"  share of the gap closed  {100 * bought / required:.2f}%  "
          f"(material bar: {100 * MATERIAL_SHARE:.0f}%)")
    print()

    if bought < MATERIAL_SHARE * required:
        print("  🔴 VERDICT: AP AND FP/HOUR ARE DECOUPLED AT THIS OPERATING POINT.")
        print("     The largest AP gain this project has ever measured does not materially")
        print("     move the metric README §31 says decides the product. R4/R2/R3 have stated")
        print("     ceilings BELOW this probe, so they cannot close the gap either.")
        print("     Detection tuning is not the bottleneck. Fix the denominator first.")
    else:
        print("  ✅ VERDICT: AP gains DO transfer to FP/hour here. Detection work is justified.")

    print()
    print("  ⚠️  Measured on Nexar NEGATIVE clips, drawn from a collision dataset and likely")
    print("      enriched for hard cases. This is NOT a production gap. comma2k19 settles that.")
    print("  ⚠️  last_window is a PROBE, not a promotion. R1 died at gate 3a (D53);")
    print("      eval/adapters.py:129 stays np.nanmax.")
    return rows


def main():
    rows, neg_hours, n_neg = measure()
    report(rows, neg_hours, n_neg)


def _self_check():
    """Ties this module to the committed numbers. If these move, the module is wrong."""
    rows, neg_hours, n_neg = measure()
    by = {r["reduction"]: r for r in rows}

    # 1. The shipped configuration must reproduce both committed headline figures, or this
    #    module is measuring something other than the system the project actually ships.
    assert abs(by["max"]["ap"] - NEXAR_AP) < 5e-4, by["max"]["ap"]
    print(f"ok  shipped reduction reproduces the committed AP {by['max']['ap']:.4f} "
          f"(committed {NEXAR_AP})")

    assert abs(by["max"]["fp_hour"] - NEXAR_FP_HOUR) < 0.1, by["max"]["fp_hour"]
    assert by["max"]["fp"] == 83, by["max"]["fp"]
    print(f"ok  and the committed FP/hour: {by['max']['fp']} FP over {neg_hours:.3f} h "
          f"= {by['max']['fp_hour']:.1f} (committed {NEXAR_FP_HOUR})")

    # 2. The gate must be DERIVED. 0.9733 is timing.py's independently committed value, and
    #    this module reaches it through gate_at_recall without ever typing it.
    assert abs(by["max"]["threshold"] - 0.9733) < 1e-3, by["max"]["threshold"]
    print(f"ok  gate derived not typed: {by['max']['threshold']:.4f} "
          f"cross-checks eval/timing.py's 0.9733")

    # 3. 🔴 THE DECISIVE CHECK. The probe really does carry a large AP gain -- otherwise the
    #    upper-bound argument collapses and the verdict below would mean nothing.
    d_ap = by["last_window"]["ap"] - by["max"]["ap"]
    assert d_ap > 0.05, f"probe's AP gain {d_ap:+.4f} is too small to bound R4/R2/R3"
    print(f"ok  the probe carries a real AP gain: {d_ap:+.4f} "
          f"(committed paired +0.0556, CI excludes zero)")

    # 4. ...and that large AP gain still fails to move FP/hour materially. This is the finding.
    bought = by["max"]["fp_hour"] - by["last_window"]["fp_hour"]
    required = by["max"]["fp_hour"] - TARGET_FP_HOUR
    assert bought < MATERIAL_SHARE * required, (bought, required)
    print(f"ok  yet it buys only {bought:+.1f} FP/hour of the {required:.1f} required "
          f"({100 * bought / required:.2f}% of the gap)")

    # 5. An AP gain can even make FP/hour WORSE -- the clearest possible demonstration that
    #    the two metrics are not merely weakly coupled but genuinely independent here.
    assert by["last4_mean"]["ap"] > by["max"]["ap"], "expected last4_mean to gain AP"
    assert by["last4_mean"]["fp_hour"] > by["max"]["fp_hour"], "expected it to lose FP/hour"
    print(f"ok  last4_mean gains {by['last4_mean']['ap'] - by['max']['ap']:+.4f} AP while "
          f"LOSING {by['last4_mean']['fp_hour'] - by['max']['fp_hour']:+.1f} FP/hour")

    # 6. The target is arithmetically unreachable on this denominator -- the reason Step 2
    #    exists at all. Anything below 1/neg_hours reads as exactly zero.
    floor = 1 / neg_hours
    assert floor > TARGET_FP_HOUR, (floor, TARGET_FP_HOUR)
    print(f"ok  target {TARGET_FP_HOUR}/h is BELOW this corpus's {floor:.1f}/h resolution "
          f"floor -- unmeasurable here by arithmetic, not by performance")

    print("PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    _self_check() if a.self_check else main()
