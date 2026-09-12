"""The baseline table README §31 and §41 Phase 5 require: three models, one code path.

    BADAS-Open zero-shot  ·  MobileNetV2+LSTM (retired)  ·  always-negative

all scored on identical clips with identical metrics and an identical FP/hour denominator.

Usage
-----
    # cheap: the two zero-compute baselines over all 667 clips
    ~/envs/crashdet/bin/python eval/run_baselines.py --no-badas

    # verify the BADAS adapter on a balanced subset first (~1 min/clip)
    ~/envs/badas/bin/python eval/run_baselines.py --limit 6

    # the real sweep: ~10 h at stride 1, ~7.5 h with --skip-predictor
    ~/envs/badas/bin/python eval/run_baselines.py --out runs/baselines

`--limit N` takes N clips balanced across classes, so a subset run still produces a
meaningful (if tiny) AP rather than a single-class degenerate one.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import benchmark as B  # noqa: E402
from adapters import AlwaysNegative, BadasOpen, CachedScores  # noqa: E402

T3_JSON = os.path.join(ROOT, "runs", "falsification", "T3_corpus_control.json")


def balanced_ids(limit=None):
    labels = B.load_labels()
    if limit is None:
        return sorted(labels)
    pos = sorted(i for i in labels if labels[i] == 1)
    neg = sorted(i for i in labels if labels[i] == 0)
    k = max(1, limit // 2)
    return pos[:k] + neg[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="score only N clips, balanced across classes (smoke runs)")
    ap.add_argument("--out", default=None, help="directory for metrics.json per model")
    ap.add_argument("--no-badas", action="store_true",
                    help="skip BADAS (runs without torch, finishes in seconds)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--skip-predictor", action="store_true",
                    help="~25%% faster; upstream discards the predictor output anyway")
    args = ap.parse_args()

    np.seterr(all="ignore")
    ids = balanced_ids(args.limit)
    print(f"clips: {len(ids)}  (limit={args.limit})\n")

    adapters = [
        CachedScores("mobilenetv2-lstm (CCD-trained, retired)", T3_JSON),
        AlwaysNegative(),
    ]
    if not args.no_badas:
        adapters.insert(0, BadasOpen(device=args.device, stride=args.stride))

    table = []
    for ad in adapters:
        t0 = time.time()
        out_dir = os.path.join(args.out, ad.name.split("(")[0].strip()) if args.out else None
        m, _ = B.run(ad, ids=ids, out_dir=out_dir, progress_every=5)
        m["elapsed_seconds"] = round(time.time() - t0, 1)
        table.append(m)
        print(f"  {B.summarise(m)}  [{m['elapsed_seconds']}s]")
        if m["skipped"]:
            print(f"    skipped {len(m['skipped'])}: {m['skipped'][:3]}")

    print("\n" + "=" * 100)
    print(f"BASELINE TABLE  ({len(ids)} clips, identical split and denominator)")
    print("=" * 100)
    for m in sorted(table, key=lambda r: -r["average_precision"]):
        print("  " + B.summarise(m))
    print("=" * 100)
    print("FP/hour is reported with its denominator because the denominator is small:")
    print(f"  {table[0]['negative_hours']:.2f} h of negatives cannot evidence a rate below "
          f"~{1 / max(table[0]['negative_hours'], 1e-9):.1f}/h (README §34 -> UK-HN-500).")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "baseline_table.json"), "w") as f:
            json.dump(table, f, indent=2)
        print(f"\nwrote {args.out}/baseline_table.json")


if __name__ == "__main__":
    main()
