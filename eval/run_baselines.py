"""The baseline table README §31 and §41 Phase 5 require: three models, one code path.

    BADAS-Open zero-shot  ·  MobileNetV2+LSTM (retired)  ·  always-negative

all scored on identical clips with identical metrics and an identical FP/hour denominator.

Usage
-----
    # cheap: the two zero-compute baselines over all 667 clips
    ~/envs/crashdet/bin/python eval/run_baselines.py --no-badas

    # verify the BADAS adapter on a balanced subset first (~1 min/clip)
    ~/envs/badas/bin/python eval/run_baselines.py --limit 6

    # the real sweep: ~31 h at stride 1. MEASURED over 140 clips of the live run =
    # ~167 s/clip end-to-end. The earlier ~97 s/clip (6-clip smoke) and ~10 h
    # (compute-only) figures are both too optimistic -- do not plan against them.
    # --skip-predictor takes ~25% off.
    PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -i \
        ~/envs/badas/bin/python eval/run_baselines.py --out runs/baselines

`--limit N` takes N clips balanced across classes, so a subset run still produces a
meaningful (if tiny) AP rather than a single-class degenerate one.

Resumable: every clip is appended to <out>/<model>/scores.jsonl as it lands, so an
interrupted sweep picks up where it stopped. Delete that file to force a rescore.
"""

import argparse
import json
import os
import sys
import time
from itertools import zip_longest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import benchmark as B  # noqa: E402
from adapters import AlwaysNegative, BadasOpen, CachedScores  # noqa: E402

T3_JSON = os.path.join(ROOT, "runs", "falsification", "T3_corpus_control.json")


def balanced_ids(limit=None):
    """Clip ids in positive/negative-interleaved order.

    Deliberately NOT `sorted(labels)`. In Nexar test-public every positive id sorts below
    every negative, so a plain sort scores all 334 positives before the first negative --
    and AP, ROC-AUC and FP/hour are all undefined until both classes are present. On a
    ~30 h sweep that is ~9 h producing no computable metric and no way to catch a broken
    harness early. Interleaving makes a partial run readable from the second clip onward.

    Metrics are order-independent, so the completed numbers are identical either way.
    """
    labels = B.load_labels()
    pos = sorted(i for i in labels if labels[i] == 1)
    neg = sorted(i for i in labels if labels[i] == 0)
    ids = [i for pair in zip_longest(pos, neg) for i in pair if i is not None]
    return ids if limit is None else ids[:limit]


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
