"""Read a sweep's health mid-flight, without touching the running process.

    ~/envs/crashdet/bin/python eval/peek.py runs/baselines/badas-open/scores.jsonl

`benchmark.run()` only writes metrics.json when a sweep finishes. That is correct --
a partial metrics.json would get quoted as a result. But a ~31 h run still needs a
way to answer "is this working?" before hour 31, so this reads the append-only
scores.jsonl and prints the metrics on whatever has landed, clearly marked PARTIAL.

Partial numbers are diagnostic, not results: the clips scored so far are a prefix of
the visiting order, not a random sample. Do not commit or quote them.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402


def main(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    scored = [r for r in recs if "reason" not in r]
    skipped = [r for r in recs if "reason" in r]
    y = [r["label"] for r in scored]
    n_pos, n_neg = sum(y), len(y) - sum(y)

    print(f"PARTIAL -- {len(recs)}/667 visited, {n_pos} pos / {n_neg} neg, "
          f"{len(skipped)} skipped")
    if skipped:
        print(f"  skipped: {[r['id'] for r in skipped][:5]}")
    if not (n_pos and n_neg):
        print("  both classes needed before AP/AUC/FP-h exist -- nothing to report yet.")
        return

    np.seterr(all="ignore")
    neg_ids = [r["id"] for r in scored if r["label"] == 0]
    m = B.evaluate(y, [r["score"] for r in scored], B.hours(neg_ids))
    m["model"] = scored[0]["adapter"]
    print("  " + B.summarise(m))
    print(f"  TP {m['tp']} FP {m['fp']} FN {m['fn']} TN {m['tn']}  "
          f"@thr {m['threshold']}   P@R0.80 {m['precision_at_recall_0.80']:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "runs/baselines/badas-open/scores.jsonl")
