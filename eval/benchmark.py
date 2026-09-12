"""Nexar test-public benchmark: loader + metrics.

Model-agnostic on purpose. A "model" here is just a mapping {clip_id: score in [0,1]}.
The adapter interface (how a model produces those scores) is deliberately NOT defined
yet -- it is blocked on the BADAS-Open input contract (progress.md U-B3/U-B4).

NOT computed here: mTTA and time-to-detection. They are undeliverable on test-public --
`time_of_event` lies beyond the distributed clip for all 334 positives and the clip's
offset into the original video is not in the shipped metadata. See progress.md.

NOT computed here: raw accuracy. README §31 bans it as a headline metric.

Run `python eval/benchmark.py` for the self-check (reproduces the committed T3 numbers).
"""

import csv
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEXAR = os.path.join(ROOT, "data", "nexar")


def load_labels(usage="Public"):
    """{clip_id: 0|1} from solution.csv, filtered to one Usage split."""
    with open(os.path.join(NEXAR, "solution.csv")) as f:
        return {
            r["id"]: int(r["target"])
            for r in csv.DictReader(f)
            if r["Usage"] == usage
        }


def load_metadata():
    """{clip_id: {light_conditions, weather, scene, ...}} merged over both class dirs."""
    md = {}
    for split in ("positive", "negative"):
        path = os.path.join(NEXAR, "test-public", split, "metadata.csv")
        with open(path) as f:
            for r in csv.DictReader(f):
                md[r["file_name"].rsplit(".", 1)[0]] = r
    return md


def clip_hours(scores):
    """Total footage duration in hours, from the per-clip records used to score.

    Kept explicit and separate so FP/hour can never be reported without it.
    """
    return sum(r["decoded_frames"] / r["fps"] for r in scores) / 3600.0


def ece(y, p, bins=10):
    """Expected calibration error, equal-width bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return total


def precision_at_recall(y, p, target_recall=0.80):
    """Highest precision achievable at >= target_recall. 0.0 if unreachable."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y)
    recall = tp / max(y.sum(), 1)
    precision = tp / np.arange(1, len(y) + 1)
    ok = recall >= target_recall
    return float(precision[ok].max()) if ok.any() else 0.0


def evaluate(y, p, neg_hours, threshold=0.80):
    """Core metric set. `neg_hours` is required -- FP/hour is meaningless without it."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {
        "n": len(y),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "precision_at_recall_0.80": precision_at_recall(y, p, 0.80),
        "ece": float(ece(y, p)),
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tpr": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "fp_per_hour": fp / neg_hours,
        "negative_hours": neg_hours,
    }


def by_condition(y, p, ids, metadata, field):
    """{field_value: {n, n_pos, average_precision}} -- AP only where both classes present."""
    out = {}
    for v in sorted({metadata[i][field] for i in ids if i in metadata}):
        m = [k for k, i in enumerate(ids) if metadata.get(i, {}).get(field) == v]
        yy = np.asarray([y[k] for k in m], int)
        row = {"n": len(m), "n_pos": int(yy.sum())}
        if 0 < yy.sum() < len(yy):
            row["average_precision"] = float(
                average_precision_score(yy, [p[k] for k in m])
            )
        out[v] = row
    return out


def assert_no_leakage(train_ids, test_ids):
    """Fail loudly if any clip id appears on both sides of a split."""
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise AssertionError(
            f"leakage: {len(overlap)} ids in both splits, e.g. {sorted(overlap)[:5]}"
        )


def durations():
    """{clip_id: (decoded_frames, fps)} -- the ONE canonical duration table.

    Sourced from the committed T3 run, which actually decoded all 667 clips rather than
    trusting container headers (which lie on some of these files). Every adapter shares
    this table, so FP/hour denominators are identical across models and the rates are
    genuinely comparable. Deriving duration per-adapter would be a silent bug.
    """
    path = os.path.join(ROOT, "runs", "falsification", "T3_corpus_control.json")
    with open(path) as f:
        return {r["id"]: (r["decoded_frames"], r["fps"]) for r in json.load(f)}


def hours(ids, table=None):
    """Total footage duration in hours for `ids`, from the canonical table."""
    table = table or durations()
    return sum(table[i][0] / table[i][1] for i in ids if i in table) / 3600.0


def clip_paths():
    """{clip_id: absolute path} for Nexar test-public."""
    out = {}
    for split in ("positive", "negative"):
        d = os.path.join(NEXAR, "test-public", split)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".mp4"):
                out[os.path.splitext(fn)[0]] = os.path.join(d, fn)
    return out


def _resume(path, adapter_name):
    """Per-clip results already on disk, keyed by clip id.

    The BADAS sweep is ~18 h at ~97 s/clip. Without this, any interruption -- MPS
    fault, laptop sleep, a clip that wedges the decoder -- throws away every hour
    spent so far. Each record carries the adapter name and a mismatch raises, so a
    stride-2 resume can never silently inherit stride-1 scores.
    """
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("adapter") != adapter_name:
                raise AssertionError(
                    f"{path} holds scores from {r.get('adapter')!r}, not {adapter_name!r}. "
                    "Move it aside rather than mixing two configurations."
                )
            cache[r["id"]] = r
    return cache


def run(adapter, ids=None, out_dir=None, threshold=0.80, progress_every=25):
    """Score every clip with `adapter`, then evaluate through the shared metric path.

    Returns (metrics, records). Writes metrics.json when `out_dir` is given, and
    appends each clip to scores.jsonl as it lands so an interrupted run resumes.
    Unscorable clips are dropped from the metrics and listed in `metrics["skipped"]` --
    never silently scored as 0, which would flatter a broken adapter.
    """
    labels, paths, table = load_labels(), clip_paths(), durations()
    ids = list(ids) if ids is not None else sorted(labels)

    cache, log = {}, None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "scores.jsonl")
        cache = _resume(log_path, adapter.name)
        if cache:
            print(f"  resuming: {len(cache)}/{len(ids)} already scored in {log_path}",
                  flush=True)
        log = open(log_path, "a")

    records, skipped = [], []
    for n, cid in enumerate(ids, 1):
        r = cache.get(cid)
        if r is None:
            if cid not in paths:
                r = {"id": cid, "reason": "no video file"}
            else:
                s = adapter.score(paths[cid])
                if s is None or not np.isfinite(s):
                    r = {"id": cid, "reason": f"adapter returned {s!r}"}
                else:
                    r = {"id": cid, "label": labels[cid], "score": float(s)}
            r["adapter"] = adapter.name
            if log:
                log.write(json.dumps(r) + "\n")
                log.flush()
                os.fsync(log.fileno())  # 18 h of work; an fsync per 97 s clip is free
        (skipped if "reason" in r else records).append(r)
        if progress_every and n % progress_every == 0:
            print(f"  {adapter.name}: {n}/{len(ids)}", flush=True)
    if log:
        log.close()

    y = [r["label"] for r in records]
    p = [r["score"] for r in records]
    neg_ids = [r["id"] for r in records if r["label"] == 0]
    m = evaluate(y, p, hours(neg_ids, table), threshold=threshold)
    m["model"] = adapter.name
    m["skipped"] = skipped

    md = load_metadata()
    m["by_condition"] = {
        f: by_condition(y, p, [r["id"] for r in records], md, f)
        for f in ("weather", "scene", "light_conditions")
    }

    assert not {k for k in m if "accur" in k.lower()}, \
        "raw accuracy must not appear in any report (README §31)"

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({"metrics": m, "records": records}, f, indent=2)
    return m, records


def summarise(metrics):
    """One comparable line per model -- the baseline table README §31 requires."""
    return (f"{metrics['model']:<46} AP {metrics['average_precision']:.4f}  "
            f"AUC {metrics['roc_auc']:.4f}  "
            f"FP/h {metrics['fp_per_hour']:7.1f} over {metrics['negative_hours']:.2f}h  "
            f"ECE {metrics['ece']:.4f}")


def _self_check():
    """Reproduce the committed T3 numbers through this code path."""
    np.seterr(all="ignore")
    recs = json.load(
        open(os.path.join(ROOT, "runs", "falsification", "T3_corpus_control.json"))
    )
    labels = load_labels()
    ids = [r["id"] for r in recs]
    assert all(labels[i] == r["label"] for i, r in zip(ids, recs)), \
        "solution.csv disagrees with the labels T3 recorded"

    y = [r["label"] for r in recs]
    p = [r["score"] for r in recs]
    neg_hours = clip_hours([r for r in recs if r["label"] == 0])
    m = evaluate(y, p, neg_hours)

    assert abs(m["roc_auc"] - 0.5339) < 5e-4, m["roc_auc"]
    assert abs(m["average_precision"] - 0.5218) < 5e-4, m["average_precision"]
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (332, 325, 2, 8), m
    assert abs(neg_hours - 0.90) < 0.02, neg_hours
    print("ok  T3 reproduced: AUC %.4f  AP %.4f  TP/FP/FN/TN %d/%d/%d/%d"
          % (m["roc_auc"], m["average_precision"], m["tp"], m["fp"], m["fn"], m["tn"]))
    print("ok  FP/hour %.1f over %.2f h of negatives" % (m["fp_per_hour"], neg_hours))
    print("ok  precision@recall0.80 %.4f   ECE %.4f"
          % (m["precision_at_recall_0.80"], m["ece"]))

    md = load_metadata()
    assert len(md) == 667, len(md)
    for field in ("weather", "scene", "light_conditions"):
        cond = by_condition(y, p, ids, md, field)
        print("ok  by %-17s %s" % (field, {k: v["n"] for k, v in cond.items()}))

    try:
        assert_no_leakage(["a", "b"], ["b", "c"])
        raise SystemExit("FAIL: leakage check did not fire")
    except AssertionError:
        print("ok  leakage check fires on overlap")

    assert not {k for k in m if "accur" in k}, "raw accuracy must not be reported (README §31)"
    print("PASS")


if __name__ == "__main__":
    _self_check()
