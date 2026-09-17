"""R1 gate 3a: does the score peak TRACK the annotated collision, or just drift to the end?

THE QUESTION, in one paragraph. Replacing `np.nanmax` with the last window is worth +0.0556
AP on Nexar (0.8349 -> 0.8905, paired CI [+0.0263, +0.0876]) -- the only measured detection
improvement this project has. NEW_PLAN.md §3.2 explains it: Nexar truncates its clips
500-1500 ms BEFORE the event, so for a positive the last window is the most informative one
and the peak sits at 0.975 of the clip. But there is a rival explanation that fits the same
Nexar evidence exactly: the score simply DRIFTS UPWARD THE LONGER THE MODEL WATCHES. Under
that story the last window wins for a reason having nothing to do with truncation, and
+0.0556 is a benchmark artifact. The two stories are indistinguishable on Nexar, because on
Nexar the event IS at the end. They separate on a corpus whose collisions are mid-clip.

  PASS            peaks track the annotated Time-of-collision, and sit materially before the
                  clip end. Truncation is the mechanism. R1's story survives.
  FAIL            peaks pile up at the clip end regardless of the annotation. Watch-time
                  drift. R1 dies and Nexar's +0.0556 is an artifact.
  UNINTERPRETABLE the annotation has too little variance to track. NOT a pass.

🔴 THE THIRD VERDICT IS NOT DECORATION. DAD's Time-of-collision has IQR 0.16 s -- 84% of its
clips within 0.25 s of exactly 3.00 s, because its accident is structurally frame 90 of 100.
Against a near-constant there is nothing to track, so a null correlation there means nothing
in either direction. This project has already nearly run the mechanism test on DAD and read
the result as evidence. The IQR floor below is checked BEFORE the correlation is read, and
a corpus that fails it returns UNINTERPRETABLE no matter how the numbers came out. DADA-2000
is the target because its IQR is 4.63 s (range 0.37-14.43 s).

🔴 THE CLIP-END NULL IS NOT ZERO CORRELATION, and this is the subtlety that makes a naive
version of this test wrong. If longer clips also tend to have later collisions, then a model
that ALWAYS peaks at the clip end still correlates positively with Time-of-collision -- purely
through duration. Testing r(t_peak, t_coll) > 0 would therefore "confirm" the mechanism on a
pure watch-time-drift model. So the null is constructed explicitly: give every clip a peak at
its last representable index and correlate THAT with the annotation. The quantity that
decides the gate is the PAIRED DIFFERENCE r_measured - r_null, with a bootstrap CI, on the
same resampled clips each iteration (NEW_PLAN.md §1's paired discipline).

TIME BASE. Reuses eval/timing.py::load_traces_abs, which keeps the absolute NaN offset.
⚠️ reduction_study.load_traces strips the 16 leading NaNs -- correct for ranking, and reusing
it here would silently place every timestamp 2 s early. Nothing would crash. fps is read
per-clip from the .npz, never hard-coded, because an external corpus may be scored at a
different target_fps than Nexar's 8.

BURDEN OF PROOF IS ON R1. Both criteria must hold for PASS. A split result (tracking but
peaks still at the end, or vice versa) is reported as FAIL with the reason named -- R1 has to
SURVIVE gate 3, and a mixed outcome is not survival.

    ~/envs/badas/bin/python -m eval.gate3_mechanism --self-check
    ~/envs/badas/bin/python -m eval.gate3_mechanism \
        --frames-dir runs/gate3a/frames \
        --annotation data/dada2000/gate3a/dada_gate3a_annotation.csv
"""

import argparse
import csv
import os

import numpy as np
from scipy.stats import rankdata

from eval.benchmark import ROOT
from eval.timing import load_traces_abs

# --- Thresholds, stated before any measurement so the bar cannot move afterwards. ---------
IQR_FLOOR_S = 1.0       # below this the annotation is a constant; DAD is 0.16, DADA 4.63
MIN_CLIPS = 30          # a correlation on fewer clips is not worth a verdict
NEXAR_PEAK = 0.975      # Nexar positives' median normalised peak, measured at full n=667
NORM_PEAK_MAX = 0.90    # "materially below Nexar": median normalised peak must clear this
N_BOOT = 10000
SEED = 0

# 🔴 NOT vendor/badas-open/annotation/dada2000_small_test_concensus.csv (D45). Those ids do not
# map onto the obtainable DADA archive -- 132/221 matched, and 41_007 annotates a 12.37 s collision
# in a 2.9 s clip -- so pairing on them correlates each clip's peak against a DIFFERENT clip's
# collision. This file ships beside the clips, is built from DADA's own Sheet1, and carries the
# same column names, so load_annotation() reads it unchanged. Do not "restore" the BADAS CSV.
DEFAULT_ANNOTATION = os.path.join(
    ROOT, "data", "dada2000", "gate3a", "dada_gate3a_annotation.csv")


def load_annotation(path):
    """{clip_id: (time_of_collision_s, event_type)}, ids stripped of their extension to
    match the .npz names the adapter writes."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["Time-of-collision"])
            except (KeyError, ValueError):
                continue
            out[os.path.splitext(row["id"])[0]] = (t, row.get("Event-type", "").strip())
    return out


def measure(traces, annot):
    """Per-clip measured peak time, the clip-end null, and the annotation, aligned.

    t_peak and clip_duration follow eval/timing.py exactly: the absolute index is
    offset + argmax, seconds are index / fps, and duration is (offset + len) / fps.
    """
    ids = sorted(set(traces) & set(annot))
    rows = []
    for cid in ids:
        trace, fps, offset = traces[cid]
        total = offset + len(trace)
        rows.append({
            "id": cid,
            "t_peak": (offset + int(np.argmax(trace))) / fps,
            # Watch-time drift predicts a peak at the LAST representable index. That is
            # (total - 1), not total: upstream discards the final window (timing.py trap 2).
            "t_end_null": (total - 1) / fps,
            "duration": total / fps,
            "t_coll": annot[cid][0],
            "event_type": annot[cid][1],
        })
    return rows


def _pearson(a, b):
    """Correlation, with a constant predictor defined as 0 rather than NaN.

    A constant explains nothing, so zero is the honest reading -- and it is the right answer
    for the clip-end null on a corpus of equal-length clips, where "always peak at the end"
    genuinely carries no information about collision time. NaN would instead delete those
    bootstrap iterations, quietly conditioning the CI on the resamples that happened to vary.
    A constant ANNOTATION is not handled here: that is the UNINTERPRETABLE case, and the IQR
    floor settles it before any correlation is read.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    return _pearson(rankdata(a), rankdata(b))


def correlations(rows, n_boot=N_BOOT, seed=SEED):
    """r(measured, annotation), r(clip-end null, annotation), and their paired difference
    with a 95% bootstrap CI.

    The difference is the quantity that decides the gate. Both correlations are computed on
    the SAME resampled clips each iteration, so the shared sampling noise cancels -- the same
    reason eval/reduction_study.py pairs its AP bootstrap.
    """
    t_peak = np.array([r["t_peak"] for r in rows])
    t_null = np.array([r["t_end_null"] for r in rows])
    t_coll = np.array([r["t_coll"] for r in rows])
    rng = np.random.default_rng(seed)
    n = len(rows)

    out = {}
    for name, fn in (("pearson", _pearson), ("spearman", _spearman)):
        r_m, r_n = fn(t_peak, t_coll), fn(t_null, t_coll)
        deltas = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            d = fn(t_peak[idx], t_coll[idx]) - fn(t_null[idx], t_coll[idx])
            deltas[i] = d
        deltas = deltas[~np.isnan(deltas)]
        out[name] = {
            "r_measured": r_m,
            "r_null": r_n,
            "delta": r_m - r_n,
            "ci": (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))),
            "n_boot_used": int(deltas.size),
        }
    return out


def verdict(rows, corr, annot_iqr):
    """PASS / FAIL / UNINTERPRETABLE, plus the criteria that produced it.

    Order matters: interpretability is settled BEFORE the correlation is read, so a
    near-constant annotation can never be laundered into a pass by a lucky number.
    """
    reasons = []
    if len(rows) < MIN_CLIPS:
        return "UNINTERPRETABLE", [f"only {len(rows)} clips (< {MIN_CLIPS})"]
    if annot_iqr < IQR_FLOOR_S:
        return "UNINTERPRETABLE", [
            f"annotation IQR {annot_iqr:.2f} s < {IQR_FLOOR_S:.2f} s floor -- the annotation "
            f"is effectively a constant, so there is nothing for the peak to track. This is "
            f"DAD's failure mode and is NOT a pass."]

    norm = np.array([r["t_peak"] / r["duration"] for r in rows])
    median_norm = float(np.median(norm))

    lo, hi = corr["spearman"]["ci"]
    tracks = lo > 0
    reasons.append(
        f"tracking: Spearman r_measured {corr['spearman']['r_measured']:+.3f} vs clip-end "
        f"null {corr['spearman']['r_null']:+.3f}, delta {corr['spearman']['delta']:+.3f} "
        f"CI [{lo:+.3f}, {hi:+.3f}] -> {'separates from the null' if tracks else 'does NOT separate from the null'}")

    early = median_norm < NORM_PEAK_MAX
    reasons.append(
        f"position: median normalised peak {median_norm:.3f} vs Nexar's {NEXAR_PEAK:.3f}, "
        f"bar {NORM_PEAK_MAX:.3f} -> {'materially before the clip end' if early else 'still at the clip end'}")

    if tracks and early:
        return "PASS", reasons
    return "FAIL", reasons


def report(frames_dir, annotation_path, n_boot=N_BOOT):
    traces = load_traces_abs(frames_dir)
    annot = load_annotation(annotation_path)
    rows = measure(traces, annot)
    if not rows:
        raise SystemExit(f"no clip ids shared between {frames_dir} and {annotation_path}")

    t_coll_all = np.array([a[0] for a in annot.values()])
    matched_coll = np.array([r["t_coll"] for r in rows])
    annot_iqr = float(np.percentile(matched_coll, 75) - np.percentile(matched_coll, 25))

    print(f"traces {len(traces)}   annotation rows {len(annot)}   matched {len(rows)}")
    print(f"unmatched traces: {len(traces) - len(rows)}   "
          f"unmatched annotations: {len(annot) - len(rows)}\n")

    print("=" * 78)
    print("THE ANNOTATION'S OWN VARIANCE — read this before any correlation below")
    print("=" * 78)
    print(f"  Time-of-collision   median {np.median(matched_coll):.2f} s   "
          f"IQR {annot_iqr:.2f} s   range {matched_coll.min():.2f}–{matched_coll.max():.2f} s")
    print(f"  interpretability floor {IQR_FLOOR_S:.2f} s   "
          f"-> {'ENOUGH variance to track' if annot_iqr >= IQR_FLOOR_S else '🔴 TOO LITTLE — result is UNINTERPRETABLE'}")
    print(f"  (for scale: DAD 0.16 s = a constant; DADA-2000 4.63 s = real variance)")
    if len(t_coll_all) != len(rows):
        print(f"  note: IQR above is over the {len(rows)} MATCHED clips, not all "
              f"{len(t_coll_all)} annotation rows")

    corr = correlations(rows, n_boot=n_boot)
    print("\n" + "=" * 78)
    print("DOES THE PEAK TRACK THE ANNOTATION, BEYOND WHAT CLIP LENGTH ALONE EXPLAINS?")
    print("=" * 78)
    print("  The clip-end null = every clip peaks at its last representable index. If long")
    print("  clips also have late collisions, that null ALREADY correlates with the")
    print("  annotation. Only the difference is evidence.\n")
    for name in ("pearson", "spearman"):
        c = corr[name]
        lo, hi = c["ci"]
        print(f"  {name:9s} measured {c['r_measured']:+.3f}   clip-end null {c['r_null']:+.3f}   "
              f"Δ {c['delta']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"{'  excludes zero' if lo > 0 or hi < 0 else '  INCLUDES ZERO'}")

    norm = np.array([r["t_peak"] / r["duration"] for r in rows])
    print("\n" + "=" * 78)
    print("WHERE IN THE CLIP DOES THE PEAK ACTUALLY SIT?")
    print("=" * 78)
    print(f"  measured normalised peak   median {np.median(norm):.3f}   "
          f"IQR [{np.percentile(norm, 25):.3f}, {np.percentile(norm, 75):.3f}]")
    print(f"  Nexar positives (n=667)    median {NEXAR_PEAK:.3f}   <- what truncation produces")
    print(f"  in the final 10% of clip   {100 * (norm > 0.9).mean():.1f}%  "
          f"(Nexar positives: 73.7%)")

    print("\n" + "=" * 78)
    print("BY EVENT TYPE — near-collisions kept separate, never pooled")
    print("=" * 78)
    for et in sorted({r["event_type"] for r in rows}):
        sub = [r for r in rows if r["event_type"] == et]
        sn = np.array([r["t_peak"] / r["duration"] for r in sub])
        st = np.array([r["t_coll"] for r in sub])
        siqr = float(np.percentile(st, 75) - np.percentile(st, 25)) if len(st) > 1 else 0.0
        print(f"  {et or '(unlabelled)':16s} n={len(sub):4d}   normalised peak median "
              f"{np.median(sn):.3f}   annotation IQR {siqr:.2f} s")

    v, reasons = verdict(rows, corr, annot_iqr)
    print("\n" + "=" * 78)
    print(f"GATE 3a VERDICT: {v}")
    print("=" * 78)
    for r in reasons:
        print(f"  · {r}")
    if v == "PASS":
        print("\n  R1's mechanism survives on untruncated data. Gate 3b (DAD AP) still open;")
        print("  R1 must survive BOTH before eval/adapters.py:129 changes.")
    elif v == "FAIL":
        print("\n  🔴 R1 DIES. The peak does not track the collision — the score drifts with")
        print("     watch-time. Nexar's +0.0556 AP is a benchmark artifact, not a finding.")
        print("     Keep np.nanmax. Record the kill; do not rescue it with a weaker test.")
    else:
        print("\n  Neither pass nor fail. Do not report this as evidence in either direction.")
    return {"verdict": v, "reasons": reasons, "corr": corr, "n": len(rows),
            "annotation_iqr": annot_iqr, "median_norm_peak": float(np.median(norm))}


# --------------------------------------------------------------------------------------
# Self-check: synthetic traces only. No downloaded data, no MPS time.
# --------------------------------------------------------------------------------------

def _write_npz(d, cid, scores, fps):
    np.savez(os.path.join(d, f"{cid}.npz"), scores=np.asarray(scores, float),
             target_fps=float(fps), stride=1, frame_count=16)


def _synth(d, csv_path, n, fps, peak_at, lead_nan=16, seed=1):
    """Write n synthetic clips + their annotation CSV.

    `peak_at(t_coll, n_valid, fps, offset)` returns the index (within the valid part of the
    trace) where the score peaks -- which is how the two rival stories are simulated.

    🔴 CLIP DURATION IS DELIBERATELY CORRELATED WITH Time-of-collision (each clip runs a
    little past its collision), because that is the confound the whole design exists to
    defeat. It makes the clip-end null correlate strongly with the annotation ALL BY ITSELF,
    so a test that only asked "is r(t_peak, t_coll) > 0" would pass the watch-time-drift
    model. Equal-length synthetic clips would hide exactly the bug this guards against.
    """
    rng = np.random.default_rng(seed)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "Event-type", "Time-of-collision"])
        for i in range(n):
            t_coll = float(rng.uniform(2.0, 10.0))
            n_total = int((t_coll + rng.uniform(1.5, 4.0)) * fps)
            n_valid = n_total - lead_nan
            j = int(np.clip(peak_at(t_coll, n_valid, fps, lead_nan), 0, n_valid - 1))
            trace = rng.uniform(0.05, 0.15, n_valid)
            trace[j] = 0.99
            _write_npz(d, f"clip{i:03d}", np.r_[np.full(lead_nan, np.nan), trace], fps)
            w.writerow([f"clip{i:03d}.mp4", "Collision", t_coll])


def _self_check():
    import tempfile

    fps = 8.0
    boot = 400  # the CI's correctness is numpy's job; this checks the decision logic

    # 1. A known peak position is recovered, offset included.
    with tempfile.TemporaryDirectory() as d:
        _write_npz(d, "one", np.r_[np.full(16, np.nan), np.full(64, 0.1)], fps)
        with np.load(os.path.join(d, "one.npz")) as f:
            s = f["scores"].copy()
        s[16 + 24] = 0.9
        _write_npz(d, "one", s, fps)
        tr = load_traces_abs(d)
        rows = measure(tr, {"one": (5.0, "Collision")})
        assert abs(rows[0]["t_peak"] - (16 + 24) / fps) < 1e-9, rows
        assert abs(rows[0]["duration"] - 80 / fps) < 1e-9, rows
        print(f"ok  known peak recovered at t_peak {rows[0]['t_peak']:.3f}s (absolute index 40)")

        # 5. The NaN offset is carried. Dropping it moves every timestamp 2 s early -- the
        #    trap this module's docstring names. Nothing would crash; the numbers would lie.
        naive = measure({"one": (tr["one"][0], fps, 0)}, {"one": (5.0, "Collision")})
        assert abs(rows[0]["t_peak"] - naive[0]["t_peak"] - 16 / fps) < 1e-9
        print(f"ok  NaN offset applied ({rows[0]['t_peak']:.3f}s with, "
              f"{naive[0]['t_peak']:.3f}s without — a silent 2.000s error)")

    # 6. fps is read per-clip, never hard-coded. Same index, two rates, two answers.
    with tempfile.TemporaryDirectory() as d:
        for cid, f_ in (("slow", 4.0), ("fast", 8.0)):
            s = np.r_[np.full(16, np.nan), np.full(64, 0.1)]
            s[16 + 24] = 0.9
            _write_npz(d, cid, s, f_)
        tr = load_traces_abs(d)
        rows = {r["id"]: r for r in measure(tr, {c: (5.0, "Collision") for c in tr})}
        assert abs(rows["slow"]["t_peak"] - 10.0) < 1e-9, rows["slow"]
        assert abs(rows["fast"]["t_peak"] - 5.0) < 1e-9, rows["fast"]
        print("ok  fps read per-clip: index 40 -> 10.000s at 4 fps, 5.000s at 8 fps")

    # 2. Peaks AT the annotation -> PASS.
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "a.csv")
        _synth(d, c, 60, fps, lambda t, nv, f_, off: t * f_ - off)
        r = report(d, c, n_boot=boot)
        assert r["verdict"] == "PASS", r["verdict"]
        print(f"ok  peaks tracking the annotation -> PASS "
              f"(median normalised peak {r['median_norm_peak']:.3f})")

    # 3. Peaks at the CLIP END regardless -> FAIL. This is the rival story, and it must not
    #    come back as "no correlation, inconclusive": it is a positive finding against R1.
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "a.csv")
        _synth(d, c, 60, fps, lambda t, nv, f_, off: nv - 1)
        r = report(d, c, n_boot=boot)
        assert r["verdict"] == "FAIL", r["verdict"]
        print(f"ok  peaks at the clip end regardless -> FAIL "
              f"(median normalised peak {r['median_norm_peak']:.3f})")

    # 4. A DAD-like near-constant annotation -> UNINTERPRETABLE, even though the traces here
    #    track it perfectly. Interpretability is decided before the correlation is read.
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "a.csv")
        rng = np.random.default_rng(7)
        with open(c, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "Event-type", "Time-of-collision"])
            for i in range(60):
                t = 3.0 + float(rng.uniform(-0.08, 0.08))  # DAD: IQR ~0.16 s
                s = np.r_[np.full(16, np.nan), np.full(80, 0.1)]
                s[int(t * fps)] = 0.99
                _write_npz(d, f"clip{i:03d}", s, fps)
                w.writerow([f"clip{i:03d}.mp4", "Collision", t])
        r = report(d, c, n_boot=boot)
        assert r["verdict"] == "UNINTERPRETABLE", r["verdict"]
        assert r["annotation_iqr"] < IQR_FLOOR_S, r["annotation_iqr"]
        print(f"ok  near-constant annotation (IQR {r['annotation_iqr']:.3f}s) -> "
              f"UNINTERPRETABLE, not laundered into a pass")

    print("PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", help="directory of .npz traces from score_external.py")
    ap.add_argument("--annotation", default=DEFAULT_ANNOTATION)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        _self_check()
    elif a.frames_dir:
        report(a.frames_dir, a.annotation)
    else:
        ap.error("need --frames-dir or --self-check")
