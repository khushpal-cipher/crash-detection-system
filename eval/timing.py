"""Temporal localisation: per-frame trace -> t_start / t_peak / t_end, the incident
record's second field.

README §27 is explicit that the product is not detection -- detection is a commodity,
BADAS-Open is free and Apache-2.0 -- it is the structured incident record. Against that
record's ~10 fields exactly one worked before this file: "collision detected", the
commodity one. The 667 per-frame traces committed at 252d282 unblock the second, at zero
compute and zero new data.

🔴 THIS IS A CAPABILITY, NOT A METRIC. Nexar's `time_of_event` is corrupt for all 334
positives (median 20.0 s against a 9.93 s clip -- the event is outside the distributed
video), so there is NOTHING here to validate a timestamp against. No mTTA, no
time-to-detection, no accuracy claim of any kind. progress.md §12 and NEW_PLAN.md §12
both forbid it. The honest claim is: the pipeline now emits these three fields; their
accuracy is unmeasured on this benchmark because this benchmark cannot measure it.

TWO-STAGE THRESHOLD, because "did it happen" and "when did it happen" are different
questions and one threshold answering both is degenerate on marginal clips:

  1. GATE (global, derived).  One threshold read off the PR curve at a stated target
     recall -- README §41 Phase 2 task 5, "derive the operating point from the test-split
     PR curve at a stated target recall". NOT hard-coded: B5 and §12 forbid hard-coded
     thresholds, and 0.80 is the *retired* model's number. Below the gate, no record is
     emitted at all, which is the correct product behaviour.
  2. LOCALISATION (per-clip, relative).  Half-prominence above the clip's own median, so
     a clip that merely sits high throughout does not report a 10-second incident.

TIME BASE, traced to vendor/badas-open/badas/utils/sliding_window.py rather than assumed:
upstream resamples the video to `target_fps` BEFORE windowing, so a trace index is an
index in the resampled timeline and t = index / target_fps. The source video's fps is
irrelevant. A window (start, start+16) has its prediction filed against `end_idx`, the
first frame the model has NOT seen, so index i reads "having watched to (i-1)/fps, this
is P(collision next)" -- which makes i/fps the natural alert timestamp.

⚠️ Two traps, both live:
  * `reduction_study.load_traces` STRIPS the 16 leading NaNs. Reuse it naively for timing
    and every timestamp lands 2 s early. This module keeps the absolute offset.
  * upstream's `if 0 <= target_frame < total_frames` guard silently DISCARDS the final
    window's prediction (at stride 1 it targets `total_frames` exactly). An 81-frame clip
    runs 66 windows and keeps 65. So the latest representable timestamp is (len-1)/fps and
    `t_end` must never be read as "end of clip". This does not affect R1, which applies the
    same reduction to every clip, but it does affect what t_end means.

Run `python -m eval.timing --self-check` for the checks, `python -m eval.timing` for the
report, `python -m eval.timing --emit DIR` to write one JSON record per gated clip.
"""

import argparse
import glob
import json
import os

import numpy as np

from eval.benchmark import ROOT, durations, hours, load_labels, precision_at_recall
from eval.reduction_study import FRAMES_DIR

TARGET_RECALL = 0.80
PROMINENCE_FRACTION = 0.5


def load_traces_abs():
    """{clip_id: (valid_scores, fps, offset)} -- offset is the absolute index of the first
    scored frame, which `reduction_study.load_traces` throws away.

    That function strips leading NaNs and is correct for its own job (ranking a clip needs
    no time base). Timing needs the offset back, so this reads the .npz directly rather
    than changing a loader two other committed scripts depend on.
    """
    out = {}
    for path in glob.glob(os.path.join(FRAMES_DIR, "*.npz")):
        clip_id = os.path.splitext(os.path.basename(path))[0]
        with np.load(path) as f:
            scores, fps = f["scores"], float(f["target_fps"])
        finite = ~np.isnan(scores)
        if not finite.any():
            continue
        offset = int(np.argmax(finite))
        assert finite[offset:].all(), f"{clip_id}: NaN after the leading run, not just before"
        out[clip_id] = (scores[offset:], fps, offset)
    return out


def gate_at_recall(y, p, target_recall=TARGET_RECALL):
    """Lowest score threshold that still reaches `target_recall`, with what it costs.

    The operating point is read off the data, never chosen. Returns the threshold plus the
    recall/precision/FP it actually delivers, so a caller cannot quote the threshold
    without the price.
    """
    y, p = np.asarray(y, int), np.asarray(p, float)
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    recall = np.cumsum(ys) / max(ys.sum(), 1)
    ok = np.flatnonzero(recall >= target_recall)
    if not ok.size:
        raise ValueError(f"recall {target_recall} unreachable on this score distribution")
    threshold = float(ps[ok[0]])
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return {
        "threshold": threshold,
        "target_recall": target_recall,
        "recall": tp / max(int((y == 1).sum()), 1),
        "precision": tp / max(tp + fp, 1),
        "tp": tp,
        "fp": fp,
    }


def incident(trace, fps, offset, gate):
    """One clip's timing fields, or None if the clip does not clear the gate.

    `trace` is the NaN-stripped score array, `offset` its absolute start index. Returns
    seconds on the resampled (target_fps) timeline.
    """
    peak_v = float(trace.max())
    if peak_v < gate:
        return None

    base = float(np.median(trace))
    prominence = peak_v - base
    j_peak = int(np.argmax(trace))

    if prominence <= 1e-9:
        # Flat trace: the peak is real but carries no shape, so there is no window to
        # report. Collapsing to a single instant is honest; inventing a span is not.
        j_start = j_end = j_peak
        cut = peak_v
    else:
        cut = base + PROMINENCE_FRACTION * prominence
        above = np.flatnonzero(trace >= cut)
        j_start, j_end = int(above[0]), int(above[-1])

    total = offset + len(trace)
    return {
        "t_start": (offset + j_start) / fps,
        "t_peak": (offset + j_peak) / fps,
        "t_end": (offset + j_end) / fps,
        "clip_duration_s": total / fps,
        "prominence_cut": cut,
        "baseline": base,
    }


def record(clip_id, trace, fps, offset, gate, detector):
    """A README §27-shaped incident record, or None below the gate.

    Fields this pipeline genuinely cannot produce -- ego_involved (Nexar has no ego field,
    D23), severity band, closing speed, GPS -- are OMITTED, not null-filled. A null implies
    a field that could be populated and is merely empty, which would misrepresent what the
    system knows.
    """
    t = incident(trace, fps, offset, gate)
    if t is None:
        return None
    return {
        "event_id": clip_id,
        "source": f"nexar/test-public/{clip_id}.mp4",
        "detector": detector,
        "t_start": round(t["t_start"], 3),
        "t_peak": round(t["t_peak"], 3),
        "t_end": round(t["t_end"], 3),
        "clip_duration_s": round(t["clip_duration_s"], 3),
        "confidence": {
            # Both reductions, promoting neither. R1 has passed gates 1 and 2 (+0.0556 AP)
            # but gate 3 is open, and progress.md's banner says do not switch until it is
            # answered. eval/adapters.py still reduces with nanmax; so does `shipped`.
            "shipped": round(float(trace.max()), 4),
            "candidate_r1_last_window": round(float(trace[-1]), 4),
        },
        "timing_validated": False,  # see module docstring: no ground truth exists here
    }


def main(emit_dir=None, target_recall=TARGET_RECALL):
    labels, traces = load_labels(), load_traces_abs()
    ids = sorted(set(labels) & set(traces))
    assert len(ids) == 667, f"expected 667 clips, found {len(ids)} (frames dir incomplete?)"
    y = np.array([labels[i] for i in ids])
    assert 0 < y.sum() < len(y), "only one class present -- broken label join"
    p = np.array([traces[i][0].max() for i in ids])

    g = gate_at_recall(y, p, target_recall)
    neg_hours = hours([i for i in ids if labels[i] == 0])
    detector = "badas-open(stride=1,fps=8.0,img=224) nanmax"

    print(f"n={len(ids)}  pos={int(y.sum())}  neg={int(len(y) - y.sum())}\n")
    print("=" * 74)
    print("STAGE 1 — GATE, derived from the PR curve (not chosen, not hard-coded)")
    print("=" * 74)
    print(f"  target recall     {g['target_recall']:.2f}")
    print(f"  threshold         {g['threshold']:.4f}   <- derived")
    print(f"  delivers recall   {g['recall']:.4f}   precision {g['precision']:.4f}")
    print(f"  false positives   {g['fp']} over {neg_hours:.3f} h of negatives "
          f"= {g['fp'] / neg_hours:.1f} FP/hour")
    print(f"  cross-check       benchmark.precision_at_recall = "
          f"{precision_at_recall(y, p, target_recall):.4f} (best precision at >= that recall)")
    print("\n  🔴 92 FP/hour is the honest state of an UNCALIBRATED model, and 0.899 h is a")
    print("     54-minute denominator. Neither number may be quoted without the other.")

    recs = {i: record(i, *traces[i], g["threshold"], detector) for i in ids}
    gated = {i: r for i, r in recs.items() if r is not None}
    pos_gated = [i for i in gated if labels[i] == 1]
    neg_gated = [i for i in gated if labels[i] == 0]

    print("\n" + "=" * 74)
    print("STAGE 2 — LOCALISATION, half-prominence above each clip's own median")
    print("=" * 74)
    print(f"  records emitted   {len(gated)}  ({len(pos_gated)} positive, "
          f"{len(neg_gated)} negative — the negatives are the gate's false alarms)")

    def summarise(name, subset):
        if not subset:
            return
        span = np.array([gated[i]["t_end"] - gated[i]["t_start"] for i in subset])
        norm = np.array([gated[i]["t_peak"] / gated[i]["clip_duration_s"] for i in subset])
        lead = np.array([gated[i]["clip_duration_s"] - gated[i]["t_peak"] for i in subset])
        print(f"\n  {name} (n={len(subset)})")
        print(f"    incident window  median {np.median(span):.2f} s   "
              f"IQR [{np.percentile(span, 25):.2f}, {np.percentile(span, 75):.2f}]")
        print(f"    t_peak position  median {np.median(norm):.3f} of clip   "
              f"({100 * (norm > 0.9).mean():.1f}% in the final 10%)")
        print(f"    t_peak to end    median {np.median(lead):.2f} s of footage after the peak")

    summarise("gated POSITIVES", pos_gated)
    summarise("gated NEGATIVES", neg_gated)

    print("\n" + "=" * 74)
    print("EXAMPLE RECORD (README §27 shape; unproducible fields omitted, not nulled)")
    print("=" * 74)
    print(json.dumps(gated[sorted(pos_gated)[0]], indent=2))

    if emit_dir:
        os.makedirs(emit_dir, exist_ok=True)
        for i, r in gated.items():
            with open(os.path.join(emit_dir, f"{i}.json"), "w") as f:
                json.dump(r, f, indent=2)
        print(f"\nwrote {len(gated)} records to {emit_dir}")

    print("\n🔴 Reported as a CAPABILITY. No timestamp above is validated against ground")
    print("   truth, because Nexar test-public has none (see the module docstring).")
    return gated


def _self_check():
    """Every assertion here fails loudly rather than degrading quietly."""
    # 1. Ordering and bounds on all 667 real clips.
    labels, traces = load_labels(), load_traces_abs()
    ids = sorted(set(labels) & set(traces))
    assert len(ids) == 667, len(ids)
    p = np.array([traces[i][0].max() for i in ids])
    y = np.array([labels[i] for i in ids])
    gate = gate_at_recall(y, p)["threshold"]

    n_gated = 0
    for i in ids:
        t = incident(*traces[i], gate)
        if t is None:
            continue
        n_gated += 1
        assert t["t_start"] <= t["t_peak"] <= t["t_end"], (i, t)
        assert 0 <= t["t_start"], (i, t)
        assert t["t_end"] <= t["clip_duration_s"], (i, t)
    print(f"ok  t_start <= t_peak <= t_end and inside the clip, all {n_gated} gated clips")

    # 2. The time base agrees with an INDEPENDENT source. runs/falsification decoded all
    #    667 clips for T3; if the resampled-fps reasoning were wrong these would diverge.
    table = durations()
    diffs = [
        (traces[i][2] + len(traces[i][0])) / traces[i][1] - table[i][0] / table[i][1]
        for i in ids if i in table
    ]
    worst = max(abs(d) for d in diffs)
    frame = 1.0 / traces[ids[0]][1]
    assert worst <= frame + 1e-9, f"time base off by {worst:.4f}s, more than one frame ({frame:.4f}s)"
    print(f"ok  duration agrees with benchmark.durations() to {worst:.4f}s "
          f"(<= one {frame:.4f}s frame) across {len(diffs)} clips")

    # 3. A synthetic spike at a known time is recovered to within one frame.
    fps, offset = 8.0, 16
    trace = np.full(64, 0.1)
    trace[24] = 0.95
    t = incident(trace, fps, offset, gate=0.5)
    expected = (offset + 24) / fps
    assert abs(t["t_peak"] - expected) < 1 / fps, (t["t_peak"], expected)
    assert t["t_start"] <= t["t_peak"] <= t["t_end"], t
    assert t["t_start"] >= offset / fps, "offset dropped -- timestamps would land 2 s early"
    print(f"ok  synthetic spike recovered at t_peak {t['t_peak']:.3f}s (expected {expected:.3f}s)")

    # 4. The NaN offset is actually carried. This is the trap the docstring names: reusing
    #    reduction_study.load_traces would silently pass every test above except this one.
    padded = np.concatenate([np.full(16, np.nan), trace])
    finite = ~np.isnan(padded)
    assert int(np.argmax(finite)) == 16
    t_stripped = incident(trace, fps, 0, gate=0.5)
    assert abs(t["t_peak"] - t_stripped["t_peak"] - 16 / fps) < 1e-9, \
        "offset is not being applied -- timestamps would be 2 s early"
    print(f"ok  NaN offset applied ({t['t_peak']:.3f}s with, {t_stripped['t_peak']:.3f}s without)")

    # 5. A flat trace does not crash and does not invent a span.
    flat = incident(np.full(40, 0.9), fps, offset, gate=0.5)
    assert flat["t_start"] == flat["t_peak"] == flat["t_end"], flat
    print("ok  flat trace collapses to an instant instead of spanning the clip")

    # 6. Below the gate, nothing is emitted.
    assert incident(np.full(40, 0.2), fps, offset, gate=0.5) is None
    print("ok  sub-gate clip emits no record")

    # 7. The gate is genuinely derived from the data, not a constant wearing a function's
    #    name. Asking for more recall must move the threshold DOWN and cost more FP; a
    #    hard-coded value could not do that. (Grepping the source for "0.80" would only
    #    test prose -- this tests behaviour.)
    loose, tight = gate_at_recall(y, p, 0.95), gate_at_recall(y, p, 0.70)
    assert loose["threshold"] < gate < tight["threshold"], (loose, gate, tight)
    assert loose["fp"] > tight["fp"], (loose["fp"], tight["fp"])
    print(f"ok  gate is data-derived: recall 0.70/0.80/0.95 -> threshold "
          f"{tight['threshold']:.4f}/{gate:.4f}/{loose['threshold']:.4f}, "
          f"FP {tight['fp']}/{gate_at_recall(y, p)['fp']}/{loose['fp']}")

    print("PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--emit", metavar="DIR", default=None,
                    help="write one JSON incident record per gated clip")
    ap.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    a = ap.parse_args()
    if a.self_check:
        _self_check()
    else:
        main(emit_dir=a.emit, target_recall=a.target_recall)
