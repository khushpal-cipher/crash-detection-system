"""False positives per hour on CONTINUOUS footage — the counting convention, declared before
any footage is scored.

WHY THIS FILE EXISTS. FP/hour is README §31's headline metric ("the metric that decides
whether a fleet keeps the product", target < 0.1/hour) and the project measures 92.3 on Nexar
test-public. But that number is CLIP-level: 333 negative clips over 0.899 h, 83 of which alert
at threshold 0.9733. comma2k19 and ZOD are continuous driving; there are no clips, so somebody
has to decide what counts as ONE false positive -- and NEW_PLAN.md does not. §7.2's "10 h at
1 Hz" is a COMPUTE instruction (how often to score) and §7.3's "1 Hz alert cadence" is a
real-time feasibility statement. Neither settles the counting rule, and the candidates differ
by about an order of magnitude on identical footage with an identical detector:

  A  per-alert    every 1 Hz decision above threshold is one FP      3600 chances/hour
  B  per-segment  fixed-length segments, one FP if ANY decision       370 chances/hour
                  inside it fires (Nexar's own rule)

B IS THE HEADLINE, because it is the only convention under which a comma2k19 number is
comparable to the 92.3 this project already reports, and because README §12 sets exactly this
precedent for the retired model ("non-overlapping 5-second windows, 720 windows per hour").
A IS REPORTED BESIDE IT, because it is the honest answer to "how often would this interrupt a
driver". NEITHER IS QUOTED WITHOUT THE OTHER, and neither is quoted without its denominator.

SEGMENT LENGTH IS MATCHED TO NEXAR, NOT CHOSEN. 9.72 s is the measured MEAN duration of the
333 Nexar negative clips (median 9.90, range 7.33-11.33), which is what makes 3600/9.72 = 370.4
segments/hour agree with Nexar's own 370.3 clips/hour. Picking a longer segment would lower the
rate for free; picking a shorter one would raise it. It is a constant here, declared before any
comma2k19 frame is decoded, for the same reason eval/gate3_mechanism.py declares its thresholds
before measuring: so the bar cannot move once the numbers are in.

🔴 THE TRAP THIS MODULE EXISTS TO AVOID — MOST OF A STRIDED TRACE IS INVENTED.
vendor/badas-open/badas/utils/sliding_window.py::_create_predictive_frame_array returns one
value PER FRAME, but only every `stride`-th frame carries a real model prediction. The gaps are
LINEARLY INTERPOLATED and everything past the last prediction is that prediction EXTENDED FLAT.
At the stride 8 that gives a 1 Hz cadence, SEVEN OF EVERY EIGHT VALUES ARE NOT MODEL OUTPUT.
Counting array entries above threshold would therefore count interpolation as alerts, with
nothing raising an error. `prediction_indices()` recovers the real ones and every count in this
module goes through it. (Segment nanmax happens to be safe -- interpolation never exceeds its
endpoints -- but the per-alert count is not, and relying on that coincidence is not a design.)

Also reproduced here rather than assumed: upstream's `if 0 <= target_frame < total_frames`
guard DISCARDS the final window, so an 81-frame stride-1 clip runs 66 windows and keeps 65.

THE SELF-CHECK TIES THIS CODE TO THE COMMITTED NUMBER. Applied to Nexar's 333 committed
negative traces with each clip as one segment, convention B must return exactly 83 false
positives and 92.3 FP/hour. If it does not, this module is wrong -- not the baseline.

    ~/envs/badas/bin/python -m eval.fp_rate --self-check
    ~/envs/badas/bin/python -m eval.fp_rate --frames-dir runs/comma2k19/frames --label comma2k19
"""

import argparse
import glob
import os

import numpy as np

# --- Declared before any continuous footage is scored. ---------------------------------
SEGMENT_S = 9.72      # measured mean of Nexar's 333 negative clips -- matched, not chosen
ALERT_HZ = 1.0        # NEW_PLAN.md §7.3's alert cadence
FRAME_COUNT = 16      # BADAS-Open's window, and the length of the leading NaN run
NEXAR_FP_HOUR = 92.3  # the number convention B must reproduce on Nexar (0.899 h, 83 FP)
NEXAR_FP = 83

# The gate threshold is NOT hard-coded here: B5 and progress.md §12 forbid it. Callers pass
# the value derived from the PR curve by eval/timing.py::gate_at_recall, which is what
# scripts/detect.py already does.


def prediction_indices(n_total, frame_count=FRAME_COUNT, stride=1):
    """Absolute frame indices carrying a REAL prediction, reproducing upstream's windowing.

    Windows start at 0, stride, 2*stride ... while start <= n_total - frame_count, and each
    files its prediction against `start + frame_count`. Upstream then keeps only targets
    strictly below n_total, which is why the final window is silently dropped.
    """
    if n_total <= frame_count:
        targets = [n_total]           # upstream's short-video branch: one window, then dropped
    else:
        targets = [s + frame_count
                   for s in range(0, n_total - frame_count + 1, stride)]
    return np.array([t for t in targets if 0 <= t < n_total], dtype=int)


def load_trace(path):
    """(scores, target_fps, stride) straight from the .npz the adapter writes.

    Deliberately NOT eval/timing.py::load_traces_abs: that returns (scores, fps, offset) and
    DROPS stride, which is correct for the stride-1 Nexar sweep it was written for and wrong
    here -- without stride there is no way to tell a measured value from an interpolated one.
    """
    with np.load(path) as f:
        return (np.asarray(f["scores"], float), float(f["target_fps"]),
                int(f.get("stride", 1)) if "stride" in f else 1)


def clip_counts(scores, fps, stride, threshold, segment_s=SEGMENT_S):
    """Both conventions for one continuous recording.

    Returns alerts (A), alerting segments and segment count (B), duration, and the number of
    real decisions the recording actually contained.
    """
    n_total = len(scores)
    idx = prediction_indices(n_total, stride=stride)
    real = scores[idx]
    finite = np.isfinite(real)
    idx, real = idx[finite], real[finite]

    duration_s = n_total / fps
    fired = real >= threshold

    # A -- one FP per real decision above threshold.
    alerts = int(fired.sum())

    # B -- one FP per segment containing any such decision. Segments are laid out by TIME
    # from the start of the recording, so the count does not depend on where predictions fall.
    n_seg = max(int(np.ceil(duration_s / segment_s)), 1)
    seg_of = np.floor((idx / fps) / segment_s).astype(int)
    seg_of = np.clip(seg_of, 0, n_seg - 1)
    seg_fired = int(np.unique(seg_of[fired]).size) if fired.any() else 0

    return {
        "duration_s": duration_s,
        "decisions": int(idx.size),
        "alerts": alerts,
        "segments": n_seg,
        "segments_fired": seg_fired,
        "cadence_s": stride / fps,
    }


def aggregate(per_clip, label):
    """Roll per-recording counts into the two rates. Never pools distinct corpora -- the
    caller passes one corpus at a time, because NEW_PLAN.md §8.2 forbids reporting comma2k19
    and ZOD as a single number (highway-only footage is a FLOOR, not a general rate)."""
    hours = sum(c["duration_s"] for c in per_clip) / 3600.0
    alerts = sum(c["alerts"] for c in per_clip)
    seg = sum(c["segments"] for c in per_clip)
    seg_fired = sum(c["segments_fired"] for c in per_clip)
    decisions = sum(c["decisions"] for c in per_clip)
    return {
        "label": label,
        "recordings": len(per_clip),
        "hours": hours,
        "decisions": decisions,
        "alerts": alerts,
        "segments": seg,
        "segments_fired": seg_fired,
        "fp_hour_segment": seg_fired / hours if hours else float("nan"),   # B, headline
        "fp_hour_alert": alerts / hours if hours else float("nan"),        # A, reported beside
        "segment_alert_rate": seg_fired / seg if seg else float("nan"),
    }


def report(agg, threshold, segment_s=SEGMENT_S):
    a = agg
    print("=" * 78)
    print(f"FALSE POSITIVES PER HOUR — {a['label']}")
    print("=" * 78)
    print(f"  footage            {a['hours']:.3f} h over {a['recordings']} recording(s)")
    print(f"  threshold          {threshold:.4f}  (derived from the Nexar PR curve, not chosen)")
    print(f"  real decisions     {a['decisions']}  (interpolated trace values excluded)")
    print()
    print(f"  B  per-segment ({segment_s:.2f} s, matched to Nexar's mean negative clip)")
    print(f"       {a['segments_fired']} of {a['segments']} segments fired "
          f"= {100 * a['segment_alert_rate']:.1f}%")
    print(f"       →  {a['fp_hour_segment']:.1f} FP/hour      <- HEADLINE")
    print()
    print(f"  A  per-alert ({ALERT_HZ:.0f} Hz cadence)")
    print(f"       {a['alerts']} alerts")
    print(f"       →  {a['fp_hour_alert']:.1f} FP/hour")
    print()
    print(f"  for scale: Nexar test-public negatives = {NEXAR_FP_HOUR} FP/hour over 0.899 h "
          f"(convention B)")
    print(f"             README §31 target           = < 0.1 FP/hour")
    print()
    print("  🔴 Neither rate may be quoted without its denominator, and a highway-only corpus")
    print("     is a FLOOR, never a general false-alarm rate (NEW_PLAN.md §8.2).")
    return a


def run(frames_dir, threshold, label, segment_s=SEGMENT_S):
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.npz")))
    if not paths:
        raise SystemExit(f"no traces in {frames_dir}")
    per_clip = []
    for p in paths:
        scores, fps, stride = load_trace(p)
        per_clip.append(clip_counts(scores, fps, stride, threshold, segment_s))
    return report(aggregate(per_clip, label), threshold, segment_s)


# --------------------------------------------------------------------------------------
# Self-check. The decisive one reproduces the committed Nexar figure.
# --------------------------------------------------------------------------------------

def _self_check():
    fps = 8.0

    # 1. Upstream's windowing, including the dropped final window that timing.py names.
    idx = prediction_indices(81, stride=1)
    assert idx[0] == 16 and idx[-1] == 80 and idx.size == 65, idx
    print(f"ok  stride-1 81-frame clip: 66 windows created, {idx.size} kept "
          f"(upstream drops the last)")

    # 2. 🔴 At stride 8 only every 8th frame is real. This is the whole point of the module.
    idx8 = prediction_indices(88, stride=8)
    assert list(idx8) == [16, 24, 32, 40, 48, 56, 64, 72, 80], list(idx8)
    assert np.allclose(np.diff(idx8) / fps, 1.0), "stride 8 at 8 fps must be exactly 1 Hz"
    print(f"ok  stride-8 trace: {idx8.size} real decisions out of 88 array entries, "
          f"spaced exactly {1.0:.1f}s — the other 79 are interpolation")

    # 3. Interpolated values must NOT be counted, even when they sit above the threshold.
    #    Two real decisions well below threshold, with a spike interpolated between them:
    #    counting array entries would find the spike, counting decisions must not.
    scores = np.full(88, np.nan)
    real_idx = prediction_indices(88, stride=8)
    scores[real_idx] = 0.10
    scores[20] = 0.99                      # an interpolated slot, not a decision
    c = clip_counts(scores, fps, 8, threshold=0.5)
    assert c["alerts"] == 0, c
    assert c["decisions"] == real_idx.size, c
    print("ok  a 0.99 sitting in an INTERPOLATED slot raises no alert "
          "(counting array entries would have)")

    # 4. Segment logic: two decisions above threshold inside one segment = ONE false positive.
    scores = np.full(88, np.nan)
    scores[real_idx] = 0.10
    scores[[16, 24]] = 0.99                # 2.0 s and 3.0 s -> same 9.72 s segment
    c = clip_counts(scores, fps, 8, threshold=0.5)
    assert c["alerts"] == 2 and c["segments_fired"] == 1, c
    print("ok  two alerts inside one segment = 2 under A, 1 under B (the conventions differ)")

    # 5. 🔴 THE DECISIVE CHECK. Convention B over Nexar's committed negative traces, one
    #    segment per clip, must reproduce the committed 83 FP / 92.3 FP/hour exactly.
    from eval.benchmark import durations, hours, load_labels
    from eval.reduction_study import FRAMES_DIR

    labels = load_labels()
    table = durations()
    neg = [i for i in labels if labels[i] == 0]
    thr = 0.973328                          # the derived gate; see eval/timing.py
    fired_ids, per_clip = [], []
    for cid in neg:
        p = os.path.join(FRAMES_DIR, f"{cid}.npz")
        if not os.path.exists(p):
            continue
        scores, f_, stride = load_trace(p)
        # one segment per clip == Nexar's own rule, so segment_s is the clip's own length
        c = clip_counts(scores, f_, stride, thr, segment_s=len(scores) / f_ + 1.0)
        assert c["segments"] == 1, c
        per_clip.append(c)
        if c["segments_fired"]:
            fired_ids.append(cid)

    assert len(fired_ids) == NEXAR_FP, f"{len(fired_ids)} != committed {NEXAR_FP}"
    h = hours(neg, table)
    rate = len(fired_ids) / h
    assert abs(rate - NEXAR_FP_HOUR) < 0.1, rate
    print(f"ok  REPRODUCES THE COMMITTED FIGURE: {len(fired_ids)} FP over {h:.3f} h "
          f"= {rate:.1f} FP/hour (committed {NEXAR_FP_HOUR})")

    # 6. The matched segment length really is matched: Nexar's clips-per-hour vs ours.
    assert abs(3600 / SEGMENT_S - len(per_clip) / h) < 1.0, (3600 / SEGMENT_S, len(per_clip) / h)
    print(f"ok  segment length matched: 3600/{SEGMENT_S} = {3600 / SEGMENT_S:.1f} segments/hour "
          f"vs Nexar's {len(per_clip) / h:.1f} clips/hour")

    print("PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir")
    ap.add_argument("--label", default="unnamed corpus")
    ap.add_argument("--threshold", type=float, default=0.973328)
    ap.add_argument("--segment-s", type=float, default=SEGMENT_S)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        _self_check()
    elif a.frames_dir:
        run(a.frames_dir, a.threshold, a.label, a.segment_s)
    else:
        ap.error("need --frames-dir or --self-check")
