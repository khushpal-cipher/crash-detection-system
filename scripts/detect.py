"""Video in -> JSON incident record out. README §27's MVP, the first rung of its product ladder.

README §27 is explicit that the product is NOT crash detection -- BADAS-Open is free,
Apache-2.0 and state of the art, so the detector is a commodity from day one. The product is
the structured incident record. The product ladder puts "CLI + API: video in -> JSON incident
records out" at weeks 1-4, and until this file every component of it existed and was
validated while the thing itself did not:

    eval/adapters.py     BadasOpen.score()        per-frame trace      ✅ committed, swept 667 clips
    eval/timing.py       gate_at_recall()         derived threshold    ✅ 7/7 self-check
    eval/timing.py       incident()               t_start/peak/end     ✅ 7/7 self-check
    eval/calibration.py  fit_beta() on tier 1     usable probability   ✅ ECE 0.33 -> ~0.05
    scripts/verify_...   sha256                   the evidence seal    ✅ 220/220 verified

This file is the glue and nothing else. It adds no model, no metric and no new science. Every
number it emits is produced by code that already had a passing self-check before it was
written, which is why it is short.

NOTHING IN eval/ IS MODIFIED. eval/{timing,adapters,benchmark,calibration}.py are four of the
five MPS regression guards; they reproduce exactly and this file is not worth risking that for.
It imports them and reduces with np.nanmax, the shipped reduction.

    R1 IS DEAD (gate 3a, 2026-09-18). Last-window is worth +0.0556 AP on Nexar but the gate
    found the peak does not track the annotated collision beyond what clip duration already
    explains, so the mechanism claim did not survive. adapters.py reduces with nanmax and so
    does this. Do not "promote" last-window here.

WHAT THE POLICY IS, AND WHY IT IS DERIVED RATHER THAN TYPED
Two numbers govern every record: the gate threshold and the calibration map. Both are fitted
artefacts of the Nexar benchmark, and B5 / progress.md §12 forbid hard-coded thresholds -- 0.80
recall is a stated target, not a chosen threshold, and the threshold that delivers it is read
off the PR curve. The map is the §6 TIER 1 map: fitted on the stratified calibration half with
seed 0, which is the only tier NEW_PLAN.md permits to be called deployable. Both are recomputed
at startup from committed data (milliseconds) and BOTH ARE EMBEDDED IN EVERY RECORD, so a
record can always be audited against the policy that produced it.

CALIBRATION CANNOT RESCUE THE FALSE-ALARM RATE, and this must not be misread. Every calibrator
in eval/calibration.py is monotone, so it cannot change AP, AUC, ranking, or FP/hour at matched
recall (D28). At the shipped operating point this system produces 92.3 FP/hour over a 0.899 h
denominator against a < 0.1 FP/hour target. The calibrated number in `confidence` is a usable
probability, not a fixed false-alarm rate.

FIELDS THAT CANNOT BE PRODUCED ARE OMITTED, NEVER NULL-FILLED, following eval/timing.py::record.
A null implies a field that could be populated and merely is not, which misrepresents what the
system knows. Omitted here and why:
  * ego_involved     Nexar ships no ego field (D23) and nothing in this pipeline infers one.
  * severity band    §27 wants bands derived from MEASURED physical quantities -- closing
                     speed, peak longitudinal accel. There is no physical channel. Deriving a
                     band from the detector score would be severity PREDICTION, which §44
                     forbids outright. It is omitted rather than invented.
  * closing speed / peak accel / GPS        no physical channel, no IMU, no GPS on this input.
  * vehicle / driver                        fleet metadata, supplied by the caller, not inferred.

TIMING IS A CAPABILITY, NOT A MEASUREMENT. Nexar's `time_of_event` is corrupt for all 334
positives (median 20.0 s against a 9.93 s clip), so there is no ground truth on this benchmark
against which a timestamp could be validated. `timing_validated: false` rides in every record
and is not decoration -- progress.md §12 and NEW_PLAN.md §12 both forbid an mTTA or
time-to-detection claim here.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ~/envs/badas/bin/python scripts/detect.py --self-check
    PYTORCH_ENABLE_MPS_FALLBACK=1 ~/envs/badas/bin/python scripts/detect.py \
        videos/safe.mp4 --out runs/incidents
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.adapters import BadasOpen  # noqa: E402
from eval.calibration import fit_beta, load_scores, tier1  # noqa: E402
from eval.timing import gate_at_recall, incident, load_traces_abs  # noqa: E402

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")

# §27 asks for a 45 s evidence clip. Split around the incident rather than centred on it:
# a reviewer needs the approach far more than the aftermath.
EVIDENCE_PRE_S = 30.0
EVIDENCE_POST_S = 15.0

SCHEMA = "incident-record/0.1"


def sha256(path, chunk=1 << 20):
    """Seal the SOURCE video, not a re-encoded cut.

    §27 asks for "a 45 s evidence clip, sealed with a hash + timestamp". ffmpeg is not
    installed in this environment and cv2's writer would re-encode, so a cut clip's hash
    would seal a lossy derivative rather than the original. Hashing the source and naming
    the window into it is both cheaper and stronger evidence: the bytes a court or an
    insurer would be shown are the ones under the hash.
    ponytail: cut the clip too once ffmpeg is available -- the window is already recorded,
    so that is an extraction step, not a redesign.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def policy(target_recall=0.80, seed=0):
    """The gate threshold and the deployable calibration map, derived from committed data.

    Returns a dict carrying both the artefacts and the provenance needed to audit a record.
    The map is NEW_PLAN.md §6 Tier 1: fitted on the stratified calibration half only, which
    is the sole tier whose calibrated numbers may be quoted as a deployed result. Fitting on
    all 667 would produce a slightly better map with no honest evaluation half left to
    support it.

    ponytail: recomputed per process (~1 s) rather than frozen to a policy file. Freeze it if
    runs/baselines/ ever moves -- every record already embeds the threshold, so a drift would
    be visible rather than silent.
    """
    ids, y, p = load_scores("nanmax")
    g = gate_at_recall(y, p, target_recall)
    cal, ev, _ = tier1(y, p, seed=seed)
    calibrate = fit_beta(p[cal], y[cal])
    return {
        "threshold": g["threshold"],
        "calibrate": calibrate,
        "provenance": {
            "gate": {
                "target_recall": g["target_recall"],
                "threshold": round(g["threshold"], 6),
                "delivers_recall": round(g["recall"], 4),
                "delivers_precision": round(g["precision"], 4),
                "fitted_on": "nexar/test-public n=667 (nanmax)",
            },
            "calibration": {
                "method": "beta",
                "tier": 1,
                "fitted_on": f"nexar/test-public stratified calibration half, "
                             f"n={len(cal)}, seed={seed}",
            },
        },
    }


def _source_path(video_path):
    """Repo-relative when the video lives under the repo, absolute otherwise, so a record
    committed here stays meaningful on another machine."""
    p = os.path.abspath(video_path)
    return os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p


def evidence_window(t_start, t_end, duration):
    """The §27 evidence span, clamped inside the video. Returns (start, end) seconds."""
    lo = max(0.0, t_start - EVIDENCE_PRE_S)
    hi = min(duration, t_end + EVIDENCE_POST_S)
    return round(lo, 3), round(hi, 3)


def build_record(clip_id, video_path, trace, fps, offset, pol, detector):
    """A README §27 incident record, or None if the clip does not clear the gate.

    Returning None below the gate is the product behaviour, not a failure: a system that
    emits a record for every video has not triaged anything.
    """
    t = incident(trace, fps, offset, pol["threshold"])
    if t is None:
        return None

    raw = float(np.max(trace))
    calibrated = float(pol["calibrate"](np.array([raw]))[0])
    ev_start, ev_end = evidence_window(t["t_start"], t["t_end"], t["clip_duration_s"])

    return {
        "schema": SCHEMA,
        "event_id": clip_id,
        "source": _source_path(video_path),
        "detector": detector,
        "t_start": round(t["t_start"], 3),
        "t_peak": round(t["t_peak"], 3),
        "t_end": round(t["t_end"], 3),
        "clip_duration_s": round(t["clip_duration_s"], 3),
        "confidence": {
            "score": round(raw, 4),
            "calibrated": round(calibrated, 4),
            "reduction": "nanmax",
        },
        "evidence": {
            "source_sha256": sha256(video_path),
            "window_start_s": ev_start,
            "window_end_s": ev_end,
            "window_duration_s": round(ev_end - ev_start, 3),
            "clip_extracted": False,  # see sha256() -- the source is the sealed artefact
        },
        "human_review_status": "pending",
        "timing_validated": False,
        "policy": pol["provenance"],
    }


def run(videos, out_dir, target_recall=0.80, device="mps", stride=1):
    pol = policy(target_recall)
    g = pol["provenance"]["gate"]
    print(f"policy  threshold {g['threshold']:.4f} at target recall {g['target_recall']:.2f}"
          f"  (delivers recall {g['delivers_recall']:.4f}, "
          f"precision {g['delivers_precision']:.4f})")
    print(f"        calibration: {pol['provenance']['calibration']['fitted_on']}\n")

    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(out_dir, exist_ok=True)
    model = BadasOpen(device=device, stride=stride, save_frames_dir=frames_dir).load()
    detector = f"{model.name} nanmax"

    scored, summary = {}, []
    for n, path in enumerate(videos, 1):
        clip_id = os.path.splitext(os.path.basename(path))[0]
        t0 = time.time()
        try:
            s = model.score(path)
        except Exception as e:  # one unreadable video must not end a batch
            print(f"  {n}/{len(videos)}  {clip_id}  FAILED {type(e).__name__}: {e}")
            summary.append({"id": clip_id, "error": f"{type(e).__name__}: {e}"})
            continue
        if s is None:
            print(f"  {n}/{len(videos)}  {clip_id}  unscorable (too short for one window)")
            summary.append({"id": clip_id, "error": "unscorable"})
            continue
        scored[clip_id] = path
        print(f"  {n}/{len(videos)}  {clip_id}  score {s:.4f}  {time.time() - t0:.1f}s")

    traces = load_traces_abs(frames_dir)
    records = []
    for clip_id, path in scored.items():
        if clip_id not in traces:
            summary.append({"id": clip_id, "error": "trace missing"})
            continue
        trace, fps, offset = traces[clip_id]
        rec = build_record(clip_id, path, trace, fps, offset, pol, detector)
        if rec is None:
            summary.append({"id": clip_id, "incident": False,
                            "score": round(float(np.max(trace)), 4)})
            continue
        dest = os.path.join(out_dir, f"{clip_id}.json")
        with open(dest, "w") as f:
            json.dump(rec, f, indent=2)
        records.append(rec)
        summary.append({"id": clip_id, "incident": True,
                        "score": rec["confidence"]["score"],
                        "calibrated": rec["confidence"]["calibrated"],
                        "record": os.path.basename(dest)})

    with open(os.path.join(out_dir, "summary.jsonl"), "w") as f:
        for row in summary:
            f.write(json.dumps(row) + "\n")

    print(f"\n{len(records)} incident record(s) from {len(videos)} video(s) -> {out_dir}/")
    if records:
        print("\nexample record:")
        print(json.dumps(records[0], indent=2))
    return records


# --------------------------------------------------------------------------------------
# Self-check: synthetic traces only. No MPS, no video decode, no downloaded data.
# --------------------------------------------------------------------------------------

def _self_check():
    import tempfile

    fps, off = 8.0, 16
    pol = policy()
    thr = pol["threshold"]
    assert 0.0 < thr < 1.0, thr
    print(f"ok  policy derived, not typed: threshold {thr:.4f} at target recall 0.80")

    # 1. The calibration map is MONOTONE, so it cannot reorder clips and therefore cannot
    #    change AP, AUC or FP/hour at matched recall. This is D28 as an assertion rather
    #    than a claim in a docstring.
    probe = np.linspace(0.001, 0.999, 200)
    mapped = pol["calibrate"](probe)
    assert np.all(np.diff(mapped) >= -1e-9), "calibration map is not monotone"
    print(f"ok  calibration monotone ({mapped[0]:.4f} -> {mapped[-1]:.4f}): "
          f"cannot change ranking, AP, or FP/hour at matched recall (D28)")

    # 2. Below the gate -> NO record. A system that emits a record per video triages nothing.
    quiet = np.full(64, max(thr - 0.2, 0.01))
    assert build_record("q", __file__, quiet, fps, off, pol, "test") is None
    print("ok  below the gate no record is emitted (the product behaviour, not a failure)")

    # 3. Above the gate -> a record whose timestamps are ordered and carry the NaN offset.
    #    Dropping that offset would place every timestamp 2 s early with nothing crashing.
    loud = np.full(64, 0.05)
    loud[40] = 0.99
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fh:
        fh.write(b"not a real video, but real bytes to hash")
        fake = fh.name
    try:
        r = build_record("loud", fake, loud, fps, off, pol, "test")
        assert r is not None
        assert r["t_start"] <= r["t_peak"] <= r["t_end"], r
        assert abs(r["t_peak"] - (off + 40) / fps) < 1e-9, r["t_peak"]
        assert abs(r["clip_duration_s"] - (off + 64) / fps) < 1e-9, r
        print(f"ok  gated record: t_peak {r['t_peak']:.3f}s carries the {off}-frame "
              f"({off / fps:.1f}s) offset")

        # 4. Unproducible fields are ABSENT, not null. A null would claim the field is
        #    merely empty rather than outside what this pipeline can know.
        for absent in ("ego_involved", "severity", "severity_band",
                       "closing_speed_mps", "gps"):
            assert absent not in r, f"{absent} must be omitted, not null-filled"
        assert r["timing_validated"] is False
        print("ok  ego_involved / severity / closing speed / gps omitted, not null-filled")

        # 5. The evidence window is clamped inside the video and sealed against the source.
        assert r["evidence"]["window_start_s"] >= 0.0
        assert r["evidence"]["window_end_s"] <= r["clip_duration_s"] + 1e-9
        assert r["evidence"]["source_sha256"] == sha256(fake)
        print(f"ok  evidence window [{r['evidence']['window_start_s']:.3f}, "
              f"{r['evidence']['window_end_s']:.3f}]s clamped inside the video, "
              f"sha256 {r['evidence']['source_sha256'][:12]}…")

        # 6. The seal actually seals: different bytes, different hash.
        before = sha256(fake)
        with open(fake, "ab") as f:
            f.write(b"tampered")
        assert sha256(fake) != before, "the seal did not detect modified bytes"
        print("ok  seal detects modified source bytes")
    finally:
        os.unlink(fake)

    # 7. Every record carries the policy that produced it, so it can be audited later.
    assert r["policy"]["gate"]["threshold"] == round(thr, 6)
    assert r["policy"]["calibration"]["tier"] == 1
    print("ok  record embeds its own gate threshold and calibration provenance")

    print("PASS")


def main():
    ap = argparse.ArgumentParser(
        description="Video in -> README §27 JSON incident record out.")
    ap.add_argument("videos", nargs="*", help="video files, or directories of them")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "incidents"))
    ap.add_argument("--recall", type=float, default=0.80,
                    help="TARGET recall; the threshold is read off the PR curve, never typed")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()

    if a.self_check:
        _self_check()
        return
    if not a.videos:
        ap.error("need at least one video, or --self-check")

    videos = []
    for v in a.videos:
        if os.path.isdir(v):
            videos += [os.path.join(v, f) for f in sorted(os.listdir(v))
                       if f.endswith(VIDEO_EXT)]
        else:
            videos.append(v)
    if not videos:
        ap.error("no videos found")

    run(videos, a.out, target_recall=a.recall, device=a.device, stride=a.stride)


if __name__ == "__main__":
    main()
