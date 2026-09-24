"""The September 30 demo: one command, the model judges a video, you watch it happen.

docs/sept30_demo.md is the plan this implements. Read §A-§D before changing anything here.

WHY TWO PASSES, AND WHY THAT IS NOT A COMPROMISE
BADAS cannot score video in real time on this Mac. MEASURED 2026-09-24 (docs/sept30_demo.md
§E): 1.33-1.58 s per window, where a window advances one frame on an 8 fps timeline.

    crash1.mov   7.38 s of video ->  43 windows ->  57.3 s to score
    safe.mp4    29.75 s of video -> 222 windows -> 351.2 s to score

So the video cannot play at normal speed while it is being scored. Any design that pretends
otherwise either stutters for minutes or quietly fakes the detection. This file therefore
does the honest thing:

    PASS 1   score the video, printing per-window progress so the viewer watches real work
    PASS 2   replay it at normal speed with the score overlaid and the alert region marked

Nothing is pre-baked. The command does the computation in front of the audience, and only
then replays. NEVER add a path that replays scores computed in an earlier run while implying
it is live -- that is the "fake functionality presented as real" failure, and it is the one
thing this demo cannot survive being caught doing.

THIS FILE IS GLUE. IT ADDS NO MODEL, NO METRIC AND NO THRESHOLD.
It imports eval/adapters.py, eval/timing.py and eval/calibration.py UNCHANGED, exactly as
scripts/detect.py does, and it reuses detect.py's own policy() rather than re-deriving a
threshold beside it. Four of the five regression guards live in eval/; nothing in a demo is
worth risking them for. The gate is DERIVED at runtime from committed Nexar scores (B5 /
progress.md §12) and is never typed here -- grep this file for 0.97 and you will not find it.

HOW THE PROGRESS COUNTER WORKS WITHOUT TOUCHING ANY COMMITTED FILE
The sliding-window loop lives in vendor/badas-open/badas/utils/sliding_window.py:124 and
calls its preprocess_fn once per window (line 133), which calls the model's `processor`.
So wrapping `model._model.processor` with a counting proxy AT RUNTIME yields an exact
per-window tick from the outside. No edit to eval/, no edit to vendor/. If the processor is
absent the demo degrades to an elapsed-time line rather than failing.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ~/envs/badas/bin/python scripts/demo.py --self-check
    PYTORCH_ENABLE_MPS_FALLBACK=1 ~/envs/badas/bin/python scripts/demo.py videos/crash1.mov
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval.timing import incident, load_traces_abs  # noqa: E402
from detect import build_record, policy  # noqa: E402  -- reuse, never re-derive

# Anthropic brand palette, as BGR because that is what cv2 speaks.
DARK = (19, 20, 20)        # #141413
LIGHT = (245, 249, 250)    # #faf9f5
ORANGE = (87, 119, 217)    # #d97757
GREY = (120, 120, 118)

FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_FRAMES = 16         # upstream's frame_count: the leading NaN run
TARGET_FPS = 8.0


# ---------------------------------------------------------------------------------------
# Pass 1 -- score, with visible per-window progress
# ---------------------------------------------------------------------------------------

class _CountingProcessor:
    """Forwards every call to the real processor and ticks a counter.

    Wraps rather than replaces: vendor code checks `if self.processor` and
    `hasattr(self.processor, "__call__")`, so this must stay truthy and callable. It changes
    no pixel and no score -- it only counts.
    """

    def __init__(self, inner, on_tick):
        self._inner = inner
        self._on_tick = on_tick
        self.count = 0

    def __call__(self, *a, **kw):
        self.count += 1
        self._on_tick(self.count)
        return self._inner(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def probe_video(path):
    """(duration_s, native_fps, frame_count_read). Reads the container, cheaply.

    🔴 The frame count in a container header can LIE. videos/crash1.mov claims 259 frames
    and yields 257; videos/crash2.mov claims 187 and yields 177 (docs/sept30_demo.md §E).
    That is why every window total this file prints is marked with a tilde.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    if fps <= 0:
        raise SystemExit(f"video reports no frame rate: {path}")
    return n / fps, fps, n


def estimate_windows(duration_s, stride):
    """Roughly how many windows pass 1 will do. APPROXIMATE -- see probe_video()."""
    frames_at_8 = int(duration_s * TARGET_FPS)
    return max(1, (frames_at_8 - WINDOW_FRAMES) // max(stride, 1) + 1)


def _bar(done, total, width=28):
    filled = min(width, int(width * done / max(total, 1)))
    return "#" * filled + "." * (width - filled)


def score_with_progress(video, device, stride, frames_dir, quiet=False):
    """PASS 1. Returns (clip_score, elapsed_s, windows_counted, detector_name).

    The model is constructed and loaded here rather than by the caller so that the ~14 s
    load is inside the progress display and does not look like a hang.
    """
    from eval.adapters import BadasOpen

    duration_s, _, _ = probe_video(video)
    expected = estimate_windows(duration_s, stride)

    print("  loading the model (V-JEPA2 ViT-L, ~14 s) ... ", end="", flush=True)
    t_load = time.time()
    model = _load_quietly(BadasOpen(device=device, stride=stride,
                                    save_frames_dir=frames_dir), quiet)
    print(f"ready in {time.time() - t_load:.1f}s\n")

    t0 = time.time()

    def tick(n):
        # The estimate comes from a container header, which can lie (see probe_video), so
        # the real count can exceed it. Grow the denominator rather than show 44/~43.
        total = max(expected, n)
        el = time.time() - t0
        left = max(total - n, 0) * (el / max(n, 1))
        print(f"\r  window {n:4d}/~{total:<4d} [{_bar(n, total)}] "
              f"{el:5.1f}s elapsed, ~{left:4.0f}s left ", end="", flush=True)

    inner = getattr(model._model, "processor", None)
    counter = None
    if inner is not None:
        counter = _CountingProcessor(inner, tick)
        model._model.processor = counter
    else:  # degrade rather than fail -- progress is presentation, not science
        print("  (no processor to hook; scoring without a window counter)")

    score = model.score(video)
    elapsed = time.time() - t0
    counted = counter.count if counter else 0
    print(f"\r  scored {counted} windows in {elapsed:.1f}s"
          f" ({elapsed / max(counted, 1):.2f}s per window){' ' * 30}")
    return score, elapsed, counted, f"{model.name} nanmax"


def _load_quietly(adapter, quiet):
    """Load the model with the transformers LOAD REPORT suppressed.

    That report is ~35 lines of UNEXPECTED-key warnings on every start. They are expected --
    the checkpoint replaces the pooler and classifier head -- but docs/sept30_demo.md §F
    forbids warning spam on screen, and a fleet manager reading 'UNEXPECTED' 35 times during
    a demo is a question you do not want to be answering.
    """
    if not quiet:
        return adapter.load()
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            import transformers
            transformers.logging.set_verbosity_error()
        except Exception:
            pass
        return adapter.load()


# ---------------------------------------------------------------------------------------
# Pass 2 -- replay with the score overlaid
# ---------------------------------------------------------------------------------------

def score_at(trace, offset, t_seconds):
    """The score the model gave for wall-clock second `t_seconds`, or None before it had one.

    The trace lives on the 8 fps resampled timeline with its leading NaN run stripped and
    `offset` remembering where it started (eval/timing.py::load_traces_abs). The first
    WINDOW_FRAMES frames have no prediction -- the model needs a past before it can judge a
    future -- and returning None for them is the honest rendering. Filling them with 0.0
    would draw a confident 'no crash' the model never actually said.
    """
    idx = int(round(t_seconds * TARGET_FPS)) - offset
    if idx < 0:
        return None
    return float(trace[min(idx, len(trace) - 1)])


def draw_overlay(frame, t, score, gate, inc, duration, trace, offset):
    """Draw the judgement onto one video frame. Returns the frame, modified in place.

    🔴 EVERY size here scales with frame height. Dashcam footage runs from 720p to 4K --
    videos/crash1.mov is 3408x1910 -- and fixed font sizes render as unreadable specks on
    the big ones, which is exactly the "nothing on screen that needs excusing" failure in
    docs/sept30_demo.md §F. 1080 is the reference height, not a magic number.
    """
    h, w = frame.shape[:2]
    k = h / 1080.0                      # one scale factor for every size below
    fs = lambda x: max(0.35, x * k)     # font scale
    th = lambda x: max(1, int(round(x * k)))
    alerting = inc is not None and inc["t_start"] <= t <= inc["t_end"]

    panel_h = int(h * 0.16)
    strip_h = int(h * 0.15)
    pad = int(28 * k)
    cv2.rectangle(frame, (0, 0), (w, panel_h), DARK, -1)
    cv2.rectangle(frame, (0, h - strip_h), (w, h), DARK, -1)

    # --- top panel: the number, and the gate it is being judged against -----------------
    if score is None:
        cv2.putText(frame, "warming up", (pad, int(panel_h * 0.55)),
                    FONT, fs(1.4), GREY, th(3), cv2.LINE_AA)
        cv2.putText(frame, "the model needs 2 seconds of past before it can judge",
                    (pad, int(panel_h * 0.85)), FONT, fs(0.7), GREY, th(2), cv2.LINE_AA)
    else:
        colour = ORANGE if score >= gate else LIGHT
        cv2.putText(frame, f"{score:.4f}", (pad, int(panel_h * 0.58)),
                    FONT, fs(2.0), colour, th(4), cv2.LINE_AA)
        cv2.putText(frame, f"alert above {gate:.4f}", (pad, int(panel_h * 0.86)),
                    FONT, fs(0.8), GREY, th(2), cv2.LINE_AA)

        # score bar, with the gate drawn on it as a line rather than described in words
        bx = int(w * 0.46)
        bw = w - bx - pad
        bh = int(34 * k)
        by = int(panel_h * 0.30)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), GREY, th(2))
        cv2.rectangle(frame, (bx, by), (bx + int(bw * min(score, 1.0)), by + bh),
                      colour, -1)
        gx = bx + int(bw * gate)
        cv2.line(frame, (gx, by - th(8)), (gx, by + bh + th(8)), LIGHT, th(3), cv2.LINE_AA)
        # above the bar, not below: below collides with the INCIDENT DETECTED label
        cv2.putText(frame, "alert line", (gx - int(58 * k), by - int(12 * k)),
                    FONT, fs(0.6), LIGHT, th(2), cv2.LINE_AA)

    if alerting:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), ORANGE, th(10))
        label = "INCIDENT DETECTED"
        (tw, _), _ = cv2.getTextSize(label, FONT, fs(1.3), th(3))
        cv2.putText(frame, label, (w - tw - pad, int(panel_h * 0.86)),
                    FONT, fs(1.3), ORANGE, th(3), cv2.LINE_AA)

    # --- bottom strip: the whole trace at once, with the alert span and a playhead -------
    sx, sw = pad, w - 2 * pad
    base_y = h - int(strip_h * 0.30)
    top_y = h - int(strip_h * 0.88)
    span = max(duration, 1e-6)

    if inc is not None:  # shade the alert region so its width is visible, not just its peak
        ax = sx + int(sw * inc["t_start"] / span)
        bx2 = sx + int(sw * inc["t_end"] / span)
        cv2.rectangle(frame, (ax, top_y), (max(bx2, ax + 2), base_y), (44, 58, 84), -1)

    gy = int(base_y - (base_y - top_y) * gate)
    cv2.line(frame, (sx, gy), (sx + sw, gy), GREY, th(2), cv2.LINE_AA)

    pts = []
    for i in range(sw):
        v = score_at(trace, offset, span * i / sw)
        if v is not None:
            pts.append((sx + i, int(base_y - (base_y - top_y) * min(v, 1.0))))
    if len(pts) > 1:
        cv2.polylines(frame, [np.array(pts, np.int32)], False, LIGHT, th(3), cv2.LINE_AA)

    px = sx + int(sw * min(t / span, 1.0))
    cv2.line(frame, (px, top_y), (px, base_y), ORANGE, th(4), cv2.LINE_AA)
    cv2.putText(frame, f"{t:5.2f}s / {duration:.2f}s", (sx, h - int(14 * k)),
                FONT, fs(0.7), GREY, th(2), cv2.LINE_AA)
    return frame


def replay(video, trace, offset, gate, inc, duration, title="crash detection"):
    """PASS 2. Plays at native speed, paced by the clock rather than by waitKey alone."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    start = None  # set after the FIRST frame is on screen: opening the window and decoding
                  # frame 0 costs ~1s, and charging that to the clock makes every later
                  # frame look late and the whole replay run ~17% slow.
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            draw_overlay(frame, t, score_at(trace, offset, t), gate, inc, duration,
                         trace, offset)
            cv2.imshow(title, frame)
            if start is None:
                start = time.time()
            # pace to real time; a slow draw shortens the wait rather than lagging behind
            wait_ms = int(max(1, (start + t - time.time()) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF in (ord("q"), 27):
                break
            i += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # macOS needs one more spin to actually close the window


# ---------------------------------------------------------------------------------------
# The incident record -- the actual product
# ---------------------------------------------------------------------------------------

def write_record(clip_id, video, trace, fps, offset, pol, detector, stride, out_dir):
    """Write the README §27 incident record. THIS is the product; detection is a commodity.

    `detect.py::build_record` is imported UNCHANGED and does all of the work -- schema,
    evidence window, source hash, calibrated confidence, policy provenance. This wrapper
    only decides where the file lands and refuses to let a strided run pass itself off as
    a clean one.

    Returns the path, or None when the clip did not clear the gate. Writing nothing below
    the gate is the product behaviour, not a failure: a system that files a report on every
    video has triaged nothing.

    It lands in runs/demo/ (gitignored), NEVER runs/incidents/, which holds detect.py's
    committed records -- a demo must not overwrite the evidence trail.
    """
    rec = build_record(clip_id, video, trace, fps, offset, pol, detector)
    if rec is None:
        return None
    if stride != 1:
        rec["demo_note"] = (
            f"PRODUCED AT --stride {stride}. Scoring was subsampled to "
            f"{TARGET_FPS / stride:.2f} Hz instead of {TARGET_FPS:.0f} Hz, which biases the "
            f"score DOWNWARD and does not reproduce the committed stride-1 numbers. "
            f"NOT a reportable record.")
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, f"{clip_id}.json")
    with open(dest, "w") as f:
        json.dump(rec, f, indent=2)
    return dest


# ---------------------------------------------------------------------------------------
# The plain-English summary
# ---------------------------------------------------------------------------------------

def summarise(name, duration, score, gate, inc, calibrated, elapsed, record_path=None):
    """What a fleet manager reads. No ids, no jargon, no field names."""
    out = []
    a = out.append
    a("")
    a("=" * 72)
    if inc is None:
        a(f"  No incident found in {name}.")
        a("")
        a(f"  The highest concern the model reached was {score:.4f}, and it raises an")
        a(f"  alert at {gate:.4f}. Nothing in this footage crossed that line.")
        a("")
        a("  So no record was written -- which is the point. A system that files a")
        a("  report on every video has not triaged anything.")
        a("=" * 72)
        return "\n".join(out)

    a(f"  INCIDENT DETECTED in {name}")
    a("")
    a(f"  Something happened around {inc['t_peak']:.1f} seconds in.")
    a(f"  The model was concerned from {inc['t_start']:.1f}s to {inc['t_end']:.1f}s"
      f" -- about {inc['t_end'] - inc['t_start']:.1f} seconds of footage.")
    a("")
    a(f"  Its confidence peaked at {score:.4f}, against an alert line of {gate:.4f}")
    a(f"  that was derived from 667 labelled clips, not chosen by hand.")
    a(f"  Calibrated against real outcomes, that works out to roughly {calibrated:.0%}")
    a("  confidence this is a genuine collision.")
    a("")
    a(f"  It watched {duration:.1f} seconds of video in {elapsed:.0f} seconds and wrote")
    if record_path:
        a("  a structured incident record -- the thing you would send to an insurer:")
        a("")
        a(f"      {record_path}")
    else:
        a("  its findings to the screen only; no record file was produced.")
    a("=" * 72)
    return "\n".join(out)


# ---------------------------------------------------------------------------------------

def run(video, device="mps", stride=1, play=True, quiet=True):
    name = os.path.basename(video)
    if not os.path.exists(video):
        raise SystemExit(f"no such video: {video}")

    duration, native_fps, n_frames = probe_video(video)
    pol = policy()
    gate = pol["threshold"]
    g = pol["provenance"]["gate"]

    print()
    print("=" * 72)
    print(f"  {name}  --  {duration:.2f}s, {n_frames} frames at {native_fps:.2f} fps")
    print("=" * 72)
    print(f"  model      BADAS-Open (V-JEPA2 ViT-L), Apache-2.0, run locally on {device}")
    print(f"  alert at   {gate:.4f}")
    print(f"  which is   the threshold that delivers {g['target_recall']:.0%} recall on "
          f"{g['fitted_on']}")
    print(f"             -- READ OFF THE DATA at startup, not typed into this file")
    print(f"             it delivers recall {g['delivers_recall']:.4f}, "
          f"precision {g['delivers_precision']:.4f}")
    if stride != 1:
        print(f"  🔴 stride  {stride} -- scoring at {TARGET_FPS / stride:.2f} Hz, NOT the "
              f"1:1 rate. Scores differ from the committed numbers.")
    print()
    print(f"  PASS 1 of 2 -- scoring. ~{estimate_windows(duration, stride)} windows to do.")
    print()

    frames_dir = os.path.join(ROOT, "runs", "demo", "frames")
    os.makedirs(frames_dir, exist_ok=True)
    score, elapsed, _, detector = score_with_progress(video, device, stride, frames_dir,
                                                      quiet=quiet)
    if score is None:
        raise SystemExit("  clip is too short to score even one window.")

    clip_id = os.path.splitext(name)[0]
    trace, fps8, offset = load_traces_abs(frames_dir)[clip_id]
    inc = incident(trace, fps8, offset, gate)
    calibrated = float(pol["calibrate"](np.array([score]))[0])

    verdict = "INCIDENT" if inc is not None else "nothing above the line"
    print(f"\n  score {score:.4f}  ->  {verdict}")

    # Written BEFORE pass 2: the record is the product, and pressing q to skip the
    # playback must never be the reason it does not exist.
    record_path = write_record(clip_id, video, trace, fps8, offset, pol, detector, stride,
                               os.path.join(ROOT, "runs", "demo", "incidents"))
    if record_path:
        rel = os.path.relpath(record_path, ROOT)
        print(f"  wrote the incident record  ->  {rel}")
        print(f"  open it with               ->  cat {rel}")

    if play:
        print(f"\n  PASS 2 of 2 -- replaying at normal speed with the score on screen.")
        print("  (press q to skip)\n")
        replay(video, trace, offset, gate, inc, duration, title=f"crash detection - {name}")

    print(summarise(name, duration, score, gate, inc, calibrated, elapsed,
                    record_path=os.path.relpath(record_path, ROOT) if record_path else None))
    print()
    return {"score": score, "incident": inc, "elapsed": elapsed, "record": record_path}


# ---------------------------------------------------------------------------------------
# Self-check: synthetic traces and synthetic frames. No model, no MPS, no video decode.
# ---------------------------------------------------------------------------------------

def _self_check():
    pol = policy()
    gate = pol["threshold"]
    assert 0.0 < gate < 1.0, gate
    print(f"ok  gate derived at runtime, not typed: {gate:.4f}")

    src = open(os.path.abspath(__file__)).read()
    body = src.split('"""', 2)[2]  # the module docstring may name numbers; code may not
    # Built by concatenation on purpose: spelling the literal here would make this check
    # trip over its own banned list, which is exactly what happened the first time.
    stem = "0.9" + "73"
    for banned in (stem + "3", stem + "33", stem + "328"):
        assert banned not in body, f"a threshold literal ({banned}) leaked into the code"
    print("ok  no threshold literal anywhere in the executable body (B5)")

    # score_at: the leading NaN run must read as "no answer yet", never as a confident zero
    trace = np.array([0.1, 0.2, 0.9, 0.3], dtype=float)
    offset = 16
    assert score_at(trace, offset, 0.0) is None
    assert score_at(trace, offset, (offset - 1) / TARGET_FPS) is None
    assert score_at(trace, offset, offset / TARGET_FPS) == 0.1
    assert score_at(trace, offset, (offset + 2) / TARGET_FPS) == 0.9
    assert score_at(trace, offset, 9999.0) == 0.3, "past the end must clamp, not crash"
    print("ok  score_at: warm-up reads as None (not 0.0), and past-the-end clamps")

    # incident() is eval/timing.py's, unmodified -- check we are calling it correctly
    loud = np.full(64, 0.02)
    loud[40] = 0.995
    inc = incident(loud, TARGET_FPS, offset, gate)
    assert inc is not None
    assert abs(inc["t_peak"] - (offset + 40) / TARGET_FPS) < 1e-9
    assert inc["t_start"] <= inc["t_peak"] <= inc["t_end"]
    print(f"ok  incident timing carries the {offset}-frame warm-up offset "
          f"(t_peak {inc['t_peak']:.3f}s)")

    quiet = np.full(64, max(gate - 0.3, 0.01))
    assert incident(quiet, TARGET_FPS, offset, gate) is None
    print("ok  a clip below the gate produces no incident and no alert region")

    # the overlay must not crash on any of: warming up, below gate, alerting, past the end
    for t in (0.0, 1.0, (offset + 40) / TARGET_FPS, 1e6):
        frame = np.zeros((360, 640, 3), np.uint8)
        out = draw_overlay(frame, t, score_at(loud, offset, t), gate, inc, 8.0, loud, offset)
        assert out.shape == (360, 640, 3)
    print("ok  overlay renders at warm-up, below gate, mid-alert and past the end")

    # the counting proxy must forward transparently and stay truthy/callable, because
    # vendor code gates on `if self.processor` and hasattr(..., "__call__")
    seen = []
    proxy = _CountingProcessor(lambda *a, **kw: ("real", a, kw), seen.append)
    assert proxy, "proxy must be truthy or vendor code silently skips preprocessing"
    assert hasattr(proxy, "__call__")
    assert proxy(1, x=2) == ("real", (1,), {"x": 2}), "proxy altered the call"
    proxy(3)
    assert proxy.count == 2 and seen == [1, 2]
    print("ok  progress proxy forwards calls unchanged and ticks once per window")

    assert estimate_windows(7.38, 1) > 0 and estimate_windows(0.1, 1) == 1
    print("ok  window estimate is positive even for a clip shorter than one window")

    # THE RECORD IS THE PRODUCT (README §27). It must actually reach disk, it must carry
    # the score, it must be absent below the gate, and a strided run must say so IN THE FILE.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        fake = os.path.join(td, "selfcheck.mp4")
        with open(fake, "wb") as f:
            f.write(b"not a real video -- only hashed, never decoded")
        out = os.path.join(td, "incidents")
        rp = write_record("selfcheck", fake, loud, TARGET_FPS, offset, pol, "d nanmax", 1,
                          out)
        assert rp and os.path.exists(rp), "the incident record never reached disk"
        rec = json.load(open(rp))
        assert rec["event_id"] == "selfcheck"
        assert rec["confidence"]["score"] == round(float(np.max(loud)), 4)
        assert rec["evidence"]["source_sha256"], "the evidence hash is empty"
        assert "demo_note" not in rec, "a stride-1 run must not be flagged as strided"
        rp8 = write_record("selfcheck", fake, loud, TARGET_FPS, offset, pol, "d nanmax", 8,
                           out)
        assert "--stride 8" in json.load(open(rp8))["demo_note"], \
            "a strided record must carry its own warning, not rely on the screen"
        assert write_record("q", fake, quiet, TARGET_FPS, offset, pol, "d nanmax", 1,
                            out) is None, "a sub-gate clip must produce no record at all"
    print("ok  incident record written, carries the score, absent below the gate, "
          "strided runs self-label")

    # the summary must read as English, must not leak field names, and must not claim a
    # record it did not write -- the bug this check exists to prevent
    s = summarise("crash1.mov", 7.38, 0.9968, gate, inc, 0.84, 57.3,
                  record_path="runs/demo/incidents/crash1.json")
    assert "INCIDENT DETECTED" in s and "t_peak" not in s and "nanmax" not in s
    assert "runs/demo/incidents/crash1.json" in s, "summary must name the record it wrote"
    ns = summarise("safe.mp4", 30.0, 0.5, gate, None, 0.1, 350.0)
    assert "No incident found" in ns and ".json" not in ns, \
        "a clip below the gate must not claim a record on screen"
    print("ok  summary is plain English in both directions, and claims no record it "
          "did not write")

    print("PASS")


def main():
    ap = argparse.ArgumentParser(
        description="Watch the crash detector judge a video. See docs/sept30_demo.md.")
    ap.add_argument("video", nargs="?", help="the video to judge")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--stride", type=int, default=1,
                    help="8 scores at 1 Hz instead of 8 Hz: ~8x faster, and the score "
                         "CHANGES, so it is printed on screen and must never be compared "
                         "against the committed stride-1 numbers")
    ap.add_argument("--no-play", action="store_true",
                    help="skip pass 2 (no window); pass 1 and the summary still run")
    ap.add_argument("--loud", action="store_true",
                    help="do not suppress the model's load-time warnings")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()

    if a.self_check:
        _self_check()
        return
    if not a.video:
        ap.error("need a video, or --self-check")
    run(a.video, device=a.device, stride=a.stride, play=not a.no_play, quiet=not a.loud)


if __name__ == "__main__":
    main()
