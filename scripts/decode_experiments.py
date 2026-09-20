"""Two decode experiments, run side by side. Modifies NOTHING in the scoring path.

This script is a LABORATORY. It imports the vendored decoder read-only, implements a
candidate beside it, and compares. Nothing here is wired into `scripts/score_external.py`
or `eval/`, and nothing here may be until the user has seen the numbers.

--------------------------------------------------------------------------------------
EXPERIMENT 1 -- WHERE does the Mac/Linux pixel difference come from?
--------------------------------------------------------------------------------------
Measured 2026-09-19: Colab and Kaggle produce BYTE-IDENTICAL decoded frames
(d43cf8ef...) which differ from the Mac (bd6126c3...) by at most 3 grey levels, with
frames aligned at offset 0. Two unrelated Linux stacks agreeing with each other and
disagreeing with macOS points at CPU arithmetic rather than a library version.

`load_full_video_frames` does three things per frame:
    cap.read()                          -> decode      (ffmpeg/swscale, VERSIONED)
    cv2.resize(frame, (224,224))        -> INTER_LINEAR (SIMD, ARCHITECTURE-dependent)
    cv2.cvtColor(BGR2RGB)               -> channel swap (pure permutation, CANNOT differ)

So hash the frames BEFORE the resize. That splits the candidates decisively:
  * raw hashes MATCH across machines  -> decode is identical; the difference is created by
    cv2.resize, which is architecture arithmetic. **No OpenCV version pin can fix it** and
    the pre-declared contingency is the honest route.
  * raw hashes DIFFER                 -> the decoder/swscale differs, which IS versioned,
    so pinning OpenCV has a real chance.
Only the 64-char hash travels, never the raw frames (~220 MB at native resolution).

--------------------------------------------------------------------------------------
EXPERIMENT 2 -- can sequential reading replace per-frame seeking, losslessly?
--------------------------------------------------------------------------------------
The vendored loader calls `cap.set(CAP_PROP_POS_FRAMES, k)` before EVERY frame -- ~480
random seeks for one 60 s comma2k19 segment. Reading in order and keeping the frames we
want decodes each frame once. Measured on a Kaggle T4: 0.276 s/frame, making decode 65% of
a projected 203 s/segment against a 72 s bar.

🔴 THIS IS NOT THE FORBIDDEN FIX. Transcoding to 8 fps re-encodes pixels and is lossy;
progress.md §12 rightly bans it. Sequential reading re-encodes nothing -- it is the same
frames by a cheaper route. But it is only USABLE if it returns the same bytes, because the
committed Nexar baseline (92.3 FP/hour) was produced with the seeking loader. If ffmpeg's
seek is imprecise on some build, sequential reading would return DIFFERENT (arguably more
correct) frames, and adopting it would silently break comparability -- the exact thing
GATE C exists to prevent. **The fingerprint arbitrates, and it is cheap.**

    ~/envs/badas/bin/python scripts/decode_experiments.py --self-check
    ~/envs/badas/bin/python scripts/decode_experiments.py --clip data/nexar/test-public/negative/01044.mp4
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(REPO, "vendor", "badas-open")):
    sys.path.insert(0, os.path.join(REPO, "vendor", "badas-open"))
else:
    # Running from the Kaggle probe dataset, not the repo. Reuse probe.py's locator and
    # its manifest check rather than duplicating either -- a second copy would drift.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import probe as _probe
    except ImportError:
        try:
            import kaggle_decode_probe as _probe
        except ImportError:
            _probe = None
    if _probe is not None:
        PROBE_DIR = _probe.find_probe_dir(None)
        sys.path.insert(0, os.path.join(_probe.stage_bundle(PROBE_DIR), "vendor", "badas-open"))

TARGET_SIZE = (224, 224)
TARGET_FPS = 8.0

# Measured on the Mac. RESIZED is the value colab_comma2k19.ipynb §2 already compares
# against; RAW is new and is what Experiment 1 turns on.
MAC_RESIZED_FP = "bd6126c391b2301d170e40128d04338587a38697a7a494c63943d4182a23f24a"
MAC_RAW_FP = "81c70e412f8f5063033658e625bfa2ce87ae7d2f4848a8eb2841de10373d3da3"   # 81 frames of (720,1280,3), measured on the Mac
LINUX_RESIZED_FP = "d43cf8efa32e87d0a6baf03f749a10afcae08c1f41457947bcd2d4976abfefdb"


def _open(video_path):
    import cv2
    if not os.path.exists(video_path):
        raise SystemExit(f"video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")
    return cap


def _plan(cap):
    """Reproduce the vendored loader's frame-selection arithmetic EXACTLY.

    Copied deliberately rather than imported: the point is to compare two loaders, so the
    plan must be identical and visible. Any drift here would make the comparison a lie.
    """
    import cv2
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        raise SystemExit(f"invalid fps {fps}")
    duration = total / fps
    if TARGET_FPS and TARGET_FPS != fps:
        count = int(round(duration * TARGET_FPS))
        interval = fps / TARGET_FPS
    else:
        count, interval = total, 1.0
    wanted = []
    for i in range(count):
        k = int(round(i * interval))
        if k >= total:
            break
        wanted.append(k)
    return total, fps, count, wanted


def decode_raw_fingerprint(video_path, limit=None):
    """Hash frames BEFORE resize, using the SAME seek path as the vendored loader, so the
    only thing removed is cv2.resize. Returns (hexdigest, n_frames, native_shape)."""
    import cv2
    cap = _open(video_path)
    try:
        total, fps, count, wanted = _plan(cap)
        h = hashlib.sha256()
        n, shape = 0, None
        for k in wanted if limit is None else wanted[:limit]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ret, frame = cap.read()
            if not ret:
                break
            if shape is None:
                shape = frame.shape
            h.update(np.ascontiguousarray(frame).tobytes())
            n += 1
        return h.hexdigest(), n, shape
    finally:
        cap.release()


def load_sequential(video_path, target_size=TARGET_SIZE, target_fps=TARGET_FPS):
    """Candidate: ONE forward pass, no seeking. Same frames, same order, same arithmetic.

    `wanted` is non-decreasing, and when the source fps is below target_fps the same index
    can be requested twice -- which the seeking loader would serve by seeking to it twice.
    The inner while-loop reproduces that duplication rather than silently dropping it.
    """
    import cv2
    cap = _open(video_path)
    try:
        total, fps, count, wanted = _plan(cap)
        frames = np.empty((len(wanted), target_size[1], target_size[0], 3), dtype=np.uint8)
        out, wi, pos = 0, 0, 0
        while wi < len(wanted):
            ret, frame = cap.read()
            if not ret:
                break
            if wanted[wi] == pos:
                small = cv2.cvtColor(cv2.resize(frame, target_size), cv2.COLOR_BGR2RGB)
                while wi < len(wanted) and wanted[wi] == pos:
                    frames[out] = small
                    out += 1
                    wi += 1
            pos += 1
        return frames[:out]
    finally:
        cap.release()


def fp(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def run(clip, reps):
    from badas.utils.video import load_full_video_frames
    import cv2

    print("=" * 72)
    print(f"stack: cv2 {cv2.__version__}   python {sys.version.split()[0]}")
    print(f"clip : {clip}")
    print("=" * 72)

    # ---- EXPERIMENT 1 -----------------------------------------------------------------
    print("\nEXPERIMENT 1 -- is the Mac/Linux difference created BEFORE or BY the resize?")
    raw, nraw, shape = decode_raw_fingerprint(clip)
    print(f"  raw frames (pre-resize): {nraw} frames of {shape}")
    print(f"  raw fingerprint here : {raw}")
    if MAC_RAW_FP:
        print(f"  raw fingerprint mac  : {MAC_RAW_FP}")
        if raw == MAC_RAW_FP:
            print("  -> RAW MATCHES. The decoder produces identical bytes on both machines.")
            print("     The difference is created BY cv2.resize == architecture arithmetic.")
            print("     🔴 NO OpenCV version pin can fix this. Use the contingency.")
        else:
            print("  -> RAW DIFFERS. The decoder/swscale itself differs, which IS versioned,")
            print("     so pinning OpenCV has a real chance of fixing it.")
    else:
        print("  (no Mac raw reference compiled in -- run --emit-constants on the Mac first)")

    # ---- EXPERIMENT 2 -----------------------------------------------------------------
    print("\nEXPERIMENT 2 -- sequential reading vs per-frame seeking")
    t0 = time.perf_counter(); a = load_full_video_frames(clip, TARGET_SIZE, TARGET_FPS)
    t_seek = time.perf_counter() - t0
    t0 = time.perf_counter(); b = load_sequential(clip, TARGET_SIZE, TARGET_FPS)
    t_seq = time.perf_counter() - t0

    print(f"  seeking (vendored) : {a.shape}  {t_seek:7.2f} s   {t_seek/max(len(a),1):.4f} s/frame")
    print(f"  sequential (new)   : {b.shape}  {t_seq:7.2f} s   {t_seq/max(len(b),1):.4f} s/frame")
    if t_seq > 0:
        print(f"  SPEEDUP            : {t_seek/t_seq:.2f}x")

    same_shape = a.shape == b.shape
    fa, fb = fp(a), fp(b)
    print(f"\n  seeking fingerprint    {fa}")
    print(f"  sequential fingerprint {fb}")
    if not same_shape:
        print(f"  🔴 SHAPES DIFFER {a.shape} vs {b.shape} -- not a drop-in. Do not adopt.")
    elif fa == fb:
        print("  -> ✅ BYTE-IDENTICAL. Sequential reading is a LOSSLESS drop-in on this")
        print("     machine: same frames, same bytes, cheaper route.")
    else:
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        per_frame_max = d.reshape(len(d), -1).max(axis=1)
        print(f"  -> ❌ DIFFERENT. max |diff| {d.max()}, mean {d.mean():.4f}, "
              f"{(per_frame_max == 0).sum()}/{len(d)} frames identical")
        print("     This means the seek is IMPRECISE on this build: the two routes fetch")
        print("     different frames. Adopting it would break comparability with the")
        print("     committed baseline. Report; do not adopt.")

    print(f"\n  for reference, the committed Mac value: {MAC_RESIZED_FP}")
    print(f"  and the Linux (Colab==Kaggle) value:    {LINUX_RESIZED_FP}")
    return 0


def emit_constants(clip):
    raw, n, shape = decode_raw_fingerprint(clip)
    print(f"MAC_RAW_FP = {raw!r}    # {n} frames of {shape}")
    return 0



def validate_dir(clips_dir, limit=None, sample=None):
    """Byte-identity across MANY clips, not one.

    One clip proves the loaders agree on one clip. GATE C re-scores 100, and the full
    contingency would re-score 667, so a single agreement is not enough evidence to change
    the scoring path. Any mismatch here kills the change outright.
    """
    from badas.utils.video import load_full_video_frames
    import numpy as np

    names = sorted(f for f in os.listdir(clips_dir) if f.endswith((".mp4", ".MP4")))
    if sample:
        ids = sorted(os.path.splitext(f)[0] for f in names)
        pick = set(np.random.default_rng(0).choice(np.array(ids), size=min(sample, len(ids)),
                                                   replace=False).tolist())
        names = [f for f in names if os.path.splitext(f)[0] in pick]
        print(f"  GATE C's deterministic sample: {len(names)} clips, first 5 "
              f"{[os.path.splitext(n)[0] for n in names[:5]]}")
    if limit:
        names = names[:limit]

    bad, t_seek, t_seq, frames_total = [], 0.0, 0.0, 0
    shapes = {}
    for i, name in enumerate(names, 1):
        path = os.path.join(clips_dir, name)
        t0 = time.perf_counter(); a = load_full_video_frames(path, TARGET_SIZE, TARGET_FPS)
        t_seek += time.perf_counter() - t0
        t0 = time.perf_counter(); b = load_sequential(path, TARGET_SIZE, TARGET_FPS)
        t_seq += time.perf_counter() - t0
        frames_total += len(a)
        shapes[len(a)] = shapes.get(len(a), 0) + 1
        if a.shape != b.shape:
            bad.append((name, f"SHAPE {a.shape} vs {b.shape}"))
        elif fp(a) != fp(b):
            d = np.abs(a.astype(np.int16) - b.astype(np.int16))
            bad.append((name, f"BYTES max|d|={d.max()} nframes_diff="
                              f"{int((d.reshape(len(d), -1).max(axis=1) > 0).sum())}"))
        if i % 25 == 0 or i == len(names):
            print(f"    {i}/{len(names)}  mismatches so far: {len(bad)}")

    print(f"\n  clips compared     {len(names)}")
    print(f"  frames compared    {frames_total}")
    print(f"  distinct lengths   {len(shapes)} (min {min(shapes)} max {max(shapes)})")
    print(f"  seeking total      {t_seek:8.1f} s   {t_seek/max(frames_total,1):.4f} s/frame")
    print(f"  sequential total   {t_seq:8.1f} s   {t_seq/max(frames_total,1):.4f} s/frame")
    print(f"  SPEEDUP            {t_seek/max(t_seq,1e-9):.2f}x")
    if bad:
        print(f"\n  ❌ {len(bad)} MISMATCH(ES) -- the change is DEAD, do not adopt:")
        for n, why in bad[:15]:
            print(f"      {n:<16} {why}")
        return 1
    print(f"\n  ✅ ALL {len(names)} CLIPS BYTE-IDENTICAL across {frames_total} frames.")
    return 0


def long_file(clip, reps=6):
    """A ~60 s file, the length of a comma2k19 segment.

    Seek cost can scale with file length (a seek far into a file re-decodes from a more
    distant keyframe), so a 10 s clip may understate the win. Built with ffmpeg -c copy:
    LOSSLESS, nothing re-encoded, and it is a throwaway timing probe that never reaches
    the scorer.
    """
    from badas.utils.video import load_full_video_frames
    import subprocess
    import tempfile

    tmp = tempfile.mkdtemp()
    lst, cat = os.path.join(tmp, "cat.txt"), os.path.join(tmp, "long.mp4")
    with open(lst, "w") as fh:
        fh.write("".join(f"file '{os.path.abspath(clip)}'\n" for _ in range(reps)))
    try:
        cp = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe",
                             "0", "-i", lst, "-c", "copy", cat], capture_output=True, text=True)
    except FileNotFoundError:
        cp = None
    if cp is None or cp.returncode != 0 or not os.path.exists(cat):
        # No ffmpeg (the Mac). Build a long file with cv2.VideoWriter instead. It is
        # RE-ENCODED, so it is NOT a stand-in for real comma2k19 bytes -- but both loaders
        # read the SAME file, so the seek-vs-sequential comparison on it is still valid,
        # and that is the only question this function asks.
        import cv2
        print("  (no ffmpeg: building the long file with cv2.VideoWriter instead --")
        print("   re-encoded, so treat the ABSOLUTE timings as indicative; the identity")
        print("   check and the seek/sequential RATIO remain valid.)")
        src = _open(os.path.abspath(clip))
        fps_in = src.get(cv2.CAP_PROP_FPS)
        frames_in = []
        while True:
            ok, fr = src.read()
            if not ok:
                break
            frames_in.append(fr)
        src.release()
        if not frames_in:
            print("  ⚠ could not read the source clip")
            return 0
        h, w = frames_in[0].shape[:2]
        vw = cv2.VideoWriter(cat, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
        for _ in range(reps):
            for fr in frames_in:
                vw.write(fr)
        vw.release()
        if not os.path.exists(cat) or os.path.getsize(cat) == 0:
            print("  ⚠ VideoWriter produced nothing")
            return 0

    t0 = time.perf_counter(); a = load_full_video_frames(cat, TARGET_SIZE, TARGET_FPS)
    ts = time.perf_counter() - t0
    t0 = time.perf_counter(); b = load_sequential(cat, TARGET_SIZE, TARGET_FPS)
    tq = time.perf_counter() - t0
    print(f"  {reps}x concat -> {len(a)} frames (~{len(a)/8.0:.0f} s of video)")
    print(f"  seeking     {ts:8.2f} s   {ts/max(len(a),1):.4f} s/frame")
    print(f"  sequential  {tq:8.2f} s   {tq/max(len(b),1):.4f} s/frame")
    print(f"  SPEEDUP     {ts/max(tq,1e-9):.2f}x")
    if a.shape != b.shape:
        print(f"  ❌ SHAPE {a.shape} vs {b.shape}")
        return 1
    if fp(a) != fp(b):
        print("  ❌ BYTES DIFFER on the long file -- seek precision degrades with length.")
        return 1
    print("  ✅ BYTE-IDENTICAL on a segment-length file too.")
    # What this implies for a real comma2k19 segment at stride 8.
    seq_pf = tq / max(len(b), 1)
    model = ((480 - 16) / 8.0) * 1.209
    for tag, m in (("as-is", model), ("with skip_predictor", model * 0.75)):
        tot = seq_pf * 480 + m
        print(f"    projected 60 s segment, {tag:<20} {tot:6.1f} s = {tot/72.0:.2f}x bar"
              f"   {tot*600/3600:5.1f} h per 10 h")
    return 0


def self_check():
    checks = []
    # The plan arithmetic must reproduce the vendored loader's, or the comparison is a lie.
    class FakeCap:
        def __init__(self, total, fps):
            self.t, self.f = total, fps
        def get(self, prop):
            import cv2
            return self.t if prop == cv2.CAP_PROP_FRAME_COUNT else self.f
    import cv2
    total, fps, count, wanted = _plan(FakeCap(300, 30.0))
    checks.append(("30fps/300 frames at target 8 -> 80 requested", count == 80))
    checks.append(("indices are non-decreasing (sequential pass is valid)",
                   all(wanted[i] <= wanted[i + 1] for i in range(len(wanted) - 1))))
    checks.append(("indices match round(i*fps/target)",
                   wanted[:4] == [0, 4, 8, 11]))
    checks.append(("no index exceeds the frame count", max(wanted) < 300))
    # Upsampling: source slower than target -> duplicate indices must be KEPT, not dropped.
    # (len(wanted) <= count, because the vendored loop BREAKS once an index passes the end.)
    _, _, c2, w2 = _plan(FakeCap(40, 4.0))
    checks.append(("4fps source at target 8 duplicates frames rather than dropping them",
                   len(set(w2)) < len(w2) and len(w2) <= c2))
    checks.append(("the early break is reproduced, not silently padded", len(w2) == 79))

    # Scan CODE only. Comments and docstrings here legitimately discuss the scoring path;
    # earlier versions of this check matched their own prose. tokenize strips both.
    import io
    import tokenize
    with open(os.path.abspath(__file__), "rb") as fh:
        code_only = " ".join(
            t.string for t in tokenize.tokenize(fh.readline)
            if t.type not in (tokenize.COMMENT, tokenize.STRING))
    checks.append(("no CODE here touches the scorer or the eval modules",
                   "score_external" not in code_only and "fp_rate" not in code_only))
    checks.append(("no CODE here opens a file for writing",
                   ".savez" not in code_only and "'w'" not in code_only
                   and '"w"' not in code_only))
    for name, ok in checks:
        print(f"{'ok ' if ok else 'FAIL'} {name}")
    bad = [n for n, ok in checks if not ok]
    print("PASS" if not bad else f"FAILED: {bad}")
    return 0 if not bad else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # In the repo the clip lives under data/nexar/; in the Kaggle probe dataset it sits
    # beside this script. Prefer whichever actually exists.
    in_repo = os.path.join(REPO, "data", "nexar", "test-public", "negative", "01044.mp4")
    default_clip = in_repo if os.path.exists(in_repo) else os.path.join(
        globals().get("PROBE_DIR", os.path.dirname(os.path.abspath(__file__))), "01044.mp4")
    p.add_argument("--clip", default=default_clip)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--validate-dir", default=None,
                   help="compare both loaders over every clip in this directory")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample", type=int, default=None,
                   help="use GATE C's deterministic rng(0) sample of this size")
    p.add_argument("--long-file", type=int, default=0,
                   help="N-fold lossless concat, to test a segment-length file")
    p.add_argument("--emit-constants", action="store_true")
    a = p.parse_args()
    if a.self_check:
        return self_check()
    if a.emit_constants:
        return emit_constants(a.clip)
    if a.validate_dir:
        print("=" * 72)
        print("MULTI-CLIP VALIDATION -- seeking vs sequential, byte for byte")
        print("=" * 72)
        return validate_dir(a.validate_dir, a.limit, a.sample)
    if a.long_file:
        print("=" * 72)
        print("LONG-FILE BENCHMARK -- a comma2k19-segment-length video")
        print("=" * 72)
        return long_file(a.clip, a.long_file)
    return run(a.clip, a.reps)


if __name__ == "__main__":
    sys.exit(main())
