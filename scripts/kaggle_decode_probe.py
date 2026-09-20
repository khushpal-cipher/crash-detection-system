"""Decode-equivalence + decode-speed probe. Runs on Kaggle, Colab, or the Mac.

WHY A SEPARATE, TINY PROBE. The user moved to Kaggle after exhausting Colab's GPU quota.
That is not a cosmetic change: GATE C (D57) exists to prove that the machine producing the
numbers reproduces the Mac, and Kaggle is a DIFFERENT machine with a different software
stack and a different GPU. **Colab's GATE C result therefore transfers to Kaggle not at
all** -- neither its failure nor any fix for it.

Rather than port the whole notebook and then discover the same wall, this probe asks the
one cheap question first: does Kaggle's cv2 decode Nexar clip 01044 to the same bytes the
Mac did? It needs ~14 MB of input, no checkpoint, no GPU, and about two minutes. If the
answer is yes, the expensive work is worth porting. If no, we have learned it for 14 MB
instead of for a 3.98 GB upload and hours of quota.

It also times decode, because GATE C's 94.7 s/clip on Colab could not be split into decode
vs model, and that split decides whether a 60 s comma2k19 segment can clear the 72 s bar.

    # locally (sanity)
    ~/envs/badas/bin/python scripts/kaggle_decode_probe.py --self-check
    ~/envs/badas/bin/python scripts/kaggle_decode_probe.py \
        --probe-dir runs/kaggle_probe

    # on Kaggle: add the probe dataset, then in a cell
    !python /kaggle/input/<dataset-slug>/probe.py
"""

import argparse
import glob
import hashlib
import os
import subprocess
import sys
import tarfile
import tempfile
import time

# Measured on the Mac, session 15, and embedded in colab_comma2k19.ipynb §2.
# load_full_video_frames(path, (224, 224), 8.0) -> sha256 of the uint8 array.
MAC_PIXEL_FP = "bd6126c391b2301d170e40128d04338587a38697a7a494c63943d4182a23f24a"
CLIP = "01044"
MAC_SHAPE = (81, 224, 224, 3)
MAC_CV2 = "5.0.0 (opencv-python-headless 5.0.0.93), avcodec 61.19.101, swscale 8.3.100"

# GATE C's own measurement on a Colab T4, 2026-09-19. Used only to split its total.
GATEC_S_PER_CLIP, GATEC_MEAN_FRAMES = 94.7, 77.1
BAR_S_PER_SEGMENT = 72.0        # NEW_PLAN §13: 12 h / 600 one-minute segments
SEG_FRAMES = 60 * 8             # a 60 s comma2k19 segment at target_fps 8.0
WINDOW = 16                     # kept windows = n_frames - 16


def find_probe_dir(explicit):
    """Locate the clip. Kaggle nests dataset inputs several levels deep and the depth is
    not predictable (observed: /kaggle/input/datasets/<user>/<slug>/), so search, don't
    assume."""
    if explicit:
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, f"{CLIP}.mp4")):
        return here
    hits = sorted(glob.glob(f"/kaggle/input/**/{CLIP}.mp4", recursive=True))
    if hits:
        return os.path.dirname(hits[0])
    raise SystemExit(
        f"Could not find the probe files ({CLIP}.mp4). Pass --probe-dir, or on Kaggle add "
        "the probe dataset to the notebook (right panel > Add Input).")


def _verify_bundle(root):
    """GATE B, in miniature. The notebook's §1 checks every bundled file against
    BUNDLE_MANIFEST.json before anything runs. When Kaggle pre-extracts the archive we skip
    tarfile entirely, which would silently skip that check too -- so do it here instead.
    An altered or truncated decoder would change the fingerprint and we would blame the
    platform for our own corrupt upload."""
    man_path = os.path.join(root, "BUNDLE_MANIFEST.json")
    if not os.path.exists(man_path):
        print("  ⚠ no BUNDLE_MANIFEST.json beside the bundle -- integrity NOT verified.")
        return
    import json
    with open(man_path) as f:
        man = json.load(f)
    bad = []
    for rel, rec in sorted(man["clips"].items()):
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            bad.append((rel, "MISSING"))
        elif os.path.getsize(p) != rec["bytes"]:
            bad.append((rel, f"SIZE {os.path.getsize(p)} != {rec['bytes']}"))
        else:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(blk)
            if h.hexdigest() != rec["sha256"]:
                bad.append((rel, "SHA-256 MISMATCH"))
    if bad:
        for rel, why in bad:
            print(f"    {rel:<50} {why}")
        raise SystemExit("bundle integrity FAILED -- the staged code is not the repo's. "
                         "Re-upload the dataset; do not improvise around this.")
    print(f"  bundle verified: {man['n']} files byte-identical to the repo ✅")


def stage_bundle(probe_dir):
    """Make the vendored code importable.

    Kaggle DECOMPRESSES archives when a dataset is uploaded, so colab_bundle.tar.gz can
    arrive as a directory named colab_bundle/ instead of a file. Handle both, and do not
    care which: the manifest check below decides whether the bytes are right.
    /kaggle/input is read-only, which is fine for importing -- Python simply skips writing
    __pycache__.
    """
    tgz = os.path.join(probe_dir, "colab_bundle.tar.gz")
    if os.path.exists(tgz) and os.path.isfile(tgz):
        root = os.path.join(
            "/kaggle/working" if os.path.isdir("/kaggle/working") else tempfile.gettempdir(),
            "probe_repo")
        os.makedirs(root, exist_ok=True)
        with tarfile.open(tgz) as tar:
            tar.extractall(root)
        print(f"  bundle extracted from the archive -> {root}")
        _verify_bundle(root)
        return root

    # Pre-extracted by the platform. Find the level that actually holds vendor/badas-open.
    marker = os.path.join("vendor", "badas-open", "badas", "utils", "video.py")
    cands = [probe_dir, os.path.join(probe_dir, "colab_bundle")]
    cands += sorted(glob.glob(os.path.join(probe_dir, "*", "")))
    cands += sorted(os.path.dirname(p) for p in
                    glob.glob(os.path.join(probe_dir, "**", marker), recursive=True))
    for c in cands:
        c = c.rstrip(os.sep)
        if os.path.exists(os.path.join(c, marker)):
            print(f"  bundle found already extracted -> {c}")
            _verify_bundle(c)
            return c
        # glob above returns the video.py's own dir; walk back up to the bundle root
        up = c
        for _ in range(4):
            up = os.path.dirname(up)
            if os.path.exists(os.path.join(up, marker)):
                print(f"  bundle found already extracted -> {up}")
                _verify_bundle(up)
                return up

    raise SystemExit(
        f"Could not find the vendored code under {probe_dir}.\n"
        f"Looked for either colab_bundle.tar.gz or a directory containing {marker}.\n"
        "Run:  ls -R /kaggle/input | head -50   and send the output.")


def report_stack():
    import cv2
    import numpy as np
    print("=" * 70)
    print("THE DECODE STACK HERE")
    print("=" * 70)
    print(f"  python         {sys.version.split()[0]}")
    print(f"  cv2            {cv2.__version__}")
    print(f"  numpy          {np.__version__}")
    bi = cv2.getBuildInformation()
    for key in ("FFMPEG", "avcodec", "avformat", "swscale"):
        for line in bi.split("\n"):
            if key.lower() in line.lower():
                print(f"    {line.strip()[:96]}")
                break
    print(f"  mac            {MAC_CV2}")
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
        print(f"  gpu            {gpu}   (torch {torch.__version__})")
        print("  🔴 WHICHEVER GPU YOU SEE HERE IS THE ONE THE WHOLE RUN MUST USE.")
        print("     GATE C validates one accelerator; switching later invalidates it.")
    except Exception:
        print("  gpu            torch unavailable (fine -- this probe needs no GPU)")


def compare(probe_dir, decoder):
    """Q1: do the decode paths agree, and if not, HOW do they differ?"""
    import numpy as np
    print("\n" + "=" * 70)
    print("Q1  DOES THIS MACHINE DECODE LIKE THE MAC?")
    print("=" * 70)
    here = decoder(os.path.join(probe_dir, f"{CLIP}.mp4"), (224, 224), 8.0)
    fp = hashlib.sha256(here.tobytes()).hexdigest()
    print(f"  clip {CLIP}   shape {here.shape}   (mac {MAC_SHAPE})")
    print(f"  fingerprint here  {fp}")
    print(f"  fingerprint mac   {MAC_PIXEL_FP}")
    if fp == MAC_PIXEL_FP:
        print("  -> ✅ IDENTICAL. This machine's decode path matches the Mac's.")
        return "identical", here
    print("  -> ❌ DIFFERENT. Diffing the arrays to find out how.")

    ref = os.path.join(probe_dir, f"{CLIP}_frames.npz")
    if not os.path.exists(ref):
        print(f"  ⚠ no reference array at {ref} -- cannot classify. Q2 still runs.")
        return "unknown", here
    mac = np.load(ref)["frames"]
    if mac.shape != here.shape:
        print(f"  🔴 SHAPES DIFFER {mac.shape} vs {here.shape} -- different frame COUNT, "
              "which would also corrupt the FP/hour denominator. Report immediately.")
        return "shape", here

    a, b = mac.astype(np.int16), here.astype(np.int16)
    d = np.abs(a - b)
    per_frame_max = d.reshape(len(d), -1).max(axis=1)
    print(f"  max |diff|            {d.max()} grey levels (of 255)")
    print(f"  mean |diff|           {d.mean():.4f}")
    print(f"  pixels differing      {100.0 * (d > 0).mean():.2f}%")
    print(f"  frames byte-identical {(per_frame_max == 0).sum()} / {len(d)}")

    print("\n  frame-alignment test -- a SHIFT, or just rounding?")
    best = {}
    for off in (-2, -1, 0, 1, 2):
        tot, n = 0.0, 0
        for i in range(len(here)):
            j = i + off
            if 0 <= j < len(mac):
                tot += float(np.abs(a[j] - b[i]).mean())
                n += 1
        best[off] = tot / max(n, 1)
    for off in sorted(best):
        mark = "  <-- best" if best[off] == min(best.values()) else ""
        print(f"    mac[k{off:+d}] vs here[k]   mean |diff| {best[off]:8.4f}{mark}")
    off = min(best, key=best.get)

    if off != 0:
        print(f"\n  🔴 SERIOUS: frames align best at offset {off:+d}, not 0. This machine's")
        print("  seeking lands on DIFFERENT frames -- it is watching different footage.")
        return "shift", here
    if d.max() <= 3:
        print(f"\n  ✅ COSMETIC: aligned at 0, max {d.max()} grey level(s). Same frames,")
        print("  SIMD/rounding noise. Matching the Mac's opencv is the targeted lever.")
        return "rounding", here
    print(f"\n  🟡 Aligned at 0 but differences reach {d.max()} grey levels -- bigger than")
    print("  rounding. Suspect decoder/colour conversion, not the seek. Do not guess a fix.")
    return "large-aligned", here


def timings(probe_dir, decoder):
    """Q2: how fast does THIS machine decode, and can a 60 s segment clear the bar?"""
    import numpy as np
    print("\n" + "=" * 70)
    print("Q2  DECODE SPEED -- CAN A 60 s SEGMENT CLEAR THE 72 s BAR?")
    print("=" * 70)
    src = os.path.join(probe_dir, f"{CLIP}.mp4")

    def timed(path):
        t0 = time.perf_counter()
        fr = decoder(path, (224, 224), 8.0)
        return len(fr), time.perf_counter() - t0

    n1, t1 = timed(src)
    print(f"  short clip   {n1:4d} frames  {t1:7.2f} s   {t1 / n1:.4f} s/frame")

    # Build a ~480-frame file LOSSLESSLY (-c copy, no re-encode). This is NOT the forbidden
    # 8 fps transcode: no pixel is re-encoded, and this file never reaches the scorer.
    tmp = tempfile.mkdtemp()
    lst, cat = os.path.join(tmp, "cat.txt"), os.path.join(tmp, "long.mp4")
    reps = max(2, int(round(SEG_FRAMES / max(n1, 1))))
    with open(lst, "w") as fh:
        fh.write("".join(f"file '{src}'\n" for _ in range(reps)))
    long_pf = None
    try:
        cp = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                             "-safe", "0", "-i", lst, "-c", "copy", cat],
                            capture_output=True, text=True)
        if cp.returncode == 0 and os.path.exists(cat):
            n2, t2 = timed(cat)
            long_pf = t2 / n2
            print(f"  long file    {n2:4d} frames  {t2:7.2f} s   {long_pf:.4f} s/frame"
                  f"   ({reps}x lossless concat ~ one comma2k19 segment)")
            print(f"  per-frame cost vs the short clip: {long_pf / (t1 / n1):.2f}x  "
                  f"({'GROWS -- seeks get dearer in longer files' if long_pf > t1 / n1 * 1.25 else 'stable'})")
        else:
            print(f"  ⚠ long probe failed: {cp.stderr[-200:]}")
    except FileNotFoundError:
        print("  ⚠ ffmpeg not on PATH -- long probe skipped (it IS present on Kaggle/Colab)")

    pf = long_pf if long_pf else t1 / n1
    decode_77 = (t1 / n1) * GATEC_MEAN_FRAMES
    windows_77 = GATEC_MEAN_FRAMES - WINDOW
    per_window = (GATEC_S_PER_CLIP - decode_77) / windows_77
    print(f"\n  splitting Colab's GATE C total (94.7 s/clip) using THIS decode rate:")
    print(f"    decode {decode_77:6.1f} s ({100 * decode_77 / GATEC_S_PER_CLIP:.0f}%)"
          f"   model {GATEC_S_PER_CLIP - decode_77:6.1f} s = {per_window:.3f} s/window")
    print("    (only meaningful if this machine's decode ~ Colab's; on Kaggle it is a"
          " cross-machine estimate, not a measurement)")

    seg_decode = pf * SEG_FRAMES
    seg_windows = (SEG_FRAMES - WINDOW) / 8.0     # stride 8 == 1 Hz
    seg = seg_decode + seg_windows * max(per_window, 0.0)
    print(f"\n  PROJECTED 60 s segment at stride 8: decode {seg_decode:.0f} s + model "
          f"{seg_windows * max(per_window, 0.0):.0f} s = {seg:.0f} s")
    print(f"    bar {BAR_S_PER_SEGMENT:.0f} s  ->  {seg / BAR_S_PER_SEGMENT:.1f}x")
    print(f"    600 segments (10 h of footage) = {seg * 600 / 3600:.0f} h"
          f"   (NEW_PLAN §13 kills comma2k19 above 12 h)")
    return seg / BAR_S_PER_SEGMENT


def self_check():
    import numpy as np
    checks = []
    rng = np.random.default_rng(0)
    f = rng.integers(0, 256, size=(8, 4, 4, 3), dtype=np.uint8)
    h = lambda x: hashlib.sha256(x.tobytes()).hexdigest()
    checks.append(("fingerprint stable on identical input", h(f) == h(f.copy())))
    checks.append(("a one-frame shift changes it", h(np.roll(f, 1, 0)) != h(f)))
    g = f.copy(); g[0, 0, 0, 0] = (int(g[0, 0, 0, 0]) + 1) % 256
    checks.append(("a one-grey-level change changes it", h(g) != h(f)))
    checks.append(("...so a hash alone cannot tell them apart -- hence the array",
                   h(np.roll(f, 1, 0)) != h(f) and h(g) != h(f)))
    checks.append(("digest pin is a full sha256", len(MAC_PIXEL_FP) == 64))
    checks.append(("bar is NEW_PLAN §13's: 12 h / 600 segments", BAR_S_PER_SEGMENT == 72.0))
    checks.append(("segment frame count is 60 s at target_fps 8.0", SEG_FRAMES == 480))
    # This probe must never score. Scoring needs the checkpoint and the threshold, and a
    # number produced here would look like evidence while bypassing GATE C entirely.
    # Search only the OPERATIONAL code -- everything above this function. Searching the
    # whole file would match the needle inside this very check (it did, twice).
    operational = open(os.path.abspath(__file__)).read().split("def self_check")[0]
    checks.append(("operational code never invokes the scorer or loads the model",
                   "score_external" not in operational and "load_vjepa" not in operational))
    for name, ok in checks:
        print(f"{'ok ' if ok else 'FAIL'} {name}")
    bad = [n for n, ok in checks if not ok]
    print("PASS" if not bad else f"FAILED: {bad}")
    return 0 if not bad else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-dir", default=None)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        return self_check()

    probe_dir = find_probe_dir(a.probe_dir)
    print(f"probe files: {probe_dir}")
    root = stage_bundle(probe_dir)
    sys.path.insert(0, os.path.join(root, "vendor", "badas-open"))
    try:
        from badas.utils.video import load_full_video_frames
    except Exception as e:
        raise SystemExit(f"could not import the vendored decoder: {e}\n"
                         "Never reimplement it -- that would test itself, not the real path.")

    report_stack()
    verdict, _ = compare(probe_dir, load_full_video_frames)
    ratio = timings(probe_dir, load_full_video_frames)

    print("\n" + "=" * 70)
    print(f"SUMMARY   decode vs Mac: {verdict}   |   projected pilot: {ratio:.1f}x the bar")
    print("Send BOTH to the user. This probe scores nothing and decides nothing on its own.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
