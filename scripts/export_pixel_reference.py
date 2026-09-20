"""Export the Mac's DECODED frames for one clip, so Colab can diff against them.

WHY THIS EXISTS. GATE C failed on 2026-09-19 with median |delta| 0.004384 against a 0.002
bar, and its pixel fingerprint DIFFERED from the Mac's:

    here  d43cf8efa32e87d0a6baf03f749a10afcae08c1f41457947bcd2d4976abfefdb
    mac   bd6126c391b2301d170e40128d04338587a38697a7a494c63943d4182a23f24a

That tells us the decode path differs -- cv2 reads different pixels on the two machines,
before the model runs -- and therefore exonerates the GPU. It does NOT tell us HOW they
differ, and the two possibilities need different responses:

  (a) cv2.resize / YUV->BGR rounding differs (ARM NEON vs x86 SIMD). Every frame is the
      SAME frame, off by 1-2 grey levels. Cosmetic: the two machines watched the same video.
  (b) cap.set(CAP_PROP_POS_FRAMES, k) lands on a DIFFERENT frame. Seek accuracy varies
      across ffmpeg builds. Colab is then watching different footage, and any "fix" that
      merely gets median |delta| under the bar would be papering over that.

A hash cannot distinguish them; only the array can. 12.2 MB uncompressed is trivially
uploadable, so ship the array and let Colab do the arithmetic (notebook cell 2b).

The digest below is a PIN, not a record: if this script ever stops reproducing it, the
Mac's own decode path has changed and every committed score is in question. That is a
stop-and-tell-the-user event, not something to work around.

    ~/envs/badas/bin/python scripts/export_pixel_reference.py --self-check
    ~/envs/badas/bin/python scripts/export_pixel_reference.py
    # then upload runs/pixel_ref/01044_frames.npz to
    #   MyDrive/crash_detection_colab/
"""

import argparse
import hashlib
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vendor", "badas-open"))

# The fingerprint embedded in scripts/colab_comma2k19.ipynb §2, measured on this Mac in
# session 15. load_full_video_frames(path, (224, 224), 8.0) -> sha256 of the uint8 array.
MAC_PIXEL_FP = "bd6126c391b2301d170e40128d04338587a38697a7a494c63943d4182a23f24a"
CLIP = "01044"
SHAPE = (81, 224, 224, 3)

# target_size is (width, height) and target_fps 8.0 -- exactly what the notebook, and
# scripts/score_external.py through the adapter, pass. Do not "tidy" these into constants
# with different values; they are the values the committed 92.3 was produced with.
TARGET_SIZE = (224, 224)
TARGET_FPS = 8.0

DEFAULT_CLIPS_DIR = os.path.join(REPO, "data", "nexar", "test-public", "negative")
DEFAULT_OUT_DIR = os.path.join(REPO, "runs", "pixel_ref")


def decode(video_path):
    """Decode via the vendored function -- the same code Colab runs. Never reimplement it."""
    from badas.utils.video import load_full_video_frames

    return load_full_video_frames(video_path, TARGET_SIZE, TARGET_FPS)


def fingerprint(frames):
    return hashlib.sha256(frames.tobytes()).hexdigest()


def export(clips_dir, out_dir, clip):
    video = os.path.join(clips_dir, f"{clip}.mp4")
    if not os.path.exists(video):
        raise SystemExit(f"clip not found: {video}")

    frames = decode(video)
    got = fingerprint(frames)

    print(f"clip        {clip}")
    print(f"shape       {frames.shape}   dtype {frames.dtype}")
    print(f"fingerprint {got}")

    if clip == CLIP:
        print(f"expected    {MAC_PIXEL_FP}")
        if frames.shape != SHAPE:
            raise SystemExit(
                f"SHAPE CHANGED: {frames.shape} != {SHAPE}. The Mac's own decode path has "
                "moved. STOP -- every committed score used the old one."
            )
        if got != MAC_PIXEL_FP:
            raise SystemExit(
                "FINGERPRINT CHANGED on the Mac itself. This is not a Colab problem.\n"
                "Something in this environment's cv2 changed since session 15, which puts "
                "every committed score in question. STOP and tell the user."
            )
        print("  -> matches the fingerprint the notebook compares against  ✅")

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{clip}_frames.npz")
    # savez_compressed, not savez: ~12.2 MB raw, and this crosses a network by hand.
    np.savez_compressed(out, frames=frames, fingerprint=np.array(got))
    mb = os.path.getsize(out) / 2**20
    print(f"\nwrote {out}  ({mb:.1f} MiB, from {frames.nbytes / 2**20:.1f} MiB raw)")
    print("\nNEXT: upload that file to MyDrive/crash_detection_colab/ then run the")
    print("notebook's §2b diagnostic cell. It needs no checkpoint and no GPU.")
    return out


def self_check():
    """Checks the reasoning, not just the plumbing: that the fingerprint is sensitive to a
    one-frame shift at all. If it were not, the whole diagnostic would be worthless."""
    checks = []

    rng = np.random.default_rng(0)
    fake = rng.integers(0, 256, size=(8, 4, 4, 3), dtype=np.uint8)

    a = fingerprint(fake)
    checks.append(("fingerprint is stable on identical input", a == fingerprint(fake.copy())))

    shifted = np.roll(fake, 1, axis=0)
    checks.append(("a ONE-FRAME shift changes it", fingerprint(shifted) != a))

    nudged = fake.copy()
    nudged[0, 0, 0, 0] = (int(nudged[0, 0, 0, 0]) + 1) % 256
    checks.append(("a ONE-GREY-LEVEL change changes it too", fingerprint(nudged) != a))

    # ...which is exactly why a hash alone cannot tell (a) from (b) above, and why this
    # script ships the array.
    checks.append(
        ("so a hash cannot distinguish a shift from rounding -- hence the array",
         fingerprint(shifted) != a and fingerprint(nudged) != a)
    )

    checks.append(("digest pin is a full sha256", len(MAC_PIXEL_FP) == 64))
    checks.append(
        ("vendored decoder is importable, not reimplemented here",
         __import__("badas.utils.video", fromlist=["load_full_video_frames"]) is not None)
    )

    for name, ok in checks:
        print(f"{'ok ' if ok else 'FAIL'} {name}")
    bad = [n for n, ok in checks if not ok]
    print("PASS" if not bad else f"FAILED: {bad}")
    return 0 if not bad else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips-dir", default=DEFAULT_CLIPS_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--clip", default=CLIP)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()

    if a.self_check:
        return self_check()
    export(a.clips_dir, a.out_dir, a.clip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
