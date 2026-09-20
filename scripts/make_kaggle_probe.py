"""Assemble the ~14 MB Kaggle probe dataset.

WHY. Moving from Colab to Kaggle changes the machine, and GATE C (D57) is a statement about
a machine. Colab's GATE C result -- its failure, and any fix for it -- transfers to Kaggle
not at all. Before porting a 23-cell notebook and uploading a 3.98 GB checkpoint, ask the
cheap question: does Kaggle's cv2 decode like the Mac's?

Everything needed for that fits in ~14 MB:
    01044.mp4              one Nexar negative -- the clip the fingerprint was measured on
    01044_frames.npz       the Mac's decoded array, so a mismatch can be CLASSIFIED
    colab_bundle.tar.gz    the vendored decoder (never reimplement it)
    probe.py               scripts/kaggle_decode_probe.py, copied in so the dataset is
                           self-contained and the Kaggle cell is a single line
    decode_experiments.py  the two follow-up experiments: WHERE the Mac/Linux pixel
                           difference comes from, and whether sequential reading can
                           replace per-frame seeking losslessly

    ~/envs/badas/bin/python scripts/make_kaggle_probe.py
    # then upload runs/kaggle_probe/ to Kaggle as a new private Dataset
"""

import hashlib
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "runs", "kaggle_probe")
CLIP = "01044"

SOURCES = {
    f"{CLIP}.mp4": os.path.join(REPO, "data", "nexar", "test-public", "negative", f"{CLIP}.mp4"),
    f"{CLIP}_frames.npz": os.path.join(REPO, "runs", "pixel_ref", f"{CLIP}_frames.npz"),
    "colab_bundle.tar.gz": os.path.join(REPO, "runs", "colab_bundle.tar.gz"),
    "probe.py": os.path.join(REPO, "scripts", "kaggle_decode_probe.py"),
    "decode_experiments.py": os.path.join(REPO, "scripts", "decode_experiments.py"),
}

README = """# crash-detection decode probe

~14 MB. Answers one question: does this machine's OpenCV decode a video to the same bytes
the Mac did? Needs no GPU, no model checkpoint, ~2 minutes.

In a Kaggle notebook, after adding this dataset as an input:

    !python /kaggle/input/<dataset-slug>/probe.py

Send the whole output back. It scores nothing and decides nothing by itself.
"""


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(block), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    missing = [n for n, p in SOURCES.items() if not os.path.exists(p)]
    if missing:
        print("MISSING inputs:")
        for n in missing:
            print(f"  {n}  -> {SOURCES[n]}")
        if f"{CLIP}_frames.npz" in missing:
            print("\nBuild the reference first:")
            print("  ~/envs/badas/bin/python scripts/export_pixel_reference.py")
        return 1

    os.makedirs(OUT, exist_ok=True)
    total = 0
    for name, src in SOURCES.items():
        dst = os.path.join(OUT, name)
        shutil.copy2(src, dst)
        n = os.path.getsize(dst)
        total += n
        print(f"  {name:<22} {n / 2**20:7.2f} MiB   {sha256(dst)[:16]}…")
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write(README)

    print(f"\n{OUT}  ({total / 2**20:.1f} MiB total)")
    print("\nNEXT")
    print("  1. kaggle.com > Datasets > New Dataset > upload this folder (private is fine)")
    print("  2. New Notebook > Add Input > your dataset. GPU NOT required for the probe.")
    print("  3. Run:  !python /kaggle/input/<dataset-slug>/probe.py")
    print("  4. Send the whole output back.")
    print("\nTHEN the two experiments (same dataset, no GPU needed):")
    print("  !python /kaggle/input/<dataset-slug>/decode_experiments.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
