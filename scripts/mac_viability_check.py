"""Does the sequential decoder reproduce the COMMITTED SCORES on the Mac, and how fast?

The fingerprint proves the frames match. This proves the whole path matches -- decode,
preprocessing, model, reduction -- by comparing the final per-clip score against
runs/baselines/badas-open/scores.jsonl, the file that produced AP 0.8349 and 92.3 FP/hour.

It ALSO answers a question the Kaggle work has made urgent: if the decode fix makes the Mac
fast enough, the whole cloud comparability problem disappears, because the committed
baseline was produced ON THIS MACHINE.

Nothing is modified on disk. The decoder is swapped in-process only.
"""
import json
import os
import sys
import time

REPO = "/Users/khushpalsinghchouhan/dev/crash_detection/crash_detection_v2"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "vendor", "badas-open"))

from scripts.decode_experiments import load_sequential  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3

ref = {}
with open(os.path.join(REPO, "runs/baselines/badas-open/scores.jsonl")) as f:
    for line in f:
        r = json.loads(line)
        if "score" in r:
            ref[r["id"]] = float(r["score"])

neg_dir = os.path.join(REPO, "data/nexar/test-public/negative")
ids = sorted(os.path.splitext(f)[0] for f in os.listdir(neg_dir) if f.endswith(".mp4"))[:N]

import badas.utils.video as bv  # noqa: E402
from eval.adapters import BadasOpen  # noqa: E402

original = bv.load_full_video_frames


def run(label, patched, skip_predictor):
    bv.load_full_video_frames = patched
    model = BadasOpen(device="mps", stride=1, skip_predictor=skip_predictor).load()
    out, t_total = {}, 0.0
    for cid in ids:
        t0 = time.perf_counter()
        out[cid] = model.score(os.path.join(neg_dir, f"{cid}.mp4"))
        t_total += time.perf_counter() - t0
    bv.load_full_video_frames = original
    print(f"\n{label}")
    worst = 0.0
    for cid in ids:
        d = abs(out[cid] - ref[cid])
        worst = max(worst, d)
        print(f"   {cid}  got {out[cid]:.10f}   committed {ref[cid]:.10f}   |d| {d:.2e}")
    print(f"   max |delta| vs committed : {worst:.3e}   {'IDENTICAL' if worst == 0 else 'DIFFERS'}")
    print(f"   {t_total/len(ids):.1f} s/clip  ({t_total:.1f} s for {len(ids)})")
    return worst, t_total / len(ids)


print(f"Mac end-to-end check on {N} clips: {ids}")
w_seek, t_seek = run("A) SEEKING decoder (the committed path), predictor ON", original, False)
w_seq, t_seq = run("B) SEQUENTIAL decoder, predictor ON", load_sequential, False)
w_both, t_both = run("C) SEQUENTIAL decoder + skip_predictor", load_sequential, True)

BAR, SEG, W = 72.0, 480, 16
windows = (SEG - W) / 8.0
print("\n" + "=" * 72)
print("WHAT THIS MEANS FOR RUNNING comma2k19 ON THE MAC")
print("=" * 72)
print(f"  A seeking            {t_seek:6.1f} s/clip   scores {'match' if w_seek == 0 else 'DIFFER'}")
print(f"  B sequential         {t_seq:6.1f} s/clip   scores {'match' if w_seq == 0 else 'DIFFER'}"
      f"   ({t_seek/max(t_seq,1e-9):.2f}x faster)")
print(f"  C seq+skip_predictor {t_both:6.1f} s/clip   scores {'match' if w_both == 0 else 'DIFFER'}"
      f"   ({t_seek/max(t_both,1e-9):.2f}x faster)")
# A Nexar clip is ~77 frames / ~61 windows at stride 1. A comma2k19 segment at stride 8 is
# 480 frames and 58 scored windows -- so per-window cost transfers, clip length does not.
for tag, t in (("B", t_seq), ("C", t_both)):
    per_window = t / 61.0
    seg = per_window * windows + 1.3      # +decode, measured ~0.0026 s/frame * 480
    print(f"  -> {tag}: {per_window:.3f} s/window -> 60 s segment {seg:5.1f} s = {seg/BAR:.2f}x bar"
          f"   {seg*600/3600:5.1f} h per 10 h of footage  {'OK' if seg*600/3600 <= 12 else 'OVER'}")
print("\n  If this fits, the Mac needs NO threshold re-derivation: the committed 92.3 was")
print("  produced here, so comparability is free rather than bought with ~10 h of quota.")
