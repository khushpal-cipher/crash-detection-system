"""CPU vs MPS throughput for one BADAS window — the measurement NEW_PLAN.md §7.3 gates on.

Every timing in this project to date has been MPS. §7.3 therefore FORBIDS the words
"real-time" or "CPU-capable" anywhere until both devices are measured and both reported
(D27). This is that measurement, and nothing here decides it in advance: the verdict is
computed from the measured numbers, not asserted.

WHAT REAL-TIME MEANS HERE, stated before measuring so the bar cannot move afterwards:
the deployment shape is a 1 Hz alert cadence -- one scored window per wall-clock second.
A window is 16 frames at 8 fps = 2.0 s of video, but the cadence is what must be kept up
with, not the window's span. So the budget is 1/cadence seconds per window, and the
system is real-time at that cadence iff per_window <= budget. Reported as a ratio, so a
reader can re-derive any other cadence.

COMPUTE ONLY, and that is a real limitation. This times the model forward pass on a
synthetic tensor, bypassing video IO -- the same method as scripts/badas_smoke.py, whose
0.856 s/window MPS figure is D10 and the basis of every sweep estimate since. End-to-end
is much worse: ~97 s/clip against ~64 windows x 0.86 s = ~55 s of compute, so DECODE IS
ALREADY THE DOMINANT COST (NEW_PLAN.md §11 point 4 flags this as the plan's least-verified
assumption). A deployed system decodes a live stream rather than seeking around a file, so
its decode cost differs from ours -- which is exactly why this reports compute, and says so,
rather than quietly extrapolating a product claim from a benchmark harness.

MEASUREMENT SPREAD, reported rather than engineered away: both devices are timed in one
process, and the second one measured runs slightly slower for it. CPU came out at 1.680 s
in a fresh process (via badas_smoke.py with BADAS_DEVICE=cpu) and 1.917 s here, after MPS;
MPS at 0.860 s and 0.871 s. The verdict is stable across that spread -- every CPU figure is
above the 1.00 s budget and every MPS figure below it -- so the conclusion does not rest on
the difference. Re-run a single device through badas_smoke.py if a tighter number is needed.

Run:  PYTORCH_ENABLE_MPS_FALLBACK=1 ~/envs/badas/bin/python scripts/hw_bench.py
      ~/envs/badas/bin/python scripts/hw_bench.py --self-check
"""

import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor", "badas-open"))

CKPT = os.path.join(ROOT, "models", "badas", "weights", "badas_open.pth")
MODEL_NAME = "facebook/vjepa2-vitl-fpc16-256-ssv2"
FRAMES, IMG = 16, 224          # badas_loader.py:44-45 -- 224, NOT 256
TARGET_FPS = 8.0               # upstream's default; §12 forbids changing it
CADENCE_HZ = 1.0               # the deployment shape §7.3 names
N_TIMED = 5


def feasible(per_window_s, cadence_hz=CADENCE_HZ):
    """(ratio, ok) for one window against a cadence budget. ratio < 1 means it keeps up."""
    budget = 1.0 / cadence_hz
    return per_window_s / budget, per_window_s <= budget


def time_window(device, n=N_TIMED):
    """Median seconds for one forward pass. Returns None if the device is unavailable."""
    if device == "mps" and not torch.backends.mps.is_available():
        return None
    from badas.models.vjepa import VJEPAModel

    m = VJEPAModel(
        model_name=MODEL_NAME, checkpoint_path=CKPT, frame_count=FRAMES, img_size=IMG,
        window_stride=1, target_fps=TARGET_FPS, use_sliding_window=True, device=device,
    )
    m.load()
    actual = str(next(m.model.parameters()).device)
    # MPS can silently fall back to CPU; assert we measured what we asked for (U-B4).
    assert actual.startswith(device), f"asked {device}, got {actual}"

    x = torch.randn(1, FRAMES, 3, IMG, IMG, device=device)
    with torch.no_grad():
        m.model(x)                                  # warm-up: never counted
        if device == "mps":
            torch.mps.synchronize()
        times = []
        for _ in range(n):
            t = time.time()
            out = m.model(x)
            if device == "mps":
                torch.mps.synchronize()
            times.append(time.time() - t)
    assert bool(torch.isfinite(out).all()), f"{device} produced non-finite logits"
    del m
    return float(np.median(times))


def self_check():
    """The arithmetic only -- no model, no device, runs anywhere in under a second."""
    r, ok = feasible(0.5, 1.0)
    assert (round(r, 6), ok) == (0.5, True), (r, ok)
    r, ok = feasible(2.0, 1.0)
    assert (round(r, 6), ok) == (2.0, False), (r, ok)
    print("ok  under budget -> ratio < 1 and feasible; over budget -> ratio > 1 and not")

    r, ok = feasible(1.0, 1.0)
    assert ok and round(r, 6) == 1.0, (r, ok)
    print("ok  exactly at budget counts as keeping up")

    # A harder cadence must never look easier.
    assert feasible(0.8, 2.0)[0] > feasible(0.8, 1.0)[0]
    assert feasible(0.8, 2.0)[1] is False and feasible(0.8, 1.0)[1] is True
    print("ok  raising the cadence raises the ratio and can flip the verdict")

    # Guard the claim direction itself: only a measured number may grant "real-time".
    assert not feasible(float("inf"))[1]
    print("ok  an unmeasurable device is never reported as real-time")
    print("PASS")


def main():
    print(f"torch {torch.__version__}  mps_avail={torch.backends.mps.is_available()}")
    print(f"window: {FRAMES} frames @ {TARGET_FPS:g} fps = {FRAMES / TARGET_FPS:.1f}s of video")
    print(f"budget: {1.0 / CADENCE_HZ:.2f}s per window at a {CADENCE_HZ:g} Hz alert cadence\n")

    results = {}
    for device in ("mps", "cpu"):
        per = time_window(device)
        results[device] = per
        if per is None:
            print(f"{device:4s}  unavailable")
            continue
        ratio, ok = feasible(per)
        print(f"{device:4s}  {per:.3f}s/window   {ratio:.2f}x budget   "
              f"{'KEEPS UP' if ok else 'TOO SLOW'} at {CADENCE_HZ:g} Hz")

    mps, cpu = results.get("mps"), results.get("cpu")
    if mps and cpu:
        print(f"\nCPU is {cpu / mps:.2f}x slower than MPS.")

    print("\nPermitted language, derived from the above and nothing else:")
    for device in ("mps", "cpu"):
        per = results.get(device)
        if per is None:
            continue
        label = "Apple Silicon GPU (MPS)" if device == "mps" else "Apple Silicon CPU"
        verb = "IS" if feasible(per)[1] else "is NOT"
        print(f"  {label}: {verb} real-time at a {CADENCE_HZ:g} Hz alert cadence, "
              f"compute only ({per:.3f}s/window measured).")
    print("  Decode cost is excluded and is already the dominant term end-to-end.")
    print("  No claim about any other hardware, cadence, or deployment shape is licensed.")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    else:
        main()
