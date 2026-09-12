"""U-B6: time ONE BADAS window on MPS before committing to a 42,700-pass sweep.

Also answers the empirical half of U-B4 (does V-JEPA2 actually run on MPS, or does it
silently fall back to CPU?) and the U-B3 normalization question (which preprocessing
path actually runs -- AutoVideoProcessor or the ImageNet manual fallback).

Run:  PYTORCH_ENABLE_MPS_FALLBACK=1 ~/envs/badas/bin/python scripts/badas_smoke.py
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
N_TIMED = 5


def main():
    device = os.environ.get("BADAS_DEVICE", "mps")
    print(f"torch {torch.__version__}  device={device}  mps_avail={torch.backends.mps.is_available()}")

    from badas.models.vjepa import VJEPAModel

    t0 = time.time()
    m = VJEPAModel(
        model_name=MODEL_NAME,
        checkpoint_path=CKPT,
        frame_count=FRAMES,
        img_size=IMG,
        window_stride=1,
        target_fps=8.0,
        use_sliding_window=True,
        device=device,
    )
    m.load()
    print(f"load: {time.time() - t0:.1f}s")

    # Which preprocessing path is live? (U-B3)
    print(f"processor: {type(m.processor).__name__ if m.processor else 'None -> ImageNet manual fallback'}")
    print(f"model device: {next(m.model.parameters()).device}")

    # One window straight through the model, bypassing video IO so this times
    # compute only. Shape is the (B, T, C, H, W) that EnhancedVideoClassifier.forward wants.
    x = torch.randn(1, FRAMES, 3, IMG, IMG, device=device)

    with torch.no_grad():
        m.model(x)  # warm-up: first MPS call compiles kernels, never count it
        if device == "mps":
            torch.mps.synchronize()

        times = []
        for _ in range(N_TIMED):
            t = time.time()
            out = m.model(x)
            if device == "mps":
                torch.mps.synchronize()
            times.append(time.time() - t)

    per = float(np.median(times))
    print(f"logits {tuple(out.shape)}  finite={bool(torch.isfinite(out).all())}")
    print(f"per-window: median {per:.3f}s  (min {min(times):.3f} max {max(times):.3f})")

    # U-B6 extrapolation. ~9.93s clip @ 8fps -> ~79 frames -> 79-16+1 windows, stride 1.
    windows = 79 - FRAMES + 1
    total = windows * 667 * per
    print(f"\nU-B6: ~{windows} windows/clip x 667 clips = {windows * 667:,} passes")
    print(f"      projected full sweep: {total / 3600:.1f} h at this rate")
    for s in (2, 4, 8):
        print(f"      stride {s}: {total / s / 3600:.1f} h  (DEVIATES from published config)")


if __name__ == "__main__":
    main()
