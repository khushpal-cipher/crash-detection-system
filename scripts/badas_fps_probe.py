#!/usr/bin/env python3
"""U-B7: is `original_fps: 4` really in conflict with `target_fps: 8.0`?

README §41 Phase 5's gate note lists this as reason 5 for restating the gate:
"`original_fps: 4` contradicts `target_fps: 8.0` in the config, and is unresolved."

It is NOT a contradiction. The two keys count different things, and every number in
the BADAS training config reconciles once that is seen:

    16 video frames @ target_fps 8.0      = 2.0 s                 <- `balanced_dataset_2s`
    V-JEPA2 tubelet_size 2                => 16 / 2 = 8 tokens
    8 tokens over 2.0 s                   = 4 tokens per second   <- `original_fps: 4`
    future_prediction_seconds 1.0 x 4     = 4 token steps ahead

So `target_fps` is the VIDEO sampling rate and `original_fps` is the post-tubelet TOKEN
rate that turns `future_prediction_seconds` into a token offset. Both are correct at once.

Why it matters: a real 2x temporal-scale mismatch here would be this project's own B1 bug
(README §15) repeated on a new backbone -- a model trained at one time-scale and run at
another. It is not. The sweep configuration is faithful.

    ~/envs/badas/bin/python scripts/badas_fps_probe.py
"""
import os
import sys

import torch
from transformers import AutoConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor", "badas-open"))

CKPT = os.path.join(ROOT, "models", "badas", "weights", "badas_open.pth")


def main():
    cfg = torch.load(CKPT, map_location="cpu", weights_only=False)["config"]
    vj = AutoConfig.from_pretrained(cfg["model_name"])
    from badas.badas_loader import preprocess_video  # upstream's own default

    up_fps = preprocess_video.__defaults__[0]  # target_fps
    frames, orig_fps = cfg["frame_count"], cfg["original_fps"]
    tubelet = vj.tubelet_size

    print(f"checkpoint : frame_count={frames}  original_fps={orig_fps}  "
          f"future_prediction_seconds={cfg['future_prediction_seconds']}")
    print(f"             data_root={cfg['data_root']}")
    print(f"vjepa2     : tubelet_size={tubelet}  frames_per_clip={vj.frames_per_clip}")
    print(f"upstream   : preprocess_video target_fps default = {up_fps}")

    seconds = frames / up_fps
    tokens = frames // tubelet
    token_rate = tokens / seconds
    print(f"\n{frames} frames @ {up_fps} fps = {seconds:.1f} s   "
          f"-> {tokens} tokens -> {token_rate:.1f} tokens/s")

    assert abs(seconds - 2.0) < 1e-6, f"{seconds} s does not match balanced_dataset_2s"
    assert "2s" in cfg["data_root"], cfg["data_root"]
    assert abs(token_rate - orig_fps) < 1e-6, \
        f"token rate {token_rate} != original_fps {orig_fps} -- the reconciliation fails"

    print(f"\nRESOLVED: original_fps {orig_fps} IS the token rate, not a rival video fps.")
    print(f"  future_prediction_seconds {cfg['future_prediction_seconds']} x {orig_fps} "
          f"= {int(cfg['future_prediction_seconds'] * orig_fps)} token steps ahead")
    print(f"  Inference at target_fps={up_fps}, frame_count={frames} is FAITHFUL to "
          "training. No B1-style temporal mismatch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
