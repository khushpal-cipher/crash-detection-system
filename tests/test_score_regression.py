"""Regression guard on the old model's scoring path: videos/safe.mp4 must score 0.7914.

Why this number is worth pinning even though the model is retired:

  * 0.7914 is the figure the ORIGINAL Keras pipeline reported as "0.79" in the pre-audit
    README, reproduced exactly by the NumPy reimplementation in
    `scripts/t124_model_falsification.py`. Two independent implementations agreeing to four
    decimals is what licenses every falsification result built on that path -- including T3.
  * It is also how T3 was falsified before its result was accepted: the same scoring code
    that collapsed to chance on Nexar still discriminates on the local videos, so the
    collapse is the model's and not the harness's (README §5, §12.3).
  * `CNN_THRESH = 0.80` clears this negative by 0.0086 -- the whole deployed operating point
    rests on this one number, so if it ever moves, something upstream broke silently.

Imports `score_clip` from the evidence script rather than reimplementing it, so this test
guards the actual committed path. progress.md §12 forbids refactoring that script; importing
it is not refactoring.

Run:  ~/envs/crashdet/bin/python tests/test_score_regression.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

EXPECTED = 0.7914
TOL = 5e-4


def main():
    np.seterr(all="ignore")  # cosmetic arm64 BLAS warnings; see progress.md §4.6

    video = os.path.join(ROOT, "videos", "safe.mp4")
    assert os.path.exists(video), f"missing {video}"

    # Module-level import loads the SavedModel feature extractor (slow, ~10 s).
    from t3_corpus_control import score_clip

    score, n_frames, fps = score_clip(video)
    print(f"safe.mp4: score={score:.4f}  frames={n_frames}  fps={fps:.1f}")

    assert score is not None, "clip was unscorable"
    assert abs(score - EXPECTED) < TOL, (
        f"REGRESSION: safe.mp4 scored {score:.4f}, expected {EXPECTED} "
        f"(tol {TOL}). The scoring path changed -- every falsification result that "
        f"depends on it, T3 included, is now suspect. Investigate before proceeding."
    )
    print(f"ok  reproduces {EXPECTED} within {TOL}")
    print(f"ok  CNN_THRESH=0.80 clears this negative by {0.80 - score:.4f}")
    print("PASS")


if __name__ == "__main__":
    main()
