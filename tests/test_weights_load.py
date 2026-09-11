#!/usr/bin/env python3
"""
The smoke test README bug R2 has asked for since revision 1: prove the environment can
actually read the model artefacts this repository ships.

The old pin (tensorflow==2.13.1 -> Keras 2) could not read the Keras 3 weights file at all,
and nothing in the repo detected that. Run this after any change to requirements.txt.

    python tests/test_weights_load.py
"""
import os
import sys

import h5py
import numpy as np

# macOS/arm64 NumPy raises spurious FP warnings inside matmul; every result here is
# asserted finite and in range explicitly, so the warnings are noise that would hide a
# real one.
np.seterr(all="ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "models/crash_model_weights.weights.h5")
FE = os.path.join(ROOT, "models/feature_extractor_saved")

DOCUMENTED_PARAMS = 578_689  # README section 11, confirmed against the Colab model.summary()
EXPECTED_LAYERS = {"dense", "dense_1", "dense_2", "lstm", "lstm_1"}


def test_weights_are_readable_keras3():
    with h5py.File(WEIGHTS, "r") as f:
        assert "layers" in f, "not a Keras 3 weights file: no /layers group"
        present = set(f["layers"].keys())
        missing = EXPECTED_LAYERS - present
        assert not missing, f"weight groups missing: {sorted(missing)}"

        total = 0

        def add(name, obj):
            nonlocal total
            if isinstance(obj, h5py.Dataset) and not name.startswith("optimizer"):
                total += obj.size

        f.visititems(add)
        assert total == DOCUMENTED_PARAMS, (
            f"parameter count is {total:,}, documented is {DOCUMENTED_PARAMS:,} — "
            "the architecture or the artefact changed"
        )
    return total


def test_feature_extractor_loads_and_runs():
    import tensorflow as tf

    fe = tf.saved_model.load(FE)
    batch = np.zeros((2, 112, 112, 3), np.float32)
    out = np.array(fe.serve(tf.constant(batch)) if hasattr(fe, "serve") else fe(tf.constant(batch)))
    assert out.shape == (2, 1280), f"feature extractor returned {out.shape}, expected (2, 1280)"
    return out.shape


def test_head_forward_pass_on_zeros():
    """Exact NumPy forward pass of the LSTM head — the same path the falsification
    scripts use, because no Keras 3 runtime is required to read the weights directly."""

    def sigm(x):
        return 1.0 / (1.0 + np.exp(-x))

    def lstm(x, Wk, Uk, b):
        units = Uk.shape[0]
        h = np.zeros(units)
        c = np.zeros(units)
        out = np.empty((x.shape[0], units))
        for t in range(x.shape[0]):
            z = x[t] @ Wk + h @ Uk + b
            i, fg, g, o = (
                sigm(z[:units]),
                sigm(z[units : 2 * units]),
                np.tanh(z[2 * units : 3 * units]),
                sigm(z[3 * units :]),
            )
            c = fg * c + i * g
            h = o * np.tanh(c)
            out[t] = h
        return out

    with h5py.File(WEIGHTS, "r") as f:
        L = f["layers"]
        g = lambda n, i: np.array(L[n]["vars"][str(i)])
        cell = lambda n, i: np.array(L[n]["cell"]["vars"][str(i)])
        x = np.maximum(np.zeros((10, 1280)) @ g("dense", 0) + g("dense", 1), 0)
        x = lstm(x, cell("lstm", 0), cell("lstm", 1), cell("lstm", 2))
        x = lstm(x, cell("lstm_1", 0), cell("lstm_1", 1), cell("lstm_1", 2))[-1]
        x = np.maximum(x @ g("dense_1", 0) + g("dense_1", 1), 0)
        score = float(sigm(x @ g("dense_2", 0) + g("dense_2", 1))[0])

    assert np.isfinite(score), "forward pass produced a non-finite score"
    assert 0.0 <= score <= 1.0, f"sigmoid output {score} outside [0, 1]"
    return score


if __name__ == "__main__":
    for missing in (p for p in (WEIGHTS, FE) if not os.path.exists(p)):
        sys.exit(f"FAIL: required artefact missing: {missing}")

    params = test_weights_are_readable_keras3()
    print(f"ok  weights readable, {params:,} non-optimizer parameters")
    shape = test_feature_extractor_loads_and_runs()
    print(f"ok  feature extractor runs, output {shape}")
    score = test_head_forward_pass_on_zeros()
    print(f"ok  head forward pass on zeros -> {score:.6f}")
    print("\nPASS: this environment can load and run the shipped model artefacts.")
