#!/usr/bin/env python3
"""U6 (README §41 Phase 0 task 7): does models/crash_model_cpu/ hold the weights that ship?

The shipped artefact is `models/crash_model_weights.weights.h5` (Keras 3). Beside it sits
`models/crash_model_cpu/`, a TensorFlow SavedModel written by Colab cell 11. README §1
finding 3 proved the .h5 came from a DIFFERENT run than the metrics recorded in the
notebook. The open question is which side of that divide the SavedModel falls on -- i.e.
whether cells 11 and 12 ran in the same kernel session.

Matched by an explicit name pair table. The SavedModel exposes opaque `variables/N` paths
with no layer names, and shape alone is NOT a safe key -- (256,) is both the dense bias and
the lstm_1 bias. The table below pairs them in architecture order (dense -> lstm -> lstm_1
-> dense_1 -> dense_2), which is the order Keras writes variables in; every pair's shapes
are asserted equal, so a wrong pairing fails loudly instead of reporting a false r.

    ~/envs/crashdet/bin/python scripts/u6_compare_weights.py
"""
import os
import sys

import h5py
import numpy as np
import tensorflow as tf

np.seterr(all="ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5 = os.path.join(ROOT, "models/crash_model_weights.weights.h5")
CKPT = os.path.join(ROOT, "models/crash_model_cpu/variables/variables")


# (SavedModel variable, .h5 dataset) in Keras variable-creation order.
PAIRS = [
    ("variables/0",  "layers/dense/vars/0"),
    ("variables/1",  "layers/dense/vars/1"),
    ("variables/3",  "layers/lstm/cell/vars/0"),
    ("variables/4",  "layers/lstm/cell/vars/1"),
    ("variables/5",  "layers/lstm/cell/vars/2"),
    ("variables/8",  "layers/lstm_1/cell/vars/0"),
    ("variables/9",  "layers/lstm_1/cell/vars/1"),
    ("variables/10", "layers/lstm_1/cell/vars/2"),
    ("variables/13", "layers/dense_1/vars/0"),
    ("variables/14", "layers/dense_1/vars/1"),
    ("variables/16", "layers/dense_2/vars/0"),
    ("variables/17", "layers/dense_2/vars/1"),
]
SUFFIX = "/.ATTRIBUTES/VARIABLE_VALUE"


def main():
    reader = tf.train.load_checkpoint(CKPT)
    f = h5py.File(H5, "r")
    same = diff = 0
    print(f"{len(PAIRS)} weight arrays compared "
          f"(optimizer state excluded from both sides)\n")
    for sm, hk in PAIRS:
        v = reader.get_tensor(sm + SUFFIX)
        w = f[hk][()]
        assert v.shape == w.shape, f"{sm} {v.shape} vs {hk} {w.shape} -- pairing is wrong"
        eq = np.allclose(v, w, atol=1e-6)
        r = float(np.corrcoef(v.ravel(), w.ravel())[0, 1]) if v.size > 1 else float("nan")
        same, diff = same + eq, diff + (not eq)
        print(f"  {str(v.shape):<14} {hk:<28} "
              f"{'IDENTICAL' if eq else 'DIFFERENT'}  r={r:+.4f}")
    f.close()

    print(f"\n{same} identical, {diff} different, of {len(PAIRS)}")
    print("VERDICT: " + (
        "SAME RUN -- models/crash_model_cpu/ holds the weights that ship. Colab cells 11 "
        "and 12 ran in one session."
        if diff == 0 else
        "DIFFERENT RUNS -- models/crash_model_cpu/ is a THIRD artefact. It does not "
        "describe the shipped weights either."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
