#!/usr/bin/env python3
"""
README §41 Phase 4 acceptance criterion 3: "The leakage test fails the build on an
injected violation." Proves two things about eval.benchmark.assert_no_leakage:

1. It does NOT fire on the real split -- Nexar's train ids (data/nexar/train/*/metadata.csv,
   scaffolding + labels only, no video downloaded yet) against test-public ids (the split
   every eval/ run actually scores).
2. It DOES fire -- raises AssertionError -- on a deliberately injected overlap.

Run:
    python tests/test_leakage.py
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

from benchmark import assert_no_leakage, load_labels  # noqa: E402

NEXAR = os.path.join(ROOT, "data", "nexar")


def _train_ids():
    ids = []
    for cls in ("positive", "negative"):
        with open(os.path.join(NEXAR, "train", cls, "metadata.csv")) as f:
            ids += [os.path.splitext(row["file_name"])[0] for row in csv.DictReader(f)]
    return ids


def test_real_split_has_no_leakage():
    train_ids = _train_ids()
    test_ids = list(load_labels())
    assert len(train_ids) == 1500, f"expected 1500 train ids, got {len(train_ids)}"
    assert len(test_ids) == 667, f"expected 667 test-public ids, got {len(test_ids)}"
    assert_no_leakage(train_ids, test_ids)  # must not raise
    return len(train_ids), len(test_ids)


def test_injected_violation_is_caught():
    test_ids = list(load_labels())
    poisoned_train = ["99999999"] + test_ids[:3]  # 3 real test ids smuggled into train
    try:
        assert_no_leakage(poisoned_train, test_ids)
    except AssertionError as e:
        assert "leakage" in str(e) and "3" in str(e), f"wrong failure message: {e}"
        return
    raise SystemExit("FAIL: assert_no_leakage did not fire on an injected violation")


if __name__ == "__main__":
    n_train, n_test = test_real_split_has_no_leakage()
    print(f"ok  real split clean: {n_train} train ids, {n_test} test-public ids, zero overlap")
    test_injected_violation_is_caught()
    print("ok  injected 3-id overlap was caught and raised")
    print("\nPASS: the leakage check is real, not vacuous.")
