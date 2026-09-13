#!/usr/bin/env python3
"""
README §41 Phase 5 task 4: "Name the authoritative published figure in metrics.json, and
record beside it every deviation." This is a documentation step, not a metric -- it adds
one "gate" key to an already-written runs/baselines/badas-open/metrics.json without
touching the metrics or records that produced it.

Run once, after eval/run_baselines.py has written that file:
    python scripts/badas_gate_provenance.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, "runs", "baselines", "badas-open", "metrics.json")

GATE = {
    "published_figures": {
        "model_card": {"source": "huggingface.co/nexar-ai/BADAS-Open README",
                        "average_precision": 0.86, "roc_auc": 0.88},
        "config_json": {"source": "vendor/badas-open/badas/config.json, performance block",
                         "average_precision": 0.832, "roc_auc": 0.85},
    },
    "measured": {"average_precision": None, "roc_auc": None},  # filled in below
    "verdict": (
        "Measured AP matches config.json's 83.2 to within 0.003 and sits ~0.02 below "
        "the model-card figure. The two published figures already disagree with each "
        "other by more than that gap, so this is not evidence of a harness defect -- "
        "see 'deviations' for every difference between this run and either source."
    ),
    "deviations": [
        "Split: 667-clip Nexar test-public (public half) here, vs the full 1,344-clip "
        "test set (test-public + test-private) in both published figures. test-private "
        "has no video files downloaded to this repo (metadata.csv only).",
        "Reduction: clip score = np.nanmax over per-frame probabilities. Upstream is "
        "self-contradictory -- badas_loader.py's per_video path uses mean, cli.py and "
        "the shipped example use max. max/nanmax was chosen to match the code path an "
        "end user actually runs (cli.py), not the alternate per_video path.",
        "Predictor path: EnhancedVideoClassifier.forward() computes V-JEPA2's "
        "predictor_output on every window (25% of compute) and never reads it -- only "
        "last_hidden_state reaches the classifier. Verified NOT a missing-weights bug "
        "(scripts/badas_predictor_probe.py: 199/199 duplicate tensors bitwise identical, "
        "load_state_dict reports missing=0). This run uses the as-shipped path "
        "(predictor computed, discarded). Consuming it via concat on the token axis is "
        "a verified 2-line change (README §41 Phase 5 gate note reason 3) but was not "
        "run for this metrics.json: with the default context/target masks the predictor "
        "reconstructs the SAME tokens it was given (future_prediction_seconds=1.0 needs "
        "training-time mask offsets that are not in this repo), so concatenating it adds "
        "no new information -- a second 17.5h sweep to test a hypothesis already refuted "
        "by the checkpoint's own mask config was judged not worth the compute.",
        "stride=1, target_fps=8.0, img_size=224, vjepa2_crop_size=256 (from the "
        "checkpoint's training config) -- all match the checkpoint's own recorded "
        "training configuration; original_fps=4 in that same config is the post-tubelet "
        "TOKEN rate, not a second video rate (resolved, scripts/badas_fps_probe.py).",
        "Temperature scaling: fixed at 2.0 (checkpoint's use_temperature_scaling=true, "
        "temperature=2.0), applied by the vendored upstream code, not refit here.",
        "0.90 hours of negative footage means FP/hour below ~1.1/h cannot be evidenced "
        "at any confidence on this split (README §34) -- true regardless of model.",
    ],
}


def main():
    with open(METRICS) as f:
        doc = json.load(f)
    m = doc["metrics"]
    GATE["measured"] = {
        "average_precision": m["average_precision"],
        "roc_auc": m["roc_auc"],
    }
    assert not any("accur" in k.lower() for k in GATE), "banned substring in a new key"
    m["gate"] = GATE
    with open(METRICS, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote gate provenance to {METRICS}")
    print(f"  measured AP {GATE['measured']['average_precision']:.4f} / "
          f"AUC {GATE['measured']['roc_auc']:.4f}")
    print(f"  vs config.json AP {GATE['published_figures']['config_json']['average_precision']} / "
          f"AUC {GATE['published_figures']['config_json']['roc_auc']}")


if __name__ == "__main__":
    main()
