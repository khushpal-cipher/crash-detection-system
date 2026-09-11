# T3 - Corpus control (Nexar test-public)

Model: `models/crash_model_weights.weights.h5` (CCD-trained). Benchmark: Nexar test-public, 667 clips (334 positive / 333 negative), positives and negatives from the same corpus (no source leakage by construction).

**ROC-AUC: 0.5339**

**AP: 0.5218**

At the deployed threshold 0.8: TP=332 FP=325 FN=2 TN=8, TPR=0.994, FPR=0.976

Runtime: 527.6s.
