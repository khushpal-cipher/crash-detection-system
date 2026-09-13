# Legacy Colab run — the training history, and why it does not describe the shipped model

Phase 0 task 4 (README §41). Extracted verbatim from `data/ccd/Untitled0.ipynb`, which is
committed with its outputs intact.

| File | Source | What it is |
|---|---|---|
| `training_log.txt` | notebook cell 8 output | 17 epochs (early-stopped from 50), per-epoch train/val accuracy, AUC, precision, recall, loss, learning rate; two `ReduceLROnPlateau` events; best epoch 9 |
| `metrics_val.txt` | notebook cell 9 output | `classification_report` on the 900-clip validation set at threshold 0.50 |

---

## ⚠️ These metrics do not describe `models/crash_model_weights.weights.h5`

**Do not quote `Best val AUC : 0.9977` as this model's performance.** Three separate
reasons, each sufficient on its own.

### 1. The numbers are from a different run than the artefact that ships

README §1 finding 3: 37 of the 38 weight arrays in the shipped
`crash_model_weights.weights.h5` differ from those in the `best_crash_model.keras`
committed at `c63e307`, and the `feat_reduce` kernel (1280×256) correlates at
**r = 0.0073**. Those are independent training runs.

**How that happened — the mechanism, now established:**

- `ModelCheckpoint` writes `best_crash_model.keras` to a **fixed path on Drive and
  overwrites it**. There are no run IDs and no artefact hashes.
- Cell 11 (CPU re-save) and cell 12 (`save_weights`) both **load that file back from
  Drive** rather than using the in-memory model. Cell 12 is literally
  `load_model(...)` → `save_weights(...)`.
- So the shipped `.h5` is a faithful copy of whatever `best_crash_model.keras` happened
  to contain **at the moment cells 11–12 last ran** — which is not necessarily the run
  whose log is in `training_log.txt`.
- `model.summary()` in the notebook prints `Model: "functional_5"`, proving at least six
  models were built in that kernel session. Only the last run's outputs survive.

**Verified 2026-09-12 (U6, `scripts/u6_compare_weights.py`):** `models/crash_model_cpu/`
and `models/crash_model_weights.weights.h5` are **bit-identical across all 12 weight
arrays**. Cells 11 and 12 read the same Drive file, so they are one artefact, not two.
That narrows the picture to exactly **two** distinct runs in evidence — the one committed
at `c63e307`, and the one that ships — not three.

### 2. Even for the run they do describe, the numbers are selection-contaminated

`train_test_split` produced **train and validation only — there is no third partition.**
`EarlyStopping` and `ModelCheckpoint` both selected on `val_auc`, and cell 9 then reports
on that same validation set. `0.9977` is a maximum over a noisy statistic on 900 samples,
quoted from the split used to choose it.

### 3. The split leaks, and the classes are corpus-aligned

- **Source leakage (T5):** the split was on filenames; `Crash-1500.txt`'s `youtubeID`
  grouping key was never read. **113 of 133 YouTube sources — 91.4% of crash clips —
  appear on both sides.**
- **Corpus confound (README N1):** every positive is a YouTube crash-compilation clip and
  every negative is a BDD100K clip. A classifier can score near-perfectly by telling the
  two corpora apart, with no understanding of collisions.

---

## What the model actually scores when those confounds are removed

Measured 2026-09-11 on Nexar test-public (667 clips, never trained on, both classes from
one corpus) — `runs/falsification/T3_corpus_control.md`:

| Metric | Value |
|---|---|
| ROC-AUC | **0.5339** |
| Average precision | **0.5218** (chance ≈ 0.5007 at this prevalence) |
| FPR @ threshold 0.80 | **97.6%** |

**That is chance.** The 0.9977 and the 0.5339 are not in tension — they are the same model
measured with and without the shortcut. This directory is kept as the historical record of
how the first number was produced, not as a performance claim.

See `runs/falsification/RESULTS.md` for the full falsification suite (T1 single-frame,
T2 temporal shuffle, T3 corpus control, T5 source leakage, T6 always-negative).
