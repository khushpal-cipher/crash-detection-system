# AI Crash Detection System

**Repository:** `khushpal-cipher/crash-detection-system`
**Document status:** Technical audit + startup strategy. Revision 2, written 2026-09-10. Supersedes the 2026-09-07 audit, which is reconciled section-by-section in [What Changed From the Previous Audit](#what-changed-from-the-previous-audit).
**Verified against:** repository commit `b539d6e`; the Google Colab training notebook `Untitled0.ipynb` (`drive/1Bmudju26-q7JQyI4FAsZDBXsJw7PRnnM`), read cell-by-cell including saved outputs on 2026-09-10; the `Cogito2012/CarCrashDataset` repository; and byte-level inspection of the model artefacts in the working tree and in git history.

**Honest one-line description of what exists today:** a clip-level binary classifier (frozen ImageNet MobileNetV2 → LSTM, 578,689 trainable parameters) trained on the public Car Crash Dataset with a random non-source-grouped 80/20 split and no held-out test set, deployed at a threshold (0.80) that appears nowhere in its training notebook, AND-ed with "≥2 YOLO vehicle boxes present", and wrapped in ~3,000 lines of 3D-geometry and visualisation code that **does not participate in the crash decision**.

> **The single most important correction in this revision:** the previous audit concluded that no training code, dataset history, split, epoch or metric information existed. **That conclusion was wrong as a global statement.** All of it exists, in Google Colab. It was absent from GitHub, which is a different — and much more fixable — problem. The training pipeline is now fully documented in [Training Pipeline](#6-training-pipeline) and [Training History](#10-training-history).
>
> **The second most important correction:** recovering that evidence did **not** improve the project's standing. It made the assessment sharper and, in three specific respects, worse. See [New Findings](#new-findings-summary).
>
> **Update, same day — the falsification tests have now been run.** Results in [`runs/falsification/RESULTS.md`](runs/falsification/RESULTS.md). Two of them failed outright: temporal shuffling changes the model's output by ≤0.0013, and 91.4% of the training split leaks by source video. What was a hypothesis in this document's first draft is now a measurement. Sections [5](#5-current-model), [12](#12-evaluation-history), [13](#13-what-has-actually-been-proven), [14](#14-what-has-not-been-proven), [16](#16-data-leakage-risks) and [23](#23-dataset-licensing--provenance) have been updated accordingly.
>
> **Update 2026-09-11 — T3, the corpus control, has now been run, and it settles the central question.** On Nexar test-public (667 clips, 334 positive / 333 negative, positives and negatives drawn from one corpus so the confound of [B4](#b4--the-positivenegative-classes-coincide-with-two-different-corpora) cannot operate) the shipped model scores **ROC-AUC 0.5339 and AP 0.5218** — chance. At the deployed threshold of 0.80 it fires on **97.6% of the negatives** and 99.4% of the positives: it is not a weak detector, it is an almost-always-positive one. Raw results: [`runs/falsification/T3_corpus_control.md`](runs/falsification/T3_corpus_control.md). **This is the first honest performance number in the project's history, and it is the number that retires the model.** The falsification suite is now closed: every test that was run, failed.

---

## Table of Contents

**Reconciliation**
- [What Changed From the Previous Audit](#what-changed-from-the-previous-audit)
- [Evidence Classification](#evidence-classification)
- [New Findings Summary](#new-findings-summary)
- [Remaining Unknowns](#remaining-unknowns)

**Part I — What you actually have**
1. [Executive Summary](#1-executive-summary)
2. [Project Status](#2-project-status)
3. [What the Current System Actually Does](#3-what-the-current-system-actually-does)
4. [Current Architecture](#4-current-architecture)
5. [Current Model](#5-current-model)
6. [Training Pipeline](#6-training-pipeline)
7. [Dataset Used](#7-dataset-used)
8. [Dataset Structure](#8-dataset-structure)
9. [Dataset Split](#9-dataset-split)
10. [Training History](#10-training-history)
11. [Current Model Artifact](#11-current-model-artifact)
12. [Evaluation History](#12-evaluation-history)
13. [What Has Actually Been Proven](#13-what-has-actually-been-proven)
14. [What Has NOT Been Proven](#14-what-has-not-been-proven)

**Part II — What is broken**
15. [Technical Problems (full bug-fix plan)](#15-technical-problems--full-bug-fix-plan)
16. [Data Leakage Risks](#16-data-leakage-risks)
17. [Reproducibility Problems](#17-reproducibility-problems)
18. [Calibration Problems](#18-calibration-problems)
19. [Current 3D/Physics Pipeline](#19-current-3dphysics-pipeline)
20. [Current Detector/Tracker](#20-current-detectortracker)
21. [Current Crash Decision Logic](#21-current-crash-decision-logic)
22. [Licensing Problems](#22-licensing-problems)
23. [Dataset Licensing / Provenance](#23-dataset-licensing--provenance)

**Part III — Where to go**
24. [Target Market](#24-target-market)
25. [Initial Country](#25-initial-country)
26. [Customer Profile](#26-customer-profile)
27. [Product Definition](#27-product-definition)
28. [Recommended Future Architecture](#28-recommended-future-architecture)
29. [Recommended Dataset Strategy](#29-recommended-dataset-strategy)
30. [Recommended Training Strategy](#30-recommended-training-strategy)
31. [Evaluation Framework](#31-evaluation-framework)
32. [Real-Time Architecture](#32-real-time-architecture)
33. [Data Collection Strategy](#33-data-collection-strategy)
34. [Hard-Negative Benchmark](#34-hard-negative-benchmark)
35. [Privacy / Legal / Compliance](#35-privacy--legal--compliance)
36. [Partnership Strategy](#36-partnership-strategy)
37. [Automotive Partnership Roadmap](#37-automotive-partnership-roadmap)
38. [Accelerator Strategy](#38-accelerator-strategy)
39. [Investor Demo Strategy](#39-investor-demo-strategy)
40. [Technical Moat](#40-technical-moat)

**Part IV — What to do**
41. [90-Day Action Plan (Phases 0–11)](#41-90-day-action-plan-phases-011)
42. [Priority Matrix](#42-priority-matrix)
43. [Exact Immediate Actions](#43-exact-immediate-actions)
44. [Things NOT to Build](#44-things-not-to-build)
45. [Final CTO Verdict](#45-final-cto-verdict)

**Appendices**
46. [Appendix A — Current Commands / Usage](#46-appendix-a--current-commands--usage)
47. [Appendix B — Current Configuration Values](#47-appendix-b--current-configuration-values)
48. [Appendix C — Known Bugs (index)](#48-appendix-c--known-bugs-index)
49. [Appendix D — Dataset References](#49-appendix-d--dataset-references)
50. [Appendix E — Sources](#50-appendix-e--sources)

---

# Evidence Classification

Every factual claim in this document is tagged to one of five evidence classes. This distinction matters more than any single finding, because the previous audit's central error was collapsing class A into "the truth".

| Class | Source | Access status | What it can and cannot prove |
|---|---|---|---|
| **A — GitHub** | Working tree + full git history at `b539d6e` | ✅ Full | Proves what is *shipped and maintainable*. Cannot prove how anything was made. |
| **B — Colab** | `Untitled0.ipynb`, 12 code cells, all with saved outputs | ✅ Full (read cell-by-cell 2026-09-10) | Proves the *training method and one recorded run*. Cannot prove which run produced the shipped weights. |
| **C — Dataset repository** | `Cogito2012/CarCrashDataset` README | ✅ Full | Proves dataset composition, annotation schema and upstream provenance. |
| **D — Model artefacts** | `.keras` / `.weights.h5` / SavedModel byte inspection | ✅ Full | Proves architecture identity and, critically, **run identity**. |
| **E — Local environment** | Working tree outside git, filesystem timestamps | ✅ Full | Proves what was downloaded when. Cannot prove why. |

**The four distinctions the previous audit conflated, now answered separately:**

| # | Question | Answer | Evidence |
|---|---|---|---|
| **A** | Does training code exist? | **YES.** 12 cells, complete and runnable. | B |
| **B** | Does training history / metrics exist? | **YES.** 17 epochs of per-epoch metrics, a confusion matrix and a classification report are saved in the notebook. | B |
| **C** | Is the exact final model reproducible? | **NO.** The shipped weights are provably from a *different* run than the one recorded, and the notebook has no run log, seed capture for the training loop, or artefact hash. | D |
| **D** | Is dataset provenance / licensing reproducible? | **PARTIALLY, and the answer is bad.** The dataset is identified with certainty, but its two classes come from two corpora with different and problematic licences. | B, C |
| **E** | Is the experiment scientifically valid? | **NO.** No held-out test set, no source grouping, and a positive/negative split that coincides with a corpus split. | B, C |

---

# What Changed From the Previous Audit

Every material conclusion from the 2026-09-07 audit, classified.

| # | Previous conclusion | Status | New evidence | Corrected conclusion |
|---|---|---|---|---|
| 1 | "No training code anywhere in repository or git history." | **PARTIALLY CONFIRMED** | Colab cells 1–12 (B); `git log --all` still shows none (A) | True of GitHub, **false of the world**. Training code exists, is complete, and is one `File → Download .ipynb` away from being committed. This was an evidence-access failure, not an absence. |
| 2 | "Provenance of the model: unrecoverable." | **CORRECTED** | Cells 2, 3, 5, 6 (B) | Dataset, preprocessing, split, hyperparameters and architecture are all now known exactly. Provenance is **recovered**. |
| 3 | "The model artefact cannot be reproduced." | **CONFIRMED — and now proven, not inferred** | 37 of 38 weight arrays in `crash_model_weights.weights.h5` differ from those in the `best_crash_model.keras` committed at `c63e307`; the `feat_reduce` kernel (1280×256) correlates at **r = 0.0073** (D) | Two *independent* training runs exist. The deployed weights are **not** the artefact whose metrics the notebook records. The pipeline is reproducible; **this specific model is not.** |
| 4 | "Cannot answer 'what is your training data licensed under?'" | **CORRECTED — you can now answer it, and the answer is a problem** | Cell 3 loads `Crash-1500.zip` + `Normal.zip` (B); CCD README (C) | Positives are YouTube-sourced crash compilations; negatives are sampled from **BDD100K**, whose basic licence is limited to personal use. See [§23](#23-dataset-licensing--provenance). |
| 5 | "The test set is three video files; there is no held-out data." | **CORRECTED at model level, CONFIRMED at system level** | Cell 6: 3,600 train / 900 val (B); `CNN_THRESH = 0.80` appears nowhere in the notebook (A, B) | The *classifier* was measured on 900 unseen CCD clips. The *deployed system* — threshold 0.80, `crash_pct ≥ 1.5`, the vehicle-count gate — was still tuned on `crash1.mov`, `crash2.mov`, `safe.mp4` and has never been measured on anything else. |
| 6 | "Thresholds are fitted to the test set." | **CONFIRMED and sharpened** | Cell 9 evaluates at 0.50; cell 10's inference snippet suggests 0.55; production uses **0.80** (A, B) | Three different thresholds exist across three artefacts. The one that ships has no derivation anywhere. |
| 7 | "There is no held-out test set." | **CONFIRMED — and worse than stated** | Cell 6 produces only train/val; cell 8 selects on `val_auc`; cell 9 reports on the *same* val set (B) | The reported metrics are **model-selection-contaminated**: the split used to pick the best epoch is the split the results are quoted from. There is no third partition. |
| 8 | "Random frame-level splits would leak." | **CORRECTED in mechanism, CONFIRMED in consequence** | `train_test_split(all_files, ..., stratify=all_labels)` splits **clips**, not frames (B) | Not frame-level. But **not source-grouped either**: CCD ships a `youtubeID` field per crash clip and an official `train.txt`/`test.txt`, and neither was used. Multiple clips cut from one YouTube video can straddle train and val. |
| 9 | "Hypothesis: the model learned clip identity / global image statistics, not collisions." | **CONFIRMED BY MEASUREMENT** | Falsification run 2026-09-10: temporal shuffle changes the mean score by ≤0.0013; a single frame tiled ×10 reproduces it to within 0.02; 91.4% of the split leaks by `youtubeID` | **It is an image classifier.** The two recurrent layers (246,528 parameters) are provably inert. The audit's original phrasing was directionally right and mechanistically understated. |
| 10 | "10 frames = 0.17–0.33 s of context." | **CONFIRMED for inference, CORRECTED for training — and this exposes a new critical bug** | Training: `np.linspace(0, total-1, 10)` over a 50-frame/10 fps clip → ~0.54 s spacing, 5 s span (B). Inference: `deque(maxlen=10)` of consecutive frames → 0.033 s spacing, 0.33 s span (A) | **A ~16× (30 fps) to ~32× (60 fps) temporal-scale mismatch between training and inference.** New finding, severity P0. See [B1](#b1--traininference-temporal-stride-mismatch). |
| 11 | "MobileNetV2 is frozen." | **CONFIRMED** | Cell 4: `base_model.trainable = False`; features precomputed to `.npy` in cell 5 (B) | Confirmed, and stronger than assumed: the backbone was never even in the training graph. Only the 578,689-parameter head was trained. |
| 12 | "The entire physics stack is excluded from the decision." | **CONFIRMED verbatim** | `crash_detection_enhanced.py:1049–1056` and `:1040–1046` (A) | Unchanged. `rule[...]` is consulted only during the 9-frame CNN warm-up; `depth_map` reaches only the renderer. |
| 13 | "C1 — focal length is not rescaled with resolution." | **CONFIRMED** | `Config.FX = FY = 460` at `:68–71`; `set_resolution()` updates only `CX_IMG`/`CY_IMG` at `:128–132` (A) | Unchanged. |
| 14 | "C3 — dashcam mode is a proximity alarm." | **CONFIRMED** | `EgoZone.check` is pure bbox overlap with a fixed rectangle; `is_crash = len(ego_hits) > 0` at `:1041` (A) | Unchanged. |
| 15 | "C7 — fault attribution is unsupportable." | **CONFIRMED** | `EgoZone.fault()` derives "from the left / from the right / head-on" from bounding-box x-position alone (A) | Unchanged, and the dashcam variant is cruder than the audit realised. |
| 16 | "C6 — `ultralytics` is AGPL-3.0." | **CONFIRMED** | `requirements.txt:3`, imported in all four forks (A) | Unchanged. |
| 17 | "`requirements.txt` pins an impossible combination." | **CONFIRMED and extended** | `tensorflow==2.13.1` vs local Python 3.14 (A); model saved by **Keras 3.13.2 / TF 2.19.0** (B, D) | Not merely un-installable — **provably wrong**. TF 2.13 ships Keras 2 and cannot read a Keras 3 weights file. The pinned environment could not load the shipped model even if it installed. New finding, [R2](#r2--pinned-tensorflow-cannot-read-the-shipped-model). |
| 18 | "Recommended architecture: V-JEPA2 + attentive probe / BADAS-Open, two heads, fused with a calibrated physical channel." | **UNCHANGED** | Nothing in the Colab alters the comparison | Seeing the actual pipeline strengthens this. See [§28](#28-recommended-future-architecture) for why, stated as an argument about *this* model rather than about model age. |
| 19 | "Dataset strategy: global for pretraining, target-market for fine-tuning and evaluation." | **UNCHANGED, with CCD added** | CCD composition (C) | CCD joins the list as a research-only corpus. It is not a commercial training asset. |
| 20 | "Business strategy: UK-first customers, Delaware incorporation, retrofit commercial fleets, incident-record product." | **UNCHANGED** | No new evidence bears on it | Retained in full. |
| 21 | "Score: ML quality 12/100 because the artefact is untraceable and n=3." | **REVISED, not raised** | B, C, D | Revised to **18/100**. The pipeline is competent, documented and reproducible-in-principle — that is worth something. The experimental design is invalid and the shipped artefact is unidentifiable — that is worth very little. |
| 22 | "Evaluation history: only three videos." | **CORRECTED** | Cell 9 (B) | There are two evaluation histories: a real one on 900 CCD clips (contaminated by model selection), and a fictional one on 3 local videos (used to set the shipped threshold). Both are documented in [§12](#12-evaluation-history). |
| 23 | "Keep the Kalman/TTC code; it is correctly implemented." | **UNCHANGED** | `compute_ttc` at `:415`, `VehicleKF` at `:436` (A) | Retained. |
| 24 | "Delete MiDaS, BEV, ego zone, fault attribution, the three forks." | **UNCHANGED** | A | Retained. |
| 25 | "The current dataset directory structure differs from the official CCD structure." | **UNKNOWN / NEEDS EVIDENCE** | No dataset exists in the local working tree or anywhere under `~/Desktop` (E) | The *Colab/Drive* structure is now fully known ([§8](#8-dataset-structure)) and it differs from the official layout for identifiable reasons. The structure in your screenshot cannot be reconciled because no such directory is present on this machine. See [Remaining Unknowns](#remaining-unknowns). |

---

# New Findings Summary

## What the Colab proved

1. **A complete, coherent, competently-written training pipeline exists.** Twelve cells: mount → configure → extract → build feature extractor → precompute features → split → build model → train → evaluate → save → CPU-compatible re-export → weights-only export. It is better engineering than the previous audit assumed possible from the absence of evidence.
2. **The dataset is identified with certainty:** the Car Crash Dataset (CCD), 1,500 crash + 3,000 normal videos, loaded from two zips on Google Drive and extracted to `/tmp`. Cell 3's output confirms 1,500 and 3,000 extracted; cell 5's output confirms 4,500 feature files written with **0 skipped**.
3. **Training history exists and is complete:** 17 epochs (early-stopped from a 50-epoch budget), per-epoch train/val accuracy, AUC, precision, recall, loss and learning rate; two `ReduceLROnPlateau` events; best epoch 9; `Best val AUC : 0.9977`, `Best val acc : 0.9822`.
4. **Evaluation exists:** a `classification_report` and confusion matrix on the 900-clip validation set at threshold 0.50 — Normal P 0.99 / R 0.97, Crash P 0.94 / R 0.98, 581 / 19 / 5 / 295.
5. **The backbone was frozen and never fine-tuned.** Features were precomputed once and cached as `.npy`; only the 578,689-parameter LSTM head was trained.
6. **`CNN_THRESH = 0.80` originates nowhere in the training process.** The notebook evaluates at 0.50 and suggests 0.55. The deployed 0.80 was invented downstream, against three local videos.
7. **There were multiple training runs.** `model.summary()` prints `Model: "functional_5"`, meaning at least six models were constructed in that kernel session. Only the last run's outputs survive in the notebook.

## What the dataset repository proved

8. **CCD's negatives are BDD100K.** The CCD README states plainly that "the 3,000 normal videos are randomly sampled from the BDD100K dataset." BDD100K's basic licence is limited to personal use.
9. **CCD's positives are YouTube-derived.** `Crash-1500.txt` carries a `youtubeID` and `startframe` per clip precisely because each clip is a window cut from a YouTube source video.
10. **CCD ships everything needed to do the split correctly, and none of it was used:** `train.txt` / `test.txt` (the official split), `youtubeID` (the correct grouping key), `binlabels` (per-frame accident labels), plus `timing`, `weather` and `egoinvolve` metadata.

## What is now worse than the previous audit believed

**N1 — The positive/negative distinction coincides exactly with a corpus distinction.**
Every positive is a YouTube crash-compilation clip. Every negative is a BDD100K clip. These two corpora differ in resolution, codec, bitrate, colour grading, camera hardware, country and upload pipeline. A classifier can reach very high AUC by recognising *which corpus a clip came from*, with zero understanding of collisions. This is the most parsimonious explanation for a val AUC of 0.998 achieved by 578,689 parameters on 3,600 examples in nine epochs. **The previous audit's "it memorised two clips" hypothesis was directionally right and mechanistically wrong; the real shortcut is larger, cleaner, and available on every single training example.**

**N2 — The model is trained and deployed at incompatible temporal scales.**
Training sequences span 5 seconds at ~0.54 s per step. Inference sequences span 0.33 seconds at 0.033 s per step. The LSTM's learned dynamics do not exist in the tensors it is shown at runtime. This is a genuine, previously-undetected P0 defect, and it means the deployed system is not running the model that was trained in any meaningful sense.

**N3 — The reported validation performance implies an unusable false-alarm rate.**
Clip-level false-positive rate on CCD normals: 19/600 = **3.17%**. Applied as a sliding 5-second window over continuous driving (720 windows per hour), that is **≈23 false alarms per driving hour** — against a shippable target of < 0.1/hour, i.e. off by more than two orders of magnitude. And this is the *optimistic* number, measured on the split used for model selection, on negatives from a corpus the model can identify by appearance.

**N4 — The shipped weights are from an unrecorded run.**
Byte comparison proves it: same architecture, same shapes, same 578,689 parameters, but 37 of 38 arrays differ and the first Dense kernel correlates at r = 0.0073. Whatever metrics you have, they do not describe the file in `models/`.

---

# Remaining Unknowns

Stated explicitly, with no filler.

| # | Unknown | Why it cannot be resolved from available evidence | How to resolve it |
|---|---|---|---|
| U1 | **Which training run produced `crash_model_weights.weights.h5`, and what its metrics were.** | Colab preserves only the last execution's outputs. `ModelCheckpoint` writes to a fixed path and overwrites. No run IDs, no experiment log. | Unresolvable retrospectively. Re-train under Phase 1 with logging, and treat the current file as disposable. |
| U2 | **Whether the notebook's saved outputs describe run A, run B, or a third run.** | `Model: "functional_5"` proves ≥6 builds; only one set of outputs survives. | Same as U1. |
| U3 | **The local dataset directory structure you referred to.** | No dataset directory exists anywhere under `~/Desktop` on this machine — `videos/` is a symlink containing only `crash1.mov`, `crash2.mov`, `safe.mp4`. I have not seen the screenshot. | Paste the screenshot or run `tree -L 3` on the directory in question. The Colab/Drive structure *is* known and is documented in [§8](#8-dataset-structure). |
| U4 | **What `models/crash_detection_model.h5` (296 MB) and `models/crash_detection_model/` (197 MB) are.** | The notebook's largest artefact is ~7 MB. These cannot have come from it. Their existence, the directory name `features_v2`, and cell 5's comment *"Old approach: np.mean(frames) → one blurry image"* jointly indicate a **prior, undocumented v1 pipeline**. No code for it survives anywhere. | If a v1 notebook exists in your Drive, export it. Otherwise mark these files as orphaned and delete them (they are gitignored already). |
| U5 | **How `models/feature_extractor_saved/` was produced.** | The notebook saves `feature_extractor.keras`; the working tree contains a TensorFlow SavedModel directory dated a month later. The conversion happened locally and is not in git. | Regenerate it deterministically from `keras.applications.MobileNetV2` in the new training script; do not preserve the artefact. |
| U6 | **Whether `models/crash_model_cpu/` shares weights with the shipped `.weights.h5`.** | Comparing a TF SavedModel checkpoint against a Keras 3 HDF5 file requires a working TensorFlow install; none is present in any environment on this machine. | One command in the Phase 0 environment: load both, compare `get_weights()`. Worth doing — it would tell you whether cells 11 and 12 ran in the same session. |
| U7 | **The exact operative licence text of CCD.** | The GitHub repository is labelled MIT; the README contains no licence section. The MIT label covers the researchers' work, not the underlying video. | Email the authors (Wentao Bao, RIT). See [§23](#23-dataset-licensing--provenance). |
| U8 | **Whether the CCD crash clips in train and val share `youtubeID`s.** | The split was performed on filenames only; `Crash-1500.txt` was never read, so the grouping was never computed. | Trivially computable in ten lines once `Crash-1500.txt` is downloaded. **Do this in Phase 0** — it converts a suspicion into a number. |

**None of these blocks the plan below.** U3 is the only one where I need something from you, and it affects one paragraph of [§8](#8-dataset-structure), not any conclusion.

---

# Part I — What You Actually Have

## 1. Executive Summary

**What you have built is a competently-engineered training pipeline attached to an invalid experiment, wrapped in a physics-themed visualisation layer that does not affect any decision.** That sentence is more generous than the previous audit's and it is still not a compliment: the middle clause is the one that matters.

Five findings dominate everything else.

1. **The experiment is not valid, for a reason that has nothing to do with effort.** The positive class is YouTube crash-compilation footage and the negative class is BDD100K. Those are two different corpora with two different visual signatures. A model can score 0.998 AUC by telling them apart. Until you run the falsification tests in [§30](#30-recommended-training-strategy), you do not know whether your classifier has ever seen a collision.

2. **There is no held-out test set, and the reported numbers are selection-contaminated.** `train_test_split` produced train and validation only. `EarlyStopping` and `ModelCheckpoint` both selected on `val_auc`. Cell 9 then reports performance on that same validation set. The best-of-17-epochs `val_auc = 0.9977` is a maximum over a noisy statistic on 900 samples, quoted from the split used to choose it.

3. **The model you ship is not the model you measured.** Byte-level comparison of `models/crash_model_weights.weights.h5` against the `best_crash_model.keras` committed at `c63e307` shows 37 of 38 weight arrays differ, with the first Dense kernel correlating at r = 0.0073. These are independent runs. No metrics exist for the one that ships.

4. **Training and inference operate at incompatible temporal scales.** The LSTM was trained on 10 frames spanning 5 seconds (≈0.54 s apart) and is run on 10 consecutive frames spanning 0.33 seconds. Whatever temporal structure it learned is absent at runtime by a factor of 16–32×.

5. **The perception stack remains decorative, and the licensing landmine remains armed.** `crash_detection_enhanced.py:1050` still reads `cnn >= 0.80 and len(vehs) >= 2`. The Kalman filter, ground projection, TTC and MiDaS depth are computed every frame and discarded. `ultralytics` is still AGPL-3.0.

**The good news is real and should not be lost in the above.** You have a documented, runnable pipeline; a known dataset; recorded hyperparameters; and a training loop with sensible callbacks and class weighting. The gap between here and a defensible experiment is a **week of work on splits, held-out data and falsification tests** — not a research programme. That is a far better position than the previous audit's "provenance unrecoverable" implied.

**The strategic situation is unchanged and remains more interesting than the code situation.** Nexar released [BADAS-Open](https://huggingface.co/nexar-ai/BADAS-Open) under **Apache 2.0** — a V-JEPA2-based ego-centric collision predictor reporting AP 0.86 / AUC 0.88 on their own benchmark, free for commercial use. Your model cannot compete with it and does not need to: **the model is not the moat.** A commercially-usable state-of-the-art baseline being free lets you skip two years of model R&D and compete where competition is actually possible — on target-market data, on false-positive measurement, and on the incident workflow.

**Recommended market (unchanged):** incorporate in the **United States** (Delaware C-corp); design for and land first pilots in the **United Kingdom**; sell **structured incident records** to **retrofit commercial fleets of 30–300 vehicles**, not crash detection to OEMs.

**Time to a credible investor demo:** ~90 days. **Time to a defensible company:** 18–24 months, and only with a proprietary data feedback loop.

### Scores (0–100), revised against the new evidence

| Dimension | Prev. | Now | One-line justification |
|---|---:|---:|---|
| Technical maturity | 22 | **26** | Training pipeline exists and is coherent — but it lives in an untracked notebook, not the repo |
| ML quality | 12 | **18** | Competent mechanics; invalid design (no test set, no source grouping, corpus-aligned classes) |
| Dataset quality | 5 | **20** | A real 4,500-clip public dataset — with a licence problem and a class/corpus confound |
| Reproducibility | — | **15** | Pipeline reproducible in principle; the shipped artefact is provably not |
| Product quality | 15 | **15** | A CLI and an OpenCV window; no API, no storage, no UI |
| Scalability | 10 | **10** | Single-process, single-video, no service boundary |
| Startup potential | 55 | **55** | Real market, real regulatory tailwind, founder demonstrably ships code |
| Investor readiness | 8 | **10** | You can now answer "how did you train it?" — you still cannot answer "how well does it work?" |
| Partnership readiness | 10 | **10** | Nothing here can be piloted |
| Defensibility | 5 | **5** | A free Apache-2.0 model beats yours today |

---

## 2. Project Status

**Stage:** research prototype with a recovered training pipeline. Not a validated system. Not pilot-ready. Not investor-ready.

| Aspect | Status |
|---|---|
| Runs from a clean clone | ❌ No — see [§17](#17-reproducibility-problems) |
| Training code in version control | ✅ Yes — `data/ccd/Untitled0.ipynb`, committed at `c9a6fda` with outputs intact |
| Dataset identified | ✅ Yes — CCD (as of this revision) |
| Dataset licensed for commercial use | ❌ No — see [§23](#23-dataset-licensing--provenance) |
| Held-out test set | ❌ No |
| Source-grouped splits | ❌ No |
| Falsification tests run | ✅ **Yes — T1, T2, T3, T5, T6 and B1. The model failed every one.** |
| Measured on a corpus-controlled benchmark | ✅ **Yes — Nexar test-public: ROC-AUC 0.5339 (chance)** |
| Calibration measured | ❌ No |
| False-positives-per-hour measured | ❌ No (implied ≈23/h; target < 0.1/h) |
| Deployed threshold derived from data | ❌ No |
| Shipped weights traceable to a run | ❌ No |
| Detector licence resolved | ❌ No — AGPL-3.0 |
| Any test in the repository | 🟡 One — `tests/test_weights_load.py` (weights load + forward pass, R2's verification). No CI, no other coverage. |
| Persistence / API / UI | ❌ None |

**Repository inventory (verified at `b539d6e`):**

```
crash_detection_v2/
├── README.md                        (this document)
├── requirements.txt                 (7 lines, 1 pin, and that pin is impossible — R1/R2)
├── camera_detect.py                 (552 lines — 4th fork, Raspberry Pi / picamera2)
├── yolov8n.pt                       (6.5 MB, gitignored)
├── LocateAnything-3B_Guide.pdf      (untracked)
├── code/
│   ├── crash_detection_enhanced.py  (1,334 lines — THE main pipeline, v17.0)
│   ├── crash_detection.py           (1,032 lines — 2nd fork, v12/v13)
│   ├── crash_detection_linux.py     (  980 lines — 3rd fork, v14.1)
│   ├── depth_estimator.py           (  180 lines — MiDaS wrapper)
│   ├── bev_renderer.py              (  107 lines — top-down minimap)
│   ├── ORBSTACK_SETUP.md            (  150 lines)
│   └── setup_orbstack.sh            (  122 lines)
├── models/
│   ├── crash_model_weights.weights.h5   (6,990,088 B — THE deployed LSTM head, tracked)
│   ├── feature_extractor_saved/         (19 MB — MobileNetV2 SavedModel, tracked)
│   ├── feature_extractor.keras          (9.2 MB, gitignored — the Colab artefact)
│   ├── crash_model_cpu/                 (4.5 MB, gitignored — Colab cell 11 export)
│   ├── crash_model_saved/               (4.5 MB, gitignored — unexplained)
│   ├── crash_detection_model.h5         (296 MB, gitignored — ORPHANED, see U4)
│   └── crash_detection_model/           (197 MB, gitignored — ORPHANED, see U4)
└── videos -> ~/dev/crash_detection/videos
    ├── crash1.mov  (48 MB, 3408×1910)
    ├── crash2.mov  (48 MB, 3408×1910)
    └── safe.mp4    (94 MB, 3840×2160)
```

**16 tracked files. 6 Python files. 3 videos. 493 MB of orphaned model artefacts. Zero tests. Zero training code.**

**Off-repository assets (newly catalogued):**

| Asset | Location | Status |
|---|---|---|
| Training notebook | Colab `Untitled0.ipynb`, drive id `1Bmudju26-q7JQyI4FAsZDBXsJw7PRnnM` | ✅ Accessible; **not in git** |
| Dataset zips | `My Drive/CarCrashDetection/data/dataset/{Crash-1500,Normal}.zip` | Present per cell 3's output |
| Cached features | `My Drive/CarCrashDetection/data/features_v2/*.npy` (4,500 files) | Present per cell 5's output |
| Model checkpoints | `My Drive/CarCrashDetection/models/` | Present per cell 8's output |
| Result plots | `My Drive/CarCrashDetection/results/{confusion_matrix,training_curves}.png` | Written by cell 9 |

**Immediate consequence:** the single highest-leverage 30-second action available to you is `File → Download .ipynb` in that Colab tab, followed by `git add`. Everything in [§17](#17-reproducibility-problems) begins there.

---

## 3. What the Current System Actually Does

Answering the capability questions directly, with the Colab evidence folded in.

| Question | Answer |
|---|---|
| Is it detecting crashes? | **Unknown, and now known to be *harder* to establish than assumed.** It separates CCD positives from CCD negatives — but those are two different corpora, so separation is not evidence of collision understanding. |
| Is it detecting vehicles? | Yes — YOLOv8n does, competently. Still the only reliable component. |
| Is it detecting collision *events*? | No. There is no event. There is a per-window score and a percentage-of-frames threshold. |
| Is it localising crashes in time? | **No — and the labels to learn it were available and discarded.** CCD ships `binlabels`, a per-frame binary accident annotation for all 50 frames. Cell 6 assigns one label per clip and ignores the file entirely. |
| Is it detecting damaged vehicles? | No. Nothing models damage. |
| Object detection? | Yes (YOLOv8n, COCO classes). |
| Video classification? | **Yes — this is what the system fundamentally is.** |
| Temporal action recognition? | No. |
| Object tracking? | Yes, a fragile greedy centroid tracker that resets on empty frames. |
| Trajectory analysis? | Implemented (Kalman + TTC), correctly, and **not used in the decision**. |
| Uses temporal information? | **Nominally, and incoherently.** Trained on 5 s spans, run on 0.33 s spans. See [B1](#b1--traininference-temporal-stride-mismatch). |
| Ego-involvement? | **No — and the labels were available and discarded.** CCD ships an `egoinvolve` boolean per crash clip. |
| Weather / lighting conditioning? | **No — and the labels were available and discarded.** CCD ships `timing` (Day/Night) and `weather` (Normal/Snowy/Rainy). |
| Near-miss vs collision? | No. No such label exists in CCD or in the code. |
| Crash severity? | No. "Fault detection" is not severity. |
| Detects false positives? | No. The CNN *is* the false-positive filter, and its measured clip-level FPR implies ≈23 false alarms per driving hour. |
| Real-time capable? | ~8 fps on M1 at 4K per the prior README — i.e. no, not at source frame rate. |

**Verdict: a short-clip binary appearance classifier with a vehicle-count gate, trained on a corpus-confounded dataset and deployed at an underived threshold and an incompatible frame rate.** Calling it crash detection is a naming choice, not a technical claim.

**The most striking thing the Colab reveals is what was thrown away.** CCD gave you frame-level accident timing, ego-involvement, weather, lighting, YouTube source IDs for grouping, and an official train/test split. The pipeline used exactly one bit of that: the directory a file came out of. Every hard limitation in the table above is a direct consequence of collapsing a richly-annotated dataset into a filename prefix.

---

## 4. Current Architecture

This is the **as-built** architecture, traced from `code/crash_detection_enhanced.py`. Note carefully which paths reach the verdict and which do not.

```
                             VIDEO FILE or WEBCAM
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ cv2.VideoCapture          │
                        │ NO RESIZE (native 4K)     │  ← enhanced.py:1010
                        │ Config.set_resolution()   │  ← sets cx,cy to NATIVE res
                        │   BUT fx,fy stay at 460   │     while fx,fy stay 640×480
                        └─────────────┬─────────────┘     ✗ CRITICAL DEFECT (C1)
                                      │
              ┌───────────────────────┼────────────────────────┐
              ▼                       ▼                        ▼
   ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────────┐
   │ apply_road_roi()   │  │ MiDaS v3.1 small   │  │ NeuralCrashDetector  │
   │ mask top 40% +     │  │ estimate_metric()  │  │ 112×112, 10-frame    │
   │ bottom 10%         │  │ ~30 ms/frame       │  │ CONSECUTIVE buffer   │
   └─────────┬──────────┘  └─────────┬──────────┘  └──────────┬───────────┘
             ▼                       │                        │  ✗ B1: trained
   ┌────────────────────┐            │                        │    on 0.54 s
   │ YOLOv8n  (AGPL!)   │            │                        │    strides
   │ conf 0.5 + NMS     │            │                        ▼
   └─────────┬──────────┘            │             ┌──────────────────────┐
             ▼                       │             │ MobileNetV2 (frozen  │
   ┌────────────────────┐            │             │ ImageNet) → 1280-d   │
   │ scene_validate()   │            │             └──────────┬───────────┘
   │ aspect ratio ≥0.4  │            │                        ▼
   └─────────┬──────────┘            │             ┌──────────────────────┐
             ▼                       │             │ LSTM head (578,689p) │
   ┌────────────────────┐            │             │ D256→LSTM128→LSTM64  │
   │ CentroidTracker    │            │             │ →D64→sigmoid         │
   │ greedy, resets on  │            │             └──────────┬───────────┘
   │ ANY empty frame    │            │                        │
   └─────────┬──────────┘            │                        │ cnn ∈ [0,1]
             ▼                       │                        │
   ┌────────────────────┐            │                        │
   │ pixel_to_ground()  │            │                        │
   │ + VehicleKF (4-D)  │            │                        │
   └─────────┬──────────┘            │                        │
             ▼                       │                        │
   ┌────────────────────┐            │                        │
   │ RuleCollision      │            │                        │
   │ 3-D dist + TTC     │            │                        │
   └─────────┬──────────┘            │                        │
             │                       │                        │
             ▼                       ▼                        ▼
   ╔═════════════════════════════════════════════════════════════════════╗
   ║                        THE ACTUAL DECISION                          ║
   ║                                                                     ║
   ║  standard mode  (enhanced.py:1050):                                 ║
   ║      is_crash = (cnn >= 0.80) and (len(vehs) >= 2)                  ║
   ║                          ▲                                          ║
   ║                          └── 0.80 appears NOWHERE in training.      ║
   ║                              Colab evaluated at 0.50, suggested 0.55║
   ║                                                                     ║
   ║  dashcam mode   (enhanced.py:1041):                                 ║
   ║      is_crash = len(EgoZone.check(...)) > 0                         ║
   ║                                                                     ║
   ║  video verdict  (enhanced.py:1273):                                 ║
   ║      is_crash = neural_confirmed and (crash_pct >= 1.5)             ║
   ╚═════════════════════════════════════════════════════════════════════╝
             │                       │                        │
             │  ✗ rule[] UNUSED      │  ✗ depth_map           │
             │    except during the  │    ONLY drawn as a     │
             │    9-frame CNN        │    heatmap overlay     │
             │    warm-up            │    (line 1089)         │
             ▼                       ▼                        ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │ Display.draw_with_overlays()  → BEV minimap, depth heatmap,         │
   │ bounding boxes, "CRASH DETECTED" banner, FaultDetector attribution  │
   └─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                         stdout report + optional JPEG frames
                         (no database, no API, no queue, no UI)
```

**Read that diagram again.** Everything in the left and middle columns — the projection, the Kalman filter, the TTC, the depth network — is computed, costs latency, and is then thrown away. It exists to be drawn on the screen.

### Component-by-component assessment

| Component | File:line | What it does | Correct? | Production-worthy? | Severity |
|---|---|---|---|---|---|
| Frame ingest | `enhanced.py:1010` | Native-resolution frames, no resize | No resolution normalisation | No | **CRITICAL** |
| ROI mask | `enhanced.py:402` | Blanks top 40% / bottom 10% before YOLO | Works; fixed fractions assume one camera | No | MEDIUM |
| Detector | `enhanced.py:263` | YOLOv8n, conf 0.5, cross-class NMS | Functionally fine | **No — AGPL-3.0** | **CRITICAL (legal)** |
| Scene validator | `enhanced.py:303` | Drops boxes with aspect ratio < 0.4 | Crude; drops head-on motorcycles | No | MEDIUM |
| Tracker | `enhanced.py:329` | Greedy centroid matching | **Resets all IDs on any empty frame** (`:335`) | No | HIGH |
| Ground projection | `enhanced.py:381` | Pinhole ray-ground intersection | **Math right, calibration wrong** | No | **CRITICAL** |
| Kalman filter | `enhanced.py:436` | 4-D constant-velocity, proper Q/R | Correctly implemented | Would be, given valid input | HIGH (wasted) |
| TTC | `enhanced.py:415` | Vector-projected closing speed | Correctly implemented | Output unused | HIGH (wasted) |
| Depth | `depth_estimator.py:130` | MiDaS → "metric" depth | **Inverse-depth error** | No | HIGH |
| BEV renderer | `bev_renderer.py:86` | Top-down minimap | **Double km/h conversion** | Demo only | MEDIUM |
| Feature extractor | `enhanced.py:174` | `tf.saved_model.load(feature_extractor_saved).serve` | Matches training preprocessing | Provenance undocumented (U5) | MEDIUM |
| Frame buffering | `enhanced.py:220` | `deque(maxlen=10)`, consecutive frames | **Wrong temporal stride vs training** | No | **CRITICAL (B1)** |
| Neural classifier | `enhanced.py:166` | MobileNetV2+LSTM over 10 frames | Loads correctly; weights untraceable | **Unknown quality** | **CRITICAL** |
| Decision gate | `enhanced.py:1050` | `cnn ≥ 0.80 and n_veh ≥ 2` | Threshold has no derivation | No | **CRITICAL** |
| Fault attribution | `enhanced.py:622`, `:761` | "Who hit whom" from boxes/velocities | Built on invalid geometry | **No — legally dangerous** | **CRITICAL** |
| Ego zone | `enhanced.py:732` | Fixed rectangle = "your bumper" | Not a bumper; not calibrated | No | **CRITICAL** |
| Reporting | `enhanced.py:1252` | Prints to stdout | Works | No persistence at all | HIGH |

---

## 5. Current Model

### What it is, exactly

| Property | Value | Evidence |
|---|---|---|
| Family | Frozen CNN feature extractor + trained recurrent head | Colab cells 4, 7 |
| Backbone | `keras.applications.MobileNetV2`, ImageNet weights, `include_top=False` | Cell 4 |
| Backbone trainable | **No** — `base_model.trainable = False`, and features were precomputed offline | Cells 4, 5 |
| Pooling | `GlobalAveragePooling2D` → 1280-d per frame | Cell 4, output `(None, 1280)` |
| Input to head | `(10, 1280)` float32 | Cell 7, `model.summary()` |
| Head | Dense 256 relu → Dropout 0.3 → LSTM 128 (seq) → Dropout 0.3 → LSTM 64 → Dropout 0.3 → Dense 64 relu → Dropout 0.2 → Dense 1 sigmoid | Cell 7 |
| Parameters | **578,689 — all trainable, 0 non-trainable** | `model.summary()`; independently confirmed by summing non-optimizer arrays in the shipped `.weights.h5` |
| Output | single sigmoid, P(crash) for the clip | Cell 7 |
| Serialisation | Keras **3.13.2**, saved 2026-04-17 08:48:28 UTC (for the archived `.keras`) | `metadata.json` inside `best_crash_model.keras` at commit `c63e307` |

### Why the classifier is probably not learning collisions

The previous audit's hypothesis was that the model learned "this looks like `crash1`/`crash2`". The Colab shows the real mechanism is broader and cleaner, and therefore more likely.

**The classes are corpora.** All 1,500 positives are YouTube crash-compilation clips (CCD's `Crash-1500`, each cut from a YouTube source at a recorded `startframe`). All 3,000 negatives are BDD100K clips. These differ systematically in:

- **encoding** — compilation uploads are re-encoded, often multiple times; BDD100K is a single controlled capture pipeline
- **resolution and aspect** — heterogeneous vs uniform
- **colour grading and exposure** — consumer dashcams worldwide vs one research collection
- **geography and road furniture** — global YouTube crash footage vs Berkeley-collected US driving
- **camera hardware and mounting** — arbitrary vs consistent

A 578,689-parameter head sitting on top of ImageNet features has more than enough capacity to separate those two distributions using **the first frame alone**, and would reach very high AUC doing so. That is the most parsimonious explanation for reaching val AUC 0.994 **in a single epoch** (cell 8, epoch 1) and 0.998 by epoch 9.

### ✅ RESULT — the tests were run on 2026-09-10, and the model failed them

Full method and raw output: [`runs/falsification/RESULTS.md`](runs/falsification/RESULTS.md). Features extracted once per video with the deployed `feature_extractor_saved`; the 578,689-parameter head evaluated by an exact NumPy forward pass read straight from the shipped Keras-3 HDF5 weights.

| Video | A · deployed (10 consecutive) | B · **temporally shuffled** | C · training-matched stride | D · **single frame tiled ×10** |
|---|---|---|---|---|
| `crash1.mov` | **0.9997** — 100% ≥0.80 | **0.9998** — 100% | 0.9998 — 100% | **0.9798** — 98% |
| `crash2.mov` | **0.9640** — 94.1% | **0.9649** — 91.2% | 0.9996 — 100% | **0.9460** — 94.1% |
| `safe.mp4` | **0.0241** — 0% (max **0.7914**) | **0.0228** — 0% | 0.0003 — 0% | 0.0512 (max **0.9695**) |

**Test 2 — temporal shuffle: FAILED.** Randomly permuting the ten frames inside the window changes the mean score by **≤ 0.0013 on every video**. A model that had learned how a collision unfolds over time cannot be invariant to the order of its own input. **The two LSTM layers — 246,528 of the 578,689 parameters — contribute nothing measurable.**

**Test 1 — single-frame: FAILED.** One frame repeated ten times reproduces the deployed score to within 0.02 on both crash videos. **This is a per-frame appearance classifier.**

**Test 4 — crash excision: not run** (needs the cached features, which live in Drive). But [§7](#7-dataset-used) now makes it near-redundant: the accident begins at frame 37.2 of 50 on average, so **72% of the frames the model was shown for a positive clip contain no accident** and were labelled 1 anyway.

**Test 3 — corpus control: RUN 2026-09-11, and FAILED.** On Nexar test-public — 667 clips, 334 positive / 333 negative, one corpus, one anonymisation pipeline, so no class/corpus confound is available to exploit — the shipped model scores **ROC-AUC 0.5339, AP 0.5218**. Chance is 0.50. The score distributions are near-identical and near-saturated: median **0.9998 for positives and 0.9998 for negatives**, means 0.998 and 0.977. At the deployed threshold of 0.80 the model fires on **325 of 333 negatives (FPR 97.6%)** and 332 of 334 positives (TPR 99.4%).

**This is the decisive measurement, and it is worse than "does not generalise."** The model does not rank Nexar clips at all; it emits near-1.0 on essentially everything it is shown from a corpus it was not trained on. The 0.9977 val AUC on CCD was therefore measuring the corpus boundary, exactly as [B4](#b4--the-positivenegative-classes-coincide-with-two-different-corpora) predicted.

**The measurement was itself falsified before being accepted.** Running the identical scoring path over `safe.mp4` and `crash1.mov` reproduces the T1/T2 figures **exactly** (0.7914 and 0.9998) — so the code discriminates when discrimination is present, and the Nexar collapse is a property of the model under corpus shift, not an artefact of the harness. Raw results: [`runs/falsification/T3_corpus_control.md`](runs/falsification/T3_corpus_control.md); method: `scripts/t3_corpus_control.py`.

**Two incidental findings that indict the deployed threshold directly.** `safe.mp4` peaks at **0.7914** — reproducing the prior README's "0.79" exactly, meaning `CNN_THRESH = 0.80` clears the only negative ever tested by **0.0086**. And under single-frame sampling the same "safe" video reaches **0.9695**: the model does emit confident crash scores on it, hidden only by the particular averaging the deployed code happens to perform.

---

## 6. Training Pipeline

Reconstructed cell-by-cell from the Colab notebook `Untitled0.ipynb`. **This section did not exist in the previous audit and is the principal addition of this revision.**

### 6.1 Pipeline in one diagram

```
Google Drive: My Drive/CarCrashDetection/
  └── data/dataset/Crash-1500.zip   (1,500 crash videos, CCD)
      data/dataset/Normal.zip       (3,000 normal videos, CCD ← BDD100K)
                    │
                    │  CELL 3: zipfile.extractall + flatten subfolders
                    ▼
  /tmp/crash_extracted/   (1,500 .mp4)     ← ephemeral, lost on runtime restart
  /tmp/normal_extracted/  (3,000 .mp4)
                    │
                    │  CELL 5: per video —
                    │    total = CAP_PROP_FRAME_COUNT                     (= 50)
                    │    indices = np.linspace(0, total-1, 10, int)       → 0,5,10,16,21,27,32,38,43,49
                    │    cap.set(CAP_PROP_POS_FRAMES, idx); cap.read()
                    │    cv2.resize(frame, (112,112))    ← aspect NOT preserved
                    │    cv2.cvtColor(BGR → RGB)
                    │    mobilenet_v2.preprocess_input   ← scales to [-1, 1]
                    │    feature_extractor.predict(batch) → (10, 1280)
                    ▼
  My Drive/.../data/features_v2/crash_000000.npy … crash_001499.npy      (1,500)
                                normal_000000.npy … normal_002999.npy    (3,000)
                    │
                    │  CELL 6: labels from FILENAME PREFIX ONLY
                    │    crash_* → 1     normal_* → 0
                    │    train_test_split(test_size=0.2, random_state=42, stratify=y)
                    │    ✗ no source grouping   ✗ no test set   ✗ Crash-1500.txt never read
                    ▼
  train_gen: 3,600 samples, batch 32, shuffle=True,  augment=True  (feat += N(0, 0.01))
  val_gen  :   900 samples, batch 32, shuffle=False, augment=False
                    │
                    │  CELLS 7–8: LSTM head, Adam 1e-3, BCE, class_weight {0:0.75, 1:1.5}
                    │    EarlyStopping(val_auc, patience 8, restore_best)
                    │    ReduceLROnPlateau(val_loss, ×0.5, patience 4)
                    │    ModelCheckpoint(val_auc, save_best_only)  → FIXED PATH, overwritten each run
                    ▼
  models/best_crash_model.keras
                    │
        ┌───────────┼───────────────┬──────────────────────┐
        ▼           ▼               ▼                      ▼
   CELL 9        CELL 10        CELL 11                CELL 12
   evaluate      save FE        rebuild on CPU,        save_weights →
   on val_gen    .keras         .export() SavedModel   crash_model_weights.weights.h5
   @ 0.50        (thresh 0.55                                  │
   ✗ same split   suggested)                                   ▼
     used for                                          THE FILE THAT SHIPS
     model selection
```

### 6.2 Training reconstruction table

Every value below is quoted from the notebook. `UNKNOWN — evidence not found` means exactly that; nothing here is inferred.

| Item | Actual current value | Evidence | Reproducible? |
|---|---|---|---|
| **Dataset** | Car Crash Dataset (CCD), `Cogito2012/CarCrashDataset` | Cell 3 loads `Crash-1500.zip` + `Normal.zip`; counts match CCD exactly | **Yes** — public Google Drive download |
| **Positive samples** | 1,500 videos → 1,500 feature files, 0 skipped | Cell 3 output `Extracted 1500 videos`; cell 5 output `Done: 1500 saved, 0 skipped` | Yes |
| **Negative samples** | 3,000 videos → 3,000 feature files, 0 skipped | Cell 3 output `Extracted 3000 videos`; cell 5 output `Done: 3000 saved, 0 skipped` | Yes |
| **Total samples** | 4,500 | Cell 5 output `Total: 1500 crash + 3000 normal = 4500 samples` | Yes |
| **Class balance** | 1:2 positive:negative (33.3% positive) | Cell 6 output `Crash samples : 1500 / Normal samples: 3000` | Yes |
| **Class weights** | `{0: 0.75, 1: 1.5}` (inverse-frequency) | Cell 6 output `Class weights: {0: 0.75, 1: 1.5}` | Yes |
| **Split method** | `sklearn.train_test_split`, `test_size=0.2`, `random_state=42`, `stratify=all_labels` | Cell 6 lines 26–31 | **Yes** — seed fixed |
| **Split sizes** | **3,600 train / 900 validation. No test set.** | Cell 6 output `Train: 3600 \| Val: 900` | Yes |
| **Official CCD split used?** | **No.** `train.txt` / `test.txt` never referenced | Cell 6 lists `FEATURES_DIR` by filename prefix only | n/a |
| **Source grouping** | **None.** `youtubeID` never read | `Crash-1500.txt` appears nowhere in the notebook | n/a |
| **Frames per video** | 10 | `FRAMES_PER_VIDEO = 10` (cell 2) | Yes |
| **Frame sampling** | `np.linspace(0, total-1, 10, dtype=int)` — uniform across the whole clip | Cell 5 line 21 | Yes |
| **Effective temporal stride (training)** | ≈5.44 frames @ 10 fps ⇒ **≈0.54 s between steps, ≈4.9 s span** | CCD clips are 50 frames @ 10 fps (CCD README) | Yes |
| **Effective temporal stride (inference)** | 1 frame, consecutive ⇒ **0.033 s @30 fps / 0.017 s @60 fps, 0.33 s / 0.17 s span** | `enhanced.py:167` `deque(maxlen=10)`, `:220` appends every frame | Yes — **and it does not match training (bug B1)** |
| **Input resolution** | 112 × 112, aspect ratio **not** preserved (`cv2.resize` to a square) | `FRAME_SIZE = 112` (cell 2); cell 5 line 30 | Yes |
| **Colour conversion** | `cv2.cvtColor(frame, COLOR_BGR2RGB)` | Cell 5 line 31 | Yes |
| **Normalisation** | `keras.applications.mobilenet_v2.preprocess_input` (scales to [-1, 1]) | Cell 5 line 12, 41 | Yes |
| **Backbone** | MobileNetV2, ImageNet weights, `include_top=False`, + `GlobalAveragePooling2D` | Cell 4 | Yes |
| **Backbone frozen?** | **Yes** — `base_model.trainable = False`, and features precomputed offline to `.npy` so the backbone was never in the training graph | Cell 4 line 14; cell 5 | Yes |
| **Feature dimension** | 1,280 | `FEATURE_DIM = 1280` (cell 2); cell 4 output `(None, 1280)` | Yes |
| **Sequence length** | 10 | Cell 7, `frame_sequence (InputLayer) (None, 10, 1280)` | Yes |
| **LSTM usage** | Dense 256 relu → Dropout 0.3 → **LSTM 128 `return_sequences=True`** → Dropout 0.3 → **LSTM 64 `return_sequences=False`** → Dropout 0.3 → Dense 64 relu → Dropout 0.2 → Dense 1 sigmoid | Cell 7 | Yes |
| **Total parameters** | **578,689**, all trainable, 0 non-trainable | `model.summary()` output; confirmed against the shipped `.weights.h5` | Yes |
| **Labels / class mapping** | `crash_*` → **1**, `normal_*` → **0**; derived from filename prefix | Cell 6 lines 8–12 | Yes |
| **Temporal annotations used?** | **No.** CCD `binlabels` (per-frame accident flags) never read | Cell 6 assigns one label per clip | n/a |
| **Ego-involvement labels used?** | **No.** CCD `egoinvolve` never read | Same | n/a |
| **Weather / lighting labels used?** | **No.** CCD `timing`, `weather` never read | Same | n/a |
| **Loss** | `binary_crossentropy` | Cell 7 line 28; confirmed in `compile_config` of the archived `.keras` | Yes |
| **Optimizer** | `keras.optimizers.Adam`, β₁ 0.9, β₂ 0.999, ε 1e-07, no weight decay, no amsgrad | Cell 7 line 27; `compile_config` in the archived `.keras` | Yes |
| **Learning rate** | `1e-3` initial; reduced to `5e-4` at epoch 11 and `2.5e-4` at epoch 15 by `ReduceLROnPlateau` | `LR = 1e-3` (cell 2); cell 8 output | Yes |
| **Batch size** | 32 | `BATCH_SIZE = 32` (cell 2); cell 6 output `X=(32, 10, 1280)` | Yes |
| **Epochs** | Budget **50**; actually ran **17**, early-stopped; best weights restored from **epoch 9** | `EPOCHS = 50` (cell 2); cell 8 output `Epoch 17: early stopping`, `Restoring model weights from the end of the best epoch: 9.` | Yes |
| **Steps per epoch** | 113 (⌈3600/32⌉) | Cell 8 output `113/113` | Yes |
| **Wall-clock per epoch** | 17–27 s on a Colab **T4** GPU | Cell 8 output; cell 1 output `PhysicalDevice(... GPU:0)` | Yes |
| **Callbacks** | `EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True)`; `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)`; `ModelCheckpoint(monitor='val_auc', save_best_only=True, mode='max')` | Cell 8 | Yes |
| **Augmentation** | **Additive Gaussian noise on the cached features only**: `feat += np.random.normal(0, 0.01, feat.shape)`, train split only. No pixel-space augmentation of any kind. | Cell 6 lines 54–55 | Yes — **but see bug B6; σ=0.01 is negligible against MobileNetV2 GAP feature magnitudes** |
| **Augmentation RNG seed** | **UNKNOWN — evidence not found.** `np.random` is never seeded; only `train_test_split` is seeded. | Cell 6 | **No** — run-to-run nondeterminism |
| **Training-loop / init seed** | **UNKNOWN — evidence not found.** No `tf.random.set_seed`, no `keras.utils.set_random_seed` | Whole notebook | **No** |
| **Metrics tracked** | accuracy, AUC (ROC, 200 thresholds), precision, recall — train and val | Cell 7 lines 29–32 | Yes |
| **Evaluation threshold** | **0.50** in cell 9 | Cell 9 line 17 `(all_preds >= 0.5)` | Yes |
| **Suggested inference threshold** | **0.55** in the generated inference snippet | Cell 10 `CRASH_THRESH = 0.55` | Yes |
| **Deployed threshold** | **0.80** | `enhanced.py:109` `CNN_THRESH = 0.80` | **No derivation exists in the notebook** |
| **Final model artefacts** | `best_crash_model.keras` (cell 8 checkpoint); `feature_extractor.keras` (cell 10); `crash_model_cpu/` SavedModel (cell 11); `crash_model_weights.weights.h5` (cell 12) | Cells 8, 10, 11, 12 outputs | Partly — see [§11](#11-current-model-artifact) |
| **Number of training runs** | **≥ 6 model constructions in the recorded kernel session** (`Model: "functional_5"`), and **≥ 2 distinct completed runs** proven by weight comparison | `model.summary()` output; weight-array comparison (D) | n/a |
| **Which run produced the shipped weights** | **UNKNOWN — evidence not found.** Not the one recorded in the notebook. | 37/38 arrays differ; r = 0.0073 on the first Dense kernel | **No** |
| **Framework versions** | TensorFlow **2.19.0**, Keras **3.13.2** | Cell 1 output `TensorFlow: 2.19.0`; `metadata.json` in the archived `.keras` | Yes |
| **Hardware** | Colab T4 GPU | Cell 1 output | Yes |
| **Experiment tracking** | **None.** No W&B, no MLflow, no run directory, no metrics file. | Whole notebook | **No** |

### 6.3 Assessment of the pipeline as engineering

Credit where it is earned, because it changes what should be *kept*:

**Sound choices.** Precomputing frozen features to disk is exactly right for a small dataset and a weak GPU budget — it makes each epoch 18 seconds instead of minutes. `stratify=all_labels` is correct. Inverse-frequency class weighting is correct. Monitoring `val_auc` rather than accuracy is correct and notably better than the previous audit's expectation. `restore_best_weights=True` is correct. The `Sequence` generator is a reasonable memory-bounded loader. `preprocess_input` is applied consistently in training *and* inference — a genuinely common bug that this pipeline does **not** have.

**Unsound choices, in order of severity.** No test set. No source grouping. Per-frame labels discarded. Uniform frame sampling with no event-centring. Feature-space noise as the only augmentation. No seeding of anything except the split. Fixed checkpoint path with no run isolation. No experiment log.

**The pipeline is worth keeping as a starting point and is not worth defending as an experiment.** Phase 1 of the plan rewrites it as a versioned script; roughly 60% of the code survives that rewrite intact.

---

## 7. Dataset Used

**Identified with certainty:** the **Car Crash Dataset (CCD)**, from Bao, Yu & Kong, *Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning*, ACM Multimedia 2020 — repository [`Cogito2012/CarCrashDataset`](https://github.com/Cogito2012/CarCrashDataset).

| Property | Value |
|---|---|
| Crash videos | 1,500 (`Crash-1500`) |
| Normal videos | 3,000 (`Normal`) |
| Frames per video | 50 |
| Frame rate | 10 fps |
| Clip duration | 5.0 s |
| Crash video source | **YouTube** — each clip is a window cut from a source video, with `youtubeID` and `startframe` recorded |
| Normal video source | **BDD100K**, randomly sampled |
| Provided features | VGG-16, 4,096-d, from Cascade R-CNN (ResNeXt-101) top-19 boxes per frame |
| Provided splits | `vgg16_features/train.txt`, `vgg16_features/test.txt` |
| Provided annotations | `Crash-1500.txt`: `vidname`, `binlabels` (50 per-frame binary accident flags), `startframe`, `youtubeID`, `timing` (Day/Night), `weather` (Normal/Snowy/Rainy), `egoinvolve` (bool) |
| Repository licence label | MIT (GitHub sidebar). **The README contains no licence section.** |

### What the project actually consumed

| Question | Answer | Evidence |
|---|---|---|
| Trained from raw MP4 videos? | **Yes** — from `videos/Crash-1500` and `videos/Normal`, repackaged as two zips | Cell 3 |
| Trained from extracted frames? | Frames were decoded transiently in memory; never written to disk | Cell 5 |
| Trained from the provided VGG16 features? | **No.** `vgg16_features/` was not used at all | No reference anywhere in the notebook |
| Trained from another representation? | **Yes** — a *custom* representation: MobileNetV2 GAP features at 112×112, 10 frames per clip, cached as `.npy` in `features_v2/` | Cell 5 |
| Official train/test split used? | **No** | Cell 6 |
| Own split created? | **Yes** — random stratified 80/20 on filenames, seed 42 | Cell 6 |
| Temporal annotations used? | **No** — `Crash-1500.txt` never opened | Whole notebook |
| Labels correctly interpreted? | **Yes at the clip level** (`Crash-1500` → 1, `Normal` → 0 is the correct CCD semantics), **no at the frame level** (every frame of a crash clip is implicitly treated as positive, including the ~4 seconds before impact) | Cells 5–6 |
| Positive/negative balanced? | **No, deliberately** — 1:2, compensated with class weights. Reasonable. But note the *deployment* prior is ~1 crash per 50,000 km, so neither 1:2 nor 1:1 resembles reality. | Cell 6 |

### Why using CCD at all was a defensible choice, and why it cannot continue

**Defensible:** it is free, well-known, correctly sized for a first experiment, and shipped in a form (two zips of MP4s) that a solo founder can actually work with. Choosing it was not the mistake.

**Cannot continue:** (a) the class/corpus confound in [§5](#5-current-model) makes any result on it uninterpretable without a corpus control; (b) the licence chain in [§23](#23-dataset-licensing--provenance) does not support commercial use; (c) the road environment is predominantly East Asian and US, not UK; (d) it has no near-miss labels, and near-miss/collision separation is the product.

**Keep it as a research and sanity-check corpus. Do not train a shipped model on it.**

---

## 8. Dataset Structure

### 8.1 Official CCD structure (evidence class C)

```
CarCrash/
├── codes/
├── vgg16_features/
│   ├── positive/   000001.npz … 001500.npz     (1,500)
│   ├── negative/   000001.npz … 003000.npz     (3,000)
│   ├── train.txt                                ← official split, UNUSED
│   └── test.txt                                 ← official split, UNUSED
├── videos/
│   ├── Normal/      000001.mp4 … 003000.mp4     ← from BDD100K
│   ├── Crash-1500/  000001.mp4 … 001500.mp4     ← from YouTube
│   └── Crash-1500.txt                           ← per-frame + metadata, UNUSED
└── README.md
```

### 8.2 Structure actually used, in Google Drive and Colab (evidence class B)

```
My Drive/CarCrashDetection/                      ← BASE_PATH
├── data/
│   ├── dataset/
│   │   ├── Crash-1500.zip                       ← repackaged from CarCrash/videos/Crash-1500/
│   │   └── Normal.zip                           ← repackaged from CarCrash/videos/Normal/
│   └── features_v2/                             ← FEATURES_DIR — the actual training input
│       ├── crash_000000.npy … crash_001499.npy     each (10, 1280) float32
│       └── normal_000000.npy … normal_002999.npy
├── models/
│   ├── best_crash_model.keras                   ← ModelCheckpoint target (FIXED PATH — overwritten every run)
│   ├── feature_extractor.keras                  ← cell 10
│   ├── crash_model_cpu/                         ← cell 11, SavedModel
│   └── crash_model_weights.weights.h5           ← cell 12 — THE FILE THAT SHIPS
└── results/
    ├── confusion_matrix.png                     ← cell 9
    └── training_curves.png                      ← cell 9

/tmp/crash_extracted/    1,500 .mp4              ← ephemeral, destroyed on runtime restart
/tmp/normal_extracted/   3,000 .mp4              ← ephemeral
```

### 8.3 Reconciliation — why the structures differ

The differences are fully explained by four identifiable, deliberate actions. **None of them is corruption.**

| Difference | Cause | Evidence |
|---|---|---|
| `vgg16_features/` absent | **Not downloaded / not used.** The project chose its own representation (MobileNetV2 @112) rather than CCD's VGG-16 @4096. | Cell 4 builds a fresh extractor; no reference to `.npz` anywhere |
| `train.txt` / `test.txt` absent | **Not downloaded / not used.** A new random split was generated instead. | Cell 6 |
| `Crash-1500.txt` absent | **Not downloaded / not used.** Per-frame and metadata annotations were never read. | Whole notebook |
| `videos/Crash-1500/` and `videos/Normal/` became `Crash-1500.zip` and `Normal.zip` | **Repackaging for Google Drive.** Two zips upload and mount far faster than 4,500 individual files, and Drive's per-file overhead makes directory-of-videos access painfully slow from Colab. This is a sensible operational choice. | Cell 3 unzips exactly these two names |
| Videos live in `/tmp`, not Drive | **Deliberate performance choice.** Decoding 4,500 videos over the Drive FUSE mount would be an order of magnitude slower than from local SSD. | Cell 3 `CRASH_VIDEOS_DIR = '/tmp/crash_extracted'` |
| A new directory `features_v2/` exists | **Custom preprocessing output**, and the `_v2` suffix plus cell 5's comment (*"Old approach: np.mean(frames) → one blurry image"*) indicate a superseded v1 feature set. | Cell 2, cell 5 |
| Nested subfolders flattened | Cell 3 explicitly walks the extraction tree and `shutil.move`s every video to the top level. | Cell 3 lines 13–20 |

**Summary: the structure changed because of (1) selective download, (2) repackaging for Drive, (3) a `/tmp` performance copy, and (4) custom feature extraction into a new directory. It did not change because of renaming accidents, copying errors, or corruption.**

### 8.4 The local structure — UNKNOWN

You referred to a current dataset structure visible in your environment or a screenshot. **I cannot reconcile it: no dataset directory exists anywhere on this machine.** `find ~/Desktop -type d -iname '*carcrash*' -o -iname '*dataset*' -o -iname 'vgg16_features'` returns nothing; `videos/` is a symlink to a directory containing only `crash1.mov`, `crash2.mov` and `safe.mp4`. Either the dataset lives only in Google Drive (which the Colab evidence supports), or it is on a device or path I have not been shown. **Send the screenshot or a `tree -L 3` of that directory and I will reconcile it against §8.1 and §8.2.** No conclusion in this document depends on the answer.

### 8.5 What to retain and what can be regenerated

| Asset | Retain? | Reason |
|---|---|---|
| `Crash-1500.zip`, `Normal.zip` | **Retain** | Re-downloadable, but slowly; and you need byte-stability for reproducibility |
| **`Crash-1500.txt`, `train.txt`, `test.txt`** | **DOWNLOAD THESE NOW — you do not have them** | They contain the grouping key, the official split and the per-frame labels. This is the highest-value missing file in the project. |
| `features_v2/*.npy` (4,500 files) | **Regenerable** — delete once the extraction script is in git and deterministic | Pure function of the videos + the extraction code |
| `best_crash_model.keras` | Retain the copy in Drive as a historical artefact; **do not ship it** | Provenance unclear |
| `crash_model_weights.weights.h5` | Retain until Phase 2 measures it; then **discard** | Untraceable run |
| `feature_extractor.keras`, `feature_extractor_saved/` | **Regenerable** in three lines from `keras.applications` | Not worth versioning |
| `crash_model_cpu/`, `crash_model_saved/` | **Regenerable / orphaned** | Export artefacts |
| `crash_detection_model.h5` (296 MB), `crash_detection_model/` (197 MB) | **Orphaned — delete after confirming nothing loads them** | Not producible by this notebook; see U4 |
| `/tmp/*_extracted/` | Ephemeral by design | — |

---

## 9. Dataset Split

```python
# Colab cell 6, verbatim
crash_files  = sorted([f for f in os.listdir(FEATURES_DIR) if f.startswith('crash_')])
normal_files = sorted([f for f in os.listdir(FEATURES_DIR) if f.startswith('normal_')])
all_files  = crash_files + normal_files
all_labels = [1] * len(crash_files) + [0] * len(normal_files)

X_train_files, X_val_files, y_train, y_val = train_test_split(
    all_files, all_labels,
    test_size=TEST_SIZE,        # 0.2
    random_state=RANDOM_SEED,   # 42
    stratify=all_labels
)
# output: Train: 3600 | Val: 900
```

| Property | Value |
|---|---|
| Method | `sklearn.model_selection.train_test_split`, stratified |
| Seed | 42 (fixed — the split itself **is** reproducible) |
| Train | 3,600 clips (1,200 crash / 2,400 normal) |
| Validation | 900 clips (300 crash / 600 normal) — confirmed by cell 9's `support` column |
| **Test** | **None. There is no third partition.** |
| Grouping key | **None** |
| Official CCD split | **Not used** |
| Calibration split | **None** |

### Was the split done correctly?

**No, in three distinct ways.**

**(1) There is no test set.** `train_test_split` yields two partitions. `EarlyStopping(monitor='val_auc')` selects the stopping epoch from partition 2. `ModelCheckpoint(monitor='val_auc', save_best_only=True)` selects the saved weights from partition 2. `restore_best_weights=True` restores from partition 2. Cell 9 then reports final performance on **partition 2**. Every number the project has ever produced about this model is a number from the split that chose the model. `Best val AUC : 0.9977` is a maximum over 17 noisy estimates on 900 samples — a selection statistic, not a performance estimate.

**(2) There is no source grouping — and the key was sitting in a file that was never downloaded.** CCD's `Crash-1500.txt` records a `youtubeID` per crash clip precisely because clips were cut from longer YouTube videos. Multiple clips from one source video can therefore be split across train and validation, letting the model recognise a source rather than a collision. The negatives inherit the same problem from BDD100K's per-drive video structure. **You cannot currently quantify this**, because the grouping file has never been downloaded — which is why "download `Crash-1500.txt` and count cross-split `youtubeID` collisions" is a Phase 0 task with a concrete numeric output.

**(3) The split is meaningless against the deployment distribution anyway.** Validation prevalence is 33% positive. Real driving prevalence is on the order of one collision per 50,000 km. Precision measured at 33% prevalence tells you essentially nothing about precision in production; see [§18](#18-calibration-problems).

### Is there data leakage?

**Cross-split source leakage: probable but unquantified** (blocked on `Crash-1500.txt`). **Model-selection leakage: certain and proven** — the reported metrics come from the selection split. **Threshold leakage into the local test videos: certain** — `CNN_THRESH = 0.80` was fitted against `crash1/crash2/safe`, the only three videos ever used to evaluate the assembled system. Full treatment in [§16](#16-data-leakage-risks).

---

## 10. Training History

**This section is entirely new. The previous audit stated no training history existed; it exists, in full, in the notebook's saved output for cell 8.**

Recorded run: 17 epochs of a 50-epoch budget, Colab T4, 113 steps/epoch, 17–27 s/epoch (≈5.5 minutes total).

| Epoch | train acc | train AUC | train loss | **val acc** | **val AUC** | val loss | val prec | val rec | LR | Event |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.8919 | 0.9561 | 0.2674 | 0.9389 | **0.9942** | 0.1590 | 0.8510 | 0.9900 | 1.0e-3 | ✅ checkpoint |
| 2 | 0.9608 | 0.9917 | 0.1087 | 0.9267 | **0.9962** | 0.1695 | 0.8250 | 0.9900 | 1.0e-3 | ✅ checkpoint |
| 3 | 0.9733 | 0.9952 | 0.0775 | 0.9633 | 0.9954 | 0.0801 | 0.9265 | 0.9667 | 1.0e-3 | |
| 4 | 0.9814 | 0.9964 | 0.0568 | 0.9767 | 0.9943 | 0.0654 | 0.9761 | 0.9533 | 1.0e-3 | |
| 5 | 0.9783 | 0.9977 | 0.0579 | 0.9800 | 0.9949 | 0.0658 | 0.9764 | 0.9633 | 1.0e-3 | |
| 6 | 0.9822 | 0.9976 | 0.0489 | 0.9778 | 0.9941 | 0.0766 | 0.9930 | 0.9400 | 1.0e-3 | |
| 7 | 0.9817 | 0.9984 | 0.0439 | 0.9822 | 0.9946 | 0.0627 | 0.9830 | 0.9633 | 1.0e-3 | best val acc |
| 8 | 0.9878 | 0.9990 | 0.0288 | 0.9733 | 0.9860 | 0.1172 | 0.9929 | 0.9267 | 1.0e-3 | |
| **9** | 0.9914 | 0.9992 | 0.0262 | 0.9733 | **0.9977** | 0.0633 | 0.9395 | 0.9833 | 1.0e-3 | ✅ **best — weights restored from here** |
| 10 | 0.9908 | 0.9987 | 0.0249 | 0.9789 | 0.9875 | 0.0936 | 0.9930 | 0.9433 | 1.0e-3 | |
| 11 | 0.9792 | 0.9979 | 0.0507 | 0.9756 | 0.9918 | 0.0866 | 0.9603 | 0.9667 | 1.0e-3 | ⤵ LR → 5.0e-4 |
| 12 | 0.9961 | 0.9997 | 0.0101 | 0.9756 | 0.9897 | 0.1061 | 0.9513 | 0.9767 | 5.0e-4 | |
| 13 | 0.9972 | 1.0000 | 0.0071 | 0.9778 | 0.9875 | 0.0989 | 0.9636 | 0.9700 | 5.0e-4 | |
| 14 | 0.9992 | 0.9998 | 0.0050 | 0.9767 | 0.9821 | 0.1435 | 0.9964 | 0.9333 | 5.0e-4 | |
| 15 | 0.9986 | 1.0000 | 0.0033 | 0.9744 | 0.9857 | 0.1316 | 0.9571 | 0.9667 | 5.0e-4 | ⤵ LR → 2.5e-4 |
| 16 | 0.9994 | 1.0000 | 0.0015 | 0.9789 | 0.9819 | 0.1268 | 0.9828 | 0.9533 | 2.5e-4 | |
| 17 | 1.0000 | 1.0000 | 3.4e-4 | 0.9800 | 0.9811 | 0.1318 | 0.9764 | 0.9633 | 2.5e-4 | ⏹ early stopping |

```
Best val AUC : 0.9977      Best val acc : 0.9822
```

### Reading this honestly

**The training curve is the strongest single piece of evidence that the task, as posed, is trivial.** Consider what epoch 1 says: **val AUC 0.9942 after a single pass**, from a randomly-initialised 578k-parameter head over frozen ImageNet features. A genuinely difficult perceptual discrimination — "did two vehicles collide in this 5-second clip?" — does not fall to 0.994 in 27 seconds. A corpus discrimination does.

**The model memorises the training set completely by epoch 17** (train acc 1.0000, train loss 3.4e-4, train AUC 1.0000) while val loss *rises* from 0.063 to 0.132 after epoch 9. That is textbook overfitting, correctly caught by early stopping. It also means the effective capacity vastly exceeds what the task requires — consistent with a shortcut being learned.

**Val AUC never meaningfully improves after epoch 1** (0.9942 → 0.9977 is +0.0035 over eight epochs, well inside the noise band of 900 samples). The `ReduceLROnPlateau` events and the eight epochs of patience bought essentially nothing. Whatever the model needed to learn, it learned in one epoch.

**All of the above is consistent with, and predicted by, the corpus-confound hypothesis in [§5](#5-current-model).** None of it proves that hypothesis. The four falsification tests do.

### What is missing from this history

No test-set curve (no test set exists). No calibration curve. No per-condition breakdown, though CCD ships `timing` and `weather` and it would have been ten lines. No confidence intervals. No run identifier. No seed for anything but the split. No record of the other ≥5 runs in the session. **No way to tell whether this history describes the model in `models/`** — and it provably does not describe the archived one.

---

## 11. Current Model Artifact

### What ships

`models/crash_model_weights.weights.h5` — 6,990,088 bytes, tracked in git, loaded at `crash_detection_enhanced.py:196–214` by name into a locally-reconstructed architecture.

| Property | Value | How established |
|---|---|---|
| Format | Keras 3 weights-only HDF5 | Group layout `layers/<name>/vars/<i>`, `layers/lstm*/cell/vars/<i>` |
| Datasets in file | 38 | `h5py` traversal |
| Non-optimizer parameters | **578,689** | Sum of array sizes excluding `optimizer/*` — **exactly matches `model.summary()`** |
| Includes optimizer state | **Yes** (`optimizer`, `adam` groups present) | Explains the 6.99 MB size for a 2.21 MB model |
| Produced by | Colab cell 12, `trained.save_weights(...)` | Cell 12 |
| Downloaded to this machine | 2026-04-20 11:28 | Filesystem mtime |
| Committed | `4afb0c1`, 2026-04-20 11:25 | git log |
| **Traceable to a recorded run** | **NO** | See below |

### The run-identity finding

Commit `c63e307` (the initial commit) contained `models/best_crash_model.keras`, 7,000,132 bytes, later deleted in `86c06b2` but recoverable from history. It is a ZIP containing `metadata.json`, `config.json` and `model.weights.h5`.

```
metadata.json: {"keras_version": "3.13.2", "date_saved": "2026-04-17@08:48:28"}

config.json layers:
  InputLayer  frame_sequence  batch_shape=[None, 10, 1280]
  Dense       feat_reduce     units=256  activation=relu
  Dropout     dropout         rate=0.3
  LSTM        lstm_1          units=128
  Dropout     dropout_1       rate=0.3
  LSTM        lstm_2          units=64
  Dropout     dropout_2       rate=0.3
  Dense       dense_1         units=64   activation=relu
  Dropout     dropout_3       rate=0.2
  Dense       output          units=1    activation=sigmoid

compile_config: Adam(lr=0.0010000000474974513, beta_1=0.9, beta_2=0.999, eps=1e-07),
                loss="binary_crossentropy", metrics=["accuracy", AUC(name="auc"), ...]
```

**This confirms the architecture and compile configuration match Colab cell 7 exactly**, including the custom layer names `feat_reduce`, `lstm_1`, `lstm_2`, `output`. The shipped model is unambiguously a product of this notebook's design.

**But it is not a product of the recorded run.** Comparing the embedded `model.weights.h5` against the deployed `crash_model_weights.weights.h5`:

```
datasets in archived .keras : 38
datasets in deployed weights: 38
keys identical: True
identical arrays: 1 | differing: 37

layers/dense/vars/0   shape=(1280, 256)   max|Δ| = 0.28665   corr = 0.0073
```

**A correlation of 0.0073 between the first Dense kernels means these are two independently initialised, independently trained models.** They are not the same run at different epochs; they are not the same weights re-saved. There were at least two completed training runs, and the notebook retains the outputs of only one.

### Consequences

1. **The metrics in [§10](#10-training-history) cannot be attributed to the shipped weights.** They may be similar; they are not known to be.
2. **`ModelCheckpoint` writing to a fixed path is the root cause.** Every run silently overwrites `best_crash_model.keras`. There is no run directory, no timestamp, no hash.
3. **Any claim you make about this model's performance is currently unsupportable**, not because the measurement was bad, but because the measurement and the artefact are not linked.
4. **The fix is trivial and belongs in Phase 1:** checkpoint to `runs/<timestamp>-<git-sha>/`, write `metrics.json` beside the weights, and record the SHA-256 of the artefact in the run log.

### Other artefacts

| File | Origin | Status |
|---|---|---|
| `models/feature_extractor_saved/` (19 MB, tracked) | **UNKNOWN — evidence not found** (U5). The notebook saves `feature_extractor.keras`; this is a TF SavedModel dated a month later. Converted locally, undocumented. | Regenerable; do not version |
| `models/feature_extractor.keras` (9.2 MB) | Colab cell 10 | Regenerable |
| `models/crash_model_cpu/` (4.5 MB) | Colab cell 11 `.export()`; signature `(None, 10, 1280) → (None, 1)` | Regenerable; comparing its weights against the shipped `.h5` would resolve U6 |
| `models/crash_model_saved/` (4.5 MB) | **UNKNOWN — evidence not found** | Orphaned |
| `models/crash_detection_model.h5` (296 MB) | **UNKNOWN — evidence not found** (U4). Cannot come from this notebook. | Orphaned; delete after verification |
| `models/crash_detection_model/` (197 MB) | **UNKNOWN — evidence not found** (U4) | Orphaned; delete after verification |

---

## 12. Evaluation History

There are **two** evaluation histories, and conflating them was the previous audit's second error.

### 12.1 Model-level evaluation — real, but invalid as reported

Colab cell 9, on the 900-clip validation split, at threshold **0.50**:

```
              precision    recall  f1-score   support
      Normal       0.99      0.97      0.98       600
       Crash       0.94      0.98      0.96       300
    accuracy                           0.97       900
   macro avg       0.97      0.98      0.97       900
weighted avg       0.97      0.97      0.97       900

Confusion matrix        Predicted
                     Normal   Crash
   True  Normal        581      19
         Crash           5     295
```

Derived quantities:

| Quantity | Value |
|---|---|
| True negatives / False positives | 581 / **19** |
| False negatives / True positives | 5 / 295 |
| **Clip-level false-positive rate** | 19 / 600 = **3.17%** |
| Recall (crash) | 295 / 300 = 98.3% |
| Precision (crash) at 33% prevalence | 295 / 314 = 94.0% |
| Best val AUC over 17 epochs | 0.9977 |

**Why this is not a valid performance estimate:**

1. **The split reported on is the split that selected the model.** Both `EarlyStopping` and `ModelCheckpoint` monitor `val_auc`; the reported figures come from the same 900 clips.
2. **The negatives are a different corpus from the positives.** See [§5](#5-current-model).
3. **The split is not source-grouped.** See [§9](#9-dataset-split).
4. **The prevalence is fictitious.** 33% positive versus roughly 1-in-50,000-km in deployment.
5. **The threshold reported is not the threshold deployed.** 0.50 here; 0.80 in production.

**The number that should terrify you is the 3.17%.** Projected onto continuous driving as non-overlapping 5-second windows — 720 windows per hour — a 3.17% clip-level FPR implies **≈23 false alarms per driving hour**. The shippable target in [§31](#31-evaluation-framework) is **< 0.1/hour**. That is a gap of more than two orders of magnitude, and it is measured on the *favourable* split. This single derived number, not the 97% accuracy, is the honest summary of where the model stands.

### 12.2 System-level evaluation — not an evaluation

The assembled pipeline (`crash_detection_enhanced.py`) has only ever been run against three files:

| File | Resolution | Role |
|---|---|---|
| `crash1.mov` | 3408×1910 | positive |
| `crash2.mov` | 3408×1910 | positive |
| `safe.mp4` | 3840×2160 | negative |

Reported previously: `safe.mp4` scored 0.79 against a threshold of 0.80 — a **0.01 margin on a sample size of one negative**. Crash-frame percentages of 21% / 14% / 0% against a gate of 1.5%. Both thresholds sit in the gaps between three data points. Additionally `--max-frames` defaults to 500, so at 60 fps only the first ~8 seconds of any clip is ever judged.

**This is threshold fitting on n=3, and it is where `CNN_THRESH = 0.80` came from.** The Colab never produced, suggested or validated 0.80.

### 12.3 Falsification results — measured 2026-09-10

| Test | Result | Verdict |
|---|---|---|
| T2 temporal shuffle | mean score changes by ≤0.0013 on all three videos | **FAILED — the LSTM is inert** |
| T1 single frame tiled ×10 | reproduces the deployed score within 0.02 | **FAILED — it is an image classifier** |
| T5 source leakage | 113/133 source videos split across train and val; **91.4% of clips implicated** | **FAILED — severe** |
| T5b official CCD split | 107/133 sources on both sides | **also not source-grouped** |
| T5c label quality | accident onset at frame 37.2/50; **72% of sampled positive frames are pre-accident** | labels are wrong for most frames |
| T6 always-negative | 0 FP/hour vs the model's ≈23 FP/hour | **the trivial baseline wins** |
| B1 stride | `safe.mp4` 0.0241 → 0.0003 at the training stride | real, but second-order given T2 |
| **T3 corpus control** | **Nexar test-public, 667 clips: ROC-AUC 0.5339, AP 0.5218. FPR 97.6% / TPR 99.4% at threshold 0.80** | **FAILED — chance-level; the decisive result** |

Raw output and method: [`runs/falsification/RESULTS.md`](runs/falsification/RESULTS.md), `scripts/t5_source_leakage.py`, `scripts/t124_model_falsification.py`.

### 12.4 What has never been measured, at either level

ROC or PR curves · average precision · false positives per hour on real driving · calibration / ECE · time-to-detection · performance by weather, lighting, road type or crash type · ego-involved versus non-ego · any comparison against BADAS-Open · any comparison against an always-negative baseline · any measurement on a source-grouped held-out set · any measurement of the *deployed* weights at all.

---

## 13. What Has Actually Been Proven

Stated conservatively. Each item is something you could defend in a diligence conversation today.

1. **A complete training pipeline exists, is documented, and runs.** Twelve Colab cells with saved outputs. (B)
2. **The dataset is CCD**, 1,500 crash + 3,000 normal, all 4,500 successfully converted to features with zero failures. (B, C)
3. **The preprocessing is internally consistent between training and inference** with respect to resize, colour order and normalisation — a common bug this project does not have. (A, B)
4. **The architecture is exactly as documented**: 578,689 trainable parameters, verified independently from the notebook's `model.summary()` and from the parameter count in the shipped weights file. (B, D)
5. **The backbone was frozen**, never fine-tuned. (B)
6. **One training run is fully recorded**: 17 epochs, early-stopped at epoch 9's weights, with per-epoch metrics. (B)
7. **On its own validation split, that model separates CCD positives from CCD negatives very well** — AUC 0.9977, F1 0.96 at threshold 0.5. (B)
8. **The split is seeded and therefore reproducible.** (B)
9. **The shipped weights implement the notebook's architecture.** (D)
10. **The shipped weights are from a different run than the archived checkpoint.** (D)
11. **YOLOv8n detects vehicles competently.** Unchanged from the previous audit. (A)
12. **The Kalman filter and TTC computation are mathematically correct implementations.** Unchanged. (A)
13. **The LSTM layers contribute nothing.** Temporal shuffling moves the output by ≤0.0013; a single tiled frame reproduces it within 0.02. Measured, not inferred. (Falsification run)
14. **The training split leaks by source at 91.4%.** 1,500 crash clips come from 133 YouTube videos; a random 80/20 clip split puts 113 of those 133 on both sides. Measured exactly from `Crash-1500.txt`. (Falsification run)
15. **72% of the frames shown for a positive clip contain no accident**, because the accident starts at frame 37.2 of 50 on average and every sampled frame was labelled 1. (Falsification run)
16. **The Nexar dataset permits commercial use** — "use, copy, modify, and distribute", with attribution, no resale of the dataset itself, and ethical-use restrictions. (`data/nexar/LICENSE`)
17. **On a corpus-controlled benchmark the shipped model performs at chance.** Nexar test-public, 667 clips: ROC-AUC **0.5339**, AP **0.5218**, FPR **97.6%** at the deployed threshold. The harness was validated by reproducing the T1/T2 local-video figures exactly through the same code path. **This is the only fair evaluation the model has ever had, and it is the evidence on which it can be retired.** (T3, 2026-09-11)

---

## 14. What Has NOT Been Proven

1. ~~That the model detects collisions.~~ ~~What remains unmeasured is how it performs on a corpus-controlled benchmark.~~ **Both settled, and negatively.** It is a per-frame appearance classifier (T1, T2), and on a corpus-controlled benchmark it performs **at chance — ROC-AUC 0.5339** (T3). Nothing about collision detection remains to be established in its favour.
2. ~~That the LSTM contributes anything.~~ **Settled: it does not** (T2, ≤0.0013 change under shuffling).
3. ~~That temporal information is used at all.~~ **Settled: it is not.**
4. **Any performance figure for the weights that actually ship.** No run is linked to that file.
5. **Generalisation beyond CCD.** No cross-dataset evaluation exists.
6. **Any false-positive rate on real driving.** The only derivable figure, ≈23/hour, is an extrapolation from a favourable split.
7. **That `CNN_THRESH = 0.80` is a sensible operating point.** No ROC curve was ever computed.
8. **That the output probability means anything.** Never calibrated; see [§18](#18-calibration-problems).
9. **That the split is free of source leakage.** Unquantified, and unquantifiable until `Crash-1500.txt` is downloaded.
10. **That the system works end-to-end on unseen footage.** Three videos, two positive.
11. **That any 3D/physics output is numerically valid.** Focal length is wrong by ~4× at the test resolutions (C1).
12. **That the pipeline is real-time.** ~8 fps at 4K on M1, against 30–60 fps sources.
13. **That the training data may be used commercially.** See [§23](#23-dataset-licensing--provenance).
14. **That the model outperforms an always-negative baseline on a realistic prevalence.** At 1-in-50,000-km prevalence, always-negative achieves near-perfect accuracy and zero false alarms; your model achieves ≈23 false alarms per hour. **On the metric that decides whether a fleet keeps the product, the trivial baseline currently wins.**

---

# Part II — What Is Broken

## 15. Technical Problems — Full Bug-Fix Plan

Every problem below carries: location, evidence, why it is wrong, severity, which of the five concerns it damages, the exact fix, how to verify the fix, the expected result, dependencies, and a priority. Bugs prefixed **B** and **R** are *new in this revision* (discovered in the Colab or by artefact inspection). Bugs prefixed **C**, **H**, **M**, **L**, **D** are carried forward from the 2026-09-07 audit and re-verified.

**Legend for the Affects line:** `train` = corrupts training · `infer` = corrupts inference · `eval` = corrupts measurement · `repro` = blocks reproducibility · `legal` = licensing or liability exposure.

---

### P0 — Must be fixed before any other work

#### B1 — Train/inference temporal stride mismatch
**Where:** training `Colab cell 5 line 21`; inference `code/crash_detection_enhanced.py:167, 220–226`.
**Evidence:** training uses `indices = np.linspace(0, total-1, FRAMES_PER_VIDEO, dtype=int)` over CCD clips of 50 frames at 10 fps → indices `0,5,10,16,21,27,32,38,43,49`, i.e. **≈0.54 s between steps spanning ≈4.9 s**. Inference uses `self.frame_buffer = deque(maxlen=10)` with `frame_buffer.append(frame)` on **every** frame → **0.033 s between steps spanning 0.33 s** at 30 fps, or 0.017 s / 0.17 s at 60 fps.
**Why it is wrong:** an LSTM models the *dynamics of the sequence it is shown*. Trained on half-second increments, it learned what half a second of change looks like. At inference it receives thirty-millisecond increments — an essentially static sequence by comparison. The temporal ratio is **≈16× at 30 fps and ≈33× at 60 fps**. Whatever the recurrent layers contribute in training is not merely degraded at runtime; it is out of distribution.
**Severity:** CRITICAL.
**Affects:** `infer` ✅ · `eval` ✅ (system-level results are meaningless) · `train` ❌ · `repro` ❌ · `legal` ❌
**Fix:** make the stride an explicit, shared constant. Add `FRAME_STRIDE` to a single config module imported by both the training script and the inference engine. Two acceptable resolutions, pick one deliberately:
&nbsp;&nbsp;**(a) Match inference to training (cheap, do this first).** Buffer at the source frame rate but sample every *n*-th frame into the model window, where `n = round(fps * 0.544)`. At 30 fps, `n = 16`; the model window then spans 4.9 s exactly as in training. Implement as a `deque(maxlen=10)` fed from a frame counter, not from every frame.
&nbsp;&nbsp;**(b) Match training to the deployment target (correct long-term).** Retrain with a stride matched to the operating frame rate and a window length chosen for the product (BADAS uses 16 frames over ~2 s). This is the Phase 6 path.
**Verify:** assert in code that `abs(train_stride_seconds - infer_stride_seconds) < 0.05`, and fail startup otherwise. Then re-run the three local videos and compare score trajectories before/after — they should change materially. If they do **not** change, that is itself a finding: it means the LSTM is contributing nothing (see the ablation in [§30](#30-recommended-training-strategy)).
**MEASURED 2026-09-10:** they move. `safe.mp4` falls from mean 0.0241 to **0.0003** at the training-matched stride; `crash2.mov` rises from 0.9640 to 0.9996. So the mismatch is real. **But the same run also showed the LSTM is order-invariant (B14), so this is a change in *which* frames get averaged, not a restoration of temporal reasoning.** Fix it for correctness; do not expect it to fix the model.
**Blocked by:** nothing. Fixable today.
**Priority: P0.**

#### B2 — No held-out test set; reported metrics come from the selection split
**Where:** `Colab cell 6` (split), `cell 8` (callbacks), `cell 9` (evaluation).
**Evidence:** `train_test_split` returns two partitions. `EarlyStopping(monitor='val_auc', restore_best_weights=True)` and `ModelCheckpoint(monitor='val_auc', save_best_only=True)` both select on partition 2. Cell 9 reports `classification_report` on partition 2.
**Why it is wrong:** the reported `val AUC 0.9977` is the **maximum over 17 noisy estimates** on 900 samples, taken from the split used to choose the epoch and the weights. It is a selection statistic, not a generalisation estimate. Every downstream claim inherits the optimism.
**Severity:** CRITICAL.
**Affects:** `eval` ✅ · `train` ✅ (model selection is part of training) · `infer` ❌ · `repro` ❌ · `legal` ❌
**Fix:** three partitions minimum, four preferred: **train / validation (model selection) / calibration (temperature scaling) / test (frozen, touched once)**. Freeze and commit the test manifest — a text file of clip IDs — before the first training run. Add a CI check that the test manifest's hash has not changed.
**Verify:** the training script must refuse to run if `test_manifest.sha256` differs from the committed value. Report test metrics separately from val metrics in `metrics.json`.
**Expected result after fix:** test AUC will be **lower** than 0.9977. Expect a drop; the size of the drop is information.
**Blocked by:** B3 (the split must be source-grouped when it is rebuilt — do both at once).
**Priority: P0.**

#### B3 — No source grouping; the official split and the grouping key were both ignored
**Where:** `Colab cell 6 lines 26–31`.
**Evidence:** `train_test_split(all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels)` operates on filenames. CCD ships `vgg16_features/train.txt` and `test.txt` (the official split) and `Crash-1500.txt` (which records `youtubeID` and `startframe` per clip). Neither file appears anywhere in the notebook, and neither is present on this machine.
**Why it is wrong:** CCD's crash clips are **windows cut from longer YouTube videos**. Two clips from the same source can land on opposite sides of a random split, letting the model recognise a source video rather than a collision. The negatives inherit the equivalent problem from BDD100K's per-drive structure. A random clip split is *better* than a random frame split, and still leaks. Separately, not using the official split means **your numbers are not comparable to any published CCD result**, which removes your only free external sanity check.
**Severity:** CRITICAL.
**Affects:** `eval` ✅ · `train` ✅ · `repro` ✅ (results not comparable) · `infer` ❌ · `legal` ❌
**Fix:** (1) download `Crash-1500.txt`, `train.txt`, `test.txt`. (2) Build a manifest: `clip_id, class, youtubeID, timing, weather, egoinvolve, split`. (3) Group by `youtubeID` for positives and by BDD100K source video for negatives, then split **groups**, not clips — `sklearn.model_selection.GroupShuffleSplit` or `StratifiedGroupKFold`. (4) Additionally report results on the **official** `train.txt`/`test.txt` split so you have a comparable number.
**Verify:** a test that fails the build if `set(train.youtubeID) & set(val.youtubeID)` is non-empty, likewise for test. Before fixing, **run the diagnostic and record the number** — count how many `youtubeID`s currently appear on both sides of the seed-42 split. That number belongs in the repo.
**Expected result after fix:** AUC drops again. Combined with B2, expect the honest test figure to be materially below 0.99. If it stays above 0.99 after both fixes, that is strong evidence for B4 and you should be *more* worried, not less.
**Blocked by:** downloading three small text files.
**Priority: P0.**

#### B4 — The positive/negative classes coincide with two different corpora
**Where:** dataset design, `Colab cell 3` + CCD composition.
**Evidence:** all 1,500 positives are YouTube-sourced crash clips (`Crash-1500`, each with a `youtubeID`); all 3,000 negatives are sampled from BDD100K (CCD README, verbatim: *"The 3,000 normal videos are randomly sampled from the BDD100K dataset"*). Cell 8: val AUC **0.9942 after epoch 1**.
**Why it is wrong:** the two classes differ systematically in encoding history, resolution, colour grading, exposure, camera hardware, geography and road furniture — none of which is about collisions. A classifier that learns "is this a re-encoded YouTube compilation clip or a BDD100K capture?" achieves near-perfect AUC while being useless. This is the single most likely explanation for the training curve in [§10](#10-training-history), and it is a property of the **dataset design**, not of the code.
**Severity:** CRITICAL — this is the finding that determines whether anything else matters.
**Affects:** `train` ✅ · `eval` ✅ · `infer` ✅ · `repro` ❌ · `legal` ❌
**Fix:** you cannot fix CCD; you can only measure the damage and change corpus. (1) Run the **corpus-control test**: evaluate on Nexar, where positives and negatives come from one corpus and one anonymisation pipeline. (2) Run the **single-frame test** and the **temporal shuffle test** to bound how much temporal information is used at all. (3) For any shipped model, train on a dataset whose classes are *not* corpus-aligned — Nexar is 50/50 from one driver community and is the right choice. (4) If you must keep CCD for pretraining, mix negatives from the same sources as positives (CCD gives you `startframe`, so non-accident windows can be cut from the *same* YouTube videos as the positives — this is the correct within-corpus negative and it costs you an afternoon).
**Verify:** the corpus-control test is the verification. Report Nexar AP and AUC beside the CCD figures in the same table, permanently.
**Expected result after fix:** a large drop on any corpus-controlled benchmark. Plan for it emotionally now; it is much cheaper to discover this week than after a pilot.
**Blocked by:** ~~Nexar dataset access and its licence question (D1)~~ — **licence resolved 2026-09-10, access confirmed (public, 31.4 GB, 2,844 clips).** Only the download remains.
**STATUS 2026-09-10:** T1/T2/T5 have run and failed. The corpus-control test is the last one outstanding, and it is now confirmatory rather than decisive.
**STATUS 2026-09-11 — CONFIRMED BY MEASUREMENT, AND THE BUG IS CLOSED AS "DIAGNOSED, NOT FIXABLE".** T3 ran on Nexar test-public: **ROC-AUC 0.5339, AP 0.5218, FPR 97.6%** at the deployed threshold. Removing the corpus confound removes essentially all of the model's apparent performance. This bug cannot be fixed on this model or this dataset — fix (3), *change corpus*, is the only remaining option, and fix (4), cutting within-corpus negatives from the same YouTube sources, is now moot because the model has no temporal or collision-specific signal to train against (B14). **Resolution: retire the model; retain CCD as a research corpus only.**
**Priority: P0 — CLOSED (diagnosed).**

#### B5 — The deployed threshold has no derivation
**Where:** `code/crash_detection_enhanced.py:109` (`CNN_THRESH = 0.80`), `:1050`, `:1186`, `:1273` (`crash_pct >= 1.5`), `:1275` (`crash_pct >= 5.0`), `code/crash_detection.py:122`.
**Evidence:** the Colab evaluates at **0.50** (cell 9) and its generated inference snippet suggests **0.55** (cell 10). Neither 0.80, nor 1.5%, nor 5.0% appears anywhere in the training pipeline. The prior README documents that `safe.mp4` scored 0.79 against a 0.80 gate — a 0.01 margin against a single negative.
**Why it is wrong:** an operating point must come from a curve on held-out data at a stated target (e.g. "the threshold achieving 80% recall"), not from the gap between three observations. Three mutually inconsistent thresholds across three artefacts is itself the tell.
**Severity:** CRITICAL.
**Affects:** `infer` ✅ · `eval` ✅ · `repro` ✅ · `train` ❌ · `legal` ❌
**Fix:** delete the constant. Compute the operating point in the evaluation harness from the frozen test split: fit the threshold that achieves the target recall, record it in `metrics.json` alongside the resulting precision and FP/hour, and load it from that file at inference time. Never hand-edit it. Apply the same treatment to `crash_pct`, which is a second, undocumented, unfitted threshold sitting on top of the first.
**Verify:** grep the codebase for numeric literals compared against a model score; there should be exactly zero. The inference engine should fail to start if `metrics.json` is missing.
**Expected result after fix:** a threshold that is defensible in one sentence, and a system-level FP/hour number you can quote.
**Blocked by:** B2, B3 (you need a real test split to fit against).
**Priority: P0.**

#### B14 — The LSTM layers are inert *(measured, not inferred)*
**Where:** `Colab cell 7` (architecture); the deployed head in `code/crash_detection_enhanced.py:185–192`.
**Evidence:** falsification run 2026-09-10. Randomly permuting the ten frames within the model's input window changes the mean output by **≤0.0013** across all three local videos (0.9997→0.9998, 0.9640→0.9649, 0.0241→0.0228). Replacing the sequence with a **single frame tiled ten times** reproduces the deployed score to within 0.02 (0.9997→0.9798, 0.9640→0.9460).
**Why it is wrong:** `lstm_1` (128 units) and `lstm_2` (64 units) account for **246,528 of the 578,689 parameters — 43% of the model** — and are provably order-invariant in practice. Every claim the project has made about "learning what a crash looks like over time" is false. It also means B1 (stride) and any future temporal-window tuning cannot help: there is no temporal function to tune.
**Severity:** CRITICAL — this is the finding that settles [§5](#5-current-model).
**Affects:** `train` ✅ · `infer` ✅ · `eval` ✅ · `repro` ❌ · `legal` ❌
**Fix:** do not repair it. A frozen ImageNet backbone with `GlobalAveragePooling2D` discards spatial layout, so relative motion between two vehicles is not representable in the 1,280-d input the LSTM receives — no recurrent layer on top can recover it. Replace the representation, per [§28](#28-recommended-future-architecture). Retain the *pattern* (frozen backbone, cached features), not the backbone.
**Verify:** the shuffle test is the verification. Keep it as a permanent regression test: any future model must show a **material** score change under temporal shuffling, or it is not using time.
**Expected result after fix:** a model whose output actually depends on frame order — measurable as a large shuffle-induced score drop.
**Blocked by:** nothing to measure; [§28](#28-recommended-future-architecture) to replace.
**Priority: P0 (as a decision input; the "fix" is replacement, not repair).**

#### B10 — Training code is not in version control
**Where:** the whole repository. `git log --all -p` contains no notebook, no training script, no dataset manifest.
**Evidence:** the pipeline exists only as `Untitled0.ipynb` in one Google Drive account, under a default name, with no revision discipline.
**Why it is wrong:** this is the root cause of the previous audit's most serious error, and of B8 and U1/U2. A single account deletion or an accidental "Run all" destroys or overwrites the only record of how the model was made. It also means no reviewer, collaborator or acquirer can see the pipeline.
**Severity:** CRITICAL.
**Affects:** `repro` ✅ · `eval` ✅ · `legal` ✅ (diligence) · `train` ❌ · `infer` ❌
**Fix:** **today, 30 seconds:** `File → Download .ipynb` in Colab, then `git add train/colab_original.ipynb && git commit`. Commit it *verbatim, outputs included* — the outputs are the training history and they are evidence. **Then, Phase 1:** port it to `train/` as plain Python modules (`data.py`, `features.py`, `model.py`, `train.py`, `evaluate.py`) with a config file, so it is diffable and testable.
**Verify:** `git log -- train/` is non-empty; a fresh clone contains the pipeline.
**Expected result after fix:** the class-A/class-B evidence gap closes permanently.
**Blocked by:** nothing. **This is the single cheapest high-value action in this document.**
**Priority: P0.**

#### R1 — `requirements.txt` specifies an impossible environment
**Where:** `requirements.txt:1`.
**Evidence:** pins `tensorflow==2.13.1`, which supports Python 3.8–3.11. The `venv/` in the working tree is **Python 3.14.2 with only `pip` installed**. On Apple Silicon `tensorflow-macos` would additionally be required and is not mentioned. `timm` (needed by MiDaS via `torch.hub`), `torchvision` and `matplotlib` are all missing.
**Why it is wrong:** a repository a collaborator or investor cannot run is a repository that does not exist.
**Severity:** CRITICAL.
**Affects:** `repro` ✅ · `infer` ✅ · `train` ❌ · `eval` ❌ · `legal` ❌
**Fix:** pin Python 3.11 in `.python-version`; generate a real lockfile (`uv lock` or `pip-compile`); add a `Dockerfile`; add `make run`. Include every transitive requirement.
**Verify:** `docker build . && docker run … --video crash1 --no-display --max-frames 50` succeeds from a **fresh clone in a clean container**. Add this to CI.
**Expected result after fix:** one command, clean clone, working system.
**Blocked by:** nothing.
**Priority: P0.**

#### R2 — The pinned TensorFlow cannot read the shipped model
**Where:** `requirements.txt:1` versus `models/crash_model_weights.weights.h5`.
**Evidence:** the model was produced by **TensorFlow 2.19.0 / Keras 3.13.2** (Colab cell 1 output; `metadata.json` in the archived `.keras`). The weights file uses the Keras 3 group layout `layers/<name>/vars/<i>`, which `crash_detection_enhanced.py:196–214` reads directly by that path. TensorFlow 2.13 ships **Keras 2**, which uses a different layout entirely.
**Why it is wrong:** even if R1 were fixed by making the pin installable, the pinned stack **still could not load the shipped weights**. The pin is not merely stale, it is contradicted by the artefact it is supposed to serve. This is a new finding and it explains why the environment question is not just hygiene.
**Severity:** CRITICAL.
**Affects:** `repro` ✅ · `infer` ✅ · `train` ❌ · `eval` ❌ · `legal` ❌
**Fix:** pin the stack that actually produced the model — `tensorflow==2.19.*` (or `tensorflow-macos`/`tensorflow-metal` equivalents on Apple Silicon) with `keras>=3.13`. Record the training-time versions in `metrics.json` and assert them at load time.
**Verify:** a smoke test that loads the weights and runs one forward pass on zeros, asserting output shape `(1, 1)`. This is a five-line test and the repository currently has nothing like it.
**Expected result after fix:** the environment can load the model it ships.
**Blocked by:** R1.
**Priority: P0.**

#### C6 — `ultralytics` (YOLOv8) is AGPL-3.0
**Where:** `requirements.txt:3`; imported in all four forks (`enhanced.py:39` et al.).
**Evidence:** Ultralytics distributes YOLOv8 under AGPL-3.0 and sells an Enterprise Licence by quotation.
**Why it is wrong:** AGPL's network clause obliges a hosted service built on it to offer its complete corresponding source to users. This is incompatible with a commercial SaaS incident product.
**Severity:** CRITICAL (legal).
**Affects:** `legal` ✅ · `infer` ✅ (if the detector must be swapped) · `train` ❌ · `eval` ❌ · `repro` ❌
**Fix:** choose one this quarter — (a) buy the Ultralytics Enterprise Licence; (b) migrate to **RT-DETR / RT-DETRv2** (Apache-2.0) as the closest drop-in, with YOLOX (Apache-2.0) or torchvision detectors as alternatives; (c) drop object detection from the shipped path entirely, which the recommended architecture in [§28](#28-recommended-future-architecture) makes viable.
> ⚠️ **Trap:** take RT-DETR from the original authors' repository (`lyuwenyu/RT-DETR`, Apache-2.0), **not** from the `ultralytics` package. Ultralytics ships its own RT-DETR wrapper under AGPL-3.0, so importing it via `ultralytics` reintroduces exactly the obligation you are escaping. Verify the licence of the specific repository and checkpoint, not of the architecture's name.
**Verify:** `pip-licenses` in CI with an allowlist; the build fails on any (A)GPL dependency.
**Expected result after fix:** a licence chain explainable in one slide.
**Blocked by:** nothing (both options can be started in parallel).
**Priority: P0.**

#### C7 — Fault attribution is unsupportable and legally hazardous
**Where:** `code/crash_detection_enhanced.py:622–725` (`FaultDetector`), `:761–785` (`EgoZone.fault`).
**Evidence:** `EgoZone.fault()` derives `"from the left" / "from the right" / "head-on"` purely from the offending box's x-centre relative to the frame centre, with a 12%-of-width margin, and emits `"V3 struck your car from the left (42 km/h)"` — where the speed comes from the broken projection of C1.
**Why it is wrong:** liability attribution from uncalibrated monocular video is not defensible. Shipping it to an insurer or fleet invites a negligence claim the first time a determination is wrong.
**Severity:** CRITICAL (legal + product).
**Affects:** `legal` ✅ · `infer` ✅ · `train` ❌ · `eval` ❌ · `repro` ❌
**Fix:** delete both. Output *evidence* — clip, timestamps, tracks, measured closing speed with an uncertainty band — and let a human adjudicate.
**Verify:** grep for `at_fault`, `collision_type`, `FaultDetector`; zero hits.
**Expected result after fix:** a smaller codebase and a defensible product claim. See [§27](#27-product-definition).
**Blocked by:** nothing.
**Priority: P0.**

#### C3 — Dashcam mode is a proximity alarm, not a crash detector
**Where:** `code/crash_detection_enhanced.py:732–760` (`EgoZone`), `:1040–1046`.
**Evidence:** `is_crash = len(ego_hits) > 0`, where `ego_hits` is any vehicle box overlapping a fixed rectangle at 96% frame height and 20% frame width (clamped to 250×80 px). `EgoZone.overlaps` is pure axis-aligned box intersection.
**Why it is wrong:** this fires for a car changing into your lane, a vehicle stopped at a red light, a bus pulling away, and any close-following in traffic. The prior claim that webcam mode produced "no false triggers" is not evidence — indoors there are no vehicles to trigger it.
**Severity:** CRITICAL.
**Affects:** `infer` ✅ · `eval` ✅ · `train` ❌ · `repro` ❌ · `legal` ❌
**Fix:** delete the ego zone entirely. Ego-involvement is a **learned** property — this is precisely BADAS's contribution — or a calibrated-geometry property. It is never a fixed screen rectangle.
**Verify:** `--dashcam` either routes to the learned ego head or is removed from the CLI.
**Expected result after fix:** one fewer false-positive generator, and one fewer claim you cannot defend.
**Blocked by:** nothing to delete it; [§28](#28-recommended-future-architecture) to replace it.
**Priority: P0.**

#### D1 — The training data is not licensed for commercial use
**Where:** dataset choice; see [§23](#23-dataset-licensing--provenance) for the full treatment.
**Evidence:** CCD negatives are BDD100K (CCD README); BDD100K's basic licence is limited to personal use. CCD positives are YouTube-derived (`youtubeID` field), so the repository's MIT label covers the researchers' annotations, not the video pixels.
**Why it is wrong:** every model trained on this corpus carries an unresolved provenance question into diligence — and unlike most such questions, this one has a documented paper trail pointing at the answer.
**Severity:** CRITICAL (legal).
**Affects:** `legal` ✅ · `train` ✅ (choice of corpus) · `infer` ❌ · `eval` ❌ · `repro` ❌
**Fix:** treat CCD as research-only. For any shipped model, retrain on a corpus with clean provenance — Nexar first-party consented footage plus your own UK collection. Maintain a `data/manifest.csv` mapping every clip to `source, licence, consent_status, split`, and make the training script **refuse to train on any clip without a licence entry**.
**Verify:** the manifest check runs in CI; `train.py` exits non-zero on an unlicensed clip.
**Expected result after fix:** you can answer "what is your training data licensed under?" in one sentence, with a file to point at.
**Blocked by:** ~~the Nexar licence clarification~~ — **RESOLVED 2026-09-10.** Nexar's licence permits commercial use (attribution, no resale of the dataset, ethical-use restrictions). The replacement corpus is available and legally clear.
**Priority: P0.**

---

### P1 — Fix during Phases 1–2

#### B8 — `ModelCheckpoint` writes to a fixed path; runs silently overwrite each other
**Where:** `Colab cell 8 line 14` — `filepath=f'{MODELS_DIR}/best_crash_model.keras'`.
**Evidence:** the deployed weights differ from the archived checkpoint in 37 of 38 arrays (r = 0.0073 on the first Dense kernel) — two runs, one path, no record of which is which. `Model: "functional_5"` shows ≥6 builds in one session.
**Why it is wrong:** it makes the artefact untraceable, which is the direct cause of U1 and U2 and of the fact that no metric in this document can be attributed to the file that ships.
**Severity:** HIGH.
**Affects:** `repro` ✅ · `eval` ✅ · `train` ❌ · `infer` ❌ · `legal` ✅ (diligence)
**Fix:** checkpoint into `runs/{UTC timestamp}-{git short sha}/`. Write `metrics.json`, `config.json`, `manifest.sha256` and `weights.sha256` into the same directory. Never write to a path that already exists.
**Verify:** two consecutive runs produce two directories; the inference engine records the run id it loaded and prints it at startup.
**Expected result after fix:** every model file is attributable to a run, a commit and a dataset manifest.
**Blocked by:** B10.
**Priority: P1.**

#### B9 — Nothing is seeded except the split
**Where:** `Colab cell 6` (`np.random.shuffle`, `np.random.normal`), cell 7 (weight init).
**Evidence:** `RANDOM_SEED = 42` is passed only to `train_test_split`. There is no `tf.random.set_seed`, no `keras.utils.set_random_seed`, no `np.random.seed`.
**Why it is wrong:** weight initialisation, batch shuffling and the augmentation noise are all nondeterministic. Two runs of identical code produce different models — which is exactly what B8's evidence shows happened.
**Severity:** HIGH.
**Affects:** `repro` ✅ · `train` ✅ · `eval` ❌ · `infer` ❌ · `legal` ❌
**Fix:** `keras.utils.set_random_seed(SEED)` at the top of the training script, plus `tf.config.experimental.enable_op_determinism()` where the performance cost is acceptable. Record the seed in `metrics.json`.
**Verify:** run training twice with the same seed on the same manifest; assert the final test AUC matches to 4 decimal places (or document the residual GPU nondeterminism if op-determinism is disabled).
**Expected result after fix:** run-to-run reproducibility, and B8's ambiguity becomes impossible in future.
**Blocked by:** B10.
**Priority: P1.**

#### C1 — Focal length is not rescaled with resolution; all metric output is fictional
**Where:** `code/crash_detection_enhanced.py:68–71` (`FX = FY = 460`, commented "OV5647 at 640×480"), `:128–132` (`set_resolution` updates only `CX_IMG`/`CY_IMG`), `:381` (`pixel_to_ground`), `:984`.
**Evidence:** `set_resolution()` moves the principal point to the native frame centre and leaves focal length at its 640×480 value. On a 3408×1910 video the true focal length for the same lens would be ≈1,830 px; the code uses 460. Numerically:

| Source resolution | y = 55% height | y = 70% | y = 85% | y = 95% |
|---|---:|---:|---:|---:|
| 640×480 (calibration res) | 14.33 m | 5.09 m | 3.08 m | 2.44 m |
| 3408×1910 (`crash1/2.mov`) | 5.12 m | **1.40 m** | **0.80 m** | **0.61 m** |
| 3840×2160 (`safe.mp4`) | 4.60 m | **1.24 m** | **0.70 m** | **0.54 m** |

**Why it is wrong:** `DIST_CONTACT = 1.5 m`. On the actual test videos, **any vehicle below ~65% of frame height is inside "contact range" by construction.** The headline measurements in the original README ("Closest = 0.83 m", "Closest = 0.34 m") are projection artefacts. Every downstream number — speed in km/h, TTC in seconds, BEV positions — inherits the error.
**Severity:** CRITICAL (as a defect), P1 (as work, because C2 means nothing currently consumes the output).
**Affects:** `infer` ✅ · `eval` ✅ · `train` ❌ · `repro` ❌ · `legal` ✅ (fault claims built on it)
**Fix:** calibrate per camera (`cv2.calibrateCamera` on a checkerboard); scale intrinsics on resize: `fx' = fx * (w' / w_calib)`; store intrinsics **per camera**, never as a global constant.
**Verify:** add an assertion that `fy / frame_height` falls in ≈0.7–1.5 (typical dashcam FOV) and refuse to emit metric output otherwise. Validate against two objects at known measured distances.
**Expected result after fix:** metric outputs that are within a stated error band, making Channel B in [§28](#28-recommended-future-architecture) viable.
**Blocked by:** H5 (do it once, in the single surviving pipeline).
**Priority: P1.**

#### C2 — The entire physics stack is excluded from the decision
**Where:** `code/crash_detection_enhanced.py:1049–1056`, `:1185–1190`; `depth_map` at `:1022–1026` used only at `:883`.
**Evidence:**
```python
if cnn >= 0:
    if (cnn >= Config.CNN_THRESH) and (len(vehs) >= 2):
        neural_confirmed = True
        is_crash = True
    else:
        is_crash = False
else:
    is_crash = rule['is_sustained'] and (len(vehs) >= 2)   # only during buffer warm-up
```
`rule` — the output of the Kalman filter, 3-D distance and TTC — is consulted **only in the first 9 frames**, before the CNN buffer fills. `depth_map` reaches the renderer at `:1089` and nowhere else. `bev_img` is display-only.
**Why it is wrong:** ~30 ms/frame of MiDaS plus the full projection/Kalman/TTC path is spent producing numbers that are then discarded. It also makes the architecture diagram in every prior README a description of something that does not happen.
**Severity:** CRITICAL (as a design defect).
**Affects:** `infer` ✅ (latency) · `eval` ✅ (the system is not what it is described as) · `train` ❌ · `repro` ❌ · `legal` ❌
**Fix:** decide deliberately. Either (a) delete the physics stack and be honest that this is a learned classifier, or (b) fix C1 and fuse the signals properly per [§28](#28-recommended-future-architecture). **Do not keep computing it for display.**
**Verify:** ablation test — delete the physics path and assert that verdicts on a fixed clip set are byte-identical. If they are (and they will be), the deletion is proven safe.
**Expected result after fix:** ~30 ms/frame recovered, ~800 lines removed, and an architecture diagram that is true.
**Blocked by:** nothing.
**Priority: P1.**

#### H3 — Silent exception swallowing masks total failure
**Where:** `code/crash_detection_enhanced.py:266–267` (`except Exception: return []` inside the detector), `:1105–1106` (a broad handler around the whole loop, after which a verdict is still printed on truncated data).
**Evidence:** a crashed detector returns an empty detection list, which is indistinguishable from an empty road.
**Why it is wrong:** the system can report "NO CRASH" for a run in which perception never worked. For a safety-adjacent product this is the most dangerous class of bug, because it fails silently and confidently.
**Severity:** HIGH.
**Affects:** `infer` ✅ · `eval` ✅ · `train` ❌ · `repro` ❌ · `legal` ✅
**Fix:** catch narrowly, log structured errors with counts, and **never emit a verdict from a run that terminated abnormally**. Count and report dropped frames and detector failures in the run summary.
**Verify:** inject a detector exception in a test and assert the process exits non-zero with no verdict printed.
**Expected result after fix:** failures are visible.
**Blocked by:** nothing.
**Priority: P1.**

---
### P2 — Fix during Phases 3–5

#### B7 — Rich CCD annotations were discarded; the model is clip-level only
**Where:** `Colab cells 5–6`; `Crash-1500.txt` never opened.
**Evidence:** CCD provides per-frame `binlabels` (50 flags), `egoinvolve`, `timing` (Day/Night) and `weather` (Normal/Snowy/Rainy). Cell 6 derives its single label from a filename prefix.
**Why it is wrong:** three product-critical capabilities were available for free and were dropped. **Temporal localisation** (`binlabels`) is what turns a score into an *event* with `t_start`/`t_peak`/`t_end` — the core of the incident record in [§27](#27-product-definition). **Ego-involvement** (`egoinvolve`) is exactly what the ego-zone hack (C3) was trying to approximate with a rectangle. **Weather and lighting** are exactly the per-condition breakdown that [§31](#31-evaluation-framework) requires and that no current report contains. It also means every crash clip trains the model to call the ~4 seconds *before* impact "crash".
**Severity:** HIGH.
**Affects:** `train` ✅ · `eval` ✅ · `infer` ✅ (no event boundaries) · `repro` ❌ · `legal` ❌
**Fix:** parse `Crash-1500.txt` into the manifest. Add an ego-involvement head. Report metrics stratified by `timing` and `weather`. Use `binlabels` either for event-boundary supervision or, minimally, to centre the sampled window on the annotated accident frames.
**Verify:** `metrics.json` contains per-condition rows; the manifest has non-null `egoinvolve` for all 1,500 positives.
**Expected result after fix:** a model that can say *when*, not just *whether* — and per-condition numbers you can show a fleet manager.
**Blocked by:** downloading `Crash-1500.txt` (Phase 0).
**Priority: P2.**

#### H1 — MiDaS output is inverse depth, used as if it were depth
**Where:** `code/depth_estimator.py:124–130`.
**Evidence:** MiDaS emits *disparity* (larger = nearer). The code computes `scale = median(analytical / midas)` then `metric_depth = midas_relative * scale`, which maps near objects to "far". The correct relation is `depth ≈ scale / (disparity + shift)`.
**Why it is wrong:** the prior README's claim that depth is "calibrated to meters using ground-plane anchoring" is false. Consequence is currently limited only because the output feeds nothing but a heatmap (C2).
**Severity:** HIGH (as a defect); low urgency because the recommendation is to delete MiDaS entirely.
**Affects:** `infer` ✅ · `eval` ❌ · `train` ❌ · `repro` ❌ · `legal` ❌
**Fix:** delete `depth_estimator.py`. If depth is ever genuinely needed, use a metric-depth model and invert correctly, with a shift term.
**Verify:** the file is gone and no verdict changes.
**Priority: P2 (as deletion).**

#### H2 — The tracker destroys all identities on a single empty frame
**Where:** `code/crash_detection_enhanced.py:334–337` — `if not dets: self.objects = {}`.
**Evidence:** one missed detection (occlusion, motion blur — i.e. exactly what happens during an impact) renumbers every vehicle. The prior claim that "Track TTL keeps lost vehicles alive for 2 seconds" is false: `TRACK_TTL_SEC` governs only the Kalman dictionary in `SpeedEstimator`, not track identity.
**Why it is wrong:** identity churn destroys per-track velocity and therefore every physical signal, precisely at the moment the physical signal matters most.
**Severity:** HIGH.
**Affects:** `infer` ✅ · `eval` ✅ · `train` ❌ · `repro` ❌ · `legal` ❌
**Fix:** replace with **ByteTrack** or **OC-SORT** (both permissively licensed) — age/TTL, Hungarian assignment, motion-model gating.
**Verify:** on a clip with a deliberate 3-frame detection gap, assert track IDs survive.
**Priority: P2.**

#### H4 — Resolution-dependent constants treated as absolute
**Where:** `MIN_BOX_AREA = 2000`, `TRACK_MAX_DIST = 80`, `EGO_ZONE_MAX_W/H`, `ROI_SKY_CUT`, `ROI_HOOD_CUT` — all in pixels, applied to frames from 640×480 to 3840×2160.
**Why it is wrong:** a 2,000 px² minimum box is a large vehicle at VGA and a distant speck at 4K. Behaviour changes silently with input resolution.
**Severity:** HIGH.
**Affects:** `infer` ✅ · `eval` ✅ · `train` ❌ · `repro` ✅ · `legal` ❌
**Fix:** express every spatial threshold as a fraction of frame dimension, or in metric units after C1 is fixed.
**Verify:** run the same clip at 480p, 720p and native; assert the verdict is stable.
**Priority: P2.**

#### H5 — Four divergent forks of the same program
**Where:** `code/crash_detection_enhanced.py` (1,334), `code/crash_detection.py` (1,032), `code/crash_detection_linux.py` (980), `camera_detect.py` (552) — 3,898 lines total.

| Pair | Shared non-comment lines | % of smaller file |
|---|---:|---:|
| `enhanced` ↔ `linux` | 200 | 30% |
| `enhanced` ↔ `camera_detect` | 128 | 32% |
| `crash_detection` ↔ `linux` | 159 | 24% |

**Why it is wrong:** each fork has its own `Config`, `Tracker` and `FaultDetector`, and its own drift. `crash_detection.py` resizes to 640×480; `enhanced.py` does not. `camera_detect.py` still uses the deprecated pixels-per-metre model. Comments claim "synced with enhanced.py SSOT" — there is no single source of truth; there are four. This is why a bug like C1 can exist in one fork and not another and go unnoticed.
**Severity:** HIGH.
**Affects:** `repro` ✅ · `infer` ✅ · `eval` ✅ · `train` ❌ · `legal` ❌
**Fix:** one package, one `Config`, one pipeline; platform differences behind small adapters. Delete the other three.
**Verify:** `git rm` the three forks; the test suite and the demo still pass.
**Priority: P2.**

#### H6 — No persistence, no API, no service boundary
**Where:** `code/crash_detection_enhanced.py:1252` — results go to stdout; optional JPEGs to `crash_outputs/`.
**Why it is wrong:** there is nothing to integrate with, nothing to review, and no way to accumulate the labelled events that constitute the moat in [§40](#40-technical-moat).
**Severity:** HIGH.
**Affects:** `infer` ✅ · `eval` ✅ · `train` ✅ (no feedback loop) · `repro` ❌ · `legal` ❌
**Fix:** the incident schema, Postgres + object store, ingest API and minimal review UI of Phase 9. See [§32](#32-real-time-architecture).
**Priority: P2.**

#### M3 — `--max-frames` defaults to 500 and silently truncates
**Where:** `code/crash_detection_enhanced.py:1315`.
**Why it is wrong:** at 60 fps that is 8.3 seconds. Every "verdict" on a long clip is a verdict on its opening seconds, unannounced.
**Severity:** MEDIUM.
**Affects:** `eval` ✅ · `infer` ✅ · `train` ❌ · `repro` ✅ · `legal` ❌
**Fix:** default to the whole clip; log loudly when truncation is requested.
**Verify:** processed-frame count equals the clip's frame count by default.
**Priority: P2.**

#### M4 — `--test` asserts nothing; there are no tests at all
**Where:** `code/crash_detection.py:929` prints `SELF-TEST PASSED ✅` unconditionally.
**Why it is wrong:** it cannot fail, so it certifies nothing while creating the impression of coverage. The repository contains zero real tests.
**Severity:** MEDIUM.
**Affects:** `repro` ✅ · `eval` ✅ · `infer` ✅ · `train` ✅ · `legal` ❌
**Fix:** delete it. Add real tests: weight-load smoke test (R2), split-leakage test (B3), stride-consistency assertion (B1), a golden-output test on a 50-frame clip.
**Verify:** `pytest` runs in CI and fails when any of the above is broken.
**Priority: P2.**

---

### P3 — Cleanup

#### B6 — The only augmentation is a no-op
**Where:** `Colab cell 6 lines 54–55` — `feat += np.random.normal(0, 0.01, feat.shape)`.
**Why it is wrong:** MobileNetV2 global-average-pooled activations have typical magnitudes of order 0.1–5. Additive noise at σ = 0.01 is far below the signal and contributes essentially nothing. Worse, augmenting in *feature* space cannot simulate any of the transformations that actually matter — weather, lighting, motion blur, camera shake, horizontal flip — because the frozen backbone has already discarded that structure. Since features are precomputed once, pixel-space augmentation is architecturally impossible in this pipeline.
**Severity:** MEDIUM (a missed opportunity rather than a defect).
**Affects:** `train` ✅ · `eval` ❌ · `infer` ❌ · `repro` ❌ · `legal` ❌
**Fix:** move augmentation to pixel space. Either extract features on the fly (slower per epoch, but the T4 handles it) or precompute *k* augmented feature sets per clip. Prioritise the augmentations that target your known failure modes: camera shake / ego-motion, weather and lighting simulation, motion blur, horizontal flip (which also helps bridge right-hand to left-hand traffic).
**Verify:** train with and without; compare test AUC and, more importantly, hard-negative FP rate.
**Priority: P3** (rises to P1 once a real model is being trained in Phase 6).

#### B11 — Square resize destroys aspect ratio
**Where:** `Colab cell 5 line 30` and `enhanced.py:220` — `cv2.resize(frame, (112, 112))`.
**Why it is wrong:** a 16:9 frame squashed to 1:1 distorts every shape. It is applied **consistently** in training and inference, so it is not a mismatch bug — but it discards geometry for no benefit and will interact badly with any future model that reasons about shape.
**Severity:** LOW.
**Affects:** `train` ✅ · `infer` ✅ · `eval` ❌ · `repro` ❌ · `legal` ❌
**Fix:** letterbox (resize preserving aspect, pad to square) in both paths, behind one shared function.
**Verify:** one preprocessing function, imported by both; a test asserting identical output for identical input.
**Priority: P3.**

#### B12 — Random frame seeking is codec-dependent
**Where:** `Colab cell 5 lines 25–26` — `cap.set(cv2.CAP_PROP_POS_FRAMES, idx)` then `cap.read()`.
**Why it is wrong:** random seeking is unreliable on long-GOP encodings; OpenCV may return the nearest keyframe rather than the requested frame. It happened to work here (0 skipped across 4,500 videos, because CCD clips are 50 frames), but it will not hold on 40-second UK footage.
**Severity:** LOW now, HIGH for the UK corpus.
**Affects:** `train` ✅ · `repro` ✅ · `eval` ❌ · `infer` ❌ · `legal` ❌
**Fix:** decode sequentially and keep the frames you want, or use PyAV/decord for reliable indexed access.
**Verify:** assert the returned frame index matches the requested index.
**Priority: P3** (P1 before Phase 7).

#### B13 — 493 MB of orphaned model artefacts
**Where:** `models/crash_detection_model.h5` (296 MB), `models/crash_detection_model/` (197 MB), `models/crash_model_saved/`.
**Evidence:** the notebook's largest artefact is ~7 MB; these cannot originate from it. `FEATURES_DIR = features_v2` and cell 5's comment about the "old approach" indicate a prior, undocumented v1 pipeline (U4).
**Fix:** grep for any code path that loads them (there is none in `enhanced.py`); then delete. They are already gitignored.
**Verify:** the system runs unchanged after deletion.
**Priority: P3.**

#### M1 — BEV double-converts speed
**Where:** `code/bev_renderer.py:86–94`. Reads `v['speed']`, which `enhanced.py:524` already set in **km/h**, then multiplies by 3.6 again. A 40 km/h vehicle is labelled 144 km/h. Line 88's `if speed_ms > 0.5` also compares km/h against an m/s threshold.
**Why it is wrong:** visible in any demo, and it is the kind of error a technical evaluator notices in the first thirty seconds.
**Fix:** delete the BEV renderer as a pipeline component; if kept as demo eye-candy, fix the conversion first. **Priority: P3.**

#### M2 — Dead configuration
`MIN_VEHICLE_Z`, `CONF_CONTINUE`, `TTC_WARN`, `SPEED_SMOOTH_FRAMES`, `PIXELS_PER_METER` are declared and never used in the decision path. They mislead readers into believing guards exist. **Fix:** delete. **Priority: P3.**

#### M5 — `torch.load` monkey-patched to `weights_only=False`
**Where:** `code/crash_detection_enhanced.py:244–250`. Correct for your own weights; a remote-code-execution vector if a checkpoint path ever becomes user-supplied. **Fix:** scope the patch to the specific MiDaS load, or remove it with MiDaS. **Priority: P3.**

#### M6 — The CNN and the detector see different frames
**Where:** YOLO receives `roi_frame` (ROI-masked); the CNN receives the raw `frame` (`:1029`). Defensible, but undocumented and apparently unintentional. **Fix:** document the decision explicitly, or unify. **Priority: P3.**

---

### P4 — Hygiene

**L1** — `.DS_Store` committed under `code/` and `models/`. Add to `.gitignore` and `git rm --cached`.
**L2** — Emoji-based logging; no `logging` module, no levels, no structured output. Blocks any production observability.
**L3** — Version numbers in docstrings disagree (`v12.0`, `v13.0`, `v16.0`, `v17.0` across four files) — a symptom of H5.
**L4** — The README has claimed MIT while no `LICENSE` file exists. Add one, or make the repository explicitly private.
**L5** — The Colab notebook is named `Untitled0.ipynb`. Rename it, and commit it (B10).

---

## 16. Data Leakage Risks

Ranked by certainty.

| # | Leakage type | Status | Mechanism | Impact on reported metrics | Fix |
|---|---|---|---|---|---|
| **1** | **Model-selection leakage** | **PROVEN** | `EarlyStopping` and `ModelCheckpoint` both select on `val_auc`; cell 9 reports on that same split | The headline 0.9977 is a max over 17 estimates on the selection split — optimistic by an unmeasured amount | B2: add a frozen test split and a separate calibration split |
| **2** | **Corpus/class confound** | **PROVEN, AND PROVEN TO HAVE BEEN EXPLOITED** | 100% of positives from YouTube compilations, 100% of negatives from BDD100K | **Explains essentially all of the 0.99 AUC. Removing the confound (Nexar test-public, one corpus) drops ROC-AUC from 0.9977 to 0.5339 — chance.** | B4: corpus change. **Measured 2026-09-11 (T3); not fixable on CCD** |
| **3** | **Threshold fitted to the evaluation videos** | **PROVEN** | `CNN_THRESH = 0.80` chosen against `safe.mp4` scoring 0.79 — one negative | System-level results are circular | B5: fit the operating point on the frozen test split |
| **4** | **Cross-split source leakage (`youtubeID`)** | **PROVEN AND QUANTIFIED — 91.4%** | 1,500 crash clips come from **133** YouTube videos (mean 11.3 each, max 34). A random 80/20 clip split puts **113 of 133 sources on both sides**, implicating **1,372/1,500 clips**; ≈274 of the 300 val crash clips have a sibling in train | **Severe.** This alone can account for most of the reported AUC | B3: group-wise split on `youtubeID`. **The official CCD split does not fix it — 107/133 sources appear on both sides there too.** |
| **5** | **Cross-split source leakage (BDD100K drives)** | **POSSIBLE, UNQUANTIFIED** | BDD100K clips come from journeys; multiple clips may share a drive | Unknown | B3, using BDD100K video IDs |
| **6** | **Near-duplicate clips** | **UNMEASURED** | Crash compilations frequently re-upload the same incident; no perceptual-hash dedup was performed | Inflates both train and val | Perceptual-hash dedup in the manifest builder |
| **7** | **Calibration leakage** | **STRUCTURAL** | No calibration split exists, so any future temperature scaling would have to reuse val or test | Would invalidate ECE | B2: create a fourth partition |
| **8** | **Fusion-layer leakage (future)** | **PREVENTABLE** | Fitting a fusion layer on training data leaks the learned model's train-set overconfidence | n/a yet | Fit fusion on validation only, never on train |

### The leakage rules to enforce in code

```
GROUPING KEY = (source_dataset, youtubeID | bdd_video_id, journey_id, event_id)

Rules — all enforced in code, with a test that fails the build if violated:
  1. All clips sharing a source video → exactly one split.
  2. All clips from one journey/session → exactly one split.
  3. All augmentations of a clip → the same split as the original.
  4. Near-duplicate clips (perceptual hash) → the same split.
  5. Test split frozen, hashed and committed once; never re-tuned against.
  6. A separate CALIBRATION split, distinct from both val and test.
  7. Thresholds are fitted on test-adjacent data exactly once, and recorded.
```

Random *frame*-level splitting would place frames 100 and 101 of one crash in train and test — a spectacular and meaningless score. Random *clip*-level splitting, which is what this project does, still leaks whenever clips share a source. **The current pipeline is one level better than the worst case and two levels short of correct.**

---

## 17. Reproducibility Problems

### Can another engineer reproduce your current model today?

| Starting from | Can they reproduce the pipeline? | Can they reproduce the model? | Blocker |
|---|---|---|---|
| **GitHub alone** | ❌ No | ❌ No | No training code, no dataset reference, no manifest, no hyperparameters (B10) |
| **GitHub + Colab** | ✅ **Yes, approximately** | ❌ No | Split seeded, hyperparameters explicit — but init/shuffle/augmentation unseeded (B9), and the shipped weights are from an unrecorded run (B8) |
| **GitHub + Colab + CCD** | ✅ Yes | ❌ No | Same |
| **GitHub + Colab + CCD + the model file** | ✅ Yes | ❌ No — you can *run* it, not *derive* it | The mapping artefact→run does not exist |

**The pipeline is reproducible in principle. The specific artefact in `models/` is not reproducible at all, and this is proven rather than suspected.**

### Exactly what is missing

| # | Missing | Consequence | Fix |
|---|---|---|---|
| 1 | Training code in git | Nobody but you can see the pipeline | B10 — `File → Download .ipynb`, commit today |
| 2 | Dataset manifest (clip → source → licence → split) | Cannot audit leakage or licensing | B3, D1 |
| 3 | `Crash-1500.txt`, `train.txt`, `test.txt` | Cannot group, cannot compare to literature | Download (Phase 0) |
| 4 | Global RNG seeding | Two runs of the same code differ | B9 |
| 5 | Run isolation for checkpoints | Artefacts untraceable | B8 |
| 6 | A `metrics.json` beside every artefact | No artefact→metric mapping | B8 |
| 7 | Artefact hashes | Cannot prove which file was measured | B8 |
| 8 | An installable environment | Nothing runs from a clean clone | R1 |
| 9 | An environment that matches the model's framework | Even a fixed install cannot load the weights | R2 |
| 10 | Any test | Nothing detects regressions | M4 |
| 11 | CI | Nothing enforces any of the above | Phase 1 |
| 12 | Experiment tracking | ≥6 runs, one surviving record | W&B free tier |

### The target state

```
Fresh machine
  → git clone
  → make setup            (pinned Python 3.11, lockfile, or docker build)
  → make data             (downloads CCD/Nexar, verifies checksums, builds manifest.csv)
  → make features         (deterministic; writes features/ + features.sha256)
  → make train            (seeded; writes runs/<ts>-<sha>/{weights,metrics.json,config.json})
  → make eval             (frozen test split; writes AP, AUC, FP/hour, ECE, per-condition)
  → numbers match runs/reference/metrics.json to 3 decimal places
```

**Every one of those steps is a day or less of work. None requires research.** The gap between the current state and this state is entirely mechanical, which is the most encouraging sentence in this document.

---

## 18. Calibration Problems

**Nothing about this model's output is calibrated, and three mutually inconsistent thresholds are in circulation.**

| Threshold | Value | Source | Justification |
|---|---:|---|---|
| Evaluation | 0.50 | Colab cell 9 | Default, unjustified |
| Suggested inference | 0.55 | Colab cell 10 | Comment says "tune: higher = fewer false positives" — i.e. explicitly unjustified |
| **Deployed** | **0.80** | `enhanced.py:109` | **None. Fitted against `safe.mp4` scoring 0.79.** |
| Secondary gate (video) | `crash_pct ≥ 1.5%` | `enhanced.py:1273` | Fitted against 21% / 14% / 0% on three clips |
| Secondary gate (fallback) | `crash_pct ≥ 5.0%` | `enhanced.py:1275` | Unexplained |

### Why this matters more than it appears

**1. The probability is not a probability.** No temperature scaling, no reliability diagram, no expected calibration error. A model trained to 100% train accuracy with binary cross-entropy is characteristically over-confident: it will emit 0.99 on inputs it has no business being sure about. Every downstream business decision — alert thresholds, review-queue ordering, insurer risk scores — assumes the number means what it says.

**2. The prevalence is wrong by four orders of magnitude.** Validation prevalence is 33%. Real prevalence is roughly one collision per 50,000 km. Precision at 33% prevalence tells you nothing about precision in production:

| Setting | Prevalence | FPR | Recall | Precision |
|---|---:|---:|---:|---:|
| CCD validation, threshold 0.5 | 33% | 3.17% | 98.3% | **94.0%** |
| Same model, 1 crash per 1,000 clips | 0.1% | 3.17% | 98.3% | **3.0%** |
| Same model, 1 crash per 10,000 clips | 0.01% | 3.17% | 98.3% | **0.31%** |

At realistic prevalence, **more than 99 of every 100 alerts would be false**. That is the arithmetic consequence of a 3.17% FPR, and it is why FP/hour rather than precision is the metric that decides whether a fleet keeps the product.

**3. There is no calibration split.** Even doing the right thing now would require reusing val or test, which reintroduces B2.

**4. The threshold was fitted at the wrong temporal scale.** Because of B1, the scores that 0.80 was fitted against were produced by feeding the model out-of-distribution sequences. Fixing B1 invalidates 0.80 twice over.

### Fix

1. Create a dedicated calibration split (B2).
2. Apply temperature scaling; report ECE with a target < 0.05 and publish the reliability diagram.
3. Delete every hard-coded threshold; derive the operating point from the ROC/PR curve on the frozen test split at a stated target recall, and persist it to `metrics.json` (B5).
4. Report **FP per hour of driving** as the headline, always. Never report raw accuracy.
5. Re-fit everything after B1 is resolved.

---

## 19. Current 3D/Physics Pipeline

**Status: computed every frame, excluded from every decision, and numerically invalid at the resolutions actually used.**

| Component | File:line | Implementation quality | Reaches the verdict? | Numerically valid? |
|---|---|---|---|---|
| Pinhole ground projection | `enhanced.py:381` | **Correct mathematics** | Only during the 9-frame CNN warm-up | ❌ No — C1 |
| 4-D Kalman filter (X, Z, Vx, Vz) | `enhanced.py:436` | **Correct**, sensible Q/R tuning | Same | ❌ Garbage in |
| TTC (vector-projected closing speed) | `enhanced.py:415` | **Correct** | Same | ❌ Garbage in |
| 3-D distance / `RuleCollision` | `enhanced.py:543–614` | Reasonable | Same | ❌ Garbage in |
| MiDaS depth | `depth_estimator.py:130` | ❌ Inverse-depth error (H1) | **Never** — heatmap only | ❌ No |
| BEV renderer | `bev_renderer.py:86` | ❌ Double km/h conversion (M1) | **Never** — display only | ❌ No |
| Ego zone | `enhanced.py:732` | Not a perception component at all | **Yes, in dashcam mode — it *is* the verdict** | n/a |
| Fault attribution | `enhanced.py:622`, `:761` | Heuristic on box positions | Reported to the user | ❌ No |

**Calibration constants, all assumed and never measured:** `H_CAM = 1.25 m`, `PITCH_DEG = 2.0°`, `FX = FY = 460 px` (for a 640×480 OV5647), `CX_IMG`/`CY_IMG` = frame centre (updated on resize — inconsistently with FX/FY, which is C1).

**Verdict.** The Kalman filter and TTC computation are the best code in the repository and should be **kept**. They are fed invalid geometry by C1 and then ignored by C2. Fix the calibration and wire them into the fusion layer of [§28](#28-recommended-future-architecture), where they become the explainability channel that lets an insurer believe your output. Delete MiDaS, the BEV renderer as a pipeline component, the ego zone and fault attribution.

**Do not describe this stack as functional perception in any document, demo or pitch until it demonstrably affects a validated decision.**

---

## 20. Current Detector/Tracker

### Detector

| Property | Value |
|---|---|
| Model | YOLOv8n (`yolov8n.pt`, 6.5 MB, auto-downloaded, gitignored) |
| Package | `ultralytics` — **AGPL-3.0 (C6)** |
| Confidence threshold | 0.5 (`YOLO_CONF`) |
| NMS IoU | 0.45, applied cross-class |
| Vehicle classes | `car`, `truck`, `bus`, `motorcycle`, `bicycle` |
| Object classes | `person`, `dog`, `cat`, `backpack`, `suitcase`, `chair`, `bench`, `traffic light`, `stop sign` |
| Input | ROI-masked frame (top 40% and bottom 10% blanked) |
| Scene validation | Aspect ratio ≥ 0.4; `MIN_BOX_AREA = 2000` px² |
| Error handling | `except Exception: return []` — **silent (H3)** |

**Assessment:** functionally the most reliable component in the system. Two problems: the licence (C6) and the crude aspect-ratio filter, which will drop a head-on motorcycle — the exact object you most need to detect.

### Tracker

| Property | Value |
|---|---|
| Algorithm | Greedy centroid matching, `TRACK_MAX_DIST = 80` px |
| Re-identification | None |
| Motion model | None in the tracker (the Kalman filter lives separately, in `SpeedEstimator`) |
| Occlusion handling | **None — all IDs reset on any empty frame (H2)** |
| TTL | `TRACK_TTL_SEC = 2.0` applies to the Kalman dictionary, **not** to track identity |

**Assessment:** the weakest link in the physical channel. Identity churn during occlusion destroys per-track velocity exactly when it matters. **Replace with ByteTrack or OC-SORT** — both permissively licensed, both a day's work, both strictly better.

---

## 21. Current Crash Decision Logic

Verbatim, from `code/crash_detection_enhanced.py`.

**Per-frame, standard mode (`:1049–1056`):**
```python
if cnn >= 0:
    if (cnn >= Config.CNN_THRESH) and (len(vehs) >= 2):   # 0.80 and 2
        neural_confirmed = True
        is_crash = True
    else:
        is_crash = False
else:
    is_crash = rule['is_sustained'] and (len(vehs) >= 2)   # ONLY during the 9-frame warm-up
```

**Per-frame, dashcam mode (`:1040–1046`):**
```python
ego_hits = EgoZone.check(vehs, fw, fh)   # any vehicle box overlapping a fixed rectangle
is_crash = len(ego_hits) > 0
```

**Video-level verdict (`:1273–1283`):**
```python
if neural_confirmed:
    is_crash = neural_confirmed and (crash_pct >= 1.5)
else:
    is_crash = crash_pct >= 5.0
```

### What this means

| Aspect | Reality |
|---|---|
| Signals that reach the verdict | Exactly two: the CNN score, and the count of YOLO vehicle boxes |
| Signals computed and discarded | Kalman state, ground projection, 3-D distance, TTC, closing speed, MiDaS depth, BEV |
| Thresholds with a derivation | **Zero of four** (0.80, 2, 1.5%, 5.0%) |
| Temporal reasoning | A percentage of frames flagged — no event boundaries, no onset, no peak |
| Ego-involvement | A fixed screen rectangle (C3) |
| Fault | Bounding-box x-position (C7) |
| Behaviour when the model is unavailable | Silently falls back to the physics rule, which is invalid (C1) — and this is never surfaced to the user |

**The vehicle-count gate deserves a specific note.** `len(vehs) >= 2` requires two detected vehicles to declare a crash. It therefore **structurally cannot detect** a single-vehicle collision (into a barrier, a tree, a pedestrian, or a cyclist), which is a large fraction of real incidents and the majority of vulnerable-road-user incidents. This is not a tuning problem; it is a category the system is defined to miss, and it is not documented anywhere.

**Replace the whole block** with the calibrated fusion layer of [§28](#28-recommended-future-architecture), emitting a structured event rather than a boolean.

---

## 22. Licensing Problems

| # | Issue | Where | Risk | Action | Priority |
|---|---|---|---|---|---|
| 1 | **`ultralytics` YOLOv8 is AGPL-3.0** | `requirements.txt:3`, all four forks | A hosted service must offer its complete corresponding source | Migrate to RT-DETR (Apache-2.0, from `lyuwenyu/RT-DETR`, **not** via `ultralytics`) or buy the Enterprise Licence | **P0** |
| 2 | **Training data provenance** | CCD — see [§23](#23-dataset-licensing--provenance) | Negatives are BDD100K (personal-use licence); positives are YouTube-derived | Treat CCD as research-only; retrain on clean-provenance data | **P0** |
| 3 | **Nexar licence ambiguity** | Primary future dataset | Three inconsistent descriptions of one licence | Email Nexar; get operative terms in writing | **P0** |
| 4 | **YouTube-derived academic datasets** | DoTA, DADA-2000, MM-AU, DAD, A3D, CADP, and CCD positives | Annotations licensed; pixels are not | Research/benchmark only; never in a shipped model | **P0** |
| 5 | **No `LICENSE` file** | Repository root | The README has claimed MIT; no file exists | Add one, or make the repository explicitly private | **P1** |
| 6 | **MiDaS via `torch.hub`** | `depth_estimator.py` | Check the specific checkpoint's terms | Moot if MiDaS is deleted as recommended | P3 |
| 7 | **NVIDIA LocateAnything-3B** | `LocateAnything-3B_Guide.pdf` (untracked) | "Research or evaluation purposes only"; licence silent on outputs | Do not use. Use Florence-2 (MIT) or OWLv2 (Apache-2.0) | **P1** |
| 8 | **Fault-determination claims** | `FaultDetector`, `EgoZone.fault` | Product-liability exposure | Delete (C7) | **P0** |
| 9 | **Safety-critical framing** | Marketing language | Even BADAS-Open explicitly disclaims safety-critical certification | Never claim it | **P0** |

**This section identifies risk areas. It is not legal advice.** Items 1–4, 7 and 8 require a qualified lawyer in the relevant jurisdiction.

---

## 23. Dataset Licensing / Provenance

### The chain for the model you have today

```
models/crash_model_weights.weights.h5
   └── trained on features_v2/*.npy
         └── extracted from Crash-1500.zip + Normal.zip
               └── Car Crash Dataset (Cogito2012/CarCrashDataset)
                     ├── Crash-1500 (1,500 positives)
                     │     └── SOURCE: YouTube  ← per-clip `youtubeID` + `startframe`
                     │           └── Copyright: original uploaders.
                     │               The repository's MIT label covers the researchers'
                     │               ANNOTATIONS, not the video pixels. Bulk downloading
                     │               generally contravenes YouTube's Terms of Service.
                     └── Normal (3,000 negatives)
                           └── SOURCE: BDD100K (UC Berkeley)  ← stated in the CCD README
                                 └── Licence: basic licence limited to personal use.
                                     Commercial use requires verification with BDD.
```

**Assessment:** 🔴 **RED for commercial use.** Both halves of the training set have provenance problems, and they are different problems. This is not a grey area you can paper over; it is a documented paper trail pointing at two identifiable rights-holders. **The current model must be treated as a research artefact and must not be shipped commercially.**

**Also note:** the CCD GitHub sidebar shows an MIT label but the README contains no licence section, so the operative scope of that MIT grant is unstated (U7). Ask the authors.

### GREEN — likely usable commercially after your lawyer confirms

| Asset | Licence | Note |
|---|---|---|
| **V-JEPA 2** (backbone) | **MIT** (majority; some utility files Apache-2.0) | Meta explicitly released V-JEPA 2 for commercial use, unlike V-JEPA 1 |
| **BADAS-Open** (model) | **Apache 2.0** | Commercial use with attribution; disclaimed for safety-critical use |
| **RT-DETR / RT-DETRv2** | **Apache-2.0** | From `lyuwenyu/RT-DETR`, **not** the `ultralytics` wrapper |
| **ByteTrack / OC-SORT** | Permissive | Verify the specific repository |
| **Florence-2** (auto-labelling) | **MIT** | Microsoft |
| **OWLv2** (auto-labelling) | **Apache-2.0** | Google |
| **DoTA annotations** | **MIT** (repository) | Annotations only — **not** the video pixels |
| **IDD-3D** | **CC BY 4.0** | Relevant only for a later India expansion |
| **Nexar Collision Prediction Dataset** | **Custom Nexar licence — commercial training permitted** | **RESOLVED 2026-09-10**, text retrieved to `data/nexar/LICENSE`. Grant: *"Permission is hereby granted, free of charge … to use, copy, modify, and distribute the Dataset."* **No non-commercial restriction.** Conditions: (a) attribution with the specified citation; (b) retain the notice on redistribution; (c) **No Resale** — the *Dataset* may not be sold or sublicensed for profit without written consent; (d) ethical-use restrictions: no malicious systems, deepfakes, re-identification, weaponisation, **"exploitative practices … such as unethical insurance practices"**, and compliance with law. Training a commercial model is "use" and "modify" and is permitted; No-Resale restricts redistributing the dataset, not derived models. **Have counsel confirm that last reading — but this is off the critical path.** |
| **Your own recorded footage** | Yours | With driver consent and a DPIA — the cleanest data you will ever own |

### YELLOW — needs legal review before commercial training

| Asset | Stated position | Why it is yellow |
|---|---|---|
| ~~**Nexar Collision Prediction Dataset**~~ | **RESOLVED 2026-09-10 — moved to GREEN below.** The operative text was retrieved directly from the repository (`data/nexar/LICENSE`) and is unambiguous. | — |
| **Car Crash Dataset (CCD)** | Repository labelled MIT; README has no licence section; positives YouTube-derived, negatives BDD100K | Use for research and benchmarking only. Ask the authors about scope (U7). |
| **BDD100K** | "Basic licence limited for personal use" | Verify current terms directly with Berkeley DeepDrive before any commercial use. **Note that you have already trained on it, via CCD.** |
| **DoTA / DADA-2000 / MM-AU / DAD / A3D video content** | Repositories MIT/academic; videos scraped from YouTube | Annotations ≠ pixels; see below |

### RED — do not train a commercial model on these

| Asset | Licence | Verdict |
|---|---|---|
| **nuScenes** | CC BY-NC-SA 4.0 | Non-commercial. Also ShareAlike — viral. |
| **Waymo Open Dataset** | Non-commercial licence agreement | Research only |
| **CADP** | Research / non-commercial only | Research only |
| **NVIDIA LocateAnything-3B** | NVIDIA License — *"only may be used or intended for use non-commercially… 'non-commercially' means for research or evaluation purposes only"* | Research only; the licence is **silent on model outputs**, and silence is not permission. Use Florence-2 or OWLv2 instead. |

### ⚠️ The subtlety that matters most: annotations ≠ pixels

Almost every academic crash-video dataset is built from YouTube-scraped footage. DoTA's repository is MIT-licensed and its README documents downloading clips from YouTube URLs. CADP's 1,416 segments come from YouTube. DAD, DADA-2000 and MM-AU follow the same pattern — **and so does CCD's Crash-1500, which is now confirmed to be in your model.**

**The MIT licence covers the researchers' annotations. It cannot and does not license the underlying video, whose copyright belongs to the original uploaders — and downloading it generally contravenes YouTube's Terms of Service.**

The practical consequence: **you can use these datasets for research, benchmarking and publication, but training a commercial production model on the pixels is a real and under-appreciated legal exposure.** Most startups in this space have not thought about it. An acquirer's diligence team will. **This is no longer hypothetical for you — it describes the model currently committed to your repository.**

This is precisely what makes the Nexar dataset unusually valuable: it is **first-party footage from Nexar's own consenting driver community, anonymised**, with a licence intended for open use. It is one of very few crash video corpora with a clean provenance chain — which is why nailing down its exact terms is a week-1 task.

### The manifest discipline that fixes this permanently

```
data/manifest.csv
  clip_id, dataset, source_url_or_id, licence, licence_verified_date,
  consent_status, redaction_status, split, group_key, duration_s, fps, resolution
```

`train.py` refuses to run if any clip in the training set lacks a `licence` entry. This one file, enforced in CI, converts "we think it's fine" into an auditable answer — and it is perhaps three hours of work.

---

# Part III — Where to Go

> **Nothing in the Colab or the dataset repository bears on Part III.** Every recommendation below is carried forward from the 2026-09-07 audit unchanged, except where the new evidence sharpens an argument — flagged inline. This part is preserved deliberately: re-deriving strategy from scratch each time a technical fact changes is how founders lose quarters.

## 24. Target Market

### The two regulatory facts that reshape the whole analysis

1. **Automatic crash notification is already mandated and solved in the EU.** Regulation (EU) 2015/758 requires 112-based eCall on all new M1/N1 vehicle *types* from 31 March 2018. It triggers on airbag/accelerometer signals, transmits location, and is estimated to cut emergency response times by 40% urban / 50% rural. **You cannot sell "detect the crash and call for help" into new EU/UK passenger cars.** That market was closed by legislation eight years ago.
2. **ADAS is mandated on new EU vehicles from July 2024.** GSR2 (Regulation (EU) 2019/2144) Phase Two requires AEB, Intelligent Speed Assist, Emergency Lane Keeping and Driver Drowsiness Warning; Phase Three (July 2026) adds Advanced Driver Distraction Warning and extended pedestrian/cyclist AEB. On new passenger cars, the OEM's own perception stack is the incumbent.

**Conclusion:** in every technologically mature market the addressable wedge is **not** new passenger cars. It is **retrofit commercial vehicles** — vans, trucks, buses, older fleet vehicles that will never receive GSR2 ADAS — plus **claims and evidence workflows**, which no regulation covers.

### Country comparison

Scored 1–5, weighted for a pre-seed founder with no capital and no local network. What determines survival in the first 18 months: can you get **video data**, a **paying pilot**, and a **second reference customer**.

| Market | Video data access | Legal clarity for dashcam video | Fleet buyer accessibility | Insurance/telematics maturity | Capital & accelerators | Founder-accessibility from India | **Total** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **United Kingdom** | 4 | **5** | **5** | **5** | 3 | **5** | **27** |
| **United States** | **5** | 4 | 3 | **5** | **5** | 2 | **24** |
| Netherlands | 3 | 4 | 4 | 3 | 3 | 4 | 21 |
| Canada | 3 | 4 | 4 | 3 | 3 | 3 | 20 |
| UAE | 3 | 4 | 4 | 2 | 2 | **5** | 20 |
| Australia | 3 | 4 | 4 | 3 | 2 | 3 | 19 |
| Singapore | 2 | 4 | 3 | 3 | 3 | 4 | 19 |
| Japan | **5** | 4 | 2 | 4 | 2 | 1 | 18 |
| South Korea | **5** | 4 | 2 | 3 | 2 | 1 | 17 |
| Sweden / Norway / Finland | 2 | 3 | 3 | 3 | 2 | 3 | 16 |
| Germany | 3 | **1** | 3 | 4 | 3 | 2 | 16 |
| France | 3 | 2 | 3 | 3 | 3 | 2 | 16 |
| Switzerland | 2 | 2 | 2 | 3 | 2 | 2 | 13 |

### Why the obvious candidates lose

**Germany — the best automotive ecosystem, and nearly the worst first market for a *video* company.** The Bundesgerichtshof ruled in May 2018 (VI ZR 233/17) that **continuous dashcam recording violates data-protection law** — footage may still be admitted in civil proceedings on a case-by-case balancing test, but the recording itself is a technical infringement. Austria bans dashcams outright. Building a video-first company in the jurisdiction most hostile to continuous video capture, while facing the world's longest OEM sales cycles, is a two-year detour. Germany is a **year-3** market, entered with a track record, to talk to Tier-1 suppliers.

**Japan and South Korea — the best data, the worst access.** South Korea has roughly **80% dashcam penetration**; Japan is close behind. An enormous corpus, almost entirely inaccessible to an unknown foreign founder: B2B procurement runs through established relationships and keiretsu/chaebol supplier networks, in Japanese and Korean. Great acquisition markets later; not where you land pilot #1.

**United States — the biggest prize, the wrong first step.** ~15.5 million commercial vehicles; US fleet telematics revenue ≈ $9.2 billion in 2025; ~67% GPS fleet-tracking adoption (>80% for 100+ vehicle fleets). It is also where Samsara, Motive, Netradyne, Lytx and Nauto compete with hundreds of millions in funding and mature sales organisations. For an unfunded founder at +10.5 to +13.5 hours, cold-starting enterprise pilots here is the hardest version of the problem. **The US should be your legal and financial home, not your first sales territory.**

**Nordics, Netherlands, Singapore, UAE, Switzerland** — all workable, none decisive. The Netherlands is Europe's logistics corridor and an excellent *second* market. The UAE offers fast government procurement and easy incorporation — a good *pilot-of-opportunity*. None has enough domestic fleet volume to build a company on alone.

---

## 25. Initial Country

> ### Design for and sell into the **United Kingdom** first. Incorporate in the **United States** (Delaware C-corp).

These are two different questions and conflating them is a common founder error. Your *legal and fundraising domicile* should be where the capital is. Your *first customers and first proprietary data* should be where you can realistically close them.

| Decision | Choice | Why |
|---|---|---|
| Country of incorporation | **US — Delaware C-corp** | Expected by every institutional investor and by YC's standard paperwork; formable remotely by a non-resident Indian founder with no US address, bank account or visa at formation |
| First design target (data distribution, camera geometry, road furniture) | **United Kingdom** | See below |
| First paying pilots | **UK mid-market commercial fleets** (30–300 vehicles) | Accessible, budget-holding, fast decisions |
| Second market | **Ireland → Netherlands → Nordics** | English-friendly, EU logistics corridor, similar road environment |
| Third market | **United States** | Enter with UK reference customers and measured metrics, not a cold pitch |
| Year-3 market | **Germany / Japan** | Tier-1 supplier and OEM conversations, once you have a track record |

### Why the UK — five verifiable reasons

**1. It is the only major market with an institutional pathway for dashcam evidence.** The National Dash Cam Safety Portal, operated by Nextbase, received **33,531 video submissions to police in England and Wales in 2023** and has processed over 135,000 clips in five years, working with **every police force in England**; roughly 70% of submissions lead to some police action. No other market has normalised "dashcam footage as evidence" to this degree. UK fleets, insurers and police already believe video evidence is actionable, so you are not selling the *category*, only your product. Category education is the most expensive thing a pre-seed startup can be forced to pay for.

**2. The legal position on dashcam video is clear, unlike Germany's.** UK GDPR imposes real obligations (DPIA, retention limits, transparency), but continuous dashcam recording for a legitimate commercial purpose is normal and widely practised. Legal clarity is worth more than market size at pilot stage.

**3. The buyer exists, has a budget, and is reachable.** UK fleet telematics is **£783.0 million in 2025, forecast to £1.34 billion by 2030** — large enough to sustain a company, small enough that mid-market operators are not saturated by Samsara-scale vendors. Fleets of 30–300 vehicles are typically run by an operations director who can authorise a pilot without a procurement committee. That is the single most important structural advantage over the US.

**4. Insurance telematics is culturally embedded.** The UK pioneered mainstream usage-based motor insurance, so insurers and fleets already price risk on driving-behaviour data. Your near-miss and risk-scoring outputs land on receptive ears.

**5. It is the only mature market you can genuinely work with from India.** BST/GMT is **+4.5 to +5.5 hours** from IST. A 14:00 IST call is 09:30 UK. Compare US Pacific at +12.5 hours, where every customer call is at 22:30 IST. For founder-led sales, this determines how many customer conversations per week are physically possible.

**What the UK does *not* give you.** Less venture capital, fewer top-tier accelerators, a smaller total market than the US, and real incumbents (VisionTrack, Lytx UK, Samsara UK). This is why the recommendation is *split*: sell in the UK, incorporate in the US, and use UK traction as the evidence that unlocks US capital in year 2.

### Global expansion ladder

```
  PUBLIC DATA
     │  Nexar (commercially licensable, pending confirmation) + DoTA/DADA (pretraining) + BADAS-Open baseline
     │  ── GATE: reproduce BADAS-Open's numbers on the Nexar benchmark. If you cannot
     │           reproduce a published result, you cannot claim to beat one.
     ▼
  SELF-COLLECTED UK DATA
     │  50–100 hours of your own UK dashcam footage; a UK hard-negative benchmark
     │  ── GATE: a UK-specific eval set nobody else has, with measured FP/hour.
     ▼
  PILOT DATA  (2–3 UK fleets, 30–300 vehicles each, free or near-free)
     │  ── GATE: 90 days continuous running, FP/hour below the fleet's tolerance,
     │           and a signed DPA letting you retain derived events.
     ▼
  PARTNER DATA  (UK dashcam OEM / video telematics vendor / broker-insurer)
     │  ── GATE: 2 paying reference customers who will take a call from a prospect.
     ▼
  PROPRIETARY DATA  (your own labelled incident corpus + feedback loop)
     │  ── GATE: >10,000 human-verified incidents across >5 fleets and >2 countries.
     ▼
  EU EXPANSION  (Ireland → Netherlands → Nordics)
     │  ── GATE: performance holds across countries without retraining.
     ▼
  US SCALE-UP
     │  ── GATE: US-based sales hire, SOC 2, a US reference customer.
     ▼
  TIER-1 / OEM  (Germany, Japan)
        ── GATE: functional-safety story, ISO 26262 awareness, ~100k+ validated
                 vehicle-hours, and a benchmark they cannot reproduce.
```

**Each rung is unlocked by *evidence*, not by effort.** Nobody at a Tier-1 cares how hard you worked. They care whether you have field data they do not have and a validation methodology they respect. Every gate above produces exactly one such artefact.

---

## 26. Customer Profile

| Segment | Ease of entry | Willingness to pay | Data access | Sales cycle | Competition | Fit | **Rank** |
|---|---|---|---|---|---|---|---|
| **UK mid-market commercial fleets (30–300 vehicles)** | High | Medium-High | **Excellent** | 4–10 weeks | Medium | **Excellent** | **① FIRST** |
| UK video-telematics / dashcam vendors | Medium | Medium | **Excellent** | 3–6 months | Medium | Very good | ② |
| Commercial motor insurers / MGAs / brokers | Low | **Very high** | Good | 6–18 months | Medium | Good | ③ |
| Logistics & last-mile delivery operators | Medium | Medium | Excellent | 2–4 months | Medium | Good | ④ |
| Bus / coach / municipal fleets | Low | Medium | Good | 9–18 months | Low | Fair | ⑤ |
| Driving schools | **Very high** | Low | Good | 1–2 weeks | None | Data source, not revenue | ⑥ |
| Automotive OEMs / Tier-1s | **Very low** | Very high | Poor | 2–5 years | High | **Not now** | ⑦ |
| Smart cities / traffic authorities | Very low | Medium | Medium | 12–36 months | Medium | Not now | ⑧ |
| Consumers | Medium | Very low | Poor | — | Very high | **Never** | ⑨ |

### Target #1, specifically

**A UK commercial fleet of 30–300 vehicles that already runs dashcams and currently reviews footage manually.**

- **They already have cameras** — you are not selling hardware, which removes capital cost, installation logistics and the longest part of the sales cycle.
- **They already have the pain** — someone is paid to scrub video, or nobody watches it and the footage is worthless.
- **They own their data** and can grant processing rights without a committee.
- **One person can say yes.** An operations or fleet director at this size authorises a pilot directly. Above ~500 vehicles you hit procurement; below ~30 there is no budget.
- **They are measurable.** Insurance premium, claims frequency and incident review hours are quantifiable within a 90-day pilot — exactly the evidence you need for the next customer and for investors.

**The pitch:** *"You are already recording. You are not watching. We turn your footage into a reviewed incident queue and a claims-ready evidence pack, automatically — and we will prove our false-alarm rate on 500 UK clips before you pay us anything."*

That last clause gets the meeting, because every fleet manager who has trialled an AI dashcam has been burned by false alerts, and you would be the first vendor to lead with the number they actually care about.

---

## 27. Product Definition

### The product is not crash detection

Detection is an input. Given that BADAS-Open is free, Apache-2.0 and state of the art, **the detector is a commodity from day one.** Building a company whose value proposition is "we detect crashes" means competing on a feature anyone can download.

**The product is the structured incident record.**

```
RAW DASHCAM FOOTAGE  (unstructured, enormous, unsearchable, nobody watches it)
                              │
                              ▼
              ┌───────────────────────────────┐
              │   VERIFIED INCIDENT RECORD    │
              ├───────────────────────────────┤
              │ • event_id, vehicle, driver   │
              │ • t_start / t_peak / t_end    │
              │ • ego-involved: yes/no        │
              │ • class: collision | near-miss│
              │ • severity band + confidence  │
              │ • measured closing speed,     │
              │   peak longitudinal accel     │
              │ • 45 s evidence clip, sealed  │
              │   with a hash + timestamp     │
              │ • GPS trace + map context     │
              │ • human-review status         │
              └───────────────────────────────┘
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
  FLEET SAFETY          CLAIMS / FNOL          RISK ANALYTICS
  coaching queue,       evidence pack for      near-miss heatmaps,
  incident review       insurer within         driver/route risk
  in minutes not hours  minutes of the event   scoring over time
```

The unit of value is **an incident a human can act on in under two minutes**, with the evidence attached. Fleets today either pay humans to watch footage (Lytx's model) or never review it at all. Insurers wait days for a claim narrative that video could have settled in minutes.

**Severity banding, not severity prediction.** Commit to coarse, defensible bands — *contact / minor / significant / severe* — derived from measured physical quantities with stated uncertainty. Do not predict injury, do not predict repair cost, and above all **do not attribute fault** (C7). Sell evidence; let humans adjudicate.

> **Sharpened by the new evidence:** the incident record needs `t_start` / `t_peak` / `t_end`. Your current model cannot produce them — but **CCD ships per-frame `binlabels` and you already have the pipeline to consume them** (B7). Temporal localisation is not a research problem for you; it is a parsing problem you skipped. The same applies to `ego_involved`, for which CCD ships a label and against which you built a rectangle (C3).

### Product ladder

| Stage | What it is | Who it is for | Timeline |
|---|---|---|---|
| **MVP** | CLI + API: video in → JSON incident records out. Reproducible, benchmarked. | You, and the first technical evaluator | Weeks 1–4 |
| **Prototype V1** | Batch service + minimal review UI. Nexar-benchmarked, UK-HN-500 measured. | First design-partner conversations | Weeks 5–8 |
| **Prototype V2** | Edge inference on Jetson + cloud sync + evidence export + fleet dashboard | First pilot deployment | Weeks 9–14 |
| **Investor demo** | Live, side-by-side against baselines on unseen footage, with the FP benchmark shown | YC / Techstars / angels | Week 12+ |
| **Pilot-ready** | Multi-vehicle, offline-capable, DPA-compliant, SLA-backed, human review loop | 2–3 UK fleets | Months 4–9 |
| **Production** | Multi-tenant, SOC 2 path, model registry, drift monitoring, partner API | Vendor/insurer partnerships | Months 9–18 |

---

## 28. Recommended Future Architecture

> **You asked me not to recommend replacing your model merely because a newer architecture exists. I am not.** The argument below is entirely about properties of *your* model and *your* task, established from the Colab. If those properties were different, the recommendation would be different. Where MobileNetV2+LSTM should be kept, I say so.

### Is MobileNetV2 + LSTM actually stronger than the previous audit believed?

**Partly yes, and it does not change the conclusion.**

**Stronger than believed, in three specific ways.** The backbone is properly frozen and features are cached — an efficient, sensible design for a small dataset. Preprocessing is consistent between training and inference. The training loop monitors AUC rather than accuracy, uses class weights, early-stops on the right metric, and restores best weights. This is better practice than the previous audit inferred from an absence of evidence, and the person who wrote it understood what they were doing.

**Not stronger where it matters, for four reasons that are specific to this model.**

1. **The 578,689-parameter head sits on frozen *ImageNet classification* features.** MobileNetV2's GAP output is optimised to describe *what object is in a picture*. Collision detection is about *how things move relative to each other*. Global average pooling destroys spatial layout entirely — after the GAP, the model cannot know *where* in the frame anything was, so relative motion between two vehicles is not representable. No LSTM on top can recover information the pooling threw away. **This is an architectural ceiling, not a tuning problem.**
2. **The temporal window is incoherent** (B1) and, even when fixed, 10 samples is a coarse description of a 5-second event.
3. **It cannot produce the product's required outputs.** No event boundaries, no ego-involvement, no near-miss/collision separation, no severity. Adding them means changing the architecture anyway.
4. **A free, Apache-2.0, commercially usable model reports AP 0.86 / AUC 0.88 on a corpus-controlled benchmark, with mTTA 4.9 s.** Your model has no corpus-controlled measurement at all.

**What should be retained from the current model:** the frozen-backbone-plus-cached-features *pattern* (it is exactly how you should train the V-JEPA2 probe on a budget); the consistent preprocessing discipline; the callback configuration; and the `Sequence` loader. **What should be discarded:** the ImageNet backbone, the GAP bottleneck, the LSTM head, and the single-output design.

**And a genuine possibility you should test rather than assume:** if the falsification tests in [§30](#30-recommended-training-strategy) show the CCD result was a corpus artefact, then MobileNetV2+LSTM has never been fairly evaluated on this task at all. In that case, **re-evaluating it properly on Nexar is a legitimate, cheap experiment** and belongs in your baseline table beside BADAS-Open and always-negative. Do not delete it before it has been measured once, honestly. Measure it, record the number, then decide.

### Approach comparison

| Approach | Accuracy potential | Data needed | Compute | Latency | Edge-deployable | Explainable | Pretrained available | Verdict |
|---|---|---|---|---|---|---|---|---|
| YOLO + rules (**what you have**) | Low | Low | Low | Good | Yes | High | Yes | **Insufficient** — brittle, uncalibratable |
| CNN + LSTM (**what you have**) | Low–Med | Medium | Low | Good | Yes | Low | Yes | **Ceiling-limited** for this task (see above) |
| 3D CNN (I3D, SlowFast, X3D) | Medium | High | Medium | Medium | Marginal | Low | Yes | Superseded |
| TimeSformer / ViViT | Med–High | High | High | Poor | No | Low | Yes | Too heavy for edge |
| VideoMAE / VideoMAEv2 | High | Medium (SSL pretrained) | High | Medium | Marginal | Low | Yes | Strong, but no task-specific checkpoint |
| **V-JEPA2 + probe head** | **High (proven)** | **Low (fine-tune)** | Medium | Medium | With distillation | Low | **Yes — Apache/MIT chain** | **✅ Recommended core** |
| Trajectory/TTC reasoning | Medium | Low | Low | Excellent | Yes | **Very high** | N/A | **✅ Recommended as a parallel calibrated channel** |
| Optical flow | Low alone | Low | Medium | Medium | Marginal | Medium | Yes | Redundant with a video backbone |
| Monocular depth (MiDaS / Depth-Anything) | — | — | High | Poor | No | Medium | Yes | **Cut.** Cost without decision value |
| Multimodal video + IMU + GPS | **Highest** | Medium | Low-ish | Excellent | Yes | High | Partial | **✅ Recommended for v2** |

### The recommendation

```
┌─────────────────────────────────────────────────────────────────────┐
│  CHANNEL A — LEARNED (primary)                                      │
│                                                                     │
│  16-frame clip @ 256×256, ~2 s temporal window, EXPLICIT STRIDE     │
│      → V-JEPA2 ViT-L encoder  (MIT)                                 │
│      → attentive probe + MLP head  (init from BADAS-Open, Apache-2) │
│      → P(ego-involved collision), P(near-miss), event boundaries    │
│                                                                     │
│  ↑ Multi-head, NOT one. Splitting collision from near-miss is the   │
│    single most valuable change you can make, because it is what     │
│    Nexar collapses and what your customers need apart.              │
│  ↑ The stride is a named constant shared with the data loader,      │
│    so B1 cannot recur.                                              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│  CHANNEL B — PHYSICAL (corroboration + explanation)                 │
│                                                                     │
│  RT-DETR (Apache-2.0) → ByteTrack → CALIBRATED pinhole projection   │
│      → per-track [X, Z, Vx, Vz] Kalman → TTC, closing speed         │
│      → IMU (|Δa|, jerk) + GPS (Δspeed) where available              │
│                                                                     │
│  ↑ This is your existing physics stack — RESURRECTED, but only      │
│    after C1 (calibration) is fixed. Its job is NOT to decide.       │
│    Its job is to (a) corroborate, (b) suppress false positives,     │
│    and (c) produce the human-readable "why".                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│  FUSION — a calibrated logistic layer, not a hand-tuned threshold   │
│                                                                     │
│  features: [P_collision, P_nearmiss, TTC_min, closing_speed,        │
│             |Δa|_peak, Δspeed_gps, track_stability, n_vehicles]     │
│      → logistic regression / small GBM (fully interpretable)        │
│      → temperature scaling for a CALIBRATED probability             │
│      → temporal confirmation over a sliding window                  │
│      → EVENT: {t_start, t_peak, t_end, p, severity_band, evidence}  │
└─────────────────────────────────────────────────────────────────────┘
```

**Why this, and not something newer:**

1. **The commercial licence chain is clean end-to-end.** V-JEPA2 (MIT) → BADAS-Open (Apache-2.0) → RT-DETR (Apache-2.0) → ByteTrack. No AGPL, no research-only weights, no YouTube-pixel ambiguity. Given [§23](#23-dataset-licensing--provenance), this is worth more to you than two points of AP.
2. **You start from a published SOTA result rather than from scratch.** Reproducing BADAS-Open's AP 0.86 on day 1 puts you further ahead than six months of your own architecture work.
3. **Channel B is your differentiator, not your baseline.** BADAS is a pure learned model. A calibrated physical channel gives you false-positive suppression a black box cannot provide, explainability insurers require, and graceful degradation out of distribution.
4. **Explainability is a product requirement.** "Closing speed went from 12 m/s to 0 in 180 ms and peak longitudinal acceleration was 4.1 g" is an insurable statement. "Our neural network said 0.94" is not.
5. **A multi-head output matches the business.** Collisions are claims events. Near-misses are risk-scoring events with 100× the volume — and they are how fleets justify the subscription between crashes.

**What to delete outright:** MiDaS/depth estimation, the BEV renderer as a pipeline component, the ego zone, the fault attribution module, and — after one honest measurement — the MobileNetV2+LSTM classifier.

### Sensor strategy

| Signal | Realistically accessible? | Acquisition path | Verdict |
|---|---|---|---|
| **Video** | ✅ Yes | The dashcam itself | **Core.** |
| **IMU (accel/gyro)** | ✅ Yes, essentially free | In every dashcam and phone; often already in the video container metadata | **Add in v2.** Highest value-per-effort on this list — the single best false-positive discriminator, because a real impact has an unmistakable acceleration signature. |
| **GPS** | ✅ Yes, essentially free | Same devices | **Add in v2.** Speed, road-type context via map matching, and location for the evidence package. |
| **OBD-II / CAN** | 🟡 Via partner | Fleet telematics vendors already read this | **v3, via partnership.** Do not build hardware. |
| **OEM ADAS / perception output** | ❌ No | Proprietary | **Do not design for this.** This is the trap in "Tesla-level" thinking. |
| **eCall / airbag deployment** | ❌ No | Type-approved OEM subsystem | Not accessible; also redundant |
| **LiDAR / radar** | ❌ No | Not in retrofit dashcams | Ignore |

**Conclusion: video + IMU + GPS.** Available on commodity hardware today, requires no OEM relationship, covers the overwhelming majority of achievable accuracy.

---

## 29. Recommended Dataset Strategy

> **Global data for pretraining. Target-market data for fine-tuning and, above all, for evaluation.**

This follows from three findings, one of which is new.

1. **Collision dynamics are physically universal; scene appearance is not.** Rapid looming, sudden ego-motion change and impact are governed by physics that does not vary by country. Road markings, signage, vehicle body styles, driving conventions and weather do. So the low-level temporal representation transfers and the appearance layer does not.
2. **BADAS demonstrates the transfer empirically.** A model trained on 1,500 Nexar videos reports strong AP across DoTA (4,677 videos), DADA-2000 and DAD — different continents, different cameras.
3. **NEW: your own experiment demonstrates the failure mode.** CCD's class/corpus confound (B4) is a concrete, first-hand demonstration of why appearance-level shortcuts dominate when the data permits them. You have already paid for this lesson; extract it. **Every dataset decision from here should ask: can a model separate my classes without solving my task?**

### The concrete recipe

| Stage | Data | Volume | Purpose |
|---|---|---|---|
| **0 — Baseline** | BADAS-Open weights | — | Reproduce published numbers. Never train before you can reproduce. |
| **1 — Pretrain / warm start** | V-JEPA2 (MIT) → BADAS-Open (Apache-2.0) | — | Do not train a video backbone from scratch. |
| **2 — Task training** | Nexar collision prediction | 1,500 clips | Primary supervised signal — **and corpus-controlled, unlike CCD** |
| **3 — Augment** | DoTA + DADA-2000 with BADAS ego-involvement re-annotations | ~6,700 clips | Diversity and ego labels. **Research/benchmark only — YouTube pixels.** |
| **3b — Sanity corpus** | **CCD** | 4,500 clips | **Research only.** Useful as a leakage/confound teaching set and for the official-split comparison. Never in a shipped model. |
| **4 — Fine-tune** | **Your own UK footage** | 50–100 h, growing | Target-domain adaptation. The step nobody else can copy. |
| **5 — Evaluate** | **Your UK hard-negative benchmark** | 500+ curated clips | The actual product differentiator |
| **6 — Compound** | Pilot fleet events, human-verified | Growing weekly | The feedback loop that becomes the moat |

**Do not** use nuScenes, Waymo, CADP, LocateAnything-3B outputs, or (pending verification) BDD100K in any model that touches a customer.

### The critical gap nobody has filled

**There is no public dashcam crash dataset from UK or Western European roads.** Every corpus above is US, Taiwanese, East Asian, or mixed-international. UK roads differ in ways that matter to a vision model: left-hand traffic, roundabouts instead of four-way stops, different lane markings and signage, a different vehicle mix (higher van share), narrow rural lanes, and a distinct weather/lighting profile (persistent overcast, low winter sun, frequent rain).

**This gap is your opening, and it is why the answer to the dataset question is not "download more data."**

---

## 30. Recommended Training Strategy

### Step −1 — Falsify the current model (do this first, it takes one day)

You now have the pipeline to run all four cheaply, because features are cached.

| # | Test | Method | What a failure means |
|---|---|---|---|
| 1 | **Single-frame** | Replace each `(10, 1280)` sequence with one frame's features tiled 10×; retrain the head | AUC stays > 0.95 → no temporal information is needed; the LSTM is decoration |
| 2 | **Temporal shuffle** | Randomly permute the 10 feature vectors at evaluation time | Score barely moves → the LSTM contributes nothing |
| 3 | **Corpus control** | Evaluate on Nexar, where positives and negatives share a corpus | AUC collapses toward 0.5 → the CCD result was a corpus artefact (B4) |
| 4 | **Crash excision** | Score CCD positives using only frames before the first `binlabels == 1` | Still scores high → not keying on the collision |
| 5 | **Source-leakage count** | Count `youtubeID`s present on both sides of the seed-42 split | Any non-zero count quantifies B3 |
| 6 | **Always-negative baseline** | Predict 0 always, at realistic prevalence | If it beats you on FP/hour — and it will — that is the headline |

**Commit the results verbatim to the repository, whatever they say.** A founder who publishes their own negative result is a founder investors believe about everything else.

### Then, the real training sequence

**Step 0 — Reproduce before you improve.** Load BADAS-Open, evaluate on the Nexar test split, confirm approximately AP 0.86 / AUC 0.88. If your number differs materially, your harness is wrong and every subsequent experiment measures your bug. One day, non-negotiable.

**Step 1 — Establish honest baselines** on your own source-grouped splits: (a) BADAS-Open zero-shot, (b) your MobileNetV2+LSTM (measured properly, once, per [§28](#28-recommended-future-architecture)), (c) a trivial always-negative predictor.

**Step 2 — Fine-tune with the head split.** Replace the single collision head with collision / near-miss / ego-involvement. Freeze the V-JEPA2 backbone initially; train only the probe and heads — **the same frozen-backbone pattern your current pipeline already uses well.** Unfreeze the last blocks only if frozen training plateaus.

**Step 3 — Augment where it targets your failure modes**, in *pixel* space (fixing B6): weather and lighting simulation, motion blur, camera shake and ego-motion perturbation, horizontal flip (which also helps bridge right-hand to left-hand traffic), random temporal crops around the event.

**Step 4 — Fine-tune on UK footage.** Small dataset, low learning rate, heavy regularisation. Monitor for catastrophic forgetting against the global test set.

**Step 5 — Calibrate.** Temperature scaling on the dedicated calibration split. Report ECE. **An uncalibrated probability is not a probability.**

**Step 6 — Fit the fusion layer** on validation data only, never on training data.

**Practical constraints.** $200–500/month of spot cloud GPU is sufficient — BADAS-Open itself trained in ~8 hours on 4× A100. Do not buy hardware. Log every run (Weights & Biases free tier). **Seed everything** (B9). Commit a data manifest and refuse to train on any clip without a licence entry (D1).

---

## 31. Evaluation Framework

### Metrics that matter

| Metric | Why | Target for a shippable v1 | Your current value |
|---|---|---|---|
| **False positives per hour of driving** | **The metric that decides whether a fleet keeps the product.** | **< 0.1 FP/hour** | **≈23/hour (derived; see [§12](#12-evaluation-history))** |
| Precision @ fixed recall (e.g. R = 0.80) | Operating-point honesty | Report the full curve | Never computed |
| Average Precision (AP) | Comparable to BADAS / Nexar literature | ≥ 0.85 on Nexar | **0.5218 on Nexar test-public (T3)** — vs BADAS-Open's published 0.86 |
| ROC-AUC | Threshold-independent ranking | ≥ 0.88 | **0.5339 on Nexar test-public (T3)**; 0.9977 on the contaminated CCD split |
| **Time-to-detection after impact** | Latency for claims/notification | < 2 s | Never computed |
| **mTTA** for warnings | Comparable to BADAS's 4.9 s | Report, don't optimise blindly | **Not computable on Nexar test-public** — clips are truncated before the event (measured 2026-09-11). Needs UK footage. |
| **Expected Calibration Error** | Is 0.8 really 80%? | < 0.05 | Never computed |
| Per-condition breakdown | Where it fails | Night / rain / low-sun / urban / rural / motorway | Never computed — **though CCD ships the labels (B7)** |
| Per-crash-type breakdown | Coverage gaps | Rear-end / side / head-on / VRU / single-vehicle | Never computed |
| Ego-involved vs non-ego | The BADAS insight | Report separately, always | Never computed — **though CCD ships the label (B7)** |

**Banned from all reporting: raw accuracy.** With realistic class balance it is dominated by the majority class and will flatter you into shipping something broken. The 97% in [§12](#12-evaluation-history) is a good illustration: it sits beside a derived ≈23 false alarms per hour.

### Leakage prevention

Enforced in code, per the rules in [§16](#16-data-leakage-risks), with a test that fails the build on violation.

### Required comparisons, in every report

| Baseline | Why it must be there |
|---|---|
| **Always-negative** | At realistic prevalence it is a genuinely strong baseline on accuracy *and* on FP/hour. If you cannot beat it on FP/hour, you have no product. |
| **BADAS-Open zero-shot** | The free Apache-2.0 SOTA. Beating it is the claim; matching it is the floor. |
| **Your MobileNetV2+LSTM** | Measured once, honestly, on the same splits. This is how you retire it with evidence rather than assertion. |
| **Your new model** | The thing you are actually proposing |

---

## 32. Real-Time Architecture

```
┌──── VEHICLE / EDGE ────────────────────────────────────────────────┐
│                                                                    │
│  Camera 30 fps ──┐                                                 │
│  IMU 100 Hz ─────┼──► Ring buffer (30 s pre / 15 s post)           │
│  GPS 1 Hz ───────┘         │                                       │
│                            ▼                                       │
│              ┌──────────────────────────────┐                      │
│              │ STAGE 1 — CHEAP TRIGGER      │  runs always         │
│              │ IMU |Δa| threshold           │  <1 ms, ~0 W         │
│              │ + optical-flow divergence    │                      │
│              │ + RT-DETR every 5th frame    │  ~15 ms              │
│              └──────────────┬───────────────┘                      │
│                             │ candidate (rare)                     │
│                             ▼                                      │
│              ┌──────────────────────────────┐                      │
│              │ STAGE 2 — LEARNED MODEL      │  runs on trigger     │
│              │ V-JEPA2 distilled/quantised  │  INT8 TensorRT       │
│              │ 16 frames @ 256×256          │  ~40–80 ms Orin      │
│              │ EXPLICIT, SHARED STRIDE      │  ← prevents B1       │
│              └──────────────┬───────────────┘                      │
│                             ▼                                      │
│              ┌──────────────────────────────┐                      │
│              │ STAGE 3 — FUSION + CONFIRM   │  <1 ms               │
│              │ calibrated p, temporal gate  │                      │
│              └──────────────┬───────────────┘                      │
│                             ▼                                      │
│              EVENT + 45 s evidence clip → local encrypted store    │
│              (operates fully offline; uploads opportunistically)   │
└─────────────────────────────┬──────────────────────────────────────┘
                              │ 4G/5G/Wi-Fi, store-and-forward
┌──── CLOUD ───────────────────▼─────────────────────────────────────┐
│  Ingest API → object store (clip) + Postgres (event record)        │
│  Re-scoring queue → full-precision model → human review queue      │
│  Review UI → verified label → training corpus  ◄── THE FEEDBACK LOOP│
│  Fleet dashboard · Claims/FNOL evidence export · Webhooks/API      │
│  Model registry · drift monitoring · per-camera performance        │
└────────────────────────────────────────────────────────────────────┘
```

**Why two stages.** Running a ViT-L on every frame is neither affordable nor necessary. Crashes are extraordinarily rare — a fleet vehicle might see one per 50,000 km. A near-free always-on trigger followed by an expensive confirmation stage cuts compute by two to three orders of magnitude with negligible recall loss. **The IMU is what makes this work**, which is the concrete reason it earns its place in the sensor set.

**Hardware.** For a pilot, an NVIDIA Jetson Orin Nano (67 TOPS; launched at $249, street prices have risen — budget $250–400) is more than sufficient. For production at scale the economics push toward the SoC already inside the dashcam (Ambarella, Qualcomm) — a partnership conversation, not a hardware project. **Do not design your own hardware.**

**Optimisation path:** PyTorch → ONNX → TensorRT, INT8 quantisation with a calibration set, knowledge distillation of the ViT-L into a smaller student if latency demands it. Measure **end-to-end** latency (capture → event), not just model inference time.

**Network.** Assume connectivity is absent when it matters — crashes happen in tunnels, underground car parks and rural blackspots. Full offline operation with encrypted local buffering and store-and-forward upload is mandatory.

**Failure behaviour.** Explicit and fail-safe: if the model fails to load, log and continue recording; if the trigger stage crashes, restart it and record the gap; **never emit a verdict from a run that terminated abnormally** — the opposite of current behaviour (H3). Count and report every dropped frame and restart.

---

## 33. Data Collection Strategy

Public data gets you to a demo. It never gets you to a defensible company, because competitors can download the same files.

### Phase 1 — Self-collected UK footage (weeks 5–8, cost: low)

1. **Buy a dashcam and pay a UK-based driver.** A consumer dashcam (£80–150) plus a paid arrangement with a UK driver — delivery, driving instructor, rideshare — for 4 weeks of continuous recording yields 100+ hours of exactly the right distribution. Written consent and a clear data agreement from day one. **The highest-value £500 you will spend.**
2. **UK dashcam enthusiast communities.** Useful for *sourcing candidate incidents*, but the annotations-vs-pixels problem applies to community footage too — obtain explicit written permission per clip.
3. **Driving schools.** UK instructors already run dashcams for liability, drive continuously in exactly the difficult urban and rural conditions you need, and are small enough to sign a one-page agreement. Also an under-appreciated route to *near-miss-rich* footage, because learner drivers generate more of them.

**Target:** 100 hours of UK footage, of which you personally review and label ~10 hours.

### Phase 2 — The hard-negative benchmark (weeks 4–10, cost: your time)

See [§34](#34-hard-negative-benchmark).

### Phase 3 — Pilot data (months 3–9)

Two or three UK fleets, 30–300 vehicles each. Offer the pilot free. **What you are buying with that free pilot is not revenue — it is the right, in writing, to retain and label detected events.** Structure the DPA so that:

- You are a **processor** for the fleet's raw video (their data, their retention, their access controls)
- You are a **controller** for the **derived, de-identified event records** you keep for model improvement
- Retention, deletion and audit obligations are explicit
- The fleet gets a genuine benefit: incident review time reduced, claims evidence packaged automatically

Get a UK-qualified data protection lawyer to draft this once. **This clause determines whether you legally own a data moat.**

### Phase 4 — Partner data (months 9–18)

UK video-telematics vendors and dashcam manufacturers have installed bases and no strong in-house crash-detection research. A revenue-share or per-seat licence gives you distribution and data simultaneously. Approach them **only** after Phase 3 produces reference customers.

### Synthetic data: a deliberate "mostly no"

- **CARLA-generated collisions** ✗ — the sim-to-real gap on impact appearance is severe, and the visual signature of a real collision (deformation, debris, violent camera motion, sensor saturation) is exactly what simulators render worst.
- **Weather/lighting augmentation of real footage** ✓ — cheap, effective, already part of BADAS's recipe. **And it directly fixes B6.**
- **Ego-motion and camera-shake augmentation** ✓ — targets the dominant false-positive mode.
- **Synthetic hard negatives via clip mixing** ✓ — splice normal driving with the motion profile of a hard brake.
- **Generative video models for crash synthesis** ✗ — not reliable enough, and provenance of generated training data is a diligence liability.

**Rule: augment real data aggressively; synthesise scenes almost never.**

---

## 34. Hard-Negative Benchmark

The single most valuable artefact you can build in 90 days, and it is achievable alone. Curate **500+ UK clips that look like crashes and are not.**

```
UK-HN-500 Benchmark  ·  500 curated non-crash clips that superficially resemble crashes
  ├─ emergency braking            (75)   ← the classic false positive
  ├─ speed bumps / potholes       (50)   ← camera shake without collision
  ├─ roundabout close-quarters    (50)
  ├─ narrow rural passing         (40)
  ├─ bus/lorry pull-out           (40)
  ├─ heavy rain / spray / wipers  (50)
  ├─ low sun / tunnel transitions (45)
  ├─ debris / stone strikes       (30)
  ├─ car park manoeuvring         (40)
  ├─ pedestrian-crossing stops    (40)
  └─ motorcycle filtering         (40)   ← visually alarming, entirely normal

Headline: false-positive rate at the operating point that achieves 80% recall
          on the Nexar collision test split.
```

Every one of these should score *low*. Any commercial system that fires on these is unshippable, and **no public benchmark measures this.** When you can say *"we measured 27 competing configurations against 500 curated UK hard negatives and here are the false-positives-per-hour,"* you are having a completely different conversation from every other pre-seed CV founder.

**This benchmark is a deliverable in its own right. Publish it. Free.** It costs you nothing competitively — anyone could build one, nobody has — and it establishes you as the person who defined how this problem is measured in your target market. That is disproportionate credibility for the effort.

> **Sharpened by the new evidence:** you now have a *first-hand* story for why this benchmark matters. "Our first model scored 0.998 AUC on a public benchmark and would have produced roughly twenty-three false alarms an hour in a real vehicle — because the benchmark's positives and negatives came from two different corpora. That is why we built UK-HN-500." That is a far more compelling origin story than an abstract argument about evaluation rigour, and it is true.

---

## 35. Privacy / Legal / Compliance

**This section identifies risk areas. It is not legal advice, and several items genuinely require a qualified lawyer in the relevant jurisdiction.**

### Immediate — resolve before any customer conversation

| # | Issue | Risk | Action |
|---|---|---|---|
| 1 | **`ultralytics` AGPL-3.0** | Obligation to release the source of a hosted service | Migrate to RT-DETR (Apache-2.0) or buy an Enterprise Licence |
| 2 | **Training-data provenance — now known and problematic** | CCD positives are YouTube pixels; CCD negatives are BDD100K (personal-use licence) | Retrain from a manifest-tracked, licensed corpus. Do not ship the current model. |
| 3 | **Nexar licence ambiguity** | Your primary future dataset has three different licence descriptions | Email Nexar in week 1; get the operative terms in writing |
| 4 | **YouTube-derived academic datasets** | Annotations licensed; pixels not | Research/benchmark only; never in a shipped model |
| 5 | **No `LICENSE` file** | The README has claimed MIT; no file exists | Add one, or make the repository explicitly private |
| 6 | **Fault-determination output** | Product-liability exposure | Delete `FaultDetector` and `EgoZone.fault` (C7) |

### Privacy — UK/EU deployment

| Area | Consideration |
|---|---|
| **Lawful basis** | Likely legitimate interests for fleet safety, but requires a documented Legitimate Interests Assessment. Driver-facing cameras raise employee-monitoring issues treated seriously in the UK and EU. |
| **DPIA** | Systematic video monitoring of individuals almost certainly requires a Data Protection Impact Assessment. Do this before the first pilot, not after. |
| **Faces and number plates** | Personal data. Blur by default at the edge for anything retained for model training; keep unredacted footage only for the specific incident, for the minimum period, under access control. |
| **Biometrics** | Driver-facing monitoring is likely special-category data — a stricter regime. Avoid in v1. |
| **Retention** | Define and enforce per purpose: e.g. 7 days rolling, 90 days for incidents, indefinite only for de-identified derived records with a documented basis. |
| **Controller/processor split** | Processor for the fleet's raw video; controller for your de-identified derived event records. **This clause determines whether you legally own a data moat.** |
| **Cross-border transfer** | Processing UK/EU video in India engages UK GDPR Chapter V. May require an International Data Transfer Agreement / SCCs plus a transfer risk assessment. **Design your architecture around this** — consider UK/EU-region processing from the start. |
| **Germany specifically** | Continuous recording is a technical data-protection violation under the 2018 BGH ruling; Austria bans dashcams. A further reason Germany is not the first market. |
| **India (DPDP Act 2023 / DPDP Rules 2025)** | Partially in force from 13 November 2025, full effect expected by 13 May 2027; consent-based regime with notice, breach-reporting and record-keeping duties. Relevant to you as an India-based operator. |

### Product-liability posture

Never claim fault determination (C7). Never claim safety-critical readiness — even BADAS-Open explicitly disclaims certification for safety-critical applications. Put clear disclaimers in the product and the contract: this is an advisory and evidentiary tool, not a safety system, and not a substitute for human judgement. Have a lawyer draft your terms of service before the first paid deployment.

### Where you need a lawyer, specifically

1. US incorporation and, if applicable, the India→US flip (FEMA/ODI, tax, valuation).
2. UK/EU data protection: DPIA, DPA templates, international transfer mechanism.
3. **Dataset licence review before commercial training — especially Nexar, BDD100K, and the status of any model already trained on CCD.**
4. Product liability and terms of service.
5. Employment/monitoring law if you add driver-facing cameras.

---

## 36. Partnership Strategy

| Organisation type | Why they would engage | What you offer | What you request | Pilot shape | Agreement needed |
|---|---|---|---|---|---|
| **Driving schools (UK)** | Liability evidence; free tooling | Free incident review, free footage storage | Consented footage for training | 4 weeks, 3–5 vehicles | Simple consent + data agreement |
| **Mid-market fleets** | Reduce review time; lower premiums; claims defence | Free 90-day pilot, measured FP rate | Right to retain de-identified events | 90 days, 20–50 vehicles | **DPA** (processor for raw, controller for derived) |
| **Video-telematics vendors** | Better detection than they can build; no R&D cost | Model as a licensable component, on-prem or API | Access to installed base + anonymised events | Shadow-mode A/B on their existing feed | Commercial licence + data-sharing |
| **Dashcam manufacturers** | Product differentiation | Edge-optimised model for their SoC | Distribution + telemetry | Reference implementation on their hardware | Licence + NDA |
| **Insurers / MGAs / brokers** | Faster FNOL, lower claims leakage, better risk pricing | Verified incident records, near-miss risk scores | Anonymised claims outcomes to validate severity | Retrospective study on historical claims | DPA + statistical validation agreement |
| **Universities (UK transport safety)** | Publications, access to novel data | Co-authorship, tooling, your UK benchmark | Credibility, annotation capacity, intros | Joint benchmark paper | Research collaboration |
| **Emergency services** | Faster, better-informed response | Structured incident data | Institutional validation | Data-sharing study, not a product | MoU |
| **Automotive Tier-1s** | Sourcing external innovation | Validated tech + field data they lack | Design-win path | Technical evaluation | NDA → evaluation agreement |

**Approach order — and this matters more than the list:** driving schools → fleets → telematics vendors → insurers → Tier-1s. Each rung supplies the proof required for the next. Skipping rungs is the most common way this kind of startup wastes a year: you get a polite meeting with a Tier-1, have nothing they cannot get elsewhere, and are never called back.

---

## 37. Automotive Partnership Roadmap

| Stage | Who | **Evidence you must already have** | Realistic timing |
|---|---|---|---|
| 0 | Yourself | Reproduced BADAS-Open numbers; UK-HN-500 built; falsification results published | Month 1–3 |
| 1 | Driving schools, individual drivers | 100 h of consented UK footage; measured FP/hour | Month 2–4 |
| 2 | 2–3 mid-market UK fleets | 90-day pilot; FP/hour under tolerance in production; signed DPA; a customer who will take a reference call | Month 4–9 |
| 3 | UK video-telematics vendor | Two paying references; model packaged for on-prem/edge; shadow-mode results on *their* data beating their incumbent | Month 9–15 |
| 4 | Commercial motor insurer / MGA | Retrospective study on their historical claims showing measurable reduction in claims leakage or handling time | Month 12–20 |
| 5 | **Tier-1 supplier** (Bosch, Continental, ZF, Denso, Aptiv) | >100k validated vehicle-hours; cross-country generalisation *without retraining*; functional-safety awareness (ISO 26262 / SOTIF ISO 21448); an eval methodology they cannot replicate | Month 18–36 |
| 6 | **OEM** | A Tier-1 relationship, or a data asset no OEM can self-generate | Month 30+ |

**The uncomfortable truth about Tesla specifically.** Tesla is vertically integrated, runs its own fleet-scale data engine, and has one of the largest real-world driving corpora in existence. They do not need a third-party crash detector and will not license your data. **Tesla is not a customer and not a partner — at best, eventually, an acquirer.** Design for the companies that *lack* an internal data engine: Tier-1 suppliers, fleet operators, insurers, dashcam manufacturers.

**What actually earns a serious automotive meeting:** field data they cannot generate themselves, plus a validation methodology they respect. Not a demo. Not a model. Not an accuracy number. *"We have 100,000 validated vehicle-hours across the UK, Ireland and the Netherlands, and a hard-negative benchmark that no public dataset covers"* gets a meeting. *"Our model achieves 95% accuracy"* does not.

---

## 38. Accelerator Strategy

### Current terms (verify at application time — these change)

| Programme | Investment | Equity / instrument |
|---|---|---|
| **Y Combinator** | $500,000 | $125k for 7% on a post-money SAFE + $375k on an uncapped MFN SAFE |
| **Techstars** | $220,000 | $20k for 5% common + $200k uncapped MFN SAFE |
| **500 Global** | $150,000 | ~6% (flagship programme) |
| **Antler** | ~$250,000 | ~9% (US), with co-founder matching |

YC's Winter 2027 cycle runs January–March in San Francisco, with an on-time deadline of 2 November and decisions on 11 December; Spring, Summer and Fall batches are available via Early Decision.

### Incorporating in the US while based in India

**YC does not formally require a Delaware C-corp at application**, but institutional investors expect one and YC guides accepted companies through incorporation before funding. For non-US founders, Delaware C-corp formation is a **remote process** — no US address, bank account or visa required at formation, and no citizenship or residency restrictions on shareholders, directors or officers. Confirm current specifics with a qualified advisor.

**Where a lawyer is genuinely required, not optional:**
- The **"flip"** — incorporating an Indian entity first and later restructuring under a US parent has real Indian regulatory and tax consequences (FEMA, ODI/FDI, transfer pricing, valuation). **Take advice before incorporating anything, not after.**
- **Founder tax residency** and how equity is held.
- **Cross-border data transfer** — UK/EU customer video processed in India engages UK GDPR Chapter V. This affects your architecture, not just your paperwork.

**Does applying from India while targeting the UK/US matter?** Less than founders fear, and in one way it helps: YC and Techstars fund distributed teams routinely, and "Indian engineering cost base, Western revenue" is a well-understood structure. What matters is that your **customers and traction are in the target market**.

### What makes your application weak *today*

1. No users, no customers, no pilot, no letters of intent.
2. No valid measured results — and now you can say precisely *why* they were invalid, which is better than not knowing, but is not traction.
3. A model that a free Apache-2.0 model outperforms.
4. No co-founder. YC funds solo founders but the bar is visibly higher.
5. No evidence of customer discovery — zero recorded conversations with fleet operators.
6. A demo that would fail live on unseen footage.
7. An unresolved AGPL dependency at the heart of the product, and an unresolved dataset-provenance problem underneath the model.

### What would make it compelling

1. **"We ran a 90-day pilot with a UK fleet of 40 vehicles. False alarms fell from 12/day to 0.4/day. They signed a paid contract."** — this single sentence outweighs everything else combined.
2. A **public UK hard-negative benchmark** you created, that others start citing.
3. **Measured superiority over BADAS-Open on your benchmark** — credible precisely because you are honest that BADAS is the baseline.
4. 30+ documented fleet-operator interviews with the pattern you found.
5. A live demo on footage the audience supplies.
6. A clean licence chain you can explain in one slide.
7. **A published write-up of the CCD corpus-confound finding.** Founders who publish their own falsified result are rare and memorable, and this one is genuinely instructive.

### What you must not claim

Do not claim Tesla-level anything. Do not claim partnerships that are conversations. Do not claim accuracy figures from a 3-video test set — **or from the CCD validation split, which is contaminated**. Do not claim fault determination. Do not claim safety-critical readiness. **Investors forgive an early-stage company for being early. They do not forgive overstatement, because it tells them what diligence will feel like.**

### Recommended timing

**Do not apply now.** Apply to the batch after you have a live pilot — realistically 4–6 months out. An application with one real pilot is dramatically stronger than a polished deck, and you only get a small number of credible attempts.

---

## 39. Investor Demo Strategy

**Principle: nothing pre-rendered, nothing cherry-picked, and lead with the failure mode.** Every CV founder shows a highlight reel. Showing your false-positive number first is memorable precisely because nobody does it.

### Minute-by-minute (12 minutes)

**0:00–0:45 — The problem, told through one number.** "UK fleets record millions of hours of dashcam footage. Almost none of it is watched. When a claim arrives six weeks later, the footage that would have settled it has been overwritten." One slide. No product yet.

**0:45–1:30 — Why this is not already solved.** eCall has handled emergency notification in new EU cars since 2018 and GSR2 mandates ADAS on new EU vehicles from July 2024 — **but none of that touches the retrofit commercial fleet, and none of it produces a claims-ready evidence record.** This slide proves you understand the regulatory landscape, which instantly separates you from founders who have not checked.

**1:30–2:15 — The result you falsified.** *(new, and put it early)* "Our first model scored 0.998 AUC on a public crash dataset. Then we noticed its positives came from YouTube and its negatives came from BDD100K — two different corpora. Corrected, the honest number was X, and the real-world false-alarm rate would have been about twenty-three an hour." **Nobody opens by disproving their own result.** It buys you total credibility for every number that follows, and it demonstrates exactly the evaluation discipline the rest of the deck depends on.

**2:15–3:00 — The honest baseline.** "Nexar open-sourced a state-of-the-art collision predictor under Apache 2.0. Here it is running. It is good. It is free. So the model is not our moat — and here is what is."

**3:00–5:00 — Live detection on unseen footage.** Ask the audience for a dashcam clip, or use a sealed clip revealed on the spot. Show the pipeline: detections → tracks → the multi-head probability curve rising → event boundary → severity band. Show the **calibrated** probability and say what calibration means.

**5:00–6:30 — The false-positive benchmark.** *(the differentiator)* Run UK-HN-500 live. Emergency braking: 0.04. Speed bump: 0.02. Roundabout close pass: 0.07. Actual collision: 0.93. Then the headline: **"0.08 false positives per hour. The naive baseline is 3.2. This benchmark did not exist before we built it."**

**6:30–8:00 — The product, not the model.** The incident record. The review queue. The evidence export. Time a reviewer from alert to decision — under two minutes, on camera.

**8:00–9:00 — Edge deployment.** The Jetson running it in real time. Pull the network cable; show it continue and then sync. Offline operation is what fleet buyers ask about first and what most demos skip.

**9:00–10:15 — Traction and data flywheel.** Pilot metrics, named fleet (with permission), hours processed, events verified, and the loop: more fleets → more verified events → better model → more fleets.

**10:15–11:00 — Market and the ladder.** UK £783M (2025) → £1.34B (2030) fleet telematics; US at $9.2B revenue in 2025 with 15.5M commercial vehicles as the expansion market. UK → EU → US → Tier-1.

**11:00–12:00 — The ask, and the risks you name yourself.** State the two things most likely to kill you and what you are doing about each. Founders who volunteer their risks are the ones investors believe about everything else.

**Tailoring.** For automotive partners, extend the edge and validation sections and add ISO 26262 / SOTIF awareness. For insurers, lead with claims-cycle time and evidence integrity. For YC, compress hard to 5 minutes and put the pilot number first.

---

## 40. Technical Moat

**An AI model is not a moat, and yours specifically is negative moat — a free Apache-2.0 alternative outperforms it today.**

| Candidate moat | Real? | Assessment |
|---|---|---|
| Model architecture | ❌ | Published, reproducible, commoditised by BADAS-Open |
| Model weights | ❌ | Replicable by anyone with the same public data |
| Public dataset access | ❌ | Anyone can download the same files |
| **Proprietary UK/EU incident corpus** | ✅ | **Strongest available to you.** Compounds; cannot be bought |
| **The UK hard-negative benchmark** | ✅ | Nobody has it. Cheap for you, valuable, and publishing it creates category authority |
| **Human-verified label flywheel** | ✅ | Every reviewed event improves the model; competitors cannot replay your pilot history |
| **Fleet/vendor integrations & switching costs** | ✅ | Slow to build, slow to displace — the classic B2B moat |
| **Evaluation methodology as reputation** | 🟡→✅ | *Upgraded.* You now have a first-hand, publishable case study in dataset-confound detection. Methodology credibility is a real asset in a field full of unfalsifiable accuracy claims. |
| Edge optimisation | 🟡 | Real engineering value; a well-funded team catches up in months |
| Clean licence provenance | 🟡 | Underrated. Genuinely matters at acquisition. Not enough alone |
| Insurance/claims workflow integration | ✅ | Deep, sticky, and the reason incumbents keep customers |
| Regulatory certification (later) | ✅ | Expensive, slow, therefore defensible — a year-3 asset |
| Geographic coverage | 🟡 | Real for a while; erodes as competitors expand |

**The three you can actually build in 12 months:**

1. **The UK/EU hard-negative benchmark** — weeks of work, immediate credibility, and it defines the metric your competitors must then answer.
2. **A human-verified incident corpus with a feedback loop** — starts at zero and compounds monthly. The only asset that gets *harder* to replicate over time.
3. **Fleet and vendor integration depth** — unglamorous, and the reason Samsara and Lytx retain customers who could switch.

**Sequencing matters more than choice.** The benchmark buys credibility, which buys pilots, which produce the corpus, which produces the flywheel. Attempting the flywheel first, without the benchmark, means you cannot convince anyone to give you a pilot.

---

# Part IV — What to Do

## 41. 90-Day Action Plan (Phases 0–11)

**The governing rule: no phase begins until the previous phase's acceptance criteria are met, in writing, in the repository.** The acceptance criteria are the point of this plan. A phase that "feels done" is not done.

> ### The governing rule orders each track, not the whole document  ·  **revised 2026-09-12**
>
> The phase numbers below are a **dependency graph, not a queue.** Read as a queue, they put the two cheapest, highest-value, least-blocked deliverables *behind* the most expensive and most-blocked one — which is the wrong shape for this project's goal and compute constraints. Three tracks run **concurrently from week 1**:
>
> | Track | Phases | Gated on | Why it runs in parallel |
> |---|---|---|---|
> | **A — Model & measurement** | 4 → 5 → 6 | The Phase 2 gate ✅ | The engineering path. Compute-bound, and under a no-GPU constraint the slowest and least certain of the three. |
> | **B — Data & benchmark** | 7 → 8 | A consent form and ~£500 | **No technical dependency on Track A.** §40 and §45 both name this as the moat. It is also the only route to a measurable headline metric: Nexar test-public holds **0.90 hours of negative footage**, and < 0.1 FP/hour cannot be demonstrated on 54 minutes at any confidence. |
> | **C — Customer discovery** | 30 fleet calls | Nothing at all | Zero code, zero cost, no dependencies. §45 Q2 and "If I were you" both treat this as the real critical path to a startup outcome. It was previously buried as task 5 of Phase 5. |
>
> **Track C's gate, since no phase owns it any more:** 30 UK fleet-operator calls written up, one question each — *"What happened the last time you trialled an AI dashcam?"* **Do not pitch.** The answers are the go-to-market.
>
> **Why this changed.** Tracks B and C were formerly sequenced after Track A. Nothing in Track A unblocks either of them, both produce evidence Track A cannot produce, and Track A now contains a phase (6) that this project's hardware cannot complete as originally written. Sequencing the moat last was the single biggest structural defect in this plan.

---

### PHASE 0 — Evidence recovery  ·  🟡 **MOSTLY COMPLETE (2026-09-10)**
> Done: notebook **committed** at `data/ccd/Untitled0.ipynb` (`c9a6fda`, outputs included); `Crash-1500.txt`, `train.txt`, `test.txt` in `data/`; Nexar licence + all 9 metadata CSVs in `data/nexar/`; falsification results in `runs/falsification/`; **U4 resolved by deletion** — the 493 MB of orphaned artefacts are gone from the tree (`ad45389`).
> Outstanding: rename the notebook in Drive · resolve U6 (`crash_model_cpu` vs the shipped weights — now one command, since a working TF 2.19 env exists) · email the CCD authors (U7) and Berkeley DeepDrive · write `runs/legacy-colab/README.md`.
**Objective:** move every off-repository artefact into version control and download the three CCD files that were never fetched, so that no future audit can reach a wrong conclusion for want of access.
**Priority: P0 · Effort: 0.5 day · Depends on: nothing**

**Exact tasks**
1. In the Colab tab: `File → Download .ipynb`. Commit as `train/colab_original.ipynb`, **outputs included** — they are the training history and they are evidence.
2. Rename the notebook in Drive from `Untitled0.ipynb` to `crash_lstm_ccd_v2.ipynb`.
3. Download from the CCD Google Drive: `videos/Crash-1500.txt`, `vgg16_features/train.txt`, `vgg16_features/test.txt`. Commit to `data/ccd/`.
4. Export the Colab cell-8 output verbatim to `runs/legacy-colab/training_log.txt` and the cell-9 report to `runs/legacy-colab/metrics_val.txt`. Add a `README.md` in that directory stating plainly that **these metrics do not describe `models/crash_model_weights.weights.h5`** and why.
5. Copy `best_crash_model.keras`, `crash_model_cpu/` and `feature_extractor.keras` out of Drive into a cold-storage archive (not git).
6. Record SHA-256 for every file in `models/`.
7. Resolve U6: load `models/crash_model_cpu/` and `models/crash_model_weights.weights.h5`, compare `get_weights()`, record the answer.
8. Resolve U4: `grep -rn "crash_detection_model" --include=*.py .` — if nothing loads them, delete the 493 MB.
9. Open the CCD provenance question with the authors (U7) and email Nexar about commercial terms (D1).

**Files affected:** `train/`, `data/ccd/`, `runs/legacy-colab/`, `models/`

**Expected output:** a repository containing the pipeline that made the model; three annotation files; an honest legacy-run record; two emails sent.

**Acceptance criteria**
- [x] A fresh clone contains the notebook — `data/ccd/Untitled0.ipynb` (`c9a6fda`). It landed in `data/ccd/` rather than `train/`, so `git log -- train/` is empty; that is a path choice, not an unmet criterion.
- [x] `data/Crash-1500.txt` exists and parses to 1,500 rows with 7 fields (path is `data/`, not `data/ccd/`)
- [ ] `runs/legacy-colab/README.md` explicitly states the artefact/metric disconnect
- [ ] U6 is answered in writing (U4 is answered by deletion, `ad45389`)
- [ ] Both emails are sent, with dates recorded

---

### PHASE 1 — Reproducibility
**Objective:** a fresh machine can clone, install, extract features and train, deterministically.
**Priority: P0 · Effort: 3 days · Depends on: Phase 0**

**Exact tasks**
1. Pin Python 3.11 (`.python-version`); pin `tensorflow==2.19.*` / `keras>=3.13` to match the artefact (R1, R2); generate a lockfile; write a `Dockerfile`; add `make setup / data / features / train / eval`.
2. Port the notebook to `train/`: `config.py`, `data.py`, `features.py`, `model.py`, `train.py`, `evaluate.py`. Keep the frozen-backbone-plus-cached-features pattern — it is sound.
3. **Seed everything** (B9): `keras.utils.set_random_seed(SEED)` plus op-determinism where affordable.
4. **Run isolation** (B8): checkpoint to `runs/{ts}-{git_sha}/` with `metrics.json`, `config.json`, `manifest.sha256`, `weights.sha256`.
5. Build `data/manifest.csv` from `Crash-1500.txt` — `clip_id, dataset, youtubeID, timing, weather, egoinvolve, licence, split, group_key` (B7, D1).
6. Add the first real tests (M4): weights-load smoke test, forward-pass shape test, manifest-completeness test.
7. Add CI running lint, tests and `pip-licenses` with an allowlist that fails on (A)GPL (C6).

**Files affected:** `train/*`, `Dockerfile`, `Makefile`, `requirements*.txt`, `.python-version`, `tests/`, `.github/workflows/`

**Expected output:** `make train` reproduces a run end-to-end from a clean container.

**Acceptance criteria**
- [ ] `docker build . && make train` succeeds from a fresh clone with no manual steps
- [ ] Two runs with the same seed produce test AUC matching to 4 decimal places (or the residual GPU nondeterminism is documented)
- [ ] Every run directory contains `metrics.json` and `weights.sha256`
- [ ] `pytest` passes in CI and CI fails on an AGPL dependency
- [ ] `train.py` exits non-zero if any training clip lacks a `licence` entry

---

### PHASE 2 — Current model validation  ·  🟡 **PARTIALLY COMPLETE (2026-09-10)**
> Done: T1 single-frame, T2 temporal shuffle, T5 source-leakage count, T6 always-negative, B1 stride measurement — see [`runs/falsification/RESULTS.md`](runs/falsification/RESULTS.md). **The model failed T1, T2 and T5.**
> **Done 2026-09-11: T3 corpus control — ROC-AUC 0.5339, AP 0.5218, FPR 97.6% on Nexar test-public (667 clips).** See [`runs/falsification/T3_corpus_control.md`](runs/falsification/T3_corpus_control.md). **The falsification suite is closed; every test run has failed. The gate below is satisfied: the model now has a measured AP, ROC-AUC and false-positive rate on a genuinely held-out, corpus-controlled benchmark it was not trained on.**
> **Dropped as dead work, with reasons:** T4 crash excision (would confirm via a fourth route a conclusion already established three ways); the source-grouped CCD split, the B1 stride fix and the `CNN_THRESH` derivation (all three exist only to evaluate or operate a CCD-trained MobileNetV2+LSTM, which §44 and T3 jointly retire — there is no operating point on a chance-level ranker). Phase 1's port of the Colab pipeline to `train/` is likewise dropped: the notebook is retained as the historical record, not as a pipeline to be maintained.
**Objective:** find out whether the model has ever detected a collision.
**Priority: P0 · Effort: 3 days · Depends on: Phase 1**

**Exact tasks**
1. Rebuild the split as **source-grouped** with a **frozen test partition** and a **separate calibration partition** (B2, B3). Commit the test manifest hash.
2. **Quantify B3 first:** count `youtubeID`s appearing on both sides of the original seed-42 split. Record the number.
3. Run all six falsification tests from [§30](#30-recommended-training-strategy) — single-frame, temporal shuffle, corpus control, crash excision, source-leakage count, always-negative baseline.
4. Fix B1: define `FRAME_STRIDE` in the shared config; make inference sample at the training stride; assert equality at startup.
5. Delete `CNN_THRESH` and every hard-coded score threshold (B5); derive the operating point from the test-split PR curve at a stated target recall; persist to `metrics.json`.
6. Measure the deployed weights, once, honestly, on the new frozen test split. Record the number whatever it is.
7. Write `runs/falsification/RESULTS.md` and commit it verbatim.

**Files affected:** `train/data.py`, `train/evaluate.py`, `code/crash_detection_enhanced.py`, `tests/test_split_leakage.py`, `runs/falsification/`

**Expected output:** a document that says, with evidence, whether the current model works.

**Acceptance criteria — the gate for everything downstream**
- [ ] Source-grouped, frozen test split exists, is hashed and is committed
- [ ] A leakage test fails the build on any cross-split group collision
- [ ] All six falsification results are committed verbatim, including unfavourable ones
- [ ] **The current model has a measured AP, ROC-AUC and FP/hour on a genuinely held-out source-grouped test set**
- [ ] The always-negative baseline is measured on the same split and reported beside it
- [ ] The train/inference stride assertion is in code and passing
- [ ] Zero hard-coded thresholds remain (`grep` proves it)

> **Do not proceed to Phase 5 until the current model's performance has been measured on a genuinely held-out source-grouped test set.** This is the single most important gate in the document.

---

### PHASE 3 — Dataset/licensing cleanup
**Objective:** be able to answer "what is your training data licensed under?" in one sentence, with a file to point at.
**Priority: P0 · Effort: 2 days + external response time · Depends on: Phase 0 (emails sent)**

**Exact tasks**
1. Record the Nexar response; classify the licence definitively (D1).
2. Record the CCD authors' response; mark CCD **research-only** in the manifest regardless.
3. Verify BDD100K's current terms directly with Berkeley DeepDrive.
4. Decide and execute on C6: RT-DETR spike (from `lyuwenyu/RT-DETR`, **not** via `ultralytics`) measured for detection parity on 100 frames, in parallel with an Ultralytics enterprise quote request.
5. Add a `LICENSE` file, or make the repository private (L4).
6. Delete `FaultDetector` and `EgoZone` (C7, C3).
7. Add the `licence`-gate to `train.py` (already scaffolded in Phase 1) and turn it on.

**Files affected:** `data/manifest.csv`, `LICENSE`, `requirements.txt`, `code/`, `docs/licensing.md`

**Expected output:** a one-page `docs/licensing.md` with the chain from every dataset and dependency to a licence, dated and sourced.

**Acceptance criteria**
- [ ] Every dataset in the manifest has a `licence` and a `licence_verified_date`
- [ ] No AGPL dependency remains, **or** an Enterprise Licence is purchased and recorded
- [ ] `LICENSE` exists and is consistent with the dependency tree
- [ ] `grep -rn "at_fault\|FaultDetector\|EgoZone"` returns nothing
- [ ] No model intended for a customer is trained on CCD or BDD100K

---

### PHASE 4 — Evaluation harness
**Objective:** a single command that produces every metric that matters, for any model, on any split.
**Priority: P0 · Effort: 4 days · Depends on: Phase 2**

**Exact tasks**
1. `eval/` package: source-grouped splitter, leakage test (CI-enforced), metric suite — AP, ROC-AUC, precision @ fixed recall, **FP/hour (always with its denominator)**, ECE.
   ⚠️ **`time-to-detection` and `mTTA` are NOT computable on Nexar test-public** and are excluded from this phase. Measured 2026-09-11: `time_of_event` lies beyond the distributed clip for **all 334 positives** (median 20.0 s vs 9.93 s clip), and the clip's offset into the original video is not in the shipped metadata. Timing metrics need a split that carries usable timestamps — the UK footage of Phases 7–8.
2. Per-condition breakdown driven by the manifest: `timing`, `weather`, and later road type (B7).
3. Ego-involved vs non-ego reporting (B7).
4. Reliability diagrams and PR curves written to `runs/<id>/plots/`.
5. A model-agnostic adapter interface so BADAS-Open, your old model and any new model are all evaluated by identical code.
6. **Ban accuracy from all reports** — assert it is absent from `metrics.json`.

**Files affected:** `eval/*`, `tests/test_leakage.py`, `Makefile`

**Expected output:** `make eval MODEL=<adapter> SPLIT=test` → `metrics.json` + plots, for any model.

**Acceptance criteria**
- [ ] Three different models evaluate through one code path
- [ ] `metrics.json` contains AP, AUC, FP/hour, ECE, per-condition and ego/non-ego rows
- [ ] The leakage test fails the build on an injected violation
- [ ] Raw accuracy appears nowhere in any generated report

---

### PHASE 5 — BADAS-Open reference baseline  ·  **TRACK A**
**Objective:** a defensible BADAS-Open number on *your* split, with every deviation from the published setup named. **Not** a bit-for-bit reproduction of someone else's figure — see the gate note.
**Priority: P0 · Effort: 2 days for the harness path, plus ~18 h of unattended sweep compute (measured end-to-end, see task 2) · Depends on: Phase 4's adapter, and the Phase 2 gate**

> #### The gate was restated on 2026-09-12, and this is the most important revision in Part IV
>
> It previously read: *"within ~0.02 AP of the named published figure. If it is not, the harness is wrong — stop and fix it. Do not proceed."* **That gate is unpassable by construction**, for five reasons that have nothing to do with your harness being wrong:
>
> 1. **The published number is disputed.** Model card: AP 0.86 / AUC 0.88. BADAS's own `config.json`: Nexar AP 83.2 / AUC 0.85.
> 2. **The split differs.** Both published figures are on the full 1,344-clip test set; you hold the 667-clip public half.
> 3. **The published inference code computes the future-prediction pathway and then throws it away.** `EnhancedVideoClassifier.forward()` consumes only `last_hidden_state` (the encoder output) and discards `predictor_output`, even though training used `predictor_combination_method: "concat"` with `future_prediction_seconds: 1.0`. So the pathway is loaded, executed, billed for ~25% of every window, and ignored. **Measured 2026-09-12 — and note this corrects the earlier reading of the same symptom:** the predictor is **not** an unimplemented module and **no weights are lost**. It is V-JEPA2's own predictor, already in `transformers` as `VJEPA2Model.predictor`, and the checkpoint stores it **twice** — embedded at `backbone.predictor.*` *and* duplicated at a top-level `predictor.*`. The embedded copy loads correctly (`load_state_dict` reports **missing 0**); the 199 "unexpected" keys are the **bitwise-identical duplicate** (verified with `torch.equal` across all 199 pairs). Evidence: `scripts/badas_predictor_probe.py`.
> 4. **Clip-score reduction is ambiguous in their own code** — `per_video` uses mean; `cli.py` and the example use max.
> 5. **`original_fps: 4` contradicts `target_fps: 8.0`** in the config, and is unresolved.
>
> **And the gate's purpose is already served by other evidence.** It exists to prove the harness is correct. `eval/benchmark.py` reproduces the committed T3 figures exactly, and the T3 scoring path was itself falsified against the three local videos (`safe.mp4` → **0.7914**, reproducing the original Keras pipeline's 0.79) *before* its result was accepted. The harness has been validated twice, independently of BADAS. **Do not block the project on a number nobody can currently define.**

**Exact tasks**
1. Obtain the Nexar collision-prediction dataset. test-public is on disk (667 clips, verified). Consider test-private (677 clips, ~3 GB) for a like-for-like 1,344-clip comparison.
2. ~~Time one forward pass before committing to any sweep (U-B6).~~ **ANSWERED 2026-09-12 — the phase is a GO, at ~18 h.** Two figures, and the difference between them matters:
   - **Compute only: 0.856 s/window** on MPS at 16×224×224 (`scripts/badas_smoke.py`, which feeds a synthetic tensor and deliberately bypasses video IO). `skip_predictor=True` gives **0.629 s/window**, with `last_hidden_state` bit-identical — so the no-predictor baseline is **~25% cheaper on compute for free**.
   - **End-to-end: ~97 s/clip measured** over a 6-clip run (`eval/run_baselines.py --limit 6`, 583.7 s). Decode plus `VJEPA2VideoProcessor` adds ~1.8× on top of compute. **A full 667-clip sweep at stride 1 is therefore ≈18 h, not the ≈10 h the compute-only figure implies.** Quote the end-to-end number when planning; the per-window number is for comparing model configurations only.
   
   18 h is two overnight runs or one long one — feasible, and not a blocker. **Stride 2 halves it to ≈9 h** and is legitimate **only if the deviation is recorded in `metrics.json`**. Model load is **8.5 s**, not the ~9 minutes previously recorded (that was a one-time HF download).
3. **Consume `predictor_output` in the forward pass** (gate note, reason 3). This is a **forward-path change, not a reimplementation** — the module and its trained weights are already loaded. Apply `predictor_combination_method: "concat"`; **do not guess the concat axis** — the token axis is the hypothesis, the feature axis is ruled out because `temporal_processor` takes 1024, not 2048. Evaluate **with and without** and report both: the delta measures what the published code discards.
4. Name the authoritative published figure in `metrics.json`, and record beside it **every** deviation: split size, stride, score reduction (use `np.nanmax`, not builtin `max` — the first 16 frames are NaN), predictor present/absent, fps.
5. Record the full baseline table: BADAS-Open zero-shot · your MobileNetV2+LSTM (**already measured: AP 0.5218 / ROC-AUC 0.5339**) · always-negative.

**Files affected:** `eval/adapters/badas.py`, `runs/baselines/`, `vendor/badas-open/` (patched, every patch marked as a vendored-upstream change)

**Expected output:** `runs/baselines/metrics.json` with three models on one split, plus a named deviation list.

**Acceptance criteria**
- [ ] A forward-pass timing is recorded, and the chosen stride is justified against it
- [ ] BADAS-Open is evaluated **both with and without** the predictor path, and both numbers are recorded
- [ ] `metrics.json` names which published figure it is compared against, and lists every deviation from that setup
- [ ] All three baselines are recorded on identical splits with identical code
- [ ] **If the BADAS AP lands materially below the named published figure, the deviation list explains why — or the harness is wrong and you fix it.** This is a judgement call on documented evidence, *not* a hard numeric stop.

---

### PHASE 6 — Match Channel A, then beat it as a system  ·  **TRACK A**
**Objective:** the outputs the product needs, and a *system* that beats BADAS-Open zero-shot even where the learned channel only matches it.
**Priority: P1 · Effort: 5 days for the CPU-feasible parts; probe training is gated on M4 Max access · Depends on: Phase 5**

> **Reframed 2026-09-12 (decision D2).** This phase previously required *"a model that beats the baselines."* **That is not achievable under this project's compute constraints, and it must not sit in the critical path as though it were.** There is no CUDA device — an M1 laptop, plus an M4 Max at college. Full V-JEPA2 backbone fine-tuning is out of reach; **probe/head training on cached frozen features is the supported path.** BADAS-Open was trained with more data and more compute than you have, so out-AP-ing its backbone is an unlikely outcome and a poor objective.
>
> **What you can actually win on, and therefore what this phase is for:** the **multi-head split** (collision / near-miss / ego-involvement — what the product needs and what Nexar collapses), **calibration**, and **Channel B false-positive suppression**. §28 already argues Channel B is the differentiator rather than the baseline. **The target is a system number, not a backbone number.**

**Exact tasks**
1. V-JEPA2 + attentive probe, warm-started from BADAS-Open.
2. **Multi-head:** collision / near-miss / ego-involvement (B7).
3. Explicit shared `FRAME_STRIDE`, enforced by assertion (B1 cannot recur).
4. Pixel-space augmentation targeting known failure modes (B6): weather, lighting, motion blur, camera shake, horizontal flip.
5. Temperature scaling on the dedicated calibration split; report ECE.
6. Fix C1 (real camera calibration); resurrect Channel B; fit the interpretable fusion layer on validation only.
7. Replace the tracker with ByteTrack (H2); replace or license the detector (C6).

**Files affected:** `train/`, `code/` (single surviving pipeline), `eval/`

**Acceptance criteria**
- [ ] The learned channel **matches** BADAS-Open zero-shot within a stated band on the frozen test split, or the gap is documented and explained
- [ ] **The fused system (Channel A + Channel B) beats BADAS-Open zero-shot on FP/hour at equal recall** — this is the phase's real claim, and the only one the hardware permits
- [ ] ECE < 0.05
- [ ] Collision and near-miss are reported as separate heads
- [ ] The metric output passes the `fy / frame_height ∈ [0.7, 1.5]` calibration assertion
- [ ] The fusion layer is interpretable and its coefficients are published in the run directory

---

### PHASE 7 — UK data collection  ·  **TRACK B — starts week 1, in parallel with Track A**
**Objective:** own footage nobody else has.
**Priority: P0 · Effort: ongoing from week 1 · Depends on: a consent form and a data agreement. NOT on Phases 4–6.**

> **Re-sequenced 2026-09-12.** This was *"ongoing from week 5, depends on Phase 3."* It has **no technical dependency on any model phase**, it is the moat per §40, and it is the only route to a measurable FP/hour: Nexar test-public holds **0.90 hours of negative footage**, so the plan's own headline target of < 0.1 FP/hour is undemonstrable on it at any confidence. **Start this first, not fifth.** The consent form and data agreement are the only genuine prerequisite — a day of work plus a lawyer's review, which is why they should be commissioned in week 1 rather than waited on.

**Exact tasks**
1. Buy a dashcam; arrange a paid UK driver and/or 3–5 driving schools.
2. Consent form and data agreement in place **before** the first recording.
3. Ingest pipeline with edge-side face and number-plate blurring for anything retained for training.
4. Fix B12 (reliable frame indexing) before processing long clips.
5. Target 100 hours; personally review ~10.

**Acceptance criteria**
- [ ] **First ~20 hours landed by week 4** — enough to begin Phase 8 curation, which does not need the full 100
- [ ] ≥ 100 hours of UK footage with written consent on file
- [ ] Redaction verified on a sample
- [ ] Every clip has a manifest row with `consent_status`

---

### PHASE 8 — Hard-negative benchmark  ·  **TRACK B**
**Objective:** UK-HN-500, published.
**Priority: P0 · Effort: 10 days spread over weeks 3–10 · Depends on: Phase 7's first ~20 hours — not its full 100**

> **Re-sequenced 2026-09-12.** Curation begins on a **partial** corpus; there is no reason to wait for 100 hours before categorising the first hard negatives. §34 calls this *"the single most valuable artefact you can build in 90 days,"* and it was previously gated behind the slowest track in the plan.

**Exact tasks**
1. Curate 500 clips across the eleven categories in [§34](#34-hard-negative-benchmark).
2. Human-label every clip; record the category and why it is a hard negative.
3. Wire it into the harness as a first-class benchmark.
4. Publish it with a short methodology note — including the CCD corpus-confound story.

**Acceptance criteria**
- [ ] 500 clips, all categories populated to target counts
- [ ] Every candidate model reports **FP/hour on UK-HN-500** at the operating point that achieves 80% recall on Nexar
- [ ] The benchmark is published with a documented methodology

---

### PHASE 9 — Product prototype
**Objective:** turn scores into incident records a human can act on.
**Priority: P2 · Effort: 10 days · Depends on: Phase 6**

**Exact tasks**
1. Incident schema per [§27](#27-product-definition), with `t_start` / `t_peak` / `t_end` (enabled by B7).
2. Postgres + object store; ingest API; evidence export (clip + metadata + hash).
3. Minimal review UI: queue, playback, accept/reject. **Reject feeds the training corpus — the flywheel starts here.**
4. Fix H3 and H6 in the shipped service path.

**Acceptance criteria**
- [ ] Video in → JSON incident record out, via API
- [ ] A reviewer can go from alert to decision in under two minutes, timed
- [ ] A rejected event lands in the training corpus automatically
- [ ] No verdict is ever emitted from an abnormally-terminated run

---

### PHASE 10 — Pilot
**Objective:** one UK fleet running it for 90 days.
**Priority: P2 · Effort: 10 days + 90 days elapsed · Depends on: Phases 8, 9**

**Exact tasks**
1. Jetson Orin Nano: ONNX → TensorRT, INT8; measure **end-to-end** latency.
2. Two-stage trigger with IMU; prove offline operation by pulling the network.
3. Sign pilot #1 (unpaid is fine — the DPA and data rights are what matter).
4. Deploy to 5–10 vehicles. Instrument everything. Review false positives daily.

**Acceptance criteria**
- [ ] Signed DPA with the controller/processor split correct
- [ ] 90 days continuous running
- [ ] **FP/hour below the fleet's stated tolerance, measured in production**
- [ ] A written before/after on incident review hours
- [ ] The customer will take a reference call

---

### PHASE 11 — Scale
**Objective:** second and third customers, and the flywheel turning.
**Priority: P3 · Effort: ongoing · Depends on: Phase 10**

**Exact tasks:** second and third pilots · vendor partnership conversations · SOC 2 path · model registry and drift monitoring · accelerator application · Ireland/Netherlands expansion.

**Acceptance criteria**
- [ ] Two paying reference customers
- [ ] >10,000 human-verified incidents
- [ ] Model performance holds in a second country without retraining

---

## 42. Priority Matrix

| # | Task | Impact | Effort | Risk | Depends on | Priority | Est. |
|---|---|---|---|---|---|---|---|
| 1 | **Commit the Colab notebook** | **Critical** | Trivial | None | — | **P0** | 5 min |
| 2 | Download `Crash-1500.txt` / `train.txt` / `test.txt` | **Critical** | Trivial | None | — | **P0** | 15 min |
| 3 | Email Nexar re commercial licence | **Critical** | Trivial | Low | — | **P0** | 30 min |
| 4 | Six falsification tests | **Critical** | Low | Low | 1, 2 | **P0** | 1 d |
| 5 | Fix B1 (temporal stride) | **Critical** | Low | Low | 1 | **P0** | 0.5 d |
| 6 | Source-grouped split + frozen test set | **Critical** | Med | Low | 2 | **P0** | 1 d |
| 7 | Reproducible env (lockfile + Docker + correct TF) | **Critical** | Low | Low | — | **P0** | 1 d |
| 8 | Delete hard-coded thresholds; derive from PR curve | **Critical** | Low | Low | 6 | **P0** | 0.5 d |
| 9 | Resolve AGPL / migrate to RT-DETR | **Critical** | Med | Med | — | **P0** | 3 d |
| 10 | Evaluation harness + leakage CI | **Critical** | Med | Low | 6, 7 | **P0** | 4 d |
| 11 | Delete `FaultDetector` + `EgoZone` | **Critical** | Trivial | None | — | **P0** | 1 h |
| 12 | Dataset manifest + licence gate | **Critical** | Med | Low | 2 | **P0** | 1 d |
| 13 | Reproduce BADAS-Open baseline | **Critical** | Low | Med | 10 | **P0** | 2 d |
| 14 | Run isolation + seeding (B8, B9) | High | Low | Low | 7 | **P1** | 0.5 d |
| 15 | Delete forks + dead subsystems (H5, C2, H1, M1) | High | Low | Low | 7 | **P1** | 2 d |
| 16 | UK footage collection started (**Track B, week 1**) | **Critical** | Med | Med | — | **P0** | 5 d |
| 17 | UK-HN-500 benchmark (**Track B**) | **Critical** | High | Low | 16 (first ~20 h) | **P0** | 10 d |
| 18 | Customer discovery (30 calls) (**Track C, week 1**) | **Critical** | Med | Low | — | **P0** | ongoing |
| 19 | Camera calibration fix (C1) | High | Med | Low | 15 | **P1** | 3 d |
| 20 | Multi-head fine-tune + calibration | High | Med | Med | 13, 16 | **P1** | 5 d |
| 21 | Parse CCD annotations into the manifest (B7) | High | Low | Low | 2 | **P1** | 0.5 d |
| 22 | Fusion layer | High | Med | Med | 19, 20 | **P2** | 4 d |
| 23 | Incident schema + storage + API | High | Med | Low | 20 | **P2** | 5 d |
| 24 | Review UI + label flywheel | High | Med | Low | 23 | **P2** | 5 d |
| 25 | Proper tracker (ByteTrack, H2) | Med | Low | Low | 15 | **P2** | 2 d |
| 26 | Pixel-space augmentation (B6) | Med | Med | Low | 20 | **P2** | 2 d |
| 27 | Edge deployment (Jetson/TensorRT) | Med-High | High | Med | 20 | **P2** | 7 d |
| 28 | Investor demo build | High | Med | Low | 17, 24 | **P2** | 4 d |
| 29 | Pilot #1 signed + DPA | **Critical** | Med | High | 18, 28 | **P2** | 10 d |
| 30 | Delete orphaned 493 MB (B13) | Low | Trivial | Low | — | **P3** | 15 min |
| 31 | Letterbox resize (B11), reliable seeking (B12) | Low | Low | Low | 7 | **P3** | 0.5 d |
| 32 | SOC 2 / security hardening | Med | High | Low | 29 | **P4** | later |

---

## 43. Exact Immediate Actions

> **⚠️ This list was written 2026-09-10 and items 1–8 are now done or deliberately dropped. Superseded 2026-09-12 — see the three-track block at the head of [§41](#41-90-day-action-plan-phases-011).** Kept verbatim below as the record of what the plan asked for at the time.
>
> | # | Status |
> |---|---|
> | 1 · commit the notebook | ✅ done — `data/ccd/Untitled0.ipynb` (`c9a6fda`) |
> | 2 · download the three CCD files | ✅ done — `data/{Crash-1500,train,test}.txt` |
> | 3 · email Nexar re licence | ✅ moot — licence retrieved directly to `data/nexar/LICENSE`, commercial use permitted |
> | 4 · count split leakage | ✅ done — T5, 113/133 sources, 91.4% of clips |
> | 5 · shuffle + single-frame tests | ✅ done — T1/T2, both failed |
> | 6 · pin the environment + smoke test | 🟡 partly — TF 2.19.1/Keras 3.15.1 pinned, `tests/test_weights_load.py` passes; **no lockfile, no Dockerfile, no `make run`** |
> | 7 · fix B1 (stride) | ⛔ dropped — T3 retired the model; there is no operating point on a chance-level ranker |
> | 8 · delete the forks and dead subsystems | ✅ done — `code/` holds one pipeline; three forks archived under `archive/parent_repo_v1/` |
> | 9 · build `eval/` | 🟡 in progress — `eval/benchmark.py` passes; adapter, `metrics.json` and plots outstanding |
> | 10 · reproduce BADAS-Open | ⚠️ **gate restated** — see Phase 5 |
>
> **The actual next actions are now:** finish Phase 4's adapter (Track A) · commission the consent form and start UK recording (Track B) · book the first fleet calls (Track C). All three start now, in parallel.

The next ten things, in order, starting now.

1. **Right now, 5 minutes.** In the Colab tab: `File → Download .ipynb`. `git add train/colab_original.ipynb && git commit -m "Add original Colab training notebook (with outputs)"`. **This is the highest value-per-second action available to you and it closes the gap that caused the previous audit's central error.**
2. **Right now, 15 minutes.** Download `Crash-1500.txt`, `train.txt` and `test.txt` from the CCD Google Drive into `data/ccd/`. You have never had them, and they contain your grouping key, your official split and your per-frame labels.
3. **Today, 30 minutes.** Email Nexar (via the Hugging Face dataset discussion or their research contact) asking for written confirmation that the collision-prediction dataset permits training models for commercial use. Everything in your model plan depends on the answer.
4. **Today, 1 hour.** Write `scripts/count_split_leakage.py`: parse `Crash-1500.txt`, reproduce the seed-42 split, count `youtubeID`s present on both sides. Commit the number. It converts a suspicion into a fact.
5. **Today, 2 hours.** Write `tests/test_model_is_real.py` implementing the temporal-shuffle and single-frame tests against the cached features. Run them. **Commit the output verbatim, whatever it says.**
6. **Day 2, 4 hours.** Pin Python 3.11 and `tensorflow==2.19.*`; generate a lockfile; write a `Dockerfile`; add `make run`. Verify from a clean clone in a fresh container. Include the five-line weights-load smoke test — the repository currently has no way to detect that its own model file is unloadable.
7. **Day 2, 30 minutes.** Fix B1: add `FRAME_STRIDE` to a shared config, sample the inference buffer at the training stride, and assert stride equality at startup. Re-score the three local videos and note how much the numbers move.
8. **Day 3, 1 hour.** `git rm code/crash_detection.py code/crash_detection_linux.py camera_detect.py`. Delete `depth_estimator.py`, `bev_renderer.py`, `EgoZone`, `FaultDetector`. Commit as "remove non-functional subsystems". The diff will be large; every deleted line is a line you no longer maintain, and no verdict changes.
9. **Days 4–8.** Build `eval/`: source-grouped splitter, frozen test manifest, leakage test that fails CI, and metrics — AP, AUC, FP/hour, ECE, per-condition breakdown from the CCD metadata you downloaded in step 2.
10. **Days 9–10.** Download Nexar. Load BADAS-Open. Reproduce the published AP/AUC through your harness. **Do not proceed until it matches within ~0.02 AP.** Then, in parallel, write the 10-question fleet discovery script and book 10 UK calls.

---

## 44. Things NOT to Build

Re-evaluated item by item against the new evidence. Two items changed.

| Item | Verdict | Change? |
|---|---|---|
| Full customer dashboard | ❌ Not in 90 days | Unchanged |
| Mobile apps | ❌ Not in 90 days | Unchanged |
| Multi-tenancy | ❌ Not in 90 days | Unchanged |
| Custom hardware | ❌ Never at this stage | Unchanged |
| Your own video backbone | ❌ No | Unchanged — the Colab makes this *more* emphatic: you have not yet extracted the value from a frozen off-the-shelf backbone |
| Fault attribution | ❌ Delete it | Unchanged — and the dashcam variant is cruder than previously known |
| Severity *prediction* | ❌ Bands only | Unchanged |
| Depth estimation (MiDaS) | ❌ Delete | Unchanged — H1 confirmed, output reaches nothing |
| 3D reconstruction / BEV as a pipeline component | ❌ Delete | Unchanged |
| Pitch deck before week 10 | ❌ No | Unchanged |
| India-market product | ❌ Deferred, not refuted | Unchanged |
| Tesla outreach | ❌ No | Unchanged |
| **More training on CCD** | ❌ **Stop** | **NEW.** The corpus confound (B4) and the licence chain (D1) both rule it out for anything shipped. Keep it only as a research and teaching corpus. |
| **Retraining the MobileNetV2+LSTM to "improve" it** | ❌ **Stop** | **NEW.** Measure it once honestly (Phase 2), record the number, then retire it. Tuning a model whose task validity is unestablished is the definition of wasted effort. |
| **Deleting the old model before measuring it** | ❌ **Do not** | **NEW nuance.** It has never had a fair evaluation. Measure it once on a corpus-controlled benchmark so it can be retired with evidence rather than assertion. |

---

## 45. Final CTO Verdict

**1. Is this actually startup-worthy?**
The *company* is. The *codebase* is not — it is a prototype that taught you the domain, which is a legitimate outcome but not an asset. The Colab changes the nuance: **the training pipeline is worth keeping as a foundation** (roughly 60% of it survives a rewrite), while the *experiment* it ran is not worth defending. The market is real, the regulatory situation creates a genuine retrofit-fleet gap, and you demonstrably ship code. What is missing is measurement discipline and a customer.

**2. Biggest weakness?**
Until this week, that you could not measure anything. Now: **that your only measurement is confounded.** 0.9977 AUC on a split where positives are YouTube and negatives are BDD100K is not a result, and the derived ≈23 false alarms per hour is the number that actually describes the model. Second weakness, unchanged: you have no data nobody else has.

**3. Biggest opportunity?**
No public dashcam crash dataset covers UK or Western European roads, and no public benchmark measures false positives on realistic hard negatives. Both gaps are fillable by one determined person in 90 days, and filling them creates category authority disproportionate to the effort. **New:** you also now have a genuinely instructive, publishable falsification story, which is rarer than another accuracy claim.

**4. What would kill this startup?**
(a) Continuing to build features instead of measurement. (b) The AGPL issue surfacing during acquisition diligence. (c) **The BDD100K/YouTube provenance of the current training data surfacing during diligence** — now a documented fact rather than an unknown. (d) Attempting to sell to OEMs before having fleet references. (e) Trying to out-model a free Apache-2.0 SOTA baseline instead of competing where you can win.

**5. Fastest path to a credible prototype?**
Start from BADAS-Open, not from your model. Add the multi-head split, the calibrated physical channel, and the UK hard-negative benchmark. Six weeks, not six months.

**6. What should I build next?**
The evaluation harness — but **first**, the six falsification tests, because they now cost you one day instead of one week, and they determine whether anything else is worth building.

**7. What should I NOT build?**
Dashboards, mobile apps, custom hardware, depth estimation, 3D reconstruction, fault attribution, your own video backbone, and any further training on CCD.

**8. Which dataset strategy?**
Global for pretraining (V-JEPA2 → BADAS-Open → Nexar → DoTA/DADA), target-market for fine-tuning and — critically — for evaluation. **CCD demoted to research-only.** The benchmark is the asset.

**9. Which model architecture?**
V-JEPA2 + attentive probe, warm-started from BADAS-Open, multi-head (collision / near-miss / ego-involvement), fused with a calibrated physical channel (RT-DETR + ByteTrack + calibrated projection + IMU/GPS) through an interpretable logistic layer. **Not because it is newer, but because MobileNetV2's global-average-pooled ImageNet features structurally cannot represent relative motion between two vehicles, which is the task.**

**10. Which customer first?**
UK commercial fleets of 30–300 vehicles that already run dashcams and review footage manually.

**11. Which partnership first?**
UK driving schools — consented footage, in weeks not months, with a one-page agreement. Then mid-market fleets.

**12. When to approach accelerators?**
After pilot #1 is live with measured before/after numbers. Realistically 4–6 months. Not now.

**13. What evidence before approaching Tesla-like companies?**
100k+ validated vehicle-hours, cross-country generalisation without retraining, a benchmark they cannot reproduce, and two commercial references. And even then: target Tier-1 suppliers, not Tesla.

**14. What gets a serious automotive company to take a meeting?**
Field data they cannot generate themselves plus a validation methodology they respect. Never a demo, never an accuracy number.

**15. Do I need to rebuild, and what do I keep?** *(new question, answered because the Colab makes it answerable)*

| Keep | Discard |
|---|---|
| The training pipeline's **structure** — frozen backbone, cached features, `Sequence` loader, callback configuration | The **ImageNet backbone and the GAP bottleneck** — they cannot represent relative motion |
| The **consistent preprocessing discipline** between train and inference | The **LSTM head and single-output design** |
| **`compute_ttc` and `VehicleKF`** — correct mathematics awaiting valid input | **MiDaS, BEV, EgoZone, FaultDetector, three of the four forks** |
| **YOLO detection scaffolding** (swap the detector, keep the structure) | **The CCD-trained weights** — after one honest measurement |
| The **CCD dataset as a research corpus** and its annotation schema as a template for your own manifest | **CCD as training data for anything shipped** |
| The **notebook itself**, committed, as the historical record | **`Untitled0.ipynb` as the only copy of your pipeline** |

**A full rebuild is not necessary. A rebuild of the experiment is.**

---

### If I were you

I would start by downloading a notebook and three text files, which would take twenty minutes and would close the single largest evidence gap in this project.

Then I would spend one day trying to disprove my own model. I now think the CCD result is a corpus artefact — positives from YouTube, negatives from BDD100K, 0.994 AUC after one epoch — and I would rather know that on Tuesday than after a pilot. The tests are cheap because the features are already cached. I would commit the results verbatim, including the ones that hurt.

Then I would do the least glamorous thing available: fix the split, freeze a test set, delete every hard-coded threshold, and measure the model I actually ship, once, honestly. That number becomes the first true sentence in the project's history, and everything after it can be believed.

Then I would delete. `crash_detection.py`, `crash_detection_linux.py`, `camera_detect.py`, `depth_estimator.py`, `bev_renderer.py`, the ego zone, the fault detector — roughly 2,500 of the 3,900 lines. Nothing that determines a verdict would be lost, because almost none of it ever did.

I would accept, without resentment, that the model is not the company. Nexar gave away a state-of-the-art collision predictor under Apache 2.0, and that is a gift — it saves you two years. I would build on it openly and say so publicly, because in a field full of people implying they invented their backbone, being the person who says "we start from BADAS, here is what we added" is a credibility advantage.

I would then go and get the thing nobody has: 100 hours of UK road footage and 500 curated hard negatives. I would pay a UK delivery driver a few hundred pounds for a month of recording, and I would personally watch and label every difficult clip — not to save money on annotation, but because after 500 hard negatives you will understand the failure modes of this problem better than anyone you will pitch to.

I would publish UK-HN-500 free, and I would publish the CCD corpus-confound finding alongside it. The second one costs nothing, is genuinely useful to other researchers, and is the most credible thing a first-time founder can put on the internet: a result they disproved themselves.

While that runs, I would make thirty phone calls to UK fleet operators and ask one question: *"What happened the last time you trialled an AI dashcam?"* I am confident the answer will be some version of "too many false alarms, we turned it off." That answer is your entire go-to-market, and it is why leading with your false-positive number instead of your accuracy number will feel, to them, like the first honest vendor conversation they have had.

I would incorporate in Delaware — remotely, cheaply, and only after talking to a lawyer about the India-side implications. I would target the UK because I can call them at 09:30 their time from India and because their courts, police and insurers already treat dashcam footage as evidence.

I would not contact Tesla. I would not build a dashboard. I would not write a pitch deck until week ten. I would not apply to YC until a pilot was running.

And I would hold on to two things from the current work with some pride. The Kalman filter and TTC computation in `crash_detection_enhanced.py` are **correctly implemented** — fed garbage by a calibration bug and then ignored, but the mathematics is right. And the Colab pipeline is **better engineering than the absence of it suggested**: frozen backbone, cached features, AUC-monitored early stopping, class weights, consistent preprocessing. Someone who writes that can write the correct version of it in a week.

The gap between where this repository is and where it needs to be is large. It is also mostly deletion, measurement and phone calls — which is a much better problem to have than needing a research breakthrough.

---

# Appendices

## 46. Appendix A — Current Commands / Usage

**This describes the system as it exists today, which this document recommends largely replacing.** Every flag marked ⚠ is affected by a bug documented above.

### Running it

```bash
# Video file (shortcuts: crash1, crash2, safe)
python code/crash_detection_enhanced.py --video crash1
python code/crash_detection_enhanced.py --video /path/to/video.mp4

# Live webcam
python code/crash_detection_enhanced.py --camera
python code/crash_detection_enhanced.py --camera --dashcam --record

# Headless / limited
python code/crash_detection_enhanced.py --video crash1 --no-display --max-frames 200
```

| Flag | Description |
|---|---|
| `--video <name/path>` | Process a video file |
| `--camera` | Live webcam feed |
| `--dashcam` | ⚠ Enables the ego-zone trigger — **a proximity alarm, not crash detection (C3)** |
| `--record` | Save the annotated session to MP4 |
| `--save-output` | Save crash frames as JPEG (capped at 20 files) |
| `--no-display` | No GUI window |
| `--max-frames N` | ⚠ **Defaults to 500 — silently truncates long videos (M3)** |

Keys during playback: `Q` quit · `B` toggle BEV · `D` toggle depth heatmap.

### Model loading path

```
Config.FE_PATH      = models/feature_extractor_saved/        (TF SavedModel, provenance UNKNOWN — U5)
Config.WEIGHTS_PATH = models/crash_model_weights.weights.h5  (Keras 3 HDF5, run UNKNOWN — U1)

NeuralCrashDetector.__init__:
    tf.saved_model.load(FE_PATH)                     → .serve(batch) → (10, 1280)
    _build_and_load(WEIGHTS_PATH)                    → architecture rebuilt in code,
                                                       weights set layer-by-layer by name:
                                                       dense, lstm, lstm_1, dense_1, dense_2
```

⚠ If either file is missing, the constructor logs a warning and the CNN signal is **silently disabled**, after which the system falls back to the invalid physics rule (C1, C2) and still prints a verdict (H3).

### Reproducing the training run (as it exists today)

There is no command. The pipeline lives in Google Colab at
`https://colab.research.google.com/drive/1Bmudju26-q7JQyI4FAsZDBXsJw7PRnnM`
and requires `My Drive/CarCrashDetection/data/dataset/{Crash-1500,Normal}.zip` to be present. **Fixing this is Phase 0, task 1.**

---

## 47. Appendix B — Current Configuration Values

### Inference — `Config` in `code/crash_detection_enhanced.py`

| Parameter | Value | Status |
|---|---|---|
| `H_CAM` | 1.25 m | Assumed, never calibrated |
| `PITCH_DEG` | 2.0° | Assumed, never calibrated |
| `FX`, `FY` | 460 px | **BROKEN — calibrated for 640×480, applied at native resolution (C1)** |
| `CX_IMG`, `CY_IMG` | w/2, h/2 | Updated per resolution — inconsistently with FX/FY |
| `YOLO_CONF` | 0.5 | Reasonable |
| `CONF_NEW_TRACK` | 0.50 | Used |
| `CONF_CONTINUE` | 0.30 | **Declared, never used (M2)** |
| `NMS_IOU` | 0.45 | Used |
| `TRACK_MAX_DIST` | 80 px | **Resolution-dependent (H4)** |
| `MIN_BOX_AREA` | 2000 px² | **Resolution-dependent (H4)** |
| `MIN_VEHICLE_Z` | 3.0 m | **Declared, never used (M2)** |
| `MIN_VEHICLE_AR` | 0.4 | Used — will drop head-on motorcycles |
| `MAX_SPEED` | 180 km/h | Display cap only |
| `CAMERA_FPS` | 30 | Fallback |
| `TTC_WARN` | 2.5 s | **Declared, never used (M2)** |
| `TTC_CRITICAL` | 1.2 s | Computed, then discarded (C2) |
| `DIST_CONTACT` | 1.5 m | Trivially satisfied at 4K due to C1 |
| `CLOSING_MIN` | 0.5 m/s | Used in the discarded path |
| `CRASH_WIN` | 10 frames | Used in the discarded path |
| `CRASH_MIN_FR` | 3 | Used in the discarded path |
| `KF_Q_VAR` / `KF_R_X` / `KF_R_Z` / `KF_INIT_COV` | 0.5 / 0.15 / 0.4 / 5.0 | Correctly implemented, output unused |
| `TRACK_TTL_SEC` | 2.0 s | Applies to the Kalman dict only, **not** track identity (H2) |
| `CNN_FRAMES` | 10 | Matches training count — **but not the training stride (B1)** |
| `CNN_SIZE` | 112 | Matches training |
| **`CNN_THRESH`** | **0.80** | **No derivation anywhere (B5). Colab evaluated at 0.50 and suggested 0.55.** |
| `EGO_ZONE_W/H/Y` | 0.20 / 0.09 / 0.96 | **Delete (C3)** |
| `EGO_ZONE_MAX_W/H` | 250 / 80 px | **Resolution-dependent (H4); delete (C3)** |
| `ROI_SKY_CUT` / `ROI_HOOD_CUT` | 0.40 / 0.90 | Fixed fractions assume one camera geometry |
| `crash_pct` gate (video verdict) | 1.5% / 5.0% | **Fitted to three videos (B5)** |
| `--max-frames` default | 500 | **Silently truncates (M3)** |

### Training — Colab cell 2

| Parameter | Value |
|---|---|
| `BASE_PATH` | `/content/drive/My Drive/CarCrashDetection` |
| `DATASET_DIR` | `{BASE}/data/dataset` |
| `FEATURES_DIR` | `{BASE}/data/features_v2` |
| `MODELS_DIR` | `{BASE}/models` |
| `RESULTS_DIR` | `{BASE}/results` |
| `CRASH_VIDEOS_DIR` | `/tmp/crash_extracted` (ephemeral) |
| `NORMAL_VIDEOS_DIR` | `/tmp/normal_extracted` (ephemeral) |
| `FRAMES_PER_VIDEO` | 10 |
| `FRAME_SIZE` | 112 |
| `FEATURE_DIM` | 1280 |
| `BATCH_SIZE` | 32 |
| `EPOCHS` | 50 (ran 17) |
| `LR` | 1e-3 (→5e-4 @ epoch 11, →2.5e-4 @ epoch 15) |
| `TEST_SIZE` | 0.2 |
| `RANDOM_SEED` | 42 — **applied only to `train_test_split` (B9)** |
| Framework | TensorFlow 2.19.0 / Keras 3.13.2, Colab T4 |

**Constants that must agree between training and inference, and currently do not:**

| Constant | Training | Inference | Status |
|---|---|---|---|
| Frame count | 10 | 10 | ✅ |
| Frame size | 112 | 112 | ✅ |
| Colour order | RGB | RGB | ✅ |
| Normalisation | `mobilenet_v2.preprocess_input` | `mobilenet_v2.preprocess_input` | ✅ |
| Feature dim | 1280 | 1280 | ✅ |
| **Temporal stride** | **≈0.54 s** | **0.033 s @30fps** | ❌ **B1** |
| **Threshold** | **0.50** | **0.80** | ❌ **B5** |
| Aspect handling | squash to square | squash to square | ⚠ consistent but lossy (B11) |

---

## 48. Appendix C — Known Bugs (index)

| ID | Title | Severity | Priority | Train | Infer | Eval | Repro | Legal | New? |
|---|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **B1** | Train/inference temporal stride mismatch | CRITICAL | P0 | | ✅ | ✅ | | | 🆕 |
| **B2** | No held-out test set; selection-contaminated metrics | CRITICAL | P0 | ✅ | | ✅ | | | 🆕 |
| **B3** | No source grouping; official split and `youtubeID` ignored | CRITICAL | P0 | ✅ | | ✅ | ✅ | | 🆕 |
| **B4** | Positive/negative classes coincide with two corpora | CRITICAL | P0 | ✅ | ✅ | ✅ | | | 🆕 |
| **B5** | Deployed threshold has no derivation | CRITICAL | P0 | | ✅ | ✅ | ✅ | | 🆕 (extends C4) |
| **B10** | Training code not in version control | CRITICAL | P0 | | | ✅ | ✅ | ✅ | 🆕 |
| **R1** | `requirements.txt` specifies an impossible environment | CRITICAL | P0 | | ✅ | | ✅ | | carried |
| **R2** | Pinned TensorFlow cannot read the shipped model | CRITICAL | P0 | | ✅ | | ✅ | | 🆕 |
| **C6** | `ultralytics` is AGPL-3.0 | CRITICAL | P0 | | ✅ | | | ✅ | carried |
| **C7** | Fault attribution unsupportable | CRITICAL | P0 | | ✅ | | | ✅ | carried |
| **C3** | Dashcam mode is a proximity alarm | CRITICAL | P0 | | ✅ | ✅ | | | carried |
| **D1** | Training data not licensed for commercial use | CRITICAL | P0 | ✅ | | | | ✅ | 🆕 (was "unknown") |
| **B8** | `ModelCheckpoint` fixed path; runs overwrite | HIGH | P1 | | | ✅ | ✅ | ✅ | 🆕 |
| **B9** | Nothing seeded except the split | HIGH | P1 | ✅ | | | ✅ | | 🆕 |
| **C1** | Focal length not rescaled with resolution | CRITICAL | P1 | | ✅ | ✅ | | ✅ | carried |
| **C2** | Physics stack excluded from the decision | CRITICAL | P1 | | ✅ | ✅ | | | carried |
| **H3** | Silent exception swallowing | HIGH | P1 | | ✅ | ✅ | | ✅ | carried |
| **B7** | Rich CCD annotations discarded | HIGH | P2 | ✅ | ✅ | ✅ | | | 🆕 |
| **H1** | MiDaS inverse depth used as depth | HIGH | P2 | | ✅ | | | | carried |
| **H2** | Tracker resets all IDs on an empty frame | HIGH | P2 | | ✅ | ✅ | | | carried |
| **H4** | Resolution-dependent constants | HIGH | P2 | | ✅ | ✅ | ✅ | | carried |
| **H5** | Four divergent forks | HIGH | P2 | | ✅ | ✅ | ✅ | | carried |
| **H6** | No persistence, API or service boundary | HIGH | P2 | ✅ | ✅ | ✅ | | | carried |
| **M3** | `--max-frames` default of 500 truncates | MEDIUM | P2 | | ✅ | ✅ | ✅ | | carried |
| **M4** | `--test` asserts nothing; no tests exist | MEDIUM | P2 | ✅ | ✅ | ✅ | ✅ | | carried |
| **B6** | Feature-space noise augmentation is a no-op | MEDIUM | P3 | ✅ | | | | | 🆕 |
| **B11** | Square resize destroys aspect ratio | LOW | P3 | ✅ | ✅ | | | | 🆕 |
| **B12** | Random frame seeking is codec-dependent | LOW→HIGH | P3 | ✅ | | | ✅ | | 🆕 |
| **B13** | 493 MB of orphaned model artefacts | LOW | P3 | | | | ✅ | | 🆕 |
| **M1** | BEV double-converts speed | MEDIUM | P3 | | ✅ | | | | carried |
| **M2** | Dead configuration constants | MEDIUM | P3 | | | | ✅ | | carried |
| **M5** | `torch.load` patched to `weights_only=False` | MEDIUM | P3 | | ✅ | | | ✅ | carried |
| **M6** | CNN and detector see different frames | MEDIUM | P3 | | ✅ | | ✅ | | carried |
| **L1–L5** | `.DS_Store`, logging, version drift, no `LICENSE`, `Untitled0.ipynb` | LOW | P4 | | | | ✅ | ✅ | partly 🆕 |

**Totals: 34 tracked issues — 12 at P0, 5 at P1, 8 at P2, 8 at P3, 1 group at P4. Fourteen are new in this revision, twelve of them discovered in the Colab or by artefact inspection.**

---

## 49. Appendix D — Dataset References

### Used by the current model

| Dataset | Role | Size | Source | Licence position |
|---|---|---|---|---|
| **Car Crash Dataset (CCD)** | **The training set** | 1,500 crash + 3,000 normal, 50 frames @ 10 fps | [`Cogito2012/CarCrashDataset`](https://github.com/Cogito2012/CarCrashDataset) | Repo labelled MIT; README has no licence section. **Positives YouTube-derived; negatives from BDD100K.** 🔴 research only |
| **BDD100K** (via CCD negatives) | Negative class | 3,000 clips sampled | [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu) | Basic licence limited to personal use. 🟡 verify before any commercial use |
| **ImageNet** (via MobileNetV2 weights) | Backbone pretraining | — | Keras Applications | Standard; verify the weights' terms for commercial redistribution |
| **COCO** (via YOLOv8n) | Detector pretraining | — | Ultralytics | 🔴 the *package* is AGPL-3.0 (C6) |

### Recommended going forward

| Dataset | Role | Size | Licence position |
|---|---|---|---|
| **Nexar Collision Prediction** | **Primary training + benchmark** | 1,500 train clips (2,844 total), ~40 s, 1280×720 @ 30 fps, 50/50 pos/neg, first-party consented and anonymised, with `time_of_event` / `time_of_alert` and lighting/weather/scene metadata | 🟡 `nexar-open-data-license` — three inconsistent public descriptions. **Confirm in writing, week 1.** |
| **BADAS-Open** (model) | Baseline to reproduce and warm-start from | **Read from the checkpoint itself, 2026-09-11** (`models/badas/weights/badas_open.pth`, 3.7 GB, epoch 3, val_acc 87.07). Four modules: `backbone` V-JEPA2 ViT-L (587 tensors, hidden 1024) · **`predictor`** (199 tensors, 12 layers @ 384, with `mask_tokens`; 1024→384→1024) · `temporal_processor` MultiheadAttention(1024, 8 heads)+LayerNorm+mean-pool · `classifier` 3-layer MLP (768 hidden, 2 classes). Training config: `use_future_prediction: true`, `future_prediction_seconds: 1.0`, `predictor_combination_method: "concat"`, `frame_count: 16`, `img_size: 224`, `temperature: 2.0`, trained on 2-second balanced clips. **Sliding window at inference: 16 frames @ 224×224, 8 fps, stride 1.** ⚠️ **The published inference code does NOT implement the predictor pathway** — `EnhancedVideoClassifier.forward()` is `backbone → temporal_processor → classifier` only, and `load_state_dict(strict=False)` silently discards all 199 `predictor.*` tensors. A naive load-and-run therefore evaluates a model missing its entire future-prediction pathway; any AP from it is not BADAS-Open's AP (progress.md §6.9, 2026-09-11). Reports Nexar AP 0.86 / AUC 0.88 / mTTA 4.9 s (model card) but its own `config.json` says AP 83.2 / AUC 0.85; DoTA 0.94 vs 95.9; DADA 0.87 vs 92.9; DAD 0.66 vs 60.9 | 🟢 **Apache 2.0**, commercial use with attribution; disclaimed for safety-critical use |
| **V-JEPA 2** (backbone) | Encoder | — | 🟢 **MIT** (majority; some utility files Apache-2.0) |
| **DoTA** | Pretraining + eval | 4,677 videos with temporal, spatial and categorical annotations | 🟡 repo MIT; **videos from YouTube** |
| **DADA-2000** | Augmentation | 2,000 sequences, 658,476 frames, 1584×660, ~6.1 h | 🟡 East Asian bias; YouTube-derived |
| **MM-AU / CAP-DATA** | Augmentation | 11,727 ego-view accident videos | 🟡 East Asian bias; YouTube-derived |
| **DAD** | Eval | 1,750 clips, six Taiwanese cities | 🟡 dense scooter traffic unlike the UK |
| **A3D** | Eval | 1,500 clips | 🟡 |
| **CADP** | — | 1,416 CCTV segments | 🔴 research / non-commercial only |
| **nuScenes** | — | AV benchmark | 🔴 CC BY-NC-SA 4.0 |
| **Waymo Open Dataset** | — | AV perception/motion | 🔴 non-commercial |
| **IDD / IDD-3D** | Later India expansion | 10k images, 34 classes | 🟢 IDD-3D CC BY 4.0 — deprioritised |
| **UK-HN-500** | **Your benchmark** | 500 curated UK hard negatives | Yours, to be published free |
| **Your UK corpus** | Fine-tuning | 100 h target | Yours, with consent + DPIA |

### CCD annotation schema — the fields you have not used

```
Crash-1500.txt, one row per crash clip:
  vidname       clip identifier
  binlabels     50 per-frame binary accident flags   ← temporal localisation (B7)
  startframe    offset into the source YouTube video ← provenance
  youtubeID     source video identifier              ← THE GROUPING KEY (B3)
  timing        Day | Night                          ← per-condition eval (B7)
  weather       Normal | Snowy | Rainy               ← per-condition eval (B7)
  egoinvolve    boolean                              ← ego-involvement head (B7, replaces C3)

vgg16_features/train.txt, test.txt:
  the official split — enables comparison with published CCD results (B3)
```

**Five capabilities, one grouping key and one comparable benchmark, all sitting in three text files that have never been downloaded.**

---

## 50. Appendix E — Sources

**Primary evidence for this revision**
- Google Colab notebook `Untitled0.ipynb` — `https://colab.research.google.com/drive/1Bmudju26-q7JQyI4FAsZDBXsJw7PRnnM` — 12 code cells with saved outputs, read cell-by-cell on 2026-09-10
- Repository at commit `b539d6e`; git history back to `c63e307`
- `models/crash_model_weights.weights.h5` and the `best_crash_model.keras` archived at `c63e307` — HDF5/ZIP byte inspection and weight-array comparison
- [`Cogito2012/CarCrashDataset`](https://github.com/Cogito2012/CarCrashDataset) README

**Models and datasets**
- [nexar-ai/nexar_collision_prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) · [arXiv 2503.03848](https://arxiv.org/abs/2503.03848) · [CVPR 2025 WAD paper](https://openaccess.thecvf.com/content/CVPR2025W/WAD/papers/Moura_Nexar_Dashcam_Collision_Prediction_Dataset_and_Challenge_CVPRW_2025_paper.pdf)
- [nexar-ai/BADAS-Open](https://huggingface.co/nexar-ai/BADAS-Open) · [arXiv 2510.14876](https://arxiv.org/abs/2510.14876)
- [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) · [Meta AI — V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
- Bao, Yu & Kong, *Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning*, ACM MM 2020 (the CCD paper)
- [DoTA — Detection of Traffic Anomaly](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)
- [LOTVS-DADA / DADA-2000](https://github.com/JWFangit/LOTVS-DADA) · [LOTVS-MM-AU](https://github.com/jeffreychou777/LOTVS-MM-AU)
- [CADP (arXiv 1809.05782)](https://arxiv.org/abs/1809.05782)
- [Waymo Open Dataset Terms](https://waymo.com/open/terms/) · [nuScenes Terms of Use](https://www.nuscenes.org/terms-of-use) · [BDD100K](https://bdd-data.berkeley.edu)
- [IDD — India Driving Dataset](https://arxiv.org/pdf/1811.10200) · [IDD-3D](https://idd3d.github.io/)
- [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) · [LICENSE](https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE)
- [Ultralytics License](https://www.ultralytics.com/license) · [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- [Vision-Based Traffic Accident Detection and Anticipation: A Survey (arXiv 2308.15985)](https://arxiv.org/pdf/2308.15985) · [LLMs for Crash Detection in Video: A Survey (arXiv 2507.02074)](https://arxiv.org/pdf/2507.02074)

**Regulation and legal**
- [Regulation (EU) 2015/758 — eCall](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A32015R0758) · [European Commission — interoperable EU-wide eCall](https://transport.ec.europa.eu/transport-themes/smart-mobility/road/its-directive-and-action-plan/interoperable-eu-wide-ecall_en) · [European Parliament — eCall mandatory in new car models](https://www.europarl.europa.eu/news/en/press-room/20180326IPR00510/saving-lives-ecall-mandatory-in-new-car-models-from-this-week)
- [Thatcham Research — EU General Safety Regulation (GSR2)](https://www.thatcham.org/thatcham-research-explains-new-eu-vehicle-safety-regulation-and-what-it-means-for-uk-drivers/) · [RAC — What is GSR2](https://www.rac.co.uk/drive/advice/road-safety/what-is-gsr2-important-eu-car-safety-features-explained/)
- [German court rules on GDPR compliance for dashcam recordings](https://ppc.land/german-court-rules-on-gdpr-compliance-for-dashcam-recordings/) (BGH, 15 May 2018, VI ZR 233/17)
- [Digital Personal Data Protection Rules, 2025 (India)](https://en.wikipedia.org/wiki/Digital_Personal_Data_Protection_Rules,_2025) · [DLA Piper — Data protection laws in India](https://www.dlapiperdataprotection.com/?t=law&c=IN)

**Market**
- [Nextbase National Dash Cam Safety Portal](https://nextbase.co.uk/national-dash-cam-safety-portal/) · [30% rise in dash cam submissions to police in 2023](https://emergencyservicestimes.com/2024/01/05/30-rise-in-dash-cam-submissions-to-police-in-2023/) · [Auto Express — record dash cam police reports](https://www.autoexpress.co.uk/consumer-news/363978/dash-cam-users-report-record-numbers-fellow-drivers-police)
- [MarketsandMarkets — UK Fleet Telematics Market 2025–2030](https://www.marketsandmarkets.com/Market-Reports/geography/future-commercial-vehicle-telematics-market/uk) · [IBISWorld — Fleet Telematics Systems in the US](https://www.ibisworld.com/united-states/industry/fleet-telematics-systems/4546/)
- [Video Telematics Market Report 2025 — North America and Europe](https://www.businesswire.com/news/home/20250502021623/en/Video-Telematics-Market-Report-2025-Video-Telematics-in-North-America-and-Europe-to-Hit-17-Million-by-2029---ResearchAndMarkets.com)
- [VAVA — Global dash cam usage by country](https://www.vava.com/blogs/dash-cam/dash-cam-usage-a-quick-country-wise-breakdown)
- [MoRTH — Road Accidents in India 2023 (reported)](https://www.business-standard.com/india-news/india-road-accidents-deaths-injuries-report-road-highway-ministry-nitin-gadkari-125082801527_1.html)

**Accelerators and incorporation**
- [Y Combinator — Apply](https://www.ycombinator.com/apply)
- [Flowjam — YC funding amount 2026](https://www.flowjam.com/blog/y-combinator-funding-amount-2026) · [Peony — Top startup accelerators 2026](https://www.peony.ink/blog/top-20-startup-accelerators-worldwide)
- [Do you need a Delaware C Corp to apply to YC?](https://www.wisp.blog/blog/do-you-need-a-delaware-c-corp-to-apply-to-yc-or-hustle)

**Hardware**
- [NVIDIA Jetson Orin Nano Super Developer Kit](https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/) · [Tom's Hardware — $249, 67 TOPS](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-launches-new-usd249-ai-development-board-that-does-67-tops) · [Hardware Busters — Jetson price increases](https://hwbusters.com/news/nvidia-jetson-prices-jump-up-to-101-the-249-orin-nano-super-is-now-399/)

---

*Revision 2, written 2026-09-10. Repository findings verified against commit `b539d6e` by direct code inspection, numerical testing and byte-level artefact comparison. Training-pipeline findings verified against the Colab notebook read cell-by-cell, including saved outputs. Dataset findings verified against the CarCrashDataset repository README. Market, dataset and licensing claims are sourced above; licence interpretations are engineering assessments, not legal advice, and the items flagged in [§35](#35-privacy--legal--compliance) require qualified counsel.*

*Supersedes revision 1 of 2026-09-07. Every conclusion from that revision is explicitly reconciled in [What Changed From the Previous Audit](#what-changed-from-the-previous-audit) — nothing was silently dropped.*
