# AI Crash Detection System

**Repository:** `khushpal-cipher/crash-detection-system`
**Document status:** Technical audit + startup strategy. Revision 2, written 2026-09-10. Supersedes the 2026-09-07 audit, which is reconciled section-by-section in [What Changed From the Previous Audit](#what-changed-from-the-previous-audit).
**Verified against:** repository commit `b539d6e`; the Google Colab training notebook `Untitled0.ipynb` (`drive/1Bmudju26-q7JQyI4FAsZDBXsJw7PRnnM`), read cell-by-cell including saved outputs on 2026-09-10; the `Cogito2012/CarCrashDataset` repository; and byte-level inspection of the model artefacts in the working tree and in git history.

**Honest one-line description of what exists today:** a clip-level binary classifier (frozen ImageNet MobileNetV2 → LSTM, 578,689 trainable parameters) trained on the public Car Crash Dataset with a random non-source-grouped 80/20 split and no held-out test set, deployed at a threshold (0.80) that appears nowhere in its training notebook, AND-ed with "≥2 YOLO vehicle boxes present", and wrapped in ~3,000 lines of 3D-geometry and visualisation code that **does not participate in the crash decision**.

> **The single most important correction in this revision:** the previous audit concluded that no training code, dataset history, split, epoch or metric information existed. **That conclusion was wrong as a global statement.** All of it exists, in Google Colab. It was absent from GitHub, which is a different — and much more fixable — problem. The training pipeline is now fully documented in [Training Pipeline](#6-training-pipeline) and [Training History](#10-training-history).
>
> **The second most important correction:** recovering that evidence did **not** improve the project's standing. It made the assessment sharper and, in three specific respects, worse. See [New Findings](#new-findings-summary).

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
| 9 | "Hypothesis: the model learned clip identity / global image statistics, not collisions." | **PARTIALLY CONFIRMED — mechanism identified, still requires the falsification tests** | Positives = YouTube compilations; negatives = BDD100K (B, C) | The hypothesis is now more specific and more damning: the two classes are **two different corpora**, differing in codec, resolution, colour grading, capture hardware and geography. A trivially-available shortcut explains val AUC 0.998 without any collision understanding. Test it before anything else. |
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
| Training code in version control | ❌ No — exists only in Colab |
| Dataset identified | ✅ Yes — CCD (as of this revision) |
| Dataset licensed for commercial use | ❌ No — see [§23](#23-dataset-licensing--provenance) |
| Held-out test set | ❌ No |
| Source-grouped splits | ❌ No |
| Falsification tests run | ❌ No |
| Calibration measured | ❌ No |
| False-positives-per-hour measured | ❌ No (implied ≈23/h; target < 0.1/h) |
| Deployed threshold derived from data | ❌ No |
| Shipped weights traceable to a run | ❌ No |
| Detector licence resolved | ❌ No — AGPL-3.0 |
| Any test in the repository | ❌ No |
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
└── videos -> ~/Desktop/crash_detection/videos
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

**This remains a hypothesis. It is cheap to falsify and you must falsify it before building anything on top.** Four experiments, each under an hour, all now runnable *because the pipeline is recovered*:

1. **Single-frame test.** Train the identical head on a sequence of 10 *identical copies* of one frame. If AUC stays above ~0.95, there is no temporal information in the task as posed, and the LSTM is decoration.
2. **Temporal shuffle test.** Randomly permute the 10 feature vectors at evaluation. If the score barely moves, the LSTM contributes nothing.
3. **Corpus-control test — the decisive one.** Evaluate on a dataset where positives and negatives come from *the same* corpus (Nexar is exactly this: 50/50 positive/negative from one driver community, one anonymisation pipeline). If AUC collapses toward 0.5, the CCD result was a corpus artefact.
4. **Crash-excision test.** Score the CCD positives using only frames before the first `binlabels == 1` frame. If they still score high, the model is not keying on the collision.

Run 1–4 **before** writing a line of new model code. If the model fails them — and I expect it to fail 1 and 3 — you have saved yourself from building a company on a measurement artefact, at a cost of one afternoon.

---
