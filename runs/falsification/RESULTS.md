# Falsification Results — 2026-09-10

Run against commit `b539d6e`, deployed artefact `models/crash_model_weights.weights.h5`.
Scripts: `scripts/t5_source_leakage.py`, `scripts/t124_model_falsification.py`.
Raw output: `T5_source_leakage.txt`, `T124_local_videos.json`.

**Verdict: the model is an image classifier. The LSTM contributes nothing measurable, and the
training split leaks at 91%. Neither result is marginal.**

---

## T1 — Single-frame test · **FAILED**
## T2 — Temporal shuffle test · **FAILED**
## B1 — Train/inference stride · **CONFIRMED, and second-order**

Method: features extracted once per video with the deployed `feature_extractor_saved`
(MobileNetV2→GAP, 112×112, `mobilenet_v2.preprocess_input`); the 578,689-parameter LSTM head
evaluated by an exact NumPy forward pass read straight from the Keras-3 HDF5 weights.
Windows stepped every 5 frames. Mean sigmoid output, and fraction of windows ≥ `CNN_THRESH` 0.80.

| Video | A · deployed (10 consecutive) | B · **temporally shuffled** | C · training-matched stride | D · **single frame tiled ×10** |
|---|---|---|---|---|
| `crash1.mov` (35.2 fps) | **0.9997** — 100% ≥0.80 | **0.9998** — 100% | 0.9998 — 100% | **0.9798** — 98% |
| `crash2.mov` (31.2 fps) | **0.9640** — 94.1% | **0.9649** — 91.2% | 0.9996 — 100% | **0.9460** — 94.1% |
| `safe.mp4` (24.0 fps) | **0.0241** — 0% (max **0.7914**) | **0.0228** — 0% | 0.0003 — 0% | 0.0512 — 0.7% (max **0.9695**) |

**T2 — the LSTM is decoration.** Randomly permuting the ten frames inside the window changes the
mean score by **≤ 0.0013 on every video** (0.9997→0.9998, 0.9640→0.9649, 0.0241→0.0228). A network
that had learned anything about how a collision unfolds over time cannot be invariant to the order
of its own input. Two recurrent layers, 246,528 parameters, contributing nothing.

**T1 — one frame is the whole model.** Replacing the sequence with a *single* frame repeated ten
times reproduces the deployed score to within 0.02 on the crash videos (0.9997→0.9798,
0.9640→0.9460). The system is a per-frame appearance classifier with a recurrent layer bolted on.

**B1 — the stride mismatch is real but is not what is wrong.** Feeding the training-matched stride
(≈0.54 s; 19/17/13 frames at these frame rates) *does* move the numbers — `safe.mp4` collapses from
0.0241 to 0.0003. But given T2, this is only a change in *which* frames are averaged, not a
restoration of temporal reasoning. Fix it for correctness; do not expect it to fix the model.

**Two incidental findings that indict the deployed threshold directly.**
1. `safe.mp4` peaks at **0.7914** under the deployed path. `CNN_THRESH = 0.80`. The prior README's
   "0.79" is reproduced exactly — the shipped operating point clears the only negative ever tested
   by **0.0086**.
2. Under the single-frame path, `safe.mp4` reaches **0.9695**. The model *does* emit confident crash
   scores on the safe video. It is hidden only by the particular averaging the deployed code happens
   to perform.

---

## T5 — Source-leakage quantification · **FAILED, severely**

Method: parsed `data/Crash-1500.txt` (1,500 rows: `vidname, binlabels[50], startframe, youtubeID,
timing, weather, egoinvolve`); 200 Monte-Carlo random 80/20 clip splits reproducing the Colab method.

**The 1,500 crash clips come from only 133 distinct YouTube videos** — a mean of 11.3 clips per
source, maximum 34. **99.9% of clips (1,499/1,500) belong to a multi-clip source.**

| Quantity | Value |
|---|---|
| Source videos split across train and val | **113 of 133** (min 102, max 121) |
| Crash clips implicated | **1,372 / 1,500 = 91.4%** |
| Val crash clips with a sibling from the same YouTube video in train | **≈274 of 300 (91%)** |

**The official CCD split does not fix it.** `train.txt` / `test.txt` put **107 of 133 source videos on
both sides**. Using the official split would make results comparable to published CCD work, but it
would not remove the leakage. A group-wise split must be built by hand from `youtubeID`.

### Label quality — from the same file
- **Accident onset: mean frame 37.2 of 50** (median 36, min 30, max 49). On average **74% of every
  "crash" clip contains no accident at all.**
- The uniform sampling `linspace(0,49,10)` selects frames `[0,5,11,16,22,27,33,38,44,49]`, of which
  **72% fall before the accident begins** — yet every one was trained with label 1.
- Combined with T1, the model was largely trained to answer "is this a frame of YouTube dashcam
  footage?" while being told the answer is "crash".

### Metadata available and discarded
`timing` Day 1325 / Night 175 · `weather` Normal 1141 / Snowy 235 / Rainy 124 ·
**`egoinvolve` Yes 801 / No 699** — a near-balanced ego-involvement label, present for every positive
clip, never read. `EgoZone` (a fixed screen rectangle) was built to approximate it.

---

## T6 — Always-negative baseline

Val-split FPR is 19/600 = **3.17%**. Applied as non-overlapping 5-second windows over continuous
driving (720/hour) that is **≈23 false alarms per driving hour**. An always-negative predictor scores
**0**. On the metric that decides whether a fleet keeps the product, the trivial baseline wins.

---

## T3 — Corpus control · **NOT YET RUN**
Blocked on downloading the Nexar videos (31.4 GB). Licence resolved — see below. This is the one
remaining test, and after T1/T2/T5 it is now a formality rather than a question.

---

## Bonus — Nexar licence **RESOLVED** (was P0 blocker D1)
`data/nexar/LICENSE`, retrieved 2026-09-10. Operative grant:

> "Permission is hereby granted, free of charge … to use, copy, modify, and distribute the Dataset,
> subject to the following conditions"

Conditions: **attribution** (specified citation), retain the notice on redistribution, and
**No Resale** — the *Dataset* may not be sold or sublicensed for profit without written consent.
Plus ethical-use restrictions (no malicious systems, deepfakes, re-identification, weaponisation,
**"exploitative practices … such as unethical insurance practices"**, and compliance with law).

**There is no non-commercial restriction.** Training a commercial model is "use" and "modify" and is
permitted. The No-Resale clause restricts redistributing the dataset itself, not derived models —
have a lawyer confirm that reading, but the email to Nexar is no longer a blocker.
Dataset shape: 750 + 750 train, 334 + 333 public test, 338 + 339 private test, `solution.csv` labels,
`time_to_accident_test_map.csv`.

---

## What this changes

1. **Do not tune, retrain or extend the MobileNetV2+LSTM.** T1 and T2 show the recurrent layers are
   inert; the ceiling is a frozen ImageNet feature with global average pooling, which discards the
   spatial layout that relative motion is made of.
2. **Do not quote `val AUC 0.9977`, ever again.** 91.4% of the split leaks by source, and 72% of the
   positive frames contain no accident.
3. **`CNN_THRESH = 0.80` is dead.** It clears the only negative ever tested by 0.0086, and the same
   video reaches 0.9695 under a marginally different sampling rule.
4. **Nexar is unblocked and is the right corpus** — same-source positives and negatives, so the
   confound that produced this result cannot recur.
5. **The `youtubeID` field is the grouping key** and must drive every future split. The official CCD
   split is not a substitute.
