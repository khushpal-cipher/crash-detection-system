# NEW PLAN v2 — Beating the Baseline under Frozen-Data Constraints

**Written 2026-09-14 (session 7). Status: PROPOSAL — nothing implemented.**
Supersedes NEW_PLAN v1, which was red-teamed and found to have three material holes.
`README.md` remains the master plan; `progress.md` remains execution state.

> **Provenance note.** The same red-team brief was accidentally given to two Claude Code sessions in
> parallel on 2026-09-13. The other session produced `NEW_PLAN_v2.md` (now moved to
> `archive/NEW_PLAN_v2_parallel_session.md`). **This file is the merged, authoritative plan.** It
> keeps this session's measured evidence (§3 — none of which the parallel session had) and folds in
> the one genuinely additive idea from the parallel document: **R9, the IMU/CAN rejection channel**,
> together with its three-way partition discipline. Candidates in that document which this plan
> rejects, and why, are listed at the end of §4.

**Objective:** build the strongest crash-detection prototype achievable from the current baseline —
**not** a better-calibrated demo. v1 failed this test: it produced zero detection improvement.
v2 leads with detection.

---

## 1. The statistical bar — what "materially stronger" means

Measured by bootstrap on the committed 667-clip run:

| | value | 95% CI | width |
|---|---|---|---|
| AP | 0.8349 | [0.7910, 0.8734] | 0.082 |
| ROC-AUC | 0.8498 | [0.8195, 0.8774] | 0.058 |

**An unpaired improvement below ~0.04 AP is invisible at n=667.** Chasing +0.01 is self-deception.
Every comparison in this plan therefore uses a **paired bootstrap on the same clips**, which cancels
the shared difficulty term and is roughly an order of magnitude more sensitive. The success
criterion throughout is *"paired 95% CI on ΔAP excludes zero"*, with +0.03 as the target effect.

---

## 2. Red-team closure

| # | v1 hole | Closed how |
|---|---|---|
| 1 | Zero detection improvement | §4 — seven detection mechanisms, one already measured at **+0.033 AP paired, CI excludes zero** |
| 2 | No clean calibration fit-set | §6 — four-tier protocol; primary is a stratified split of test-public with the map fitted on the calibration half only |
| 3 | Compute under-scoped ~10× | §7 — recomputed per config; plus the tail-scoring insight that cuts most sweeps ~10× |
| 4 | comma2k19 treated as representative | §8 — demoted to *highway stress test*; ZOD (CC BY-SA 4.0, European, commercial-OK) added as a second, distinct negative domain |
| 5 | "CPU demo" unvalidated | §7.3 — explicit CPU-vs-MPS benchmark as a gating task; no performance claim until measured |
| 6 | External independence overstated | §8.1 — three-tier taxonomy: harness reproduction / cross-dataset generalisation / genuinely independent |
| 7 | Provisional promoted to conclusion | §3 — split-half replication run; F3 and R1 explicitly held at *provisional*, confirmation gated on n=667 |
| 8 | Track C a wish | §9 — executable 10-message protocol with a hard kill condition |
| 9 | No timeline | §7.4 — four-week schedule built on real Air/Studio availability |
| 10 | Uncertainty unquantified | §1 — the bar; paired bootstrap mandated everywhere |

---

## 3. New evidence measured this session

### 3.1 The aggregation finding (provisional, n=268 of 667)

Paired bootstrap against the current `nanmax` reduction:

| reduction | AP | AUC | ΔAP vs max | paired 95% CI |
|---|---|---|---|---|
| max (current) | 0.8809 | 0.8804 | — | — |
| **last window** | **0.9140** | **0.9030** | **+0.0330** | **[+0.0008, +0.0744]** ✓ |
| **max×last (geometric)** | **0.9158** | **0.9057** | **+0.0349** | **[+0.0027, +0.0729]** ✓ |
| last 4 mean | 0.9050 | 0.8992 | +0.0240 | [−0.0087, +0.0645] |
| top-3 mean | 0.8585 | 0.8657 | −0.0224 | [−0.0383, −0.0073] ✗ |
| p90 | 0.8154 | 0.8241 | −0.0656 | [−0.1041, −0.0293] ✗ |
| area > 0.5 | 0.6478 | 0.7246 | −0.2331 | [−0.3110, −0.1461] ✗ |

### 3.2 The mechanism, pinned

Position of each clip's score peak, normalised (0 = start, 1 = end):

- **positives: median 0.984** — the peak sits at the very end
- negatives: median 0.782
- peak in the final 10% of the clip: **83.0% of positives vs 42.5% of negatives**

Nexar test clips are truncated 500–1500 ms *before* the event, so for a positive the last window is
the most informative one and `max ≈ last`. For a negative, `max` wanders off and grabs a spurious
mid-clip peak. **Last-window wins by suppressing false peaks on negatives, not by boosting
positives.** This also makes it *more* deployment-faithful than max — in production there is no
retrospective maximum, only the current window.

### 3.3 Adversarial checks on that finding

- **Clip length does not leak the label:** AP(length) = 0.5330, AUC = 0.5483 — chance. Correlation
  with the last-window score is +0.109. The effect is not a duration artifact. ✓
- **Split-half replication:** positive in 5 of 6 halves, but spread is wide (−0.009 to +0.075,
  mean ≈ +0.034). **Direction is consistent; magnitude is not yet established.** Held as
  PROVISIONAL until all 667 traces exist.
- **Test-set selection risk is real** — this reduction was chosen by looking at test-public. Handled
  in R1's validation protocol below.

### 3.4 Calibration (unchanged from v1, still correct)

Beta calibration, 5-fold cross-fitted: **ECE 0.3286 → 0.0498**, Brier −48%, NLL −57%, AP within
0.002. Isotonic loses 0.02 AP. Temperature alone reaches only 0.20.

---

## 4. Ranked improvement candidates

Each carries the nine required fields. Ranked by *expected gain × probability ÷ cost*, not by ease.

### R1 — Deployment-faithful reduction · **TIER 0**

- **Hypothesis:** the last window (or `√(max·last)`) outranks `nanmax` at clip level.
- **Mechanism:** §3.2 — truncation places positive evidence at the clip end; max over-triggers on negatives.
- **Data:** per-frame traces already being written. No new data.
- **Compute:** **zero.**
- **Implementation:** ~20 lines in `eval/adapters.py` + a reduction study script.
- **Expected gain:** +0.033 AP measured at n=268; expect +0.02–0.04 at n=667.
- **Failure condition:** paired CI on the full 667 includes zero.
- **Leakage risk:** ⚠️ **test-set selection.** Mitigations: (a) the mechanism is derived from
  documented dataset construction, not fitted; (b) decide on a fixed held-out half of test-public and
  confirm on the other half; (c) a **falsifiable external prediction** on untruncated clips — see the
  gate-3 block below.
- **Validation:** paired bootstrap, full 667, plus the external falsification test.
- → **Model: opus · Effort: high** — the leakage argument is the whole ballgame.

#### Gate 3 — the external falsification test · **REDESIGNED 2026-09-16 (session 10), user-approved**
#### ⚠️ CORRECTED LATER THE SAME SESSION — read the correction block before the design below.

> ### Correction, session 10 — two errors in this section's first version
>
> **Error 1 — "the AP test cannot be computed" was too strong.** The vendored
> `*_concensus.csv` files are positives-only, which is true and was verified twice. But they are
> **collision-*timing* annotations**, and a negative clip has no collision to time, so their being
> positives-only says nothing about the underlying datasets. **DAD's test split is 466 clips:
> 165 positive and 301 negative** — [the authors' project page](http://aliensunmin.github.io/project/dashcam/)
> (1,750 clips total; 620 pos / 1,130 neg; train 1,284 = 455/829; test 466 = 165/301). BADAS also
> publishes **AP 0.66 / AUC 0.87 on DAD**, and both metrics require two classes. **The AP test the
> plan originally specified is therefore computable after all**, from the full DAD download.
>
> **Error 2 — DAD is the wrong dataset for the mechanism test.** Its annotated collision time is
> effectively a **constant**:
>
> | Set | n | `Time-of-collision` median | **IQR** | usable variance? |
> |---|---|---|---|---|
> | `dad_test_concensus.csv` | 165 | 2.96 s | **0.16 s** | ❌ 84% within 0.25 s of exactly 3.00 s |
> | `dada2000_small_test_concensus.csv` | 221 | 5.33 s | **4.63 s** | ✅ range 0.37–14.43 s |
> | `dota_test_concensus.csv` | 598 | 4.65 s | 1.95 s | ✅ range 0.90–14.40 s (but disk-blocked) |
>
> The mechanism test asks whether the score peak **tracks** the annotated collision time. **Against a
> constant there is nothing to track** — no correlation is estimable, and a null result would be
> uninterpretable rather than a failure. The first version of this section called DAD's range
> "narrow" and warned against reading a null as a pass; that was right in direction and far too mild.
>
> **And the constant is structural, not observed.** DAD's page states the accident occurs "at the
> last 10 frames" of a 100-frame clip — frame 90, which at 30 fps is **exactly 3.00 s**, matching the
> annotation's clustering. So DAD's collision sits at normalised **~0.90, near the clip end — much
> like Nexar's 0.975.** DAD is not an untruncated mid-clip contrast at all. It fails as the
> mechanism-test target on both counts.
>
> **Resulting split of the gate, each test on the data that suits it:**
>
> | Test | Dataset | Why | Cost — **see the correction below** |
> |---|---|---|---|
> | **3a — mechanism** (peak tracks annotation) | **DADA-2000**, 221 clips | only set with real collision-time variance (IQR 4.63 s) and genuinely mid-clip events | **~10.3 h** at 167 s/clip, Air overnight |
> | **3b — AP** (last-window should LOSE on untruncated data) | **DAD full test split**, 466 clips (165 pos + 301 neg) | restores the plan's original design; the negatives exist | **~21.6 h** at 167 s/clip — but DAD's clips are 5 s, ~half Nexar's, so likely far less |
>
> > **⚠️ COST CORRECTION (2026-09-17).** The first version of this table used **~97 s/clip**.
> > `eval/run_baselines.py`'s own docstring forbids that figure: the rate measured over 140 clips of
> > the live sweep was **~167 s/clip end-to-end**, and it states *"the earlier ~97 s/clip (6-clip
> > smoke) and ~10 h (compute-only) figures are both too optimistic — do not plan against them."*
> > Both estimates above were therefore understated by ~1.7×. **Per-clip cost scales with clip
> > duration**, so neither figure transfers cleanly between corpora — DADA's clips are longer and
> > variable, DAD's are a fixed 5 s. **Measure a 20-clip pilot and replace these numbers before
> > committing to any full run** (§7.2's own instruction, and the `badas_smoke.py` precedent).
> > Stop condition: if a pilot implies > 18 h for 221 clips, re-plan onto the Studio or a subsample
> > stratified on `Time-of-collision` — never on score.
> >
> > **🔴 §7.2's tail-scoring optimisation does NOT apply to gate 3a.** Scoring only the final ~8
> > windows is what makes most of this plan fit the hardware, but **3a asks where in the clip the
> > peak falls**, which needs the dense stride-1 trace over the whole clip. A tail-only run would
> > presuppose the answer. **3a must be a full dense sweep**, and its cost cannot be reduced that
> > way. (3b, being an AP comparison of two reductions, *could* use a tail run — but only once the
> > reductions it compares are both computable from the tail, which `max` is not.)
>
> **R1 must survive both.** They fail independently and for different reasons, which is the point.
> DoTA stays excluded on disk (~55 GB against ~30 GB free on the Air).
>
> **Access note:** DAD is distributed by a Google Form request to the authors, not a direct
> download, and its terms are not posted publicly. **Confirm the terms in writing before use** —
> §22/§23 licence discipline applies, and a research-only grant would bar it from any commercial
> claim even though it may still be used as an internal falsification test.
>
> > ### 🔴 ANNOTATION-SOURCE CORRECTION (2026-09-17, session 11 — **D45**)
> >
> > **Gate 3a does NOT use `vendor/badas-open/annotation/dada2000_small_test_concensus.csv`.**
> > That file is named throughout this block and **its ids do not map onto the obtainable DADA
> > archive.** Only **132/221** matched; nine of DADA's 61 type folders are absent from the mirror;
> > and among the matched ones **11/132 annotate a collision after the clip ends** — `41_007` claims
> > 12.37 s in an 87-frame (2.9 s) clip, and no frame rate from 10 to 60 fps reconciles them. **The
> > annotation describes different videos.** Using it would have correlated each clip's peak against
> > a *different* clip's collision time and produced a confident, meaningless number.
> >
> > **3a's annotation is DADA's own `Sheet1` (`dada标注.xlsx`)**, shipped beside the clips as
> > `data/dada2000/gate3a/dada_gate3a_annotation.csv` (220 rows, stratified on collision position,
> > seed 0 — D47). Rows failing a consistency check are **dropped, never repaired** (D48).
> > **Do not "restore" the BADAS CSV.**
> >
> > **Mapping proven independently:** the sheet's `total frames` equals the on-disk frame count for
> > **1949/1962** clips — a quantity the mapping was not fitted on.
> >
> > **This also eliminates the time-base risk below rather than merely checking it.** Position is
> > `accident_frame / total_frames`, a **ratio**, so the frame rate cancels completely. The clips
> > were stitched at 30 fps with frame counts asserted preserved, and the annotation's seconds are
> > `accident_frame / 30` — the *same* 30 — so seconds and frames cannot drift apart.
> >
> > **Corpus, measured over the 220 selected clips:** normalised collision position median **0.541**,
> > IQR 0.282, range 0.084–0.995; seconds IQR **4.57 s**; only **3.6%** in the final 10% of the clip,
> > against **Nexar positives' 0.975 / 73.7%**. That contrast is what lets 3a discriminate at all.
> >
> > 🔴 **DADA posts no licence.** Internal falsification only (D43) — never in an external write-up
> > or a commercial claim.

> **The test design — the mechanism directly, not through an AP proxy.** Every positive row ships a
> `Time-of-collision`, so ask the question §3.2 actually rests on: *where does the score peak sit?*
>
> - **§3.2's claim:** Nexar positives peak at the clip **end** (0.975 at full n) because Nexar
>   truncates 500–1500 ms *before* the event. The peak tracks **the event**, which truncation has
>   pushed to the edge.
> - **The external prediction:** on untruncated clips the event is mid-clip, so the peak should sit
>   **at the annotated `Time-of-collision`**, not at the clip end.
> - **PASS (mechanism survives):** peak position clusters at `Time-of-collision`; normalised peak
>   position is materially below Nexar's 0.975 and tracks the annotation clip-by-clip.
> - **FAIL (R1 dies):** peaks pile up at the clip end *regardless* of when the collision was
>   annotated. That would mean the score merely **drifts upward with watch-time** — last-window would
>   be winning for a reason that has nothing to do with truncation, and the Nexar +0.0556 would be a
>   benchmark artifact, not a finding.
>
> **Why 3a complements 3b rather than replacing it:** the AP version (3b) asks whether last-window
> *loses* on untruncated data — a directional check with one bit of output. 3a measures the
> mechanism's own quantity against a ground-truth timestamp, clip by clip, and can fail in a way that
> **names the alternative explanation** (watch-time drift). Run both.
>
> **Statistical care — 3a has its own traps:**
> - **Dynamic range is the whole ballgame, and it is why the target moved to DADA.** Report the
>   correlation between measured peak position and annotated collision position **with its CI, and
>   report the annotation's own IQR next to it** so a reader can see whether there was variance to
>   explain. **A null correlation against a near-constant annotation is not a pass, not a fail, and
>   must never be reported as either.** DAD (IQR 0.16 s) is exactly that case.
> - **Time-base compatibility must be checked before the correlation is believed.** Our trace
>   timestamps are real seconds (`t = index / target_fps` after upstream resamples to 8 fps). The
>   consensus annotation's seconds are only the same seconds if the annotators used each clip's true
>   frame rate. DAD's ~3.00 s clustering is consistent with frame 90 read at 30 fps, while its page
>   describes 100-frame clips as 5 s (20 fps) — **the two readings disagree by 1.5 s.** Before
>   trusting any offset on DADA, verify one decoded clip's true duration and fps against its
>   annotation. **A systematic offset would corrupt 3a silently.**
>   **→ DISCHARGED for 3a by D45 (see the correction block above): the gate turns on a ratio, so the
>   frame rate cancels and there is no offset to carry. This bullet still binds 3b on DAD.**
> - Near-collisions (23 in DADA) have no impact; keep them separate from the 198 collisions.
> - State the comparison against a **clip-end null** explicitly: what peak position would a
>   watch-time-drift model predict, and does the measurement separate from it?
> - `t_end` is `(len-1)/fps`, never "end of clip" — upstream discards the final window (§21.4 item 7).
>
> **Statistical care — 3b:**
> - Use the **paired** bootstrap on the same DAD clips, as §1 mandates everywhere.
> - DAD's 301 negatives are the denominator; report it, as with every FP/hour figure in this project.
> - A DAD AP far from BADAS's published 0.66 means our harness differs from theirs — that is a
>   **harness-reproduction** finding (§8.1 tier 1), and must be resolved before 3b's ΔAP is read as
>   evidence about R1.
>
> **Reuse, do not rebuild:** `eval/timing.py::load_traces_abs` (keeps the absolute NaN offset — the
> 2-second trap), its `t_peak` argmax, and `eval/adapters.py`'s scoring path unchanged.

### R2 — Multi-temporal-scale ensemble · **TIER 1**

- **Hypothesis:** scoring at several `target_fps` values (16 frames at 4 / 8 / 16 fps = 4 s / 2 s / 1 s
  of context) and averaging beats any single scale.
- **Mechanism:** collision dynamics occur on multiple timescales; the checkpoint is fixed at one.
  Ensembling decorrelated views reduces variance.
- **Data:** existing 667 clips.
- **Compute:** **cheap given R1** — if the reduction is tail-based we need only the final ~8 windows
  per clip, not all ~81. Estimated ~4–5 h per scale (decode-bound), versus ~20 h for a dense sweep.
- **Implementation:** `target_fps` is already a constructor parameter. Mostly orchestration.
- **Expected gain:** +0.01–0.03.
- **Failure condition:** no scale beats 8 fps and the ensemble does not beat the best single scale.
- **Leakage risk:** low — ensemble weights must be fitted on the calibration half only, or fixed at uniform.
- **Validation:** paired bootstrap vs the R1 winner.
- → **Model: sonnet · Effort: medium**

### R3 — JEPA surprise channel · **TIER 1, highest novelty**

- **Hypothesis:** V-JEPA2's *own prediction error* is a collision signal independent of the trained classifier.
- **Mechanism:** V-JEPA2 predicts future latent states. A collision is precisely a violation of
  predicted dynamics, so ‖predictor_output − actual latent‖ should spike. **The model already computes
  this on every window and throws it away** (~25% of every forward pass — documented in `progress.md`
  §6.11 and the gate block). Literature supports latent prediction error as an anomaly signal
  (MTS-JEPA; world-model failure detection).
- **Data:** existing clips.
- **Compute:** one tail-scoring pass (~4–5 h); the predictor output is already being computed.
- **Implementation:** a `register_forward_hook` on the backbone in `BadasOpen.load()` — touches no
  vendored file. Moderate.
- **Expected gain:** high variance. Could be 0; could be a genuinely independent second channel.
- **Failure condition:** surprise score AP < 0.60 standalone, **and** no paired gain when fused.
- **Leakage risk:** none — entirely unsupervised.
- **Validation:** standalone AP/AUC, correlation with the classifier score (**report correlation
  honestly — if r > 0.8 do not call it independent**), and paired fusion test.
- → **Model: opus · Effort: high** — novel, and the independence claim must not be overstated.

### R4 — Flip test-time augmentation · **TIER 1**

- **Hypothesis:** averaging the score over a horizontal flip reduces variance and raises AP.
- **Mechanism:** standard TTA. Caveat: flipping reverses driving-side convention, which may be
  semantically harmful for a road-scene model — test on a subset before committing.
- **Compute:** doubles whichever pass it rides along with (~4–5 h tail-scoring).
- **Expected gain:** +0.005–0.02.
- **Failure condition:** flipped-only AP drops more than ~0.05 (indicates orientation dependence), or no paired gain.
- **Leakage risk:** none.
- **Validation:** paired bootstrap; subset pilot (100 clips) before the full pass.
- → **Model: sonnet · Effort: low**

### R5 — Learned trace aggregator · **TIER 2**

- **Hypothesis:** a small model over trace-shape features (max, last, argmax position, slope, std,
  area) beats any hand-picked reduction. §3.2 shows **argmax position alone separates classes**
  (0.984 vs 0.782).
- **Mechanism:** R1 is one hand-chosen point in this feature space; a fitted combiner finds a better one.
- **Data:** needs labelled traces **not** from the reported set → fit on the calibration half or on external data.
- **Compute:** seconds once traces exist.
- **Expected gain:** +0.01–0.04 over R1.
- **Failure condition:** no paired gain over R1 on the held-out half.
- **Leakage risk:** ⚠️ **highest in the plan.** Six features on ~334 clips overfits easily. Mitigation:
  logistic regression only, ≤ 6 features, fitted on the calibration half, reported on the eval half.
- **Validation:** held-out half + external transfer.
- → **Model: opus · Effort: high** — this is where a plausible-looking fake gain would enter.

### R6 — Frozen-feature probe on external data · **TIER 2**

- **Hypothesis:** a CPU-trained head on pooled V-JEPA2 features, trained on DoTA/DADA/DAD, ensembles
  with BADAS to beat BADAS alone.
- **Mechanism:** BADAS's head saw only Nexar; a head trained on other corpora makes different errors.
- **Data:** 984 externally annotated clips (already vendored annotations).
- **Compute:** feature extraction on external clips (~10–20 h, Studio weekends) + minutes to train.
- **Implementation:** forward hook to capture `pooled` (B, 1024) — the same hook as R3.
- **Expected gain:** +0.01–0.03 as an ensemble member; likely negative standalone on Nexar.
- **Failure condition:** ensemble shows no paired gain.
- **Leakage risk:** low — trained on external, tested on Nexar. Clean direction.
- **Validation:** paired bootstrap on Nexar test-public.
- → **Model: sonnet · Effort: medium**

### R7 — Nexar train split, targeted extraction · **TIER 2, conditional**

- **Hypothesis:** a window-level head trained on `time_of_alert` labels beats BADAS's clip-level head.
- **Mechanism:** train clips are uncut 40 s with event *and* alert timestamps. We can extract windows
  ending exactly 0.5 / 1.0 / 1.5 s before the event — **reproducing test-public's construction
  exactly** — giving ~22,500 well-matched window-level training examples.
- **Data:** 25.5 GB download (**Studio only** — 44 GB free on the Air is too tight).
- **Compute:** naive dense extraction is ~80 h and is *not viable*. Targeted extraction (~15 windows
  per clip) is **~4–8 h** — viable on one weekend.
- **Expected gain:** highest ceiling in the plan, but the most likely to disappoint.
- **Failure condition:** probe fails to beat BADAS on the eval half.
- **Leakage risk:** ⚠️ **structural.** These 1,500 clips are BADAS-Open's own training data, so the
  backbone has seen them and their features are abnormally separable. **Evaluation on test-public
  stays valid** (it is held out), but the probe is trained on an easier distribution than it will
  face. Must be reported explicitly. **Never usable for calibration** (§6).
- **Validation:** test-public only; report train-vs-test feature separability as a diagnostic.
- → **Model: opus · Effort: high** — the leakage reasoning is subtle and easy to get wrong.

### R8 — Calibration · **TIER 0 for credibility, not for detection**

Kept from v1, correctly demoted. **It cannot change AP or AUC** — monotone maps preserve ranking by
construction. Its value is a usable probability, a principled operating point, and threshold
portability across datasets. Protocol in §6.
- → **Model: opus · Effort: high** (protocol design) then **sonnet · medium** (implementation)

### R9 — IMU/CAN false-positive rejection channel · **TIER 1 for FP/hour, not for AP**

*Merged from the parallel session's `NEW_PLAN_v2.md` (its C2), which identified this and which this
plan had missed.*

- **Hypothesis:** BADAS's false positives have a *physical* signature — hard braking, potholes, speed
  bumps, sharp turns — that is visible in inertial data but not reliably in video. A small classifier
  over IMU/CAN features can suppress those scores without touching true collisions.
- **Mechanism:** this is the one candidate that adds a signal the video-only backbone **structurally
  cannot see**. It is not "detect impact" (killed by §3.2 — the crash is never in frame); it is
  "recognise the physical signature of a non-collision event that looks alarming on camera."
  `README.md` §28 independently names IMU as the single best false-positive discriminator.
- **Data:** comma2k19 ships synced CAN bus + IMU alongside video. **Verify against the actual
  downloaded release first — some mirrors are video-only.**
- **Compute:** rides on the R-negatives scoring pass already planned (§8.2); classifier training is
  CPU-seconds.
- **Implementation:** medium-high. comma2k19's format is idiosyncratic; needs a parser plus a feature
  pass.
- **Expected gain:** potentially the largest FP/hour improvement available. **Zero AP gain on Nexar** —
  see the limitation below.
- **Failure condition:** no FP/hour improvement at matched recall on a held-out third of comma2k19
  never used for mining or fitting.
- **Leakage risk:** medium, and specifically addressed by a **three-way partition of comma2k19**:
  (a) mining set — find BADAS's hard false positives; (b) fit set — train the rejection classifier;
  (c) report set — the final FP/hour number. Collapsing any two of these inflates the headline.
- **⚠️ Limitation this plan adds, which the parallel document did not state:** **Nexar clips carry no
  IMU data.** R9 can therefore be *developed and measured* on comma2k19 but **cannot be validated on
  the Nexar benchmark at all**, and it changes no number in §1. It also assumes the eventual product
  ships on hardware exposing IMU — true of most dashcams, but an assumption, not a given. Treat R9 as
  a product-track item whose evidence lives entirely on external data.
- **Validation:** three-way split; Poisson CI on the report set; cross-dataset generalisation claim
  only (§8.1), never a UK claim.
- → **Model: opus · Effort: high** — the partition discipline is where this silently goes wrong.

### Explicitly rejected

| Rejected | Why |
|---|---|
| Temporal persistence / smoothing | Measured: harmful (§3.1). Mechanism understood. |
| Channel B as impact detection | The crash is never in frame on positives. |
| Synthetic / CARLA | No identified mechanism; large effort; would not have been found by any evidence gathered here. |
| Backbone fine-tuning | No CUDA. |
| Full-rewrite | §5. |
| **Synthetic hard negatives by clip-mixing** (parallel session's C6) | Only an amplifier for R9, and R9 is unproven. Build it only if R9 shows signal — not before. |
| **`predictor_output` as a timing/severity signal** (parallel session's C4) | Its premise is already contradicted by D22: with the checkpoint's default masks the predictor reconstructs the tokens it was given, so a time-to-event signal is unlikely. **R3 is the better form of this idea** — use prediction *error* as an anomaly score rather than its magnitude as a clock. |
| **Ensembling with the retired MobileNetV2+LSTM** (parallel session's C7) | It is chance-level (AUC 0.5339) and its errors are corpus artefacts, not complementary signal. `progress.md` §12 forbids resurrecting it. |

---

## 5. Architecture decision: keep, hybrid, or rebuild

**Decision: Option B — keep the model and harness, replace the decision layer.**

Evidence:
- The harness is model-agnostic, resumable, leakage-tested, and self-checking. Every experiment above
  plugs in through a two-member duck type. Nothing in it obstructs any R-item.
- The baseline reproduces a published figure to within 0.003 AP. Discarding a validated baseline
  requires an identified defect; none was found.
- **The defect is in the layer above the model** — the clip reduction (§3.2) and the absent
  calibration — which is exactly what Option B replaces.

**What would change this decision:** if R1–R6 all fail their paired tests, the conclusion would be
that the frozen representation carries no extractable headroom, and the case for R7 (or a different
backbone) becomes the remaining path. That is a Week-4 decision, not a now decision.

---

## 6. Calibration protocol — closing hole #2

There is no clean held-out labelled Nexar data: the train split is BADAS's own training data **and**
is not downloaded; test-private has no videos and no public labels. So the protocol must be built
from test-public alone, with external transfer as the generalisation check.

**Tier 1 — primary, deployable.** Stratified split of test-public into calibration (n≈333) and
evaluation (n≈334) halves with a fixed seed. Fit the map on the calibration half; report every
calibrated metric on the evaluation half only. Produces a real deployable map and a clean claim.
Cost: CIs widen by ≈√2.

**Tier 2 — split-luck control.** Repeat Tier 1 over 100 random stratified splits; report the
distribution of evaluation-half ECE. Prevents a lucky split being mistaken for a result.

**Tier 3 — best-estimate.** 5-fold cross-fitted out-of-fold ECE (the §3.4 numbers). Uses all data,
yields no single deployable map. Report as *"achievable calibration"*, never as a deployed result.

**Tier 4 — transfer.** Fit on DAD/DADA, apply to Nexar (and vice versa). Answers whether the map is
a property of the model or of the dataset. **A negative result here is a finding worth reporting.**

**Forbidden:** fitting on Nexar train. The backbone memorised those clips, so their score
distribution is shifted; a map fitted there will not transfer. **This is currently an argument, not
a measurement — if R7 proceeds and train scores exist, test it and convert it into one.**

**Reported metrics:** ECE (equal-width, 10 bins, matching `benchmark.ece`), adaptive ECE
(equal-mass), Brier, NLL, reliability diagram, each with bootstrap CIs.

---

## 7. Compute, hardware and schedule — closing holes #3, #5, #9

### 7.1 Real capacity

| Machine | Availability | Weekly usable |
|---|---|---|
| M4 Air | 24/7, unattended | ~120 h (after the current sweep ends) |
| M4 Max Studio | 2–4 h weekdays, ~10 h weekends | ~32 h, **in chunks** |

Every job must be chunk-friendly. The harness already is — append-only `scores.jsonl`, resume by id.
**No workflow in this plan assumes uninterrupted Studio access.**

### 7.2 Measured and estimated costs

| Job | Cost | Machine |
|---|---|---|
| Dense stride-1 sweep, 667 clips | ~18–20 h | Air, overnight |
| **Tail-only sweep (final ~8 windows)** | **~4–5 h (decode-bound)** | Air |
| Reduction / calibration / fusion studies | seconds | either |
| External benchmark sweep (984 clips) | ~20–26 h dense, ~6–8 h tail | Studio weekend |
| Nexar train targeted extraction (R7) | ~4–8 h + 25.5 GB download | Studio only |
| comma2k19 / ZOD negatives, 10 h at 1 Hz | ~6–10 h, **pilot first** | Air |

**The tail-scoring insight is what makes this plan fit the hardware.** Once the reduction is
tail-based, most sweeps drop ~10× and the binding constraint becomes video decode, not the GPU.
Every figure above is an estimate and **must be replaced by a measurement** before the full run —
the `badas_smoke.py` precedent.

### 7.3 CPU feasibility — no claims before measurement

Every timing in this project to date is MPS. Before any "CPU" or "real-time" language is used
anywhere: benchmark one window on M4 Air CPU and on MPS, and report both.

Arithmetic to be checked, not asserted: 16 frames at 8 fps = 2 s of video; at the measured 0.63 s per
window on MPS, a 1 Hz alert cadence is ~0.63× real-time — **plausibly real-time on MPS, probably not
on CPU.** Until measured, the honest phrasing is *"real-time at 1 Hz alert cadence on Apple Silicon
GPU; CPU throughput not yet measured."*

### 7.4 Four-week schedule

**Week 1 — zero-compute, highest value.** Sweep finishes. Confirm R1 on all 667 with the held-out
protocol. Build the reduction study, calibration (§6 Tiers 1–3), the operating-point policy with
Poisson CIs, and the regression test. CPU/MPS benchmark. *Studio not required.*

**Week 2 — cheap ensembles.** R4 flip TTA (100-clip pilot, then full tail pass). R2 multi-scale at
4 fps and 16 fps. R3 hook implemented and the surprise channel measured. Air runs overnight; Studio
weekday hours for the pilots.

**Week 3 — external validation (Studio weekend).** DAD first (165 clips, smallest), then DADA.
**Run the R1 falsification test here** — this is the week the reduction finding either survives or
dies. Ego/non-ego and time-to-collision analysis. R6 probe if features are captured in the same pass.

**Week 4 — negatives and consolidation.** comma2k19 pilot then run; ZOD if week 3 left room. Fuse the
surviving channels, calibrate the fused score, build the demo on real outputs. Decide on R7.

**DoTA (55 GB, 598 clips) is a full weekend on its own and is explicitly a Week-4-or-later item.**
R7 costs a second weekend. Both cannot happen in this month; choose one based on Week-3 results.

---

## 8. Data strategy — closing holes #4 and #6

### 8.1 Independence taxonomy

| Tier | What it proves | Sources |
|---|---|---|
| **Harness reproduction** | our pipeline matches a published figure | DAD / DADA / DoTA against BADAS's own published numbers on **Nexar's own consensus annotations** — tests *us*, not the model |
| **Cross-dataset generalisation** | a Nexar-trained model transfers to other corpora | same clips, but the *model* never saw them — genuine, though the labels are still Nexar's |
| **Genuinely independent** | neither trained on nor annotated by Nexar | **comma2k19 and ZOD negatives** — all-negative by construction, so no annotation dependency at all |

v1 conflated the first two. Only the third is fully independent evidence, and it exists only on the
negative side — which is precisely where the product claim lives.

### 8.2 Negative footage

| Source | Licence | Content | Role |
|---|---|---|---|
| Nexar test-public | nexar-open-data | 0.899 h, urban US | current denominator |
| **comma2k19** | **MIT** | 33 h, California **highway** | **highway stress test — NOT a general rate** |
| **ZOD** | **CC BY-SA 4.0, commercial OK** | 1,473 × 20 s + 29 multi-minute drives, European, diverse weather/light | second, distinct negative domain |
| BDD100K | research-only | 1,100 h US urban | **excluded** — licence incompatible with a commercial prototype |

**comma2k19 alone would flatter us:** highway driving is the easiest negative case — few pedestrians,
few intersections, few occlusions. Any FP/hour measured there is a **floor**. Reporting it as a
general rate would be the mirror image of the 0.90-hour problem. ZOD's European urban/rural mix is
the corrective, and both must be reported **separately, never pooled**.

---

## 9. Track C — executable or dead

v1 criticised this as a wish and then re-listed it unchanged. Fixed:

**Protocol:** 10 messages in Week 1 to UK/EU fleet safety managers via LinkedIn or published contact
addresses, each with the same three questions (how are collisions currently reviewed; how many hours
of footage per vehicle per week; what false-alarm rate would make an alert useless). Logged in one
CSV in the repo.

**Kill condition:** fewer than 3 substantive replies by end of Week 2 → **formally close Track C**,
record it in `progress.md`, and stop listing it. A P0 that has not started in six sessions is not a
priority, and pretending otherwise has already cost this project credibility with itself.

→ **Model: haiku · Effort: low** — drafting only; the sending is yours.

---

## 10. THE DECISIVE ANSWER

> *If we had one month under these exact constraints, what sequence gives the highest probability of
> a materially stronger prototype than AP 0.8349 / AUC 0.8498?*

**In order: R1 → R4 → R2 → R3 → fuse → calibrate. R6/R7 only if the first four underdeliver.**

**R9 runs on a separate track**, because the question asks specifically about beating AP 0.8349 and
R9 cannot move that number — Nexar has no IMU. It is nonetheless the strongest *product* candidate in
the plan and should take the Week-4 negatives slot, since it rides on the same comma2k19 scoring pass.

Rationale: R1 is already measured at +0.033 paired, costs nothing, and is mechanistically explained.
R4 and R2 are cheap variance reductions that the tail-scoring trick makes affordable. R3 is the one
genuinely novel idea and the best story even at modest gain. Fusion and calibration convert channels
into a product. R7 has the highest ceiling but the worst cost-to-confidence ratio and a structural
leakage caveat — it is a Week-4 contingency, not an opener.

**Honest probability assessment:**

| Outcome | Probability |
|---|---|
| Beat baseline by ≥ 0.03 AP, paired CI excludes zero | **~65%** — carried almost entirely by R1 |
| Beat by ≥ 0.05 AP | ~30% — needs R1 plus at least one ensemble win |
| No detection gain survives (calibration-only outcome) | ~25% |
| R3 yields a channel with correlation < 0.8 to the classifier | ~35% |

The single largest risk is that R1's +0.033 shrinks toward zero at n=667 — split-half already showed
one half at −0.009. **If that happens, the honest outcome of this month is a well-calibrated,
externally-validated prototype with no detection improvement**, and that must be reported as such
rather than rescued with a worse experiment.

---

## 11. Adversarial review of *this* plan

As instructed, turning the same hostility on v2.

1. **R1 was chosen by looking at the test set.** The single most serious remaining risk. Three
   mitigations are in place (mechanism-first derivation, held-out confirmation, external
   falsification), but none is as good as a dataset we never touched. **Residual risk: real.**
   *Update 2026-09-16:* mitigations (a) and (b) are now **discharged** — gate 1 passed at full n=667
   (+0.0556, CI [+0.0263, +0.0876]) and gate 2's held-out median (+0.0551) matched it with the
   effect positive in 100% of 1000 splits, so there is **no measurable winner's curse**. Both still
   live inside Nexar. Mitigation (c), the only one that leaves it, is gate 3 above — **open**.
2. **§3.1 and §3.2 rest on 268 clips in non-random order.** The sweep visits interleaved
   positive/negative by sorted id, so this is a balanced but not random prefix. If id correlates with
   collection batch, the estimate is biased. Unquantified — flagged, not solved.
3. **Weeks 3 and 4 are over-subscribed.** DoTA and R7 each need a full weekend and there are ~2
   Studio weekends in the window. The plan says choose one; it does not say which. That decision is
   deferred to evidence, which is defensible, but it means the month may end with external
   validation *or* R7, not both.
4. **R2/R3/R4 cost estimates assume the tail-scoring optimisation works.** If decode dominates more
   than expected — and the existing gap between 0.63 s/window compute and ~97–167 s/clip end-to-end
   suggests decode is already dominant — savings could be far below 10×, and Week 2 slips.
   **This is the least-verified assumption in the plan.**
5. **Fusing R1–R4 multiplies selection risk.** Each channel chosen partly on test-public performance
   compounds. Fusion weights must be fitted on the calibration half only, and even then the fused
   number is optimistic. Prefer uniform weights unless a fitted combiner clearly wins on held-out data.
6. **The ±0.04 bar cuts both ways.** Several R-items have expected gains (+0.01–0.02) *below* what an
   unpaired test could resolve. They are justified only by paired testing — which is correct, but it
   means the headline improvement may be real yet unprovable to a sceptic who demands unpaired CIs.
7. **Still no moat.** v2 improves detection but produces no proprietary data. §5 of v1 stands
   unamended, and R7 is the only item that would change it — the one item ranked last.
8. **Success on Nexar may not transfer.** Every detection gain here is tuned to a benchmark whose
   construction (truncation) is the very thing R1 exploits. If deployment clips are not truncated
   that way, **R1's advantage may not exist in production** — the streaming score is the last window
   *by definition*, so the comparison against max becomes moot rather than favourable.
   **This is the deepest caveat in the plan and belongs in any external write-up.**

---

## 12. Claims permitted and forbidden

**Permitted (after the stated validation):** reproduced a published model to within 0.003 AP ·
identified and corrected a reduction mismatch worth ~+0.03 AP paired · corrected uncalibrated
probabilities from ECE 0.33 to ≈0.05 with no ranking loss · demonstrated with numbers that temporal
smoothing harms this task · evaluated across N external benchmarks with ego/non-ego and timing
breakdowns · every metric with denominator and CI.

**Forbidden until earned:** any production FP/hour · any general false-alarm rate from highway-only
footage · "real-time" or "CPU-capable" before §7.3 is measured · "independent channel" for R3 if
correlation ≥ 0.8 · any claim from a provisional partial-sweep number · any mTTA on Nexar
test-public · proprietary-data or trained-in-house claims.

---

## 13. Kill conditions

| Item | Kill if |
|---|---|
| R1 | paired CI on 667 includes zero **or** gate 3a fails on DADA-2000 (peak sits at the clip end regardless of the annotated `Time-of-collision` — watch-time drift, not truncation) **or** gate 3b fails on DAD's 466-clip test split (last-window does *not* lose to max on untruncated data) |
| R2 | ensemble fails to beat best single scale on held-out half |
| R3 | standalone AP < 0.60 **and** no paired fusion gain |
| R4 | flipped AP drops > 0.05, or no paired gain at n=100 pilot |
| R5 | no paired gain over R1 on the held-out half |
| R7 | not started unless R1–R4 deliver < +0.02 combined |
| R9 | downloaded comma2k19 release ships no IMU/CAN, **or** no FP/hour gain at matched recall on the held-out report third |
| Track C | < 3 replies by end of Week 2 |
| comma2k19 | pilot throughput implies > 12 h for 10 h of footage |
