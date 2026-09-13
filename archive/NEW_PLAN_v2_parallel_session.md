# NEW PLAN v2 — Red-Team Revision Under Real Hardware Constraints

**Written 2026-09-13 (session 8). Status: PROPOSAL — nothing in it has been implemented.**
Supersedes `NEW_PLAN.md` as the active proposal. `README.md` remains the master plan and
`progress.md` remains the execution state; neither is modified by this document. Nothing here is
executed until this plan is reviewed and accepted.

## 0.1 A scope caveat, stated up front

The instruction that produced this document referred to "your 10 points" as an existing red-team
finding to close. **That artifact does not exist in this session's visible context** — it was not
found in `README.md`, `progress.md`, or `NEW_PLAN.md`, and no prior-session record of it was
located. Rather than fabricate a list to match, this document treats the **ten numbered
requirements in the actual instruction** as the red-team findings to close, and additionally
cross-references `NEW_PLAN.md`'s six findings (F1–F6) wherever they overlap — they turn out to map
onto exactly the same failure modes (F1↔calibration leakage, F3↔provisional-conclusion promotion,
F5/N2.1↔dataset independence, N3↔comma2k19 misuse, Track C↔point 8). If a specific prior "10
points" document exists outside this session, it should be supplied and this plan re-diffed against
it; until then, this is the closest faithful reconstruction available.

## 0.2 Repo state actually verified this session (not assumed)

| Claim | Verified state |
|---|---|
| Nexar test-public | 667 clips on disk (334 positive + 333 negative dirs, 2.7 GB), fully scored |
| Nexar train / test-private | Metadata only — 72 KB / 44 KB, **zero `.mp4`** (confirms F1) |
| `vendor/badas-open/annotation/` | `dad_test_concensus.csv` 165 rows, `dada2000_small_test_concensus.csv` 221 rows, `dota_test_concensus.csv` 598 rows (header-inclusive counts; 164/220/597 clips) |
| Committed clip-level sweep | `runs/baselines/badas-open/scores.jsonl` — **667/667 complete**, `metrics.json` present |
| Per-frame trace sweep (feeds F3) | `runs/baselines2/badas-open/scores.jsonl` — **222/667 complete as of this session** (was ~194/667 when `NEW_PLAN.md` was written — confirms F3 is still partial, not stalled) |
| Calibration code | `eval/calibration.py` **does not exist yet** — N1.1 is unbuilt, the F2 table in `NEW_PLAN.md` was produced by an ad hoc script, not a committed reproducible artifact |
| CPU/MPS throughput benchmark | `scripts/badas_fps_probe.py` exists but **only validates config coherence** (the `original_fps`/`target_fps` reconciliation) — it does not measure end-to-end throughput on any of the three real machines. The only real timing numbers on record are from `badas_smoke.py` (0.856 s/window MPS compute-only) and the 6-clip `run_baselines.py` sample (~97 s/clip end-to-end) — both presumably on whatever machine ran session 6/7, not confirmed to be either of the two machines this plan must now target |
| BADAS weights | `models/badas/` — 3.7 GB, present |
| `eval/` package | 5 files, ~1520 lines total across `eval/` + `scripts/` — functioning adapter/benchmark/plots pipeline exists and is the correct integration point for everything below |

---

## 1. Hardware reality — this supersedes every compute estimate in `README.md`, `progress.md`, and `NEW_PLAN.md`

**Every prior compute plan in this repo (`progress.md:186,1547,2565` and `NEW_PLAN.md` throughout)
assumes "M1 laptop + M4 Max at college, ~40 GB free, installs allowed" with unstated availability.**
That is not the constraint stated for this plan. The actual constraint is:

| Machine | Availability | Role |
|---|---|---|
| **M4 MacBook Air** | 24/7 | Orchestration, small CPU-only jobs, calibration fitting (seconds), all N1-scale work, monitoring long Studio jobs remotely |
| **Mac Studio M4 Max** | ~2–4 h weekdays, ~10 h weekends | The *only* machine for anything GPU/MPS-bound: BADAS inference sweeps, V-JEPA2 feature extraction, any per-frame trace work |

Weekly Studio budget: **5 weekdays × 3 h (midpoint) + 2 weekend days × 10 h = 15 h + 20 h = ~35 h/week**,
assuming every session is actually used — treat 25–30 h/week as the realistic planning number after
setup/teardown overhead and missed days. **This is a ~4–5× reduction from an implicit "10+ h/day"
assumption**, and it is the single most important number in this document: it gates every timeline
estimate in §9 and rules out several of `NEW_PLAN.md`'s Phase N2/N3 items being done in one sitting.

Consequence for the already-measured ~18 h full-stride-1 BADAS sweep (`README.md` Phase 5 task 2):
that sweep **cannot run in one weekday session** (budget: 2–4 h) and **can barely fit one weekend
day** (10 h budget vs. 18 h job) — it must span a weekend boundary or run unattended overnight
across two weekend days, or the stride must be relaxed (§3 already documents stride 2 → ~9 h as an
acceptable, disclosed deviation). **Any future full-corpus sweep at BADAS's measured throughput
should default to stride 2 unless stride 1 is specifically required**, purely as a scheduling
consequence of this hardware, not a scientific one — and this must be logged as a deviation exactly
as `README.md` Phase 5 already requires.

**MPS availability correction:** the M4 Air is Apple Silicon with an on-die GPU addressable via
PyTorch's MPS backend — it is not "CPU-only" in the way `progress.md`'s CCD-era language sometimes
implies. Any place this plan or the codebase says "CPU-only" should be read as "no CUDA, MPS where
available, degrade to CPU if MPS is unavailable or numerically unstable for a given op" — and this
must be benchmarked (§5), not assumed, on both machines separately, because MPS throughput on an
Air's smaller GPU vs. the Studio's M4 Max GPU will differ by a large, currently unmeasured factor.

---

## 2. Point-by-point closure index

| # | User's point | Closed in | One-line status |
|---|---|---|---|
| 1 | Detection improvements beyond calibration | §4 | 7 candidates, full 9-field template, ranked |
| 2 | Calibration-data protocol | §5 | Nested cross-fitting on test-public only; external transfer as a *separate*, weaker claim |
| 3 | Compute plan vs. real hardware | §1, §9 | Done above; timeline in §9 built directly from it |
| 4 | comma2k19 reclassification + better FP sources | §6 | Reclassified to hard-negative/stress-test; 3 alternative sources evaluated, 1 recommended |
| 5 | No unbenched CPU/real-time claims | §7 | Every such claim in the repo flagged; benchmark-first protocol specified |
| 6 | Dataset independence audit | §8 | Three-way table for every dataset touched or proposed |
| 7 | Revisit F3 and other provisional findings | §10 | F3 kept provisional with explicit promotion criteria; two other provisional items surfaced |
| 8 | Track C: executable or killed | §11 | Kept, made executable with a script, quota, and kill condition |
| 9 | Timeline from real compute | §9 | Week-by-week, multi-weekend items flagged explicitly |
| 10 | Quantify uncertainty | §12 | Bootstrap CIs specified for every comparison currently stated as a point estimate |
| — | Ranked plan, optimized for expected gain not ease | §13 | |
| — | Keep/hybrid/rebuild verdict | §14 | |
| — | One-month decisive answer | §15 | |
| — | Adversarial self-review | §16 | |

---

## 3. What changes from `NEW_PLAN.md`, and why

`NEW_PLAN.md`'s Phase N1–N4 structure survives largely intact — it was not wrong, it was
**incomplete on detection improvement** (it explicitly dropped every detection-improving idea: probe
training, multi-head, Channel B, synthetic data — see its §4 "Dropped" table) and **silent on
compute reality**. This plan does not discard N1–N4; it:

1. **Reopens the "Dropped" list from `NEW_PLAN.md`§4 as candidates**, re-evaluated against the real
   M4 Air/Studio budget rather than an assumed richer one, with the explicit instruction from this
   session's brief that ease-of-execution must not be the ranking criterion.
2. **Replaces the compute estimates throughout** with the §1 budget.
3. **Adds a rigor layer** (§5 calibration protocol, §8 dataset independence, §12 uncertainty) that
   `NEW_PLAN.md` gestured at (its Poisson-CI note in N1.3) but did not apply everywhere it should.

---

## 4. Detection-improvement candidates (point 1)

Every candidate uses the 9-field template as required. Ranked in §13 by *expected AP/FP-hour gain
per Studio-hour spent*, not by implementation simplicity.

### C1 — Cross-fitted logistic/GBM head on frozen BADAS features from external annotated corpora (DAD+DADA+DoTA)

- **Hypothesis:** BADAS-Open's zero-shot score is a strong but generic ranker; a small supervised
  head trained on the 984 externally-annotated clips (ego-involved subset) can shift the *decision
  boundary* toward ego-collision specifically, since Nexar training data was not ego-filtered the
  same way.
- **Mechanism:** extract BADAS's `last_hidden_state` + consumed `predictor_output` (README Phase 5
  task 3 fix) as a fixed feature per clip; train a small logistic regression or shallow GBM (≤5
  features to stay interpretable, per README §28's fusion-layer philosophy) on ego-involved DAD/DADA
  labels; evaluate held-out.
- **Data:** 984 externally annotated clips (need to download DAD 165, DADA 221, DoTA 598 videos —
  **not yet on disk**, this is new acquisition, not reuse of `test-public`).
- **Compute:** feature extraction at BADAS's measured ~97 s/clip end-to-end × 984 clips ≈ **26.5 h**
  Studio time (at stride 1; stride 2 ≈13 h). This alone consumes most of one weekend. Head training
  itself is CPU-seconds.
- **Implementation cost:** medium — reuses `eval/adapters.py` pattern, needs a new small
  `train/probe_head.py` (~150 lines) plus feature-caching (this closes F6: features must be
  returned, not discarded, requiring a ~5-line patch to `EnhancedVideoClassifier.forward()`).
- **Expected gain:** unknown, plausibly modest (0.01–0.03 AP) — this is a linear head on features not
  fine-tuned for this exact task; DAD/DADA are smaller and noisier-labeled than Nexar. Not free of
  overfitting risk given the probe's own held-out set is under 1,000 clips.
- **Failure condition:** held-out AP on external ego-collision labels does not exceed the raw BADAS
  score's AP on the same labels by more than its bootstrap CI half-width (§12) — i.e., a
  statistically indistinguishable result kills this.
- **Leakage risk:** **high, and this is the single most dangerous item in this plan.** The
  consensus CSVs were used by BADAS's own authors to report their *published* DAD/DADA/DoTA figures
  (`vendor/badas-open/badas/config.json`). If any of these 984 clips were seen during BADAS-Open's
  own pretraining/fine-tuning, features on them are not "frozen and independent" — they are
  memorized. **This must be checked before any training happens**: BADAS's model card and paper
  state training was on Nexar's 1,500 videos only, not DAD/DADA/DoTA, which supports independence —
  but this is a claim from BADAS's authors about their own training set, unverifiable from outside,
  and should be stated as an assumption, not a fact, in any report.
- **Validation:** held-out split *within* the 984 (e.g. 70/30 by clip, source-grouped by original
  video where DoTA's `ego_discrepancy` metadata allows it), bootstrap CI on the AP delta, and
  explicit statement of the unverifiable-training-independence assumption above.

### C2 — Hard-negative mining from comma2k19 to fit a CPU-only rejection classifier (Channel B replacement)

- **Hypothesis:** F3 already shows Channel B as originally conceived (impact detection) cannot work
  because Nexar positives are truncated pre-event. But a **rejection classifier** — trained to
  recognize BADAS's own *false-positive* failure modes on genuinely negative, licensed driving
  footage — is a different, achievable idea: not "detect impact," but "suppress BADAS scores that
  spike on hard braking/potholes/turns without collision."
- **Mechanism:** score comma2k19 negatives with BADAS (per §6's pilot), collect the highest-scoring
  false-positive segments, extract lightweight CV features (optical-flow magnitude, IMU-if-available
  from comma2k19's own sensor logs — comma2k19 *does* ship CAN/IMU data, which is a genuine, underused
  asset here), train a small classifier that down-weights BADAS's score when the physical signature
  looks like a false-positive class rather than collision.
- **Data:** comma2k19 (MIT, 33 h, includes synced CAN bus + IMU — verify this against the actual
  downloaded release before relying on it, some comma2k19 mirrors ship video-only).
- **Compute:** scoring 33 h of comma2k19 at deployment cadence (1 window/s, not stride-1) is the
  same measurement `NEW_PLAN.md` N3.1 already proposes — reuse it rather than duplicate. Classifier
  training is CPU-seconds once features exist.
- **Implementation cost:** medium-high — needs IMU/CAN parsing (new code, comma2k19's format is
  idiosyncratic), a feature extraction pass, and a small trained classifier with its own leakage
  discipline (never fit on the same negatives used to report the final FP/hour number — see §5's
  nested structure).
- **Expected gain:** this is the plan's best shot at actually moving FP/hour, because it is the first
  candidate that uses a signal (IMU) BADAS's video-only backbone cannot see at all — potentially
  large (README §28 independently identifies IMU as "the single best false-positive discriminator").
  But comma2k19 is highway/US driving, not UK dashcam distribution — expect the gain to be real but
  smaller when it eventually meets UK data (§8 addresses this generalization gap explicitly).
- **Failure condition:** FP/hour with the rejection classifier does not improve over raw BADAS FP/hour
  at matched recall, on a **held-out third of comma2k19 never used for mining or fitting**.
- **Leakage risk:** medium — must strictly partition comma2k19 into (a) mining set (find hard FPs),
  (b) fit set (train rejection classifier), (c) report set (final FP/hour number), three-way, or the
  headline number is inflated by fitting-to-the-test-set-by-another-name.
- **Validation:** three-way split above; Poisson CI on FP/hour on the report set; explicit statement
  that this is a **cross-dataset generalization test at best** (§8), not evidence about UK
  performance.

### C3 — Multi-frame temporal head replacing `max` reduction, trained (not just evaluated) on per-frame BADAS traces

- **Hypothesis:** F3 shows `max` beats mean/persistence *as fixed, untrained reduction rules*. A
  small *trained* temporal aggregator (e.g., a 2-layer causal 1D-conv or tiny attention pool over the
  per-frame score/feature sequence) could beat a fixed rule, since it can learn asymmetric weighting
  (e.g., recent frames matter more, consistent with truncated-clip evidence) rather than committing to
  pure max or pure mean.
- **Mechanism:** use the per-frame traces already being produced by the in-flight sweep
  (`runs/baselines2/badas-open/`, 222/667 and growing) as training data for a tiny sequence model;
  train/eval must be on disjoint clip subsets to avoid the exact leakage this plan is trying to close
  everywhere else.
- **Data:** the 667 per-frame traces once the sweep finishes (no new acquisition — this is the
  cheapest candidate in the whole list on data cost).
- **Compute:** trivial — CPU, seconds to minutes, once traces exist. The expensive part (producing
  the traces) is already sunk/in-flight.
- **Implementation cost:** low-medium — ~100 lines, reuses the existing `CachedScores`/`eval/benchmark.py`
  path per `NEW_PLAN.md` N1.1's already-planned JSONL fix.
- **Expected gain:** genuinely uncertain and possibly negative — with only 667 clips split further
  into train/held-out for this head, the effective *n* for fitting a temporal model is very small,
  and F3 already shows fixed `max` is a strong baseline (AP 0.9124 on the partial 194-clip sweep).
  The honest expectation is a **small gain or no gain**, but it is the cheapest experiment to actually
  run given the sweep is already producing the data for free, so its cost/benefit ratio is favorable
  even at low expected gain.
- **Failure condition:** trained aggregator's held-out AP does not exceed `max`'s AP by more than the
  bootstrap CI on the difference.
- **Leakage risk:** low if train/held-out split is enforced and is the *same* clip-level split used
  for calibration (§5) to avoid compounding partitions across too many small subsets.
- **Validation:** k-fold on the 667 (once complete), bootstrap CI on AP delta vs. fixed `max`.

### C4 — Near-miss / severity re-labeling from BADAS's `future_prediction_seconds` output

- **Hypothesis:** BADAS's discarded `predictor_output` (README Phase 5 task 3) encodes a
  *time-to-event* signal (`future_prediction_seconds: 1.0`) that could be repurposed as a continuous
  severity/urgency score rather than the current binary collision probability — potentially giving a
  cheap "near-miss vs. collision" distinction the product needs (README §28) without new labels.
- **Mechanism:** consume `predictor_output` per the concat fix already scheduled (README Phase 5 task
  3, not yet done); inspect whether its magnitude correlates with ground-truth `time_to_accident` in
  the Nexar metadata (`time_of_event`, `time_of_alert` fields — present per README §16 task 3, though
  noted corrupted for `time_of_event` specifically, `time_to_accident` column existence needs
  re-verification against the actual metadata.csv schema before this is trusted).
- **Data:** Nexar test-public metadata (already on disk), no new acquisition.
- **Compute:** near-zero incremental — this rides on the Phase 5 task 3 fix that's already planned
  and largely a compute-only re-run of the existing sweep.
- **Implementation cost:** low — analysis script, not a new model.
- **Expected gain:** speculative and possibly zero product-metric gain (this doesn't move AP/FP-hour,
  it adds an output field) — but if it works it's a "free" second output for the multi-head goal
  README §28 sets, at near-zero cost since Phase 5 task 3 is already required work.
- **Failure condition:** no correlation between `predictor_output` magnitude and any usable timing
  field; report as a negative result rather than force a claim.
- **Leakage risk:** none — purely observational analysis on already-held-out data.
- **Validation:** correlation with bootstrap CI; explicitly caveat the `time_of_event` corruption
  finding (README Phase 4 task 1) since any timing-based validation inherits that data quality issue.

### C5 — V-JEPA2 frozen-feature attentive probe fine-tuned on Nexar train (the idea F1/F6 killed) — **kept killed, restated why**

- **Hypothesis (as originally proposed):** a probe trained on Nexar's actual 1,500-clip train split
  would outperform BADAS's own zero-shot head, since it is task/domain-matched.
- **Mechanism:** frozen V-JEPA2 backbone → cached features → small trainable head, exactly the
  frozen-backbone pattern README §28 recommends and praises in the legacy Colab pipeline.
- **Data:** Nexar train, 1,500 clips, 25.5 GB — **not currently downloaded.**
- **Compute:** downloading is trivial; feature extraction at BADAS's measured throughput ≈ 1,500 ×
  97 s ≈ **40 h** Studio time (stride 1) — over one full weekend's budget by itself, before any
  training.
- **Implementation cost:** high — needs the same `forward()` patch as C1/C2 plus a full training loop.
- **Expected gain:** **structurally close to zero**, and this is why it stays killed: F1 establishes
  Nexar train is BADAS-Open's own training data. A probe trained on it is not learning a new signal —
  at best it is re-deriving what BADAS-Open's own head already learned from the same 1,500 clips
  (and BADAS had access to the *full* backbone gradient, not just a frozen-feature linear probe, so a
  probe should be expected to underperform BADAS's own head on its own training distribution, not
  beat it). At worst, evaluating this probe on `test-public` and reporting a number is safe (no
  leakage into the reported eval set), but the *training* signal itself is redundant with what
  produced the baseline being compared against.
- **Failure condition:** this is already the expected outcome; the item is not proposed for
  execution.
- **Leakage risk:** N/A — the risk that kills this idea is redundancy, not leakage, though F1 is
  also independently a leakage argument if this probe's output were ever used as its own calibration
  or evaluation set.
- **Validation:** N/A — **verdict: do not build.** Retained here only because the brief requires every
  "dropped in the prior plan" idea to be explicitly re-examined, not silently dropped again.

### C6 — Synthetic hard negatives via clip-mixing (splice normal driving with hard-brake motion profiles)

- **Hypothesis:** README §33 already flags this as a cheap, effective augmentation
  ("synthetic hard negatives via clip mixing ✓"). Applying it specifically to BADAS's *own*
  false-positive modes (rather than as training augmentation for a from-scratch model, which nobody
  is training) could expand the hard-negative mining set in C2 without needing more comma2k19/UK
  hours.
- **Mechanism:** take comma2k19 (or eventually UK) clips, apply time-warping / frame-rate perturbation
  to simulate a "sudden motion" signature without an actual incident, re-score with BADAS, add
  high-scoring synthetic negatives to C2's rejection-classifier training set.
- **Data:** comma2k19 (already covered), no new acquisition.
- **Compute:** low — clip mixing is cheap; re-scoring cost is bounded by however much synthetic data
  is generated, controllable.
- **Implementation cost:** medium — new augmentation code, and augmentation-correctness testing.
- **Expected gain:** small, incremental to C2 — this is an amplifier for C2, not independent. Only
  worth building *after* C2 shows the rejection-classifier approach has any signal at all.
- **Failure condition:** synthetic negatives don't resemble real BADAS false-positive score
  distributions (checked by comparing score histograms, synthetic vs. real mined hard negatives).
- **Leakage risk:** low, but synthetic negatives derived from the *same* comma2k19 clips used in C2's
  mining set must stay in the same partition as their source clip, not be treated as independent data.
- **Validation:** same three-way split discipline as C2, since this only ever feeds C2.

### C7 — Ensemble of BADAS-Open with the retired MobileNetV2+LSTM model

- **Hypothesis:** even a chance-level model (T3: AUC 0.5339) could in principle contribute
  *uncorrelated* error if its failure modes differ from BADAS's — ensembling sometimes helps even with
  a weak second model.
- **Mechanism:** weighted average or stacked logistic combination of both models' scores.
- **Data:** none new — both models' scores already exist in `runs/baselines/`.
- **Compute:** trivial.
- **Implementation cost:** trivial — tens of lines.
- **Expected gain:** **near-certainly zero to negative.** T3 already establishes the MobileNetV2+LSTM
  model is chance-level and, per the falsification tests (T1/T2), is not extracting temporal signal
  at all — it is not "differently wrong," it is uninformative noise, which ensembling cannot help
  with (a truly chance-level, feature-collapsed model contributes no signal to decorrelate against).
- **Failure condition:** expected outcome; include only as a **near-zero-cost sanity check**
  (<1 hour) precisely because it's nearly free and closes off a reviewer's obvious question ("did you
  try ensembling your own model in?") with a measured answer instead of an assertion.
- **Leakage risk:** none.
- **Validation:** AP/AUC of the ensemble vs. BADAS alone, on the same 667-clip split, bootstrap CI.

---

## 5. Calibration protocol (point 2) — resolving F1 rigorously

**The problem, restated precisely:** Nexar train is BADAS-Open's own training data (F1) and is not
downloaded anyway. Nexar test-private is metadata-only. The *only* video data with labels currently
on disk is `test-public` (667 clips) — the same 667 clips every headline AP/AUC/ECE number in this
project is computed on. Fitting a calibration map on any subset of those 667 and evaluating on
another subset of the same 667 is the textbook definition of using held-out data twice unless it's
done correctly.

**Resolution — nested, nothing-fitted-on-itself, exactly as `NEW_PLAN.md` N1.1 already specifies,
made explicit and formalized here:**

1. **Primary protocol: 5-fold, out-of-fold, cross-fitted calibration on `test-public` only.**
   Partition the 667 clips into 5 folds (source-grouped where `youtubeID`/original-video identity is
   known, to respect README §16's leakage rules even within this smaller partition). For each fold,
   fit the calibration map (beta, per F2's measured winner) on the *other 4 folds' scores*, apply to
   the held-out fold, concatenate the 5 held-out predictions. **This produces one calibrated score per
   clip, none of which were used to fit the map that calibrated them** — this is the correct,
   textbook nested design, and it is what `NEW_PLAN.md`'s F2 table already claims to have done. This
   session did not find a committed script (`eval/calibration.py` does not exist yet) — **closing this
   gap (writing and committing the script that reproduces the F2 table) is P0, not optional, before
   any calibration number is quoted again.**
2. **What this protocol proves:** that a calibration map *of this functional form*, fit on data drawn
   from the same distribution as the reported eval set, reduces ECE without materially changing AP.
   It is a valid, standard claim.
3. **What this protocol does NOT prove, and must never be stated as proving:** that the calibration
   map will hold on any other distribution (different country, different camera, different clip
   truncation convention). Every report of the calibrated ECE number must carry the sentence: *"fit
   and evaluated via 5-fold cross-fitting on Nexar test-public; not yet validated on any external or
   held-out-by-construction dataset."*
4. **External-transfer test, as a separate and explicitly weaker claim (this is `NEW_PLAN.md` N2.5,
   kept, but re-scoped):** fit the calibration map on *all* of `test-public` (no fold-holding-out
   needed here, since the target is a genuinely different dataset), apply unmodified to DAD/DADA/DoTA
   external scores once C1's data acquisition happens. Report the external ECE. **This is a
   generalization test, not a calibration-validity test** — a failure here does not invalidate point 3
   above, and a pass here does not turn point 3's "not yet validated" caveat off retroactively for
   *other* external distributions (e.g. UK footage) not tested.
5. **What would need to change if Nexar train ever becomes usable (e.g., licensed differently, or the
   project pivots to training its own head rather than calibrating BADAS's):** at that point train and
   test-public would be genuinely disjoint, and a train-fit/test-public-eval calibration would become
   valid *for that new model* — but it remains invalid for calibrating BADAS-Open's own output, since
   train is BADAS's training data regardless of who's evaluating on it. **This distinction — whose
   model is being calibrated, not just which split the data came from — is the part of F1 most likely
   to be silently violated by a future contributor**, and should be a comment in the calibration code
   itself, not just this document.

---

## 6. comma2k19 reclassification and better FP-rate sources (point 4)

**Reclassification, as instructed:** comma2k19 is **hard-negative / operational stress-test data
only.** It is US highway-heavy, single-camera-model, single-fleet-source driving. Any FP/hour number
computed on it must be labeled "comma2k19 stress-test FP/hour," never "FP/hour" unqualified, and
never cited as evidence about UK, mixed-urban, or multi-camera-model deployment conditions.

**Alternatives evaluated for a more diverse FP-rate signal:**

| Source | License | Diversity gain | Realistically acquirable this month? | Verdict |
|---|---|---|---|---|
| **comma2k19** | MIT | Highway, US, one camera | Yes — already identified, 33 h | Keep as stress-test baseline |
| **Zenseact ZOD** (README F5 table) | Permissive, commercial-allowed | Swedish roads, multi-scenario, includes some urban/rural mix — genuinely different geography and camera setup from comma2k19 | Needs verification of exact access terms and download size before committing Studio time; not yet attempted | **Recommended second source** — closest available proxy for non-US, non-highway-only conditions without touching UK-specific acquisition (README Track B) |
| **BDD100K** | Research/non-profit only (F5) | Very diverse — but **not commercially usable**, and this project's own CCD post-mortem is a first-hand cautionary tale about exactly this dataset's provenance risk | Acquirable, but **do not use for any FP/hour number that will ever inform a shipped product decision** — research/qualitative-only, clearly labeled | Usable only as a *qualitative* sanity check, never a headline number |
| **DoTA driving segments** (non-incident portions) | Repo says MIT; pixels YouTube-sourced, verify | Very diverse — many countries/cameras | 55 GB, Mac Studio only, and verification of the YouTube-pixel licensing question is unresolved per README F5 | Defer until DoTA is acquired for C1 anyway (shared cost); do not acquire *solely* for this |

**Recommendation:** run the comma2k19 pilot exactly as `NEW_PLAN.md` N3.1 specifies (throughput
gate first), and if the Studio budget allows it *after* the higher-ranked items in §13, pursue ZOD as
the second, geographically-distinct stress-test source — but do not present a single-source FP/hour
number as representative of anything beyond "US highway driving, comma2k19-specific" until a second,
different-geography source exists. This is a direct, explicit instruction from point 4 and is treated
as non-negotiable in any report this project produces.

---

## 7. No claim without a benchmark (point 5)

**Every "real-time" / "CPU-capable" claim found in the repo, and its current evidence status:**

| Claim | Location | Evidence on file | Verdict |
|---|---|---|---|
| "Beta calibration... zero GPU" | `NEW_PLAN.md` F2 | True — calibration fitting genuinely is CPU-seconds-scale, this one is safe | **Verified-by-nature**, no benchmark needed (the claim is about an operation, not a model) |
| BADAS 0.856 s/window MPS compute-only | README Phase 5 task 2 | `scripts/badas_smoke.py`, synthetic tensor, bypasses video IO | **Real, but narrow** — compute-only, not end-to-end, machine not confirmed to be either of the two hardware targets in §1 |
| BADAS ~97 s/clip end-to-end | README Phase 5 task 2 | `eval/run_baselines.py --limit 6` | **Real but a 6-clip sample** — no CI given, machine not confirmed |
| Any "CPU-only Channel B" framing (this session's brief, and implicitly `NEW_PLAN.md`) | this document, C2/C6 | **None exists yet** | **Unverified — must not be claimed as feasible until benchmarked on both M4 Air and M4 Max, separately, since the Air is the always-available machine and any Channel B meant to run continuously must work on it specifically, not just on the Studio** |
| "Real-time" anywhere in a future demo (N4.1) | `NEW_PLAN.md` Phase N4 | None | **Must carry a measured end-to-end latency number or the word "real-time" is not used**, per README §32's own rule ("measure end-to-end latency... not just model inference time") |

**New requirement, closing point 5 completely:** before N1 work proceeds, add
`scripts/hw_bench.py` — a small script that runs BADAS inference (and, once built, the C2/C3 heads)
for N=20 clips on (a) the M4 Air, MPS backend, (b) the M4 Air, forced CPU backend, (c) the M4 Studio,
MPS backend — and records wall-clock s/clip with a 95% bootstrap CI for each. **This is a
zero-Studio-cost, near-zero-effort item and belongs in Phase N1, not deferred** — every subsequent
timeline estimate in §9 depends on knowing whether the Air can do meaningful MPS work at all or is
CPU-only in practice for tensors of this size.

---

## 8. Dataset independence audit (point 6)

| Dataset | Benchmark reproduction | Cross-dataset generalization | Genuinely independent validation | Why |
|---|---|---|---|---|
| **Nexar test-public** | — | — | **No** — this is the primary dataset every current number is computed on; it cannot validate itself |
| **DAD / DADA-2000 / DoTA (via vendored consensus annotations)** | **Yes, primarily** — BADAS's own published per-dataset figures (`config.json`) exist for these exact benchmarks, so reproducing them is literally benchmark reproduction | **Also yes, secondarily** — different geography/camera than Nexar, so a *second* use (e.g. C1's ego-collision head, or the calibration-transfer test in §5.4) is legitimately cross-dataset | **No** — BADAS's authors report figures on these exact sets, so "independent" in the sense of "never touched by anyone building on this model" is not true; **the assumption that BADAS wasn't trained on them (only on Nexar) is unverifiable from outside and must be stated as an assumption every time (see C1's leakage-risk field)** |
| **comma2k19** | No BADAS figure exists on this dataset | **Yes** — this is its correct role: measuring generalization to a driving distribution BADAS was never benchmarked on | **Partially** — genuinely never used by BADAS's authors as far as public information shows, but it is US-highway-only, so "independent" does not mean "representative" |
| **Zenseact ZOD (proposed, §6)** | No | **Yes, if acquired** — same logic as comma2k19, different geography | **Partially**, same caveat as comma2k19 |
| **BDD100K** | No | Qualitative only (licensing forbids commercial-decision use) | N/A — excluded from any quantitative claim per F5 |
| **UK footage (README Track B, not yet started)** | No | **Yes, and the strongest possible cross-dataset test** since it's the actual target market | **Yes, if and when collected** — this is the only dataset in the entire list that would be genuinely independent *and* representative simultaneously. This is why README §40 calls Track B the moat, and this plan does not dispute that — it only observes Track B has been at zero for many sessions and treats it as separately tracked, not blocking, per §11 |

**The headline discipline this audit produces:** any report from this project must label every
number with one of these three category tags. A number with no tag is not acceptable output, per
this plan's own standard.

---

## 9. Timeline against real compute (points 3 + 9)

Built directly from §1's ~25–30 Studio-h/week realistic budget and 24/7 Air availability. Weeks are
calendar weeks, not "days of Air availability" (Air work is fast and fits inside any week regardless).

### Week 1 (Air-only + one weekend Studio session)
- `eval/calibration.py` (N1.1), `eval/benchmark.py` metric additions (N1.2), operating-point policy +
  Poisson CI (N1.3) — all Air, CPU-seconds, no Studio needed.
- `scripts/hw_bench.py` (§7) — Air, minutes.
- **One weekend Studio session (~10 h):** let the in-flight per-frame trace sweep finish (currently
  222/667, needs the remaining ~445 clips — at whatever the per-clip rate for the trace sweep turns
  out to be from `hw_bench.py`, this may or may not fit in one weekend day; if not, it spans into
  week 2, and that must be said explicitly rather than assumed done).
- Track C: draft the 30-call script and target list (§11) — zero compute, any day.

### Week 2
- Freeze the reduction question (N1.4) once the trace sweep lands.
- C3 (trained temporal head) — Air, CPU, once traces exist. Cheapest detection-improvement candidate,
  scheduled early deliberately.
- C7 (ensemble sanity check) — Air, <1 h.
- **Weekend Studio (~10 h):** begin comma2k19 pilot (N3.1) — 30 min at deployment cadence first, per
  its own stop condition, before committing more.
- Track C: first 10 fleet calls attempted.

### Weeks 3–4 (this is where the plan must be honest about spanning multiple weekends)
- **DAD acquisition + scoring is the first item that will not fit one weekend.** DAD alone (165
  clips) at ~97 s/clip end-to-end ≈ 4.4 h — fits a single weekend day easily. **DADA (221 clips, ≈6 h)
  and DoTA (598 clips, ≈16 h, 55 GB download) do not both fit inside one weekend alongside anything
  else.** Realistic sequencing: DAD acquisition + scoring in one weekend (week 3), DADA in the
  following weekend (week 4), **DoTA explicitly deferred to weeks 5–6** because of both its size (55
  GB download on top of ~16 h scoring) and because C1's leakage-risk field means its value is
  conditional on DAD/DADA already showing signal — do not spend the most expensive acquisition first
  if the cheaper ones might already falsify C1.
- If comma2k19's N3.1 pilot passes its throughput gate: full negative run (N3.2), one weekend,
  competing with the DAD/DADA schedule above — **these cannot both happen in the same weekend on a
  10 h budget; comma2k19 full run is lower-ranked (§13) and should yield the weekend slot to
  DAD/DADA acquisition if both are ready simultaneously.**
- Track C: remaining 20 fleet calls.

### Weeks 5–6
- DoTA acquisition + scoring (spans at minimum one full weekend day for download alone, plus a
  second for scoring — **this is explicitly a two-weekend item**, stated per point 9's instruction).
- C1 (external-corpus head) training, once DAD/DADA/DoTA features exist — CPU/Air-feasible once
  features are cached, low incremental Studio cost.
- Mid-plan checkpoint (§16-style self-check): re-run the adversarial pass on actual measured numbers,
  not the projections in this document.

### Weeks 7–8 (conditional on weeks 3–6 showing signal)
- C2 (comma2k19-derived rejection classifier) — the highest-expected-value candidate (§13), but
  correctly sequenced *after* the throughput and feature-extraction infrastructure above already
  exists, so it does not duplicate Studio time.
- ZOD acquisition, if week 3–6 budget allowed comma2k19 to show the concept has legs (§6).

**What this timeline deliberately excludes from "one month":** C5 (killed, §4), C6 (dependent on C2,
pushed beyond week 8), full UK data collection (Track B, README Phase 7 — separately tracked, not
compute-bound, not part of this Studio-hour budget at all since it depends on a driver/consent-form
arrangement, not GPU time).

---

## 10. Revisiting provisional findings (point 7)

### F3 (temporal smoothing hurts) — **stays provisional, explicit promotion criteria stated**

Current evidence: 222/667 traces (up from 194 when `NEW_PLAN.md` was written). The direction of the
result (max > mean > persistence, by a wide margin) has been stable across two snapshots (194→222)
with no sign reversal, which is mildly reassuring but **is not the same as significance on the full
n**. **F3 is promoted from provisional to a stated conclusion only when:**
1. The trace sweep reaches 667/667 (already in progress, §9 week 1–2).
2. The max-vs-second-best gap (AP 0.9124 vs. persist-k4's 0.8851, a 0.027 delta on the 194-subgroup)
   is re-measured on the full set **with a bootstrap CI**, and the CI excludes zero.
3. The result is reproduced with the JSONL-replay fix (N1.1) through the same evaluation code path
   used for every other headline number, not a separate ad hoc script.

Until all three hold, any report using F3 must say "provisional, n=222/667 as of 2026-09-13" exactly
as this document does, not "settled" as `NEW_PLAN.md`'s §1 language ("Settled: max", §4 table)
currently overstates it.

### Other items found provisional-but-treated-as-settled during this audit:

- **F2's calibration table itself** — computed by an unspecified, uncommitted process (§0.2, §5
  point 1). Not wrong, but not yet reproducible by a third party, which is the actual bar this
  project sets for itself elsewhere (README's "commit the results verbatim" rule, §30). Fix: §5's P0
  action.
- **The BADAS end-to-end timing figures (0.856 s/window, ~97 s/clip)** — a synthetic-tensor number
  and a 6-clip sample respectively, both currently cited in README Phase 5 as though final. Fix:
  §7's `hw_bench.py`, re-measured on the actual two machines this plan targets.

---

## 11. Track C — made executable (point 8)

Track C (30 UK fleet-operator calls, README §41) has been at zero for multiple sessions despite
costing nothing but time — this is itself a finding: a P0, zero-dependency, zero-cost item sitting
at zero across six-plus sessions is a scheduling failure, not a resourcing one, and no amount of
restating it as important has moved it. **Two options, decisively:**

- **Option A — execute, with forcing structure:** a fixed weekly quota (5 calls/week, independent of
  Studio/model work, scheduled on the calendar like the Studio sessions are) using the exact
  one-question script already specified in README §41 ("What happened the last time you trialled an
  AI dashcam? — Do not pitch"). Track completion in `progress.md` alongside model-work sessions, not
  separately, so it stops being invisible relative to the compute-bound work this plan otherwise
  obsesses over.
- **Option B — kill it,** on the reasoning that if it has genuinely not happened across many
  sessions despite zero cost, the actual constraint is not "when," it's an unstated blocker (contact
  list access, confidence, time-zone/scheduling friction) that this plan cannot see and re-stating
  the task will not fix.

**Verdict: Option A, with one addition.** The repeated failure to execute a zero-cost task is itself
evidence the blocker is not "it's not prioritized in the plan" (it already is, at P0, three times
over). Before assuming Option A will work this time, the actual blocker should be named explicitly —
ask directly whether it's a contact-list problem, a confidence/script problem, or a genuine
time-availability problem, since the fix differs for each. **This plan does not have that answer and
should not guess it** — it is a judgment call outside a red-team's scope to resolve, and is flagged
as an open question in the adversarial pass (§16).

---

## 12. Uncertainty quantification (point 10)

Every existing comparison in `NEW_PLAN.md` re-examined for whether its stated delta is meaningful:

| Comparison | Stated delta | Meaningful? |
|---|---|---|
| Beta vs. Platt calibration ECE (0.0498 vs 0.0524) | 0.0026 | **No** — well within the stated bootstrap CI [0.040, 0.087] on beta's own ECE; the two methods are statistically indistinguishable on this data and either is defensible. `NEW_PLAN.md` bolding "beta" as if it clearly wins overstates this. |
| Calibrated AP vs. raw AP (0.8330 vs 0.8349) | 0.0019 | **No** — this is the point (calibration should not move AP), and the delta needs its own CI to state "AP unchanged" rigorously rather than just eyeballing a small number. Not yet computed — add to §5's P0 script. |
| F3 max vs. persist-k4 AP (0.9124 vs 0.8851) | 0.0273 | **Provisional** — larger than the calibration deltas above and plausibly real, but computed on n=194 (now 222) of 667; needs the full-n bootstrap CI per §10 before being called meaningful. |
| F3 max vs. mean AP (0.9124 vs 0.7066) | 0.2058 | **Almost certainly meaningful** — this delta is large relative to any plausible sampling noise at n=194+, but should still get a CI for completeness rather than asserted by magnitude alone. |
| Isotonic "losing" 0.02 AP vs. beta | 0.02 | **Plausibly real, but small-sample-overfitting is the stated explanation, not measured directly** — a learning-curve check (does the isotonic gap shrink as fold size grows) would distinguish "isotonic is worse" from "isotonic is noisier at this n," and hasn't been done. |

**Standing rule going forward, closing point 10 completely:** no comparison in any future report from
this project states a winner without either (a) a bootstrap CI on the delta that excludes zero, or
(b) an explicit "provisional / not yet significance-tested" label. This is now a project-wide rule,
not a per-experiment choice, exactly as README already enforces "no raw accuracy" and "always quote
the denominator."

---

## 13. Ranked plan — optimized for expected gain, not ease

Explicitly not sorted by implementation cost. Ranked by (expected effect on AP/FP-hour or on the
project's credibility-per-README's-own-stated-goal) ÷ (Studio-hours required), with ties broken toward
lower leakage risk.

| Rank | Item | Studio-h | Expected value | Why this rank |
|---|---|---|---|---|
| 1 | §5 calibration protocol as committed code (P0, was already "done" only informally) | 0 | High — fixes a reproducibility gap under every existing headline number | Zero cost, blocks credibility of everything downstream; must go first regardless of any other ranking logic |
| 2 | §7 `hw_bench.py` | 0 | High — every later Studio-hour estimate in this document is currently unverified | Zero cost, de-risks every subsequent item's timeline |
| 3 | F3 completion + freeze (§10) | ~0 (sweep already in flight) | Medium-high — resolves the plan's most-cited provisional claim | Sunk cost already running; just needs to land and be re-measured with a CI |
| 4 | C3 — trained temporal head | ~0 (rides on item 3's data) | Low-medium, but essentially free | Best cost/benefit ratio in the whole candidate list |
| 5 | C7 — ensemble sanity check | <1h | Near-zero expected gain, but closes an obvious reviewer question at negligible cost | Cheap insurance, not a real bet |
| 6 | N3.1 comma2k19 pilot (throughput gate) | ~0.5h | Unlocks the denominator problem | Required before C2 can be attempted at all |
| 7 | **C2 — hard-negative rejection classifier from comma2k19 (+ IMU/CAN)** | ~10–15h once pilot passes | **Highest genuine detection/FP-hour improvement potential in this document** — uses a signal (IMU) BADAS cannot see | Ranked here, not first, because it is gated on item 6's throughput measurement and item 2's benchmark existing first — but it is the single highest-expected-value item once unblocked, consistent with the brief's instruction not to optimize for ease |
| 8 | DAD acquisition + scoring (feeds C1, N2.3) | ~4.5h | Medium — benchmark reproduction value is real regardless of C1's outcome | Cheapest of the three external corpora, good first test of whether C1 is worth pursuing further |
| 9 | DADA acquisition + scoring | ~6h | Medium | Second cheapest, same logic |
| 10 | **C1 — external-corpus supervised head** | ~26.5h (DAD+DADA+DoTA combined feature extraction) once all three land | **Second-highest genuine detection improvement potential**, but flagged with the highest leakage risk in this document (§4) | Ranked high on potential value, tempered by the honest failure-condition statement that it may show no measurable gain — this is exactly the kind of high-risk/high-reward item the brief says must not be excluded for being merely "easy" or "hard" |
| 11 | DoTA acquisition + scoring | ~16h + 55GB download | Lower marginal value than DAD/DADA (largest, least differentiated addition) | Correctly deferred per §9's timeline |
| 12 | ego/non-ego + mTTA analysis (`NEW_PLAN.md` N2.4) | ~0 once DAD/DADA/DoTA land | Medium — answers two previously "permanently blocked" questions | Free once its data dependency is paid for elsewhere |
| 13 | comma2k19 full negative run (N3.2) | ~10h | Medium — moves the FP/hour denominator, but is *stress-test only* per §6, capped value | Competes for weekend slots with higher-ranked items; explicitly instructed to yield |
| 14 | C6 — synthetic hard negatives | conditional | Low-medium, amplifier only | Correctly gated behind C2 showing signal first |
| 15 | ZOD acquisition (§6) | unknown, needs scoping | Speculative | Second-geography diversity, worth pursuing once budget allows |
| — | C5 — Nexar-train probe | — | **Near-zero, do not build** | Kept killed (§4), included only for completeness of the "re-examine everything dropped" requirement |
| — | Track C — 30 fleet calls | 0 (not Studio-bound) | Potentially the highest-value item in the *entire* project per README §45, but orthogonal to this ranking's Studio-hour metric | Tracked separately in §11, must run in parallel with everything above, not traded off against it |

---

## 14. Keep / hybrid / rebuild verdict

**Verdict: hybrid — keep BADAS-Open as the frozen backbone and primary ranker; do not fine-tune or
replace it; build a genuinely new, evidence-justified addition on top (C2's rejection classifier,
contingent on its own validation) rather than either (a) shipping calibration-only as the ceiling, or
(b) attempting a from-scratch or fine-tuned rebuild.**

Reasoning, stated against the brief's explicit instruction not to assume BADAS+calibration is the
ceiling:

- **A full rebuild is ruled out by the same evidence that ruled it out in `README.md` §45 and §28**,
  restated here because the brief asked this to be re-derived, not assumed: no CUDA device exists on
  either target machine, BADAS was trained with materially more data and compute than this project
  can reproduce, and MobileNetV2+LSTM (the only in-house alternative) is measured chance-level (T3).
  Nothing in this session's audit changes that physics.
- **Calibration-only is explicitly rejected as the ceiling**, per the brief's strongest instruction.
  C2 (hard-negative rejection using IMU, a signal channel BADAS's video-only backbone structurally
  cannot use) is the one candidate in §4 with a plausible mechanism for a *real*, not merely cosmetic,
  improvement over BADAS zero-shot + calibration — because it adds information, not just reshapes
  existing scores.
- **C1 (external-corpus supervised head) is the second-most-promising real addition**, but is
  explicitly flagged (§4, §13) as high-leakage-risk and possibly redundant with BADAS's own training
  signal — it is a legitimate bet, not a certainty, and this plan does not oversell it.
- **This is "hybrid" rather than "keep," specifically**, because "keep" (as `NEW_PLAN.md` effectively
  proposed) would mean calibration + external validation only, with no new detection mechanism — and
  this document's mandate is explicit that this is not automatically the ceiling. The hybrid
  verdict commits to attempting C2 and C1 as real, evidence-gated experiments, not merely proposing
  and then dropping them the way `NEW_PLAN.md`'s §4 "Dropped" table did with the entire prior
  candidate list.

---

## 15. The decisive one-month answer

> **If we had one month of engineering time under these exact hardware/data constraints, what
> sequence of experiments gives us the highest probability of producing a materially stronger
> prototype than AP 0.8349 / AUC 0.8498?**

**Answer:** Run, in this exact order, items 1–7 of §13's ranked list — commit the calibration
protocol and hardware benchmark (week 1, zero Studio cost), let the in-flight F3 sweep land and
freeze it with a real CI (week 1–2), train the free C3 temporal head and the near-zero-cost C7
ensemble check (week 2), then spend the *majority of the month's Studio budget* on item 7: the
comma2k19-derived, IMU-informed hard-negative rejection classifier (C2), because it is the only
candidate in this entire analysis that introduces a genuinely new information channel (IMU/physical
signal) rather than reshaping or re-deriving BADAS's own existing video-only score. **This is not
optimized for probability of a large AP jump** — nothing here plausibly moves BADAS-Open's core AP by
more than a few points given no backbone fine-tuning is possible — **it is optimized for the
metric this project's own README (§31, §45) says actually decides commercial viability: FP/hour**,
where an added physical-signal channel has a real, mechanistically justified path to improvement that
calibration alone does not.

If the month's remaining Studio budget allows (realistically weeks 3–4 under the §1/§9 schedule),
DAD acquisition and a first pass at C1 is the second bet, run in parallel with C2 rather than
sequentially, since C1 uses Studio time for feature extraction (parallelizable with C2's comma2k19
scoring only if the Studio is genuinely free for both — in practice, given the 25–30h/week budget,
these will contend for the same weekend slots and §13's ranking already resolves that contention in
C2's favor). **Track C's 30 fleet calls run throughout, unconditionally, on their own weekly quota,
because they are the only item in this entire plan with a plausible path to the single largest
missing asset the whole project lacks — a real customer conversation — at literally zero compute
cost**, and this plan treats "zero compute cost" as no excuse for continued non-execution (§11).

**What this explicitly does not attempt in one month:** DoTA (too expensive relative to its marginal
value, §9), the Nexar-train probe (killed, §4), any UK data collection (Track B — not compute-bound,
tracked separately, not this plan's bottleneck), and any claim of a "materially stronger" number
being *guaranteed* — the honest statement is that C2 has the best mechanistic case for a real
improvement, not a proven one, and this document's own §12 discipline requires that outcome be
reported with a CI, win or lose.

---

## 16. Adversarial self-review pass

A second pass against the draft above, checking specifically for the contradiction classes named in
the brief.

**1. Does §1's compute budget actually support §9's timeline?**
Found and fixed one instance: an earlier draft of §9 had DAD+DADA+DoTA all landing by week 3, which
at ~27h combined Studio time does not fit inside two weekly budgets (~50–60h) once comma2k19 (§9
week 2) and the trace-sweep tail (§9 week 1) are also accounted for. **Fixed by explicitly deferring
DoTA to weeks 5–6** and stating the comma2k19-vs-DAD/DADA weekend contention directly in §9 rather
than leaving it implicit.

**2. Does §5's calibration protocol actually avoid §4's leakage risks?**
Checked: §5's nested cross-fitting is entirely internal to `test-public` and does not touch any of
C1's external-corpus data, so the two are independent — no contradiction found. One gap closed: §5
did not originally state which split C3's temporal-head training should reuse; **added a sentence
requiring C3 to use the same clip-level partition as calibration**, to avoid the plan silently
creating three or four different small ad hoc splits of the same 667 clips, which would itself become
a leakage/multiple-comparisons problem across experiments even if each one individually looks clean.

**3. Does the Track C decision (§11) contradict the §13 ranking?**
§13 explicitly separates Track C from the Studio-hour-based ranking (noted in its final row) precisely
to avoid this: Track C is not competing for Studio time, so it cannot be "outranked" by a Studio-bound
item in a way that would justify deprioritizing it. No contradiction, but the original draft's §13
table listed Track C without this caveat, which read as though it had been ranked last; **fixed by
adding the explicit "orthogonal to this ranking" note.**

**4. Any of the ten points not fully closed?**
Re-checked against §2's index. Point 8 (Track C) is closed with a recommendation (Option A) but
**honestly flags an unresolved sub-question** (why has it actually stayed at zero) that this plan
cannot answer from the available evidence — this is disclosed as an open question rather than
papered over, which is consistent with the brief's own standard for the rest of the document (e.g.
§10's treatment of F3), not a gap unique to this section.

**5. Unjustified claims check.**
One found and fixed: an earlier draft's §14 verdict used the word "clearly" to describe C2's
advantage over C1 without the CI language §12 requires everywhere else. **Reworded to "plausible
mechanistic case," consistent with §12's standing rule**, since C2 has not actually been run yet and
no comparison of unrun experiments should be stated more confidently than a comparison of measured
ones.

**6. Does killing C5 while ranking C1 highly contradict each other, given both use external/adjacent
data to BADAS's own training?**
Checked directly: no — C5 is killed because Nexar train *is* BADAS's training data (direct identity,
F1), while C1 uses DAD/DADA/DoTA, which are *plausibly but not provably* independent of BADAS's
training (§4's leakage-risk field already states this is an unverifiable assumption from outside).
These are different confidence levels of the same concern, not a contradiction — C1 is correctly
flagged as the highest-leakage-risk item that is *still attempted*, precisely because "unverifiable
but plausible independence" is a different, weaker claim than "confirmed identity with the training
set," and the plan treats them accordingly rather than applying one rule inconsistently.

**Outcome of this pass:** three wording/structure fixes applied directly into the sections above
(§9, §5, §13); one open question flagged rather than resolved (§11); no outright contradiction found
that required abandoning a recommendation.
