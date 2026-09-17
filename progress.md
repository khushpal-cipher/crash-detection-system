# progress.md — execution state

**Living execution-state file. `README.md` is the master plan; this file records progress against it.**
Last updated: **2026-09-17, end of session 10** (session ran 2026-09-16 evening into 2026-09-17).
Six commits, all pushed, working tree clean. `NEW_PLAN.md` is the detailed/current execution plan and
must be read alongside `README.md`.

> # ▶ SESSION 10 (2026-09-16 evening → 2026-09-17) — READ THIS FIRST
>
> **Nothing is running. Nothing is partially done. Everything is committed AND pushed.**
> `git status` is clean, `git log origin/main..HEAD` is empty, HEAD = **`37f02b0`**.
>
> Session 10 cleared the commit backlog that had been pending since session 8, then executed
> `NEW_PLAN.md` Week 1 items in order. **Six commits:**
>
> | Commit | What |
> |---|---|
> | `fbf127c` | Recovered sessions 8+9's handoff — `progress.md` had been uncommitted for two sessions |
> | `024bec9` | `eval/timing.py` — session 9's deliverable, incident-record field 2 |
> | `c005757` | R1 gate-3 redesign (user-approved plan change) |
> | `d0bc237` | **Correction to that redesign** — see the finding below |
> | `faa823e` | `scripts/hw_bench.py` — CPU vs MPS, lifts §7.3's ban on "real-time" language |
> | `37f02b0` | Calibration **Tiers 1 and 2** — the deployable map, not just the achievable one |
>
> ### 🔴 THE FINDING OF THIS SESSION — two errors in a plan change, caught the same session
>
> Session 8 concluded that R1's gate 3 "cannot be computed" because DAD/DADA/DoTA are positives-only.
> That was committed as a plan change (`c005757`), then **found to be wrong in two ways** (`d0bc237`),
> by checking DAD's actual construction *before* downloading it:
>
> 1. **The vendored `*_concensus.csv` files are collision-TIMING annotations.** A negative clip has no
>    collision to time, so their being positives-only says nothing about the datasets.
>    **DAD's test split is 466 clips: 165 positive and 301 NEGATIVE.** BADAS also publishes AP 0.66 /
>    AUC 0.87 on DAD, and both metrics require two classes. **The original AP test is computable.**
> 2. **DAD is the wrong dataset for the mechanism test anyway** — its annotated collision time is
>    effectively a constant (IQR **0.16 s**; 84% within 0.25 s of exactly 3.00 s), and the test asks
>    whether the peak *tracks* that time. Against a constant there is nothing to track. The constant is
>    structural: DAD's page says the accident is in "the last 10 frames" of a 100-frame clip = frame 90
>    = exactly 3.00 s at 30 fps, putting DAD's collision at normalised **~0.90, near the clip end like
>    Nexar's 0.975.** It is not an untruncated mid-clip contrast at all.
>
> **Gate 3 now splits, and R1 must survive both** (`NEW_PLAN.md` R1, gate-3 block):
>
> | | Dataset | Why | Cost |
> |---|---|---|---|
> | **3a mechanism** | **DADA-2000**, 221 clips | only set with real collision-time variance (IQR 4.63 s) | ~6.0 h Air |
> | **3b AP** | **DAD full test**, 466 clips (165 pos + 301 neg) | restores the plan's original design | ~7.5 h Air |
>
> ### The headline numbers from this session
>
> ```
> HARDWARE (scripts/hw_bench.py, M4 Air, one window, 16 frames @ 8 fps)
>   mps   0.871 s/window   0.87x budget   KEEPS UP at a 1 Hz alert cadence
>   cpu   1.917 s/window   1.92x budget   TOO SLOW
>   CPU is 2.20x slower than MPS.   MPS 0.871 s re-confirms D10's 0.856 s.
>
> CALIBRATION TIER 1 (deployable; fit on calibration half, reported on evaluation half only)
>   uncalibrated  ECE 0.3273 [.282,.374]   AP 0.8596
>   beta          ECE 0.0809 [.062,.127]   AP 0.8596   <- THE DEPLOYABLE NUMBER
>   isotonic      ECE 0.0560 [.049,.107]   AP 0.8342   <- costs 0.025 AP
>   platt         ECE 0.0908 [.068,.134]   AP 0.8596
>   temperature   ECE 0.2209 [.177,.263]   AP 0.8596
> TIER 2 (100 random stratified splits, evaluation-half ECE)
>   beta median 0.0596  5-95% [0.0405, 0.0850]
> ```
>
> **🔴 The deployable calibration number is 0.0809, NOT the 0.0498/0.0503 this project has been
> quoting.** That lower figure is Tier 3 — cross-fitted, no shippable map, and `NEW_PLAN.md` §6
> forbids reporting it as a deployed result. Quoting it as a product figure overstates deployed
> calibration by ~60%. Tier 2 shows seed 0 is a slightly *unlucky* split, so the headline is not
> flattered by split luck.
>
> **`eval/adapters.py` is still `np.nanmax`. R1 is still NOT promoted.** That remains deliberate until
> gate 3 is answered.
>
> ### 🔴 THE BLOCKER IS DATA ACQUISITION, AND IT NEEDS THE USER
>
> Gate 3 cannot start. Both targets are blocked on acquisition, not code:
> - **DADA-2000** — single **~53 GB** Baidu Pan archive (`9pab`; a Google Drive mirror is mentioned).
>   The Air has **~30 GB free**. We need only 221 of its 2,000 clips, so **if the Drive mirror is a
>   per-file folder this becomes easy — that is the first thing to check.**
> - **DAD** — **Google Form request to the authors**, no publicly posted terms. The user must submit
>   it, and the terms must be confirmed in writing before use (§22/§23 licence discipline).
>
> **Read order for a brand-new Claude: this block → §21.5 (session 10 findings) → §13 (exact next
> action) → §12 (what not to redo) → `NEW_PLAN.md` R1 gate-3 block → §21.4 (session 9).**

> # ▶ SESSION 9 (2026-09-16, afternoon) — history, superseded by the SESSION 10 block above
>
> **Nothing is running. Nothing is partially done. Nothing was committed.**
>
> Session 9 executed §13 as session 8 wrote it: **`eval/timing.py` now exists** (331 lines,
> untracked) and turns the 667 committed per-frame traces into `t_start` / `t_peak` / `t_end`.
> `README.md` §27's incident record now has **two working fields instead of one**, and the second
> is the first non-commodity one. Zero compute, zero new data, no committed number changed.
>
> | | |
> |---|---|
> | `eval/timing.py --self-check` | **7/7 PASS** |
> | Regression guards (benchmark / calibration / reduction_study) | **all reproduce exactly** |
> | `eval/adapters.py` | **untouched** — still `np.nanmax`, R1 still unpromoted |
> | `README.md` / `NEW_PLAN.md` | **unchanged** |
>
> ### 🔴 TWO THINGS ARE UNCOMMITTED AND ONE OF THEM IS SESSION 8's ENTIRE HANDOFF
>
> ```
>  M progress.md      <- session 8's handoff + this session's. NOT committed.
> ?? eval/timing.py   <- session 9's deliverable. NOT committed.
> ```
>
> Session 8's §17-S8 claims *"Working tree clean, pushed, in sync with origin/main"*. **That is false
> about `progress.md` itself** — session 8 wrote its handoff at 16:21, *after* its 13:05 commit
> `252d282`, and never committed it. It has been one `git checkout` away from destruction ever since.
> **The user was asked to commit + push and said "wait" twice. It is still pending their go-ahead.**
> HEAD is `252d282`, `main` is otherwise in sync with `origin/main`, 0 unpushed commits.
>
> ### The headline numbers from this session
>
> ```
> Gate derived from the PR curve @ target recall 0.80  -> threshold 0.9733
>   delivers recall 0.8024, precision 0.7635
>   83 false positives over 0.899 h of negatives = 92.3 FP/hour     <- BAD, and honest
> 351 incident records emitted (268 positive, 83 the gate's false alarms)
> ```
>
> **§3.2's mechanism reproduced at full n=667 through an independent code path** (CONFIRMED, new):
> positives peak at **0.975** of clip vs negatives **0.802**; 73.7% vs 38.4% in the final 10%.
> Session 7 had 0.984 / 0.782 and 83.0% / 42.5% at n=268. Same direction, mild softening at full n.
>
> **Read order for a brand-new Claude: this block → §21.4 (session 9 findings) → §13 (exact next
> action) → §12 (what not to redo) → `NEW_PLAN.md` → §21.3 (session 8).**

> **⚠️ SESSION 7 RAN AS TWO PARALLEL CLAUDE WINDOWS BY ACCIDENT.** The user gave the same red-team
> brief to two sessions at once. Both wrote plans. They have been **reconciled** — see the
> CONSOLIDATED block below. Anywhere this file says "this session" inside the older SESSION 7 SUMMARY,
> it means *one* of the two windows, and three of its statements were false and are corrected inline.

> # ✅ THE SECOND SWEEP IS FINISHED. NOTHING IS RUNNING. — READ THIS BEFORE ANYTHING ELSE
>
> **SUPERSEDES session 7's "job still running" banner, which is now deleted.** PID 75682 completed
> and exited. Verified 2026-09-16: `ps -p 75682` returns nothing; `wc -l
> runs/baselines2/badas-open/scores.jsonl` = **667**; `ls runs/baselines2/badas-open/frames/*.npz |
> wc -l` = **667**. **Do not relaunch it. Do not look for it. There is no background job.**
>
> **All of `runs/baselines2/` is now COMMITTED** (`252d282`), including the 667 per-frame `.npz`
> traces and `sweep.log` (force-added past the `*.log` ignore rule, matching the `runs/baselines/`
> precedent, per D24). Total 3.1 MB.
>
> ### What session 8 settled: R1 passed TWO of its three promotion gates
>
> | Gate (`NEW_PLAN.md` R1) | Status |
> |---|---|
> | 1. Full-667 paired CI on ΔAP excludes zero | ✅ **PASSED** |
> | 2. Held-out-half confirmation | ✅ **PASSED** (this session) |
> | 3. External falsification on untruncated clips | ⬜ **OPEN — and its design is broken, see §21.3** |
>
> **R1 IS NOT PROMOTED.** `eval/adapters.py::BadasOpen.score` still reduces with `np.nanmax`, and
> that is deliberate — do not switch it until gate 3 is answered.
>
> ### The numbers, exactly
>
> Gate 1 (`python -m eval.reduction_study`, re-run and reproduced this session):
> ```
> n=667  pos=334  neg=333
> max            0.8349      —
> last_window    0.8905   +0.0556   [+0.0263, +0.0876] excludes zero
> max_x_last     0.8904   +0.0555   [+0.0274, +0.0865] excludes zero
> last4_mean     0.8732   +0.0383   [+0.0101, +0.0688] excludes zero
> top3_mean      0.8213   -0.0136   worse
> p90            0.7862   -0.0488   worse
> area_gt_half   0.6549   -0.1800   worse
> ```
> **Cross-check that matters:** `max` on the *second* sweep = **0.8349**, identical to the committed
> *first* sweep's AP. That independently re-confirms **D19** (`--skip-predictor` is score-identical)
> and proves the 667 traces are sound.
>
> Gate 2 (`python -m eval.heldout_half`, new this session):
> ```
> primary split (seed 0): A selects last_window; held-out B ΔAP +0.0598
>                         paired 95% CI [+0.0169, +0.1085] excludes zero
> 1000 splits:            A picks last_window 52.4% / max_x_last 47.6% — never a losing candidate
>                         held-out ΔAP median +0.0551, 5-95% [+0.0291, +0.0796], >0 in 100% of splits
> ```
> **The held-out median (+0.0551) matches the full-667 estimate (+0.0556) — no winner's-curse
> shrinkage.** The effect was not manufactured by selecting on test-public. This closes
> `NEW_PLAN.md` §11 point 1 as far as Nexar can close it.
>
> **Read order for a brand-new Claude: this block → §21.3 (session 8 findings) → §13 (exact next
> action) → §12 (what not to redo) → `NEW_PLAN.md`.**

> # SESSION 7 — CONSOLIDATED SUMMARY (2026-09-13/14) · READ THIS FIRST, then §21.1, then §13
>
> **Read order for a brand-new Claude: this block → §21.1 (the measured findings) → `NEW_PLAN.md`
> → §13 (exact next action) → §12 (what not to redo).**
>
> ### What this session was
> A **planning and measurement** session. **No production code was written. Nothing was committed.
> No model was run.** All measurements were read-only analyses over *already-committed* scores and
> the in-flight sweep's `.npz` files, executed in-conversation. The background sweep ran unattended
> throughout and advanced 14 → 377 of 667.
>
> ### The accident, and how it was resolved
> The user gave the same red-team brief to **two Claude Code windows simultaneously**. Both produced
> a revised plan, neither could see the other. Result on disk was two competing documents whose names
> implied the wrong order (`NEW_PLAN_v2.md` was *older* and parallel, not successive), plus a handoff
> written by the window that had done no measurements.
> **Resolved 2026-09-14:** merged into **one authoritative plan, `NEW_PLAN.md`**. The parallel document
> was moved to `archive/NEW_PLAN_v2_parallel_session.md` (moved, **not** deleted). Its one genuinely
> additive idea — the **IMU/CAN false-positive rejection channel** — was folded in as **R9**, with its
> three-way partition discipline. Its C4/C6/C7 were rejected with written reasons in `NEW_PLAN.md` §4.
> Three false statements in the older SESSION 7 SUMMARY block below are corrected inline (❌/✅).
>
> ### The five things that actually matter from this session
>
> 1. **A statistical bar now exists. CONFIRMED.** AP 0.8349 has a bootstrap 95% CI of
>    **[0.7910, 0.8734]**. **Any unpaired improvement under ~0.04 AP is invisible at n=667.** Several
>    claims made earlier in this project are smaller than their own error bar. All future comparisons
>    must use *paired* bootstrap. (§21.1 item 1)
> 2. **A candidate detection improvement worth ~+0.033 AP. PROVISIONAL.** Replacing the `nanmax`
>    clip reduction with **last-window** (or `√(max·last)`) beat max with a paired CI excluding zero,
>    on 268 clips. **This is the first idea in the project's history that would move AP rather than
>    presentation.** It is NOT yet confirmed — see §21.1 items 2–4 for the mechanism, the checks that
>    passed, and the selection-risk caveat.
> 3. **Calibration is solved on paper but has no committed code.** Beta calibration: ECE
>    0.3286 → 0.0498, AP unchanged. **`eval/calibration.py` DOES NOT EXIST** — the numbers came from
>    an in-conversation script. **Do not quote them in any external document until the script is
>    committed and reproducible.** Also: beta-vs-Platt (0.0026) is NOT a meaningful difference.
> 4. **Nexar's train split is BADAS-Open's own training data, and is not downloaded.** The BADAS paper
>    and model card state BADAS-Open was *"trained solely on Nexar's public dataset (1,500 videos)."*
>    On disk, `data/nexar/train/` is metadata-only (zero `.mp4`), as is `test-private/`. **It is
>    therefore unusable for calibration and handicapped for probe training.** (§21.1, `NEW_PLAN.md` F1)
> 5. **Two items this file previously recorded as permanently impossible are not.**
>    `vendor/badas-open/annotation/` already contains consensus re-annotations for DAD (165),
>    DADA2000-small (221) and DoTA (598) — **984 clips carrying ego-involvement flags and
>    collision/alert timestamps.** That is the data D23 (ego/non-ego breakdown) and the mTTA strike
>    were declared blocked for lack of. They are blocked **on Nexar**, not in general. **The videos are
>    not downloaded.** (`NEW_PLAN.md` F4)
>
> ### Status of the plan
> **`NEW_PLAN.md` is a PROPOSAL. It has not been accepted, started, or merged into `README.md`.**
> The next session must get an explicit accept/reject/revise decision from the user before executing
> any of it. `README.md` was **not** modified this session (§16).

> # SESSION 7 SUMMARY (2026-09-13, planning-only session) — the block below was written by ONE of the
> # two parallel windows; corrections are inline. The CONSOLIDATED block above supersedes it.
>
> **🔴 CORRECTED 2026-09-14. The block below was written by ONE of TWO Claude sessions that ran in
> parallel on 2026-09-13 — the user gave the same red-team brief to two windows by mistake. The
> session that wrote this had no visibility into the other, and three of its statements are FALSE.
> Corrections are inline and marked ❌/✅. Read §21 for the reconciled state.**
>
> ❌ *"No code was written or run this session. No experiments ran."* — **False for the parallel
> session.** That window ran five read-only measurement experiments whose results are now the single
> most important evidence the project has (§21.1). No files were *written* by it other than a plan,
> but real measurements were made. ✅ Correct statement: **no production code was written or
> committed this session; read-only analysis scripts were run in-conversation and their results are
> recorded in §21.1.**
>
> 1. **User asked for a general explanation of the project** (a "teach me from scratch, what to learn
>    on YouTube" style question, conversational only, no files touched) — answered in conversation:
>    pointed at CNN backbones/transfer learning, LSTM/sequence models, YOLO/object detection, ViT +
>    V-JEPA2/self-supervised video models, MiDaS/Kalman/TTC classical CV, and model calibration, in
>    that priority order, with V-JEPA2/ViT flagged as highest-leverage since it's the model actually in
>    use now. No files changed.
> 2. **User then gave a formal 10-point red-team brief** asking for `NEW_PLAN.md` (session 6's earlier
>    output — see §21 below, it predates this numbered list) to be revised: close leakage/hardware/
>    provisional-finding loopholes, find genuine detection-improvement paths beyond calibration, and
>    answer a specific "best one-month sequence" question — **explicitly against the user's REAL
>    hardware (M4 MacBook Air 24/7, Mac Studio M4 Max ~2–4h weekdays / ~10h weekends), not the richer
>    assumed budget `NEW_PLAN.md` and this file's own compute language had been implicitly using.**
>    **IMPORTANT CAVEAT the user should know:** the brief referred to "your 10 points" as a pre-existing
>    red-team finding from a prior session. **That artifact was searched for and NOT FOUND** — not in
>    this file, not in `README.md`, not in `NEW_PLAN.md`, no prior-session record located. The revision
>    was built directly from the 10 numbered requirements in the user's own brief instead, cross-
>    referenced against `NEW_PLAN.md`'s F1–F6 findings (which turned out to cover the same failure
>    modes). **If a real prior "10 points" document exists outside this session/repo, it must be
>    supplied and `NEW_PLAN_v2.md` re-diffed against it — this is an open item, not resolved.**
>
>    ❌ **RESOLVED 2026-09-14 — the "10 points" artifact was real and is NOT missing.** It was an
>    adversarial self-review of `NEW_PLAN.md` produced in the *other* parallel window, which the user
>    then quoted back. It never existed as a file, which is why the search failed. ✅ Its ten points
>    are now closed in `NEW_PLAN.md` §2 (the closure index). **This is no longer an open item — do not
>    go looking for this document again.**
> 3. **Produced `NEW_PLAN_v2.md`** (repo root, ~450 lines, untracked, proposal-only — see §21 for full
>    contents summary). It supersedes `NEW_PLAN.md` as the active proposal but, per the user's own
>    instruction, **does not modify `README.md` or this file's plan content** — planning-document
>    status only until reviewed and accepted.
>
>    ❌ **Two claims here are wrong.** (a) `NEW_PLAN.md` is **not** session 6's output — it was written
>    by the parallel session on the same day, and its current content is *newer* than this file's.
>    (b) `NEW_PLAN_v2.md` does **not** supersede it. ✅ **Reconciled 2026-09-14:** the two documents
>    were merged. **`NEW_PLAN.md` is now the single authoritative plan**; the parallel document was
>    moved to `archive/NEW_PLAN_v2_parallel_session.md`, with its one genuinely additive idea (the
>    IMU/CAN rejection channel, its C2) folded into `NEW_PLAN.md` as **R9**, and its C4/C6/C7
>    explicitly rejected with reasons in that file's §4.
> 4. **Repo/data state was verified, not assumed, as part of writing that document** — this produced
>    real findings worth keeping even independent of the plan itself:
>    - `eval/calibration.py` **does not exist**. `NEW_PLAN.md`'s F2 calibration table (beta calibration
>      ECE 0.0498 etc.) was produced by an **ad hoc, uncommitted script** — not reproducible by a third
>      party. Flagged as a P0 gap to close before quoting that table again.
>    - `scripts/badas_fps_probe.py` exists but only checks config coherence, **not** end-to-end
>      throughput. The only real timing numbers on record (0.856 s/window MPS compute-only;
>      ~97 s/clip end-to-end from a 6-clip sample) were never confirmed to have run on either of the
>      user's two actual target machines (M4 Air / M4 Max Studio). A new script, `scripts/hw_bench.py`,
>      is specified (not yet written) to close this.
>    - `vendor/badas-open/annotation/` row counts confirmed: DAD 165 (164 data rows), DADA2000-small
>      221 (220), DoTA 598 (597) — these are the externally-annotated clips proposed for the C1
>      candidate in `NEW_PLAN_v2.md`, and **none of the underlying videos are downloaded yet.**
>    - The in-flight sweep (banner above) was at 222/667 mid-session, now 261/667 at handoff —
>      confirmed alive both times via direct process/file inspection, not assumed continuing.
> 5. **User then asked to end the session and asked for this handoff.** This entry and §21 are the
>    result. `README.md` was **not** touched this session (see §16 below — NO change, correctly, per
>    the user's own rule that only master-plan-level discoveries justify a README edit, and everything
>    found this session is proposal/verification-level, living in `NEW_PLAN_v2.md` instead).
>
> **The one thing a new Claude must not do: treat `NEW_PLAN_v2.md` as accepted or in progress.** It is
> a reviewed-by-nobody-yet proposal sitting next to `NEW_PLAN.md` (also still just a proposal, itself
> never merged into README/progress). Confirm with the user whether it's accepted before starting any
> of its §13-ranked items.

> # SESSION 6 SUMMARY — READ THIS, then §3, §13
>
> 1. **The committed sweep's headline result: BADAS-Open AP 0.8349 / AUC 0.8498** on all 667 clips
>    (334 pos / 333 neg). This **matches the vendored `config.json`'s published 83.2/0.85 to within
>    0.003** — the harness is now validated a third independent way (T3 replay + T1/T2 falsification +
>    this reproduction). Committed at `e7ac3e6` with `sweep.log`, `scores.jsonl`, `metrics.json` for
>    all three models, `baseline_table.json`. **This was 18 h of GPU work sitting untracked with zero
>    backup at session start — the single highest-risk item this session opened with.**
> 2. **README §41 Phase 4 is now COMPLETE.** `eval/plots.py` ran against real data for the first time
>    (`runs/baselines/plots/{pr_curve,reliability}.png`, visually verified sane). `tests/test_leakage.py`
>    written and passing — proves `assert_no_leakage` both stays silent on the real 1500 train / 667
>    test-public ids (zero overlap) AND raises on an injected 3-id violation (previously the function
>    existed but nothing exercised the "fires on violation" half of the acceptance criterion).
> 3. **README §41 Phase 5's gate is judged: PASS, in writing, in `metrics.json`.**
>    `scripts/badas_gate_provenance.py` wrote a `gate` block into
>    `runs/baselines/badas-open/metrics.json` naming both disputed published figures (model card
>    0.86/0.88 vs config.json 83.2/0.85), the measured numbers, and every deviation (split size,
>    `np.nanmax` reduction, predictor computed-then-discarded, fps/img/crop/temperature). This closes
>    Phase 5 task 4.
> 4. **Two real bugs fixed, both verified with an A/B test before trusting them (D7):**
>    - `--skip-predictor` existed as a CLI flag since session 3 but was **never wired to the adapter**
>      — silently ignored. Fixed (threaded as a model *attribute*, not a `forward()` kwarg, because the
>      sliding-window closure calls `self.model(processed_frames)` positionally). A/B on one clip:
>      score **0.995916 both ways**, bit-identical, confirming the predictor output genuinely
>      contributes nothing (D19).
>    - Nothing persisted per-frame scores — only one `nanmax` per clip survived each sweep. Fixed via
>      `BadasOpen.save_frames_dir` (D20). Free (~1.6 MB for all 667 clips), and it's what makes the
>      reduction ambiguity (upstream uses mean in one place, max in another) answerable without
>      re-running the model.
> 5. **`eval/benchmark.py` gained `threshold_sweep()`** — every `metrics.json` now records
>    tp/fp/fn/tn/recall/precision/fp_per_hour at 9 thresholds, not just the one fixed 0.80 snapshot.
>    Regenerating all three `metrics.json` cost **0.0 seconds** — proof it replayed from the resume
>    cache and re-scored nothing (D21).
> 6. **The "consume predictor_output" research question (Phase 5 task 3, D9) was investigated and
>    DELIBERATELY NOT ACTED ON THIS SESSION — see D22, this is a decision, not an oversight.** Traced
>    the exact tensor shapes: concatenating `predictor_output` on the token axis is a verified 2-line
>    change with zero shape/weight impact. But with the checkpoint's default context/target masks, the
>    predictor reconstructs the SAME tokens it was given — `future_prediction_seconds: 1.0` needs
>    training-time mask offsets that are not in this repo and are not recoverable. Running a second
>    17.5 h sweep to test a hypothesis the checkpoint's own config already refutes was judged not worth
>    the compute. Documented as a named deviation in the `gate` block instead of measured.
> 7. **README edited — TWO lines, same shape as the mTTA finding, not a plan change.** Struck the
>    ego-involved/non-ego breakdown from Phase 4 (task 3 + its acceptance criterion): verified directly
>    that `data/nexar/{test-public,train}/*/metadata.csv` has no ego field on either split. See §16.
> 8. **A second, deliberately different sweep was launched** (see the ⏳ banner above) — **user's
>    explicit choice** among three offered options (no more compute / cheap probe / full instrumented
>    re-sweep). NOT yet finished at handoff. Everything after it is the next session's job.
> 9. **claude-mem status unknown this session** — not checked. Assume still down per session 5's note
>    until re-verified.

> # SESSION 5 SUMMARY (history — superseded by the block above)
>
> **Sessions 4 and 5 both happened after progress.md was last written. Neither was recorded until
> now.** Session 4 is reconstructed from git; session 5 is this session.
>
> 1. **⚠️ progress.md was ONE SESSION STALE.** It said the BADAS sweep was "not started, awaiting the
>    user's stride decision". **A session 4 had already launched it** (commits `e488a05`, `4204bf2`,
>    `cc13966`; sweep started 13:48). The stride question is now **moot** — stride 1 is running and
>    ~230 clips are banked at that setting (§6.15).
> 2. **🔴 THE SWEEP WAS SCORING ALL 334 POSITIVES BEFORE ANY NEGATIVE — FIXED.** `balanced_ids()`
>    returned `sorted(labels)`, and in Nexar test-public every positive id sorts below every negative.
>    AP/ROC-AUC/FP-h are undefined until both classes are present, so the run would have produced
>    **no computable metric for ~9 hours**. Now interleaved. Restarted; the 140 banked clips resumed
>    instantly. **Metrics are order-independent, so the final numbers are unchanged** (§6.16, D15).
> 3. **FIRST REAL BADAS SIGNAL — partial, at 230/667 clips (140 pos / 90 neg):
>    AP 0.9274 · ROC-AUC 0.8988 · ECE 0.2794 · TP/FP/FN/TN 138/58/2/32 @thr 0.80.**
>    **PARTIAL, NOT A RESULT — do not commit or quote it** (§6.17). But it clears README Phase 5's
>    "AP ≈ 0.5 = something is broken" branch decisively.
> 4. **U-B7 RESOLVED — `original_fps: 4` vs `target_fps: 8.0` is NOT a contradiction, and the running
>    sweep is configured correctly.** `target_fps` is the video rate; `original_fps` is the
>    post-tubelet **token** rate. 16 frames @ 8 fps = 2.0 s → tubelet 2 → 8 tokens → 4 tokens/s.
>    Every checkpoint number reconciles. **No B1-style temporal mismatch.** Evidence:
>    `scripts/badas_fps_probe.py` (§6.18).
> 5. **U6 ANSWERED — Phase 0 acceptance item closed.** `models/crash_model_cpu/` and the shipped
>    `crash_model_weights.weights.h5` are **bit-identical across all 12 weight arrays**. Evidence:
>    `scripts/u6_compare_weights.py` (§6.19).
> 6. **`runs/legacy-colab/` WRITTEN — the last Phase 0 acceptance item.** Training log + val metrics
>    extracted verbatim from the committed notebook, plus a README stating why `val AUC 0.9977` does
>    not describe the shipped weights.
> 7. **Phase 3's own grep criterion now PASSES.** Deleted the unreachable fault-reporting block in
>    `code/crash_detection_enhanced.py`. `grep -rn "at_fault\|FaultDetector\|EgoZone" code/` is clean.
> 8. **§18's documentation contradictions 1–5 are FIXED** (§18 is updated in place).
> 9. **README was edited — ONE note only** (a dated delta table under the §2 inventory). Not a plan
>    change. See §16.
> 10. **NOTHING IS COMMITTED.** HEAD is still `cc13966`. See §8.
> 11. **claude-mem is DOWN** — observer allowance exhausted since 2026-09-11T22:17Z. No memories were
>    captured this session. Do not restart the worker; it clears a backoff.

---

> # SESSION 3 SUMMARY (history — superseded by the block above)
>
> # SESSION 3 SUMMARY — READ THIS FIRST, then §3, §6.11–6.14, §13
>
> **Nothing is blocked. Phase 5 is a GO. Phase 4's headline criterion is met. Nothing is committed.**
>
> 1. **⚠️ SESSION 2's BIGGEST FINDING (§6.9) WAS WRONG, AND IS NOW CORRECTED (§6.11).** Session 2
>    concluded the BADAS checkpoint's 199-tensor `predictor` is a module "the published code does not
>    implement at all", making Phase 5 a reimplementation job of unknown size. **That is false.** The
>    predictor is **V-JEPA2's own**, already implemented in `transformers` as `VJEPA2Model.predictor`
>    (199 params, exact name+shape match). The checkpoint stores it **twice** — embedded at
>    `backbone.predictor.*` AND duplicated at top-level `predictor.*`. The embedded copy **loads
>    correctly** (`load_state_dict` reports **missing 0**) and all 199 duplicate pairs are **bitwise
>    identical** (`torch.equal`). **NO WEIGHTS ARE LOST.** Evidence: `scripts/badas_predictor_probe.py`.
> 2. **The real gap is smaller and is in the FORWARD PASS, not the load.** `EnhancedVideoClassifier.forward()`
>    consumes only `last_hidden_state` and **discards `predictor_output`**, even though
>    `skip_predictor` defaults to `False` so it is computed. Training used
>    `predictor_combination_method: "concat"`, `future_prediction_seconds: 1.0`. So the pathway is
>    loaded, executed, billed ~25% of every window, and thrown away. Phase 5 task 3 is now "consume
>    `predictor_output`", not "rebuild a module".
> 3. **U-B6 ANSWERED — the Phase 5 blocker is gone.** Patched the 1-line API drift, then measured.
>    **Compute only: 0.856 s/window** (0.629 with `skip_predictor=True`, `last_hidden_state`
>    bit-identical). **End-to-end: ~97 s/clip → a 667-clip sweep is ≈18 h, NOT ≈10 h.** Model load is
>    **8.5 s**, not the ~9 min session 2 recorded (that was a one-time HF download).
> 4. **Phase 4 acceptance criterion #1 IS MET: three models now evaluate through one code path.**
>    Built `eval/adapters.py` + `eval/run_baselines.py`, extended `eval/benchmark.py`. The retired
>    model reproduces its committed T3 numbers **through the new adapter path**.
> 5. **First real baseline table** (667 clips, identical split and FP/hour denominator):
>    old model **AP 0.5218 / AUC 0.5339 / FP-h 361.4 / ECE 0.4880**; always-negative
>    **AP 0.5007 / AUC 0.5000 / FP-h 0.0**. The retired model beats the trivial baseline by **0.02 AP**
>    and loses **361 : 0** on FP/hour. README §14 item 14, now measured rather than asserted.
> 6. **BADAS has NOT been measured.** Only a 6-clip smoke run (it works end-to-end). **Its AP 1.0000
>    at n=6 is NOT a result — do not quote it.**
> 7. **README was edited** (4 plan-level edits + stale-status fixes + 3 Phase 5 corrections). See §16.
> 8. **Documentation contradictions found** — README §2 inventory and `docs/HANDOFF.md` are stale. §18.

---

## 1. PROJECT IDENTITY

**Project: AI Crash Detection System.** A dashcam crash-detection system — video in, crash/incident
determination out.

**Repository:** `khushpal-cipher/crash-detection-system` (SSH remote: `git@github.com:khushpal-cipher/crash-detection-system.git`)
**Working directory:** `/Users/khushpalsinghchouhan/dev/crash_detection/crash_detection_v2`

> **Identity warning for a fresh session.** An earlier handoff request described this project as
> "ULPF Solo Build / Universal Log Pre-processing Framework (SIH 2026, NTRO)". **That is a different,
> unrelated project.** Nothing here relates to log pre-processing. If asked about ULPF, say there is
> zero ULPF context in this repository and ask.

**What we are trying to build:** not a crash classifier — the classifier is a commodity (see §11).
The product is the **structured incident record**: a verified, evidence-backed event a human can act
on in under two minutes, sold to UK commercial fleets (30–300 vehicles) that already run dashcams
and do not watch the footage. README §27 defines this.

**Hard project constraints (confirmed with the user this session):**

| Constraint | Value | Consequence |
|---|---|---|
| Goal | **Startup** — pilots and funding | README Part III (UK market, DPAs, fleets, YC) stays in scope |
| Compute | **M1 laptop, plus an M4 Max machine at college** (software installs allowed, ~40 GB free). **No cloud spend.** | **Probe/head training on cached frozen features: feasible. Full backbone fine-tuning: still out of reach — Apple Silicon has no CUDA.** See §11 D2 |
| Environment location | Never under `~/Desktop` | iCloud eviction caused `TimeoutError: [Errno 60]` during TF import |
| Evidence standard | Falsification over confirmation | Report failures, keep raw outputs, never soften an unfavourable number |

---

## 2. MASTER PLAN REFERENCE

**`README.md` (3,191 lines) is the authoritative master plan. The next session MUST read it before
making implementation decisions.** It is a 50-section technical audit + product strategy. Do not
copy it here; do not rewrite it to record progress.

Sections that most affected this session:

- **§41 Phase 2 — "Current model validation"** — the phase this session completed.
- **§30 Step −1** — the six falsification tests. T3 (corpus control) was the last one outstanding.
- **§41 Phase 2 gate** — *"Do not proceed to Phase 5 until the current model's performance has been
  measured on a genuinely held-out source-grouped test set."* **This gate is now satisfied** (§6, T3).
- **§28 / §40 / §45** — the argument that the model is not the moat, and that MobileNetV2+LSTM has an
  architectural ceiling (GlobalAveragePooling2D destroys the spatial layout needed to represent
  relative vehicle motion). This is why no amount of retraining is planned.
- **§44 "Things NOT to Build"** — explicitly forbids further CCD training and retraining the old model.
- **§15 bugs R1/R2** — the pinned TensorFlow could not read the shipped weights. Fixed this session.

---

## 3. CURRENT PHASE

**Updated end of SESSION 6 (2026-09-13). Sections 4.1–4.8 describe SESSION 1, §4.9 SESSION 2,
§4.10 SESSION 3, §4.11 SESSION 4, §4.12 SESSION 5, §4.13 SESSION 6 — kept for history except the
last. Read this table for the truth as of now.**

**Project:** AI Crash Detection System. **`README.md` (3,258 lines) is the master plan** — a 50-section
technical audit plus product strategy. **Currently between README §41 Phase 5 (closed, gate PASSED)
and Phase 6, Track A.** README §41 defines **three parallel tracks**, not a queue:
Track A = model/measurement (Phases 4→5→6), Track B = data/benchmark (7→8), Track C = 30 fleet calls.

**Status as of end of SESSION 6 (2026-09-13, 14:45 IST). This table is the truth; everything below
labelled "this session" for sessions 1–5 is history.**

| | |
|---|---|
| **Phase 0 — Evidence recovery** | **COMPLETE** (session 5). U6 answered; `runs/legacy-colab/README.md` written. Only the two licence emails (U7, BDD100K) remain, external, not blocking. |
| **Phase 2 — Current model validation** | **COMPLETE** (session 1). Gate satisfied by T3 (§6.1). |
| **Phase 4 — Evaluation harness** | **COMPLETE (session 6).** All 4 acceptance criteria met: three models one code path (session 3); `metrics.json` has AP/AUC/FP-h/ECE/per-condition rows (ego/non-ego struck, not computable on this data — README edited, §16); leakage test written and proven to fire on an injected violation; accuracy banned and asserted absent. Plots rendered for the first time (`runs/baselines/plots/`). |
| **Phase 5 — BADAS-Open reference baseline** | **GATE JUDGED: PASS (session 6).** Full 667-clip sweep: **AP 0.8349 / AUC 0.8498**, matching the vendored `config.json`'s published 83.2/0.85 to within 0.003. Provenance + every deviation recorded in `metrics.json`'s `gate` block (`scripts/badas_gate_provenance.py`). Committed `e7ac3e6`. |
| **Phase 5-continuation (unplanned, user's choice)** | **COMPLETE (session 8).** The second sweep (`runs/baselines2/`, `--skip-predictor` + `--save-frames-dir`) **finished at 667/667 and is committed** at `252d282`, including all 667 per-frame `.npz` traces. Its `nanmax` AP reproduces the first sweep's **0.8349 exactly**, re-confirming D19. **Nothing is running.** |
| **R1 — deployment-faithful reduction** (`NEW_PLAN.md` R1) | **2 of 3 promotion gates PASSED (session 8).** Gate 1 full-667 paired CI and gate 2 held-out-half both exclude zero; held-out median +0.0551 vs full-667 +0.0556, so no winner's-curse shrinkage. **Gate 3 (external falsification) is OPEN and its specified design is broken** — §21.3 item 2. **R1 is NOT promoted; `eval/adapters.py` still uses `np.nanmax` (D29).** |
| **Current objective** | **Build `eval/timing.py` — `t_start`/`t_peak`/`t_end` from the committed traces (§13).** Zero compute, and it is `README.md` §27's second incident-record field, of which only one currently works. Then: user sign-off on the gate-3 redesign, then DAD (~4.5 h, Air only). **README §40/§45 still argue Tracks B/C are the real critical path and they remain at zero after eight sessions.** |

**Phases with outstanding items (none blocking Phase 4 or 5):**

- **Phase 0 (Evidence recovery) — mostly complete.** Colab notebook IS committed at
  `data/ccd/Untitled0.ipynb` (verified session 3: 12 cells, 12/12 with saved outputs). **U4 is
  resolved by deletion** — `models/crash_detection_model.h5` and `models/crash_detection_model/` are
  confirmed GONE from the tree (removed in `ad45389`). Outstanding: licence emails to the CCD authors
  (U7) and Berkeley DeepDrive.
- **Phase 1 (Reproducibility) — partially done, partially dropped.** Done: Python/TF pinning,
  `.python-version`, two real tests. **Dropped as dead work:** porting the Colab notebook to
  `train/` as a maintained pipeline (decision D3, §11). Still missing: lockfile, `Dockerfile`,
  `Makefile`, CI, `data/manifest.csv`.
- **Phase 3 (Dataset/licensing) — partially done.** `FaultDetector`, `EgoZone`, and (session 5) the
  last dead fault-reporting remnant in `code/crash_detection_enhanced.py` are all deleted; Phase 3's
  own `grep -rn "at_fault\|FaultDetector\|EgoZone" code/` criterion now passes clean. Outstanding:
  licence emails, the RT-DETR-vs-Enterprise-Licence decision (C6), a `LICENSE` file (L4).
- **Tracks B and C (README §41, Phases 7–8 + the 30 fleet calls) — ZERO progress, unchanged this
  session.** No UK footage, no fleet calls. Both need no GPU and no code. README §40/§45 call these
  the actual critical path — the sequencing note in §14 still applies.

**Phases with outstanding items (none blocking Phase 4):**

- **Phase 0 (Evidence recovery) — mostly complete.** Colab notebook IS committed at
  `data/ccd/Untitled0.ipynb`. Outstanding: licence emails to the CCD authors (U7) and Berkeley
  DeepDrive; U4 (493 MB orphaned model artefacts — these are gitignored and not in the working tree
  at this path, needs confirmation); U6 (`crash_model_cpu/` vs shipped weights comparison — now
  *possible* because a working TF 2.19 env exists, but not done).
- **Phase 1 (Reproducibility) — partially done, partially dropped.** Done: Python/TF pinning,
  `.python-version`, first real test. **Dropped as dead work:** porting the Colab notebook to
  `train/` as a maintained pipeline (decision D3, §11).
- **Phase 3 (Dataset/licensing) — partially done.** `FaultDetector` and `EgoZone` were deleted in a
  prior session. `ultralytics` removed from the default install this session. Outstanding: licence
  emails, the RT-DETR-vs-Enterprise-Licence decision (C6), and adding a `LICENSE` file (L4).

---

## 4. EVERYTHING COMPLETED (§4.1–4.8 = S1 · §4.9 = S2 · §4.10 = S3 · §4.11 = S4 · §4.12 = S5 · **§4.13 = S6**)

### 4.1 State verification (no changes made)

Ran the verification block from `docs/HANDOFF.md` §1. **No discrepancies found between the handoff
and the filesystem.** Confirmed at session start:

- HEAD was `08d9453`; 5 commits unpushed; working tree clean.
- `code/` contained only `crash_detection_enhanced.py` (977 lines) — the four dead forks,
  `depth_estimator.py` and `bev_renderer.py` were already deleted and committed.
- `grep -n "self\.depth_est\|self\.bev\|EgoZone\|FaultDetector" code/crash_detection_enhanced.py`
  → **zero matches.** The dead-code deletion is genuinely finished.
- `find data/nexar/test-public -name '*.mp4' | wc -l` → **667** (334 positive + 333 negative),
  matching `data/nexar/solution.csv` exactly. Download is complete; no process running.
- Disk: **42 GB free.**

### 4.2 Environment rebuilt (README bugs R1/R2)

Created **`~/envs/crashdet`** (deliberately outside iCloud):

```bash
python3.11 -m venv ~/envs/crashdet
~/envs/crashdet/bin/pip install tensorflow==2.19.1 keras==3.15.1 \
    opencv-python-headless==4.10.0.84 scikit-learn h5py numpy
```

Verified installed: `tf 2.19.1 · keras 3.15.1 · cv2 4.10.0 · numpy 2.1.3` (Python 3.11.16).
Install took ~25 minutes (large TF wheel). **`torch` is NOT installed in this env** — deliberate.

Verified both shipped artefacts load in it:
- `models/feature_extractor_saved` → signatures `['serve', 'serving_default']`
- `models/crash_model_weights.weights.h5` → layer groups `dense, dense_1, dense_2, dropout×4, input_layer, lstm, lstm_1`

### 4.3 Scripts and tests created

- **`scripts/t3_corpus_control.py`** (new) — the T3 corpus-control experiment. Reuses the exact NumPy
  LSTM forward pass from `scripts/t124_model_falsification.py` (reading the Keras 3 HDF5 directly),
  plus the deployed SavedModel feature extractor. Scores each clip as the **max over all
  stride-1 windows of 10 consecutive frames**, which mirrors the deployed decision path
  (`Config.CNN_FRAMES = 10`, `deque(maxlen=10)` fed every frame).
  - Mid-session fix: the sweep logic was moved into a `main()` behind `if __name__ == "__main__"`,
    because importing the module for a smoke test would otherwise have launched the full 667-clip run.
- **`tests/test_weights_load.py`** (new) — **the first real test in this repository's history.**
  Implements the verification README bug R2 has asked for since revision 1. Plain `assert`-based,
  no pytest needed: `python tests/test_weights_load.py`.

### 4.4 Files modified

- **`requirements.txt`** — rewritten. Was `tensorflow==2.13.1` (ships Keras 2, **physically cannot
  read** the Keras 3 weights file this repo ships — bug R2). Now pins the versions actually verified
  to work. Removed `ultralytics` (AGPL-3.0, bug C6; needed only by the retired YOLO-gated path),
  `torch` (moved to a future separate env), and `scipy` (unused by the surviving pipeline). Each
  removal is documented in-file with its reason.
- **`.python-version`** (new) — `3.11`.
- **`README.md`** — **modified.** See §16 for the full justification and the list of edits.

### 4.5 Experiments run

See §6 for full detail. Summary: T3 executed (the headline result), a self-falsification of the T3
harness, a numerical-sanity investigation of unexpected warnings, and a footage-duration calculation
that establishes a hard limit on future FP/hour claims.

### 4.6 Bug discovered and resolved this session

**Spurious NumPy `RuntimeWarning`s** (`divide by zero` / `overflow` / `invalid value encountered in
matmul`) fire on macOS arm64 even for an all-zeros input. Investigated rather than ignored:

- `mobilenet_v2.preprocess_input` output range: `-1.0 … 1.0` — correct.
- Feature extractor output: shape `(20, 1280)`, min `0.0`, max `5.552668`, **no NaN**.
- First Dense output: min `0.0`, max `7.838482`, **no NaN**.
- Every weight array checked for NaN/inf: **none**; all magnitudes ≤ ~1.04.

**Conclusion: cosmetic, arising in the BLAS matmul path; all results are finite and correct.**
Silenced with `np.seterr(all="ignore")` in the test only (which asserts finiteness and range
explicitly). **The evidence scripts were deliberately NOT touched**, so previously committed results
remain byte-reproducible.

### 4.7 Git

- Commit **`420ef17`** — "T3 corpus control: the shipped model performs at chance (AUC 0.5339)".
  7 files changed, 5,657 insertions, 14 deletions.
- **Pushed `c995fd7..420ef17` to `origin/main`** — all 6 previously unpushed commits, with explicit
  user approval. Verified no `.mp4` was staged before committing.

### 4.8 Written outside the repository (not in git)

Project memory files under
`~/.claude/projects/-Users-khushpalsinghchouhan-dev-crash-detection/memory/`:
`crash-detection-status.md`, `crash-detection-constraints.md`, `reading-preferences.md`, `MEMORY.md`.

---

### 4.9 Everything completed in SESSION 2 (2026-09-11 → 2026-09-12)

Full narrative in §6.9/§6.10/§10; this is the compact checklist.

- **Built `eval/benchmark.py`** — Phase 4's model-agnostic loader + metrics (AP, ROC-AUC,
  precision@recall0.80, ECE, FP/hour-with-denominator, per-condition breakdown, leakage assertion).
  Self-check reproduces T3 exactly. Command: `~/envs/crashdet/bin/python eval/benchmark.py` → `PASS`.
- **Measured `time_of_event` vs clip duration** across all 334 positives → proved mTTA/time-to-detection
  are not computable on Nexar test-public (§6.7). Cross-checked against `cv2.CAP_PROP_FRAME_COUNT`.
- **Found the prior session's BADAS-Open file list and architecture description were wrong** — no
  `preprocessing.py`, no `model.safetensors`, no "12 learned queries", input is 224×224 not 256×256.
- **Bypassed the HF gate for source** by finding the ungated GitHub mirror `getnexar/BADAS-Open`
  (Apache-2.0) and vendoring it to `vendor/badas-open/` (228 KB, `.git`/`assets/` stripped).
- **Resolved U-B3 and U-B4** by reading the vendored source directly (§10).
- **User accepted the HF gate terms and logged in** (`~/envs/badas/bin/hf auth login` →
  `user=khushpal`). **Downloaded `weights/badas_open.pth` (3.7 GB)** to `models/badas/weights/`.
- **Built `~/envs/badas`**: Python 3.11 + torch 2.14.0 + transformers 5.17.0 + albumentations 2.0.8 +
  cv2 5.0.0 + huggingface_hub. Confirmed `torch.backends.mps.is_available() == True`.
- **Wrote `scripts/badas_smoke.py`** to time one BADAS forward pass on MPS (U-B6) — **not yet run
  successfully**, blocked on the API-drift bug found below.
- **Inspected the checkpoint directly with `torch.load`** (not just the code) and discovered §6.9:
  the checkpoint has a 199-tensor `predictor` module the inference code never loads (`strict=False`
  silently drops it). This is the most important finding of the session — see §6.9 in full.
- **Found and left unfixed:** `EnhancedVideoClassifier.forward()` in
  `vendor/badas-open/badas/utils/video.py:175` calls `self.backbone(pixel_values=...)`, but
  transformers 5.17.0's `VJEPA2Model.forward()` requires `pixel_values_videos`. One-line fix, not
  yet applied (§13 step 1 — exact next action).
- **Fixed a real bug: `models/badas/` (3.7 GB) was NOT gitignored** — added to `.gitignore`. Verified
  with `git check-ignore -v` after the fix.
- **README.md edited** (5 changes) — see §16 for the full list. User explicitly approved via
  AskUserQuestion: "AP only, edit README."
- **Nothing committed this session.** `git status`: `M .gitignore`, `M README.md`,
  `?? eval/`, `?? progress.md`, `?? scripts/badas_smoke.py`, `?? vendor/`, `?? models/badas/` (now
  ignored, so it will stop showing as untracked next run — not yet re-verified after the fix).

---

### 4.10 Everything completed in SESSION 3 (2026-09-12)

Full detail in §6.11–6.14. Compact checklist:

**Code written**
- **Patched `vendor/badas-open/badas/utils/video.py:175`** — `self.backbone(pixel_values=...)` →
  `pixel_values_videos=...`. Verified FIRST against
  `inspect.signature(transformers.VJEPA2Model.forward)` rather than trusting the handoff. Marked with
  a `# ponytail:` comment as a vendored-upstream patch with an upgrade path.
- **`eval/adapters.py` (new, 115 lines)** — the model-agnostic interface README §41 Phase 4 task 5
  requires. Three adapters, duck-typed (no base class): `AlwaysNegative`, `CachedScores`, `BadasOpen`.
- **`eval/run_baselines.py` (new, 103 lines)** — the baseline-table entry point. `--limit N` takes N
  clips balanced across classes; `--no-badas` runs without torch in seconds; `--skip-predictor`,
  `--stride`, `--out`.
- **`eval/benchmark.py` extended** (172 → 256 lines, additive; self-check preserved and still passes).
  Added `durations()`, `hours()`, `clip_paths()`, `run()`, `summarise()`, and `metrics.json` writing.
- **`tests/test_score_regression.py` (new, 58 lines)** — the `safe.mp4` → 0.7914 guard §14 asked for.
  Imports `score_clip` from `scripts/t3_corpus_control.py` (import is not refactoring; D5 respected).
- **`scripts/badas_predictor_probe.py` (new, 124 lines)** — settles the §6.9 predictor question. Its
  printed verdict was corrected mid-session after the duplicate-weights discovery.

**Experiments / measurements run**
- `scripts/badas_smoke.py` — first successful run. U-B6 answered (§6.12).
- `scripts/badas_predictor_probe.py` — predictor mapping settled (§6.11).
- Direct `torch.equal` comparison of the two predictor copies — 199/199 bitwise identical.
- `skip_predictor` True/False timing + output comparison (§6.12).
- `eval/run_baselines.py --no-badas` over all 667 clips — the two zero-compute baselines (§6.13).
- `eval/run_baselines.py --limit 6` — BADAS adapter end-to-end smoke (§6.14).
- `tests/test_score_regression.py` — PASSES, `safe.mp4` = 0.7914 exactly.
- `tests/test_weights_load.py` — re-run, PASSES.
- `eval/benchmark.py` self-check — re-run after extension, PASSES.

**Environment**
- **Installed `scikit-learn` 1.9.1 into `~/envs/badas`** (needed by `benchmark.py`). This is NOT a
  rebuild — §12's "do not rebuild `~/envs/badas`" still holds.
- Confirmed `~/envs/badas`: torch 2.14.0, transformers 5.17.0, MPS available.
- Confirmed `matplotlib` is absent from **both** envs (blocks Phase 4's plots).

**Files modified**
- `README.md` — see §16 (4 plan-level edits, stale-status fixes, then 3 Phase 5 corrections).
- `runs/falsification/RESULTS.md` — its T3 section still said "**NOT YET RUN** … a formality rather
  than a question". Replaced with the measured figures + the self-falsification note + B4 closure.
- `.gitignore` — `models/badas/` (carried from session 2, still uncommitted).

**Verification performed (not assumed)**
- Git history read in full, with dates and diffstats.
- `data/ccd/Untitled0.ipynb` — 12 cells, **12/12 with saved outputs**.
- `data/Crash-1500.txt` — 1,500 rows, 7 logical fields.
- `data/nexar` — 667 `.mp4` on disk; `solution.csv` = 667 Public + 677 Private, 672/672 class balance.
- Raw `T5_source_leakage.txt` and `T124_local_videos.json` read directly.
- **`code/crash_detection_enhanced.py` cannot run** — parses fine, but imports `torch`, `ultralytics`,
  `scipy`, none installed in `~/envs/crashdet`. Unchanged finding, now explicitly re-verified.

**Nothing was committed.** Git HEAD is still `420ef17`.

---

### 4.11 Everything completed in SESSION 4 (reconstructed from git — it left no handoff)

Session 4 ran ~13:36–15:22 IST on 2026-09-12, after progress.md was last written at 13:14. It
**committed three times and updated nothing in this file.** Verified from `git log`/`git show`:

| Commit | Time | What |
|---|---|---|
| `e488a05` | 13:36 | **Made the sweep resumable.** `benchmark.run()` now appends+fsyncs every clip to `<out>/scores.jsonl` and `_resume()` reads it back on restart. `_resume` raises if a record's `adapter` name differs, so a stride-2 resume can never silently inherit stride-1 scores. (+64/-11 across `eval/benchmark.py`, `eval/run_baselines.py`.) **This is what saved 6 hours of work when session 5 restarted the sweep.** |
| `4204bf2` | 13:43 | **`eval/plots.py` (119 lines)** — PR curves + reliability diagrams, README §41 Phase 4 task 4. Standalone by design: it reads a finished `metrics.json` rather than drawing from inside the scoring loop, because matplotlib lives only in `~/envs/crashdet` while the sweep runs in `~/envs/badas`. An 18 h sweep can therefore never die on a plotting import. |
| `cc13966` | 15:22 | **Committed `progress.md`** (1,809 lines) to git for the first time — but committed it *unchanged*, still stamped "end of session 3". |

Session 4 also **launched the BADAS sweep at 13:48** at stride 1, without recording the launch
anywhere. Session 5 found it running.

---

### 4.12 Everything completed in SESSION 5 (2026-09-12, ~19:50–23:15 IST) — THIS SESSION

Full detail in §6.15–6.19. Compact checklist:

1. **Deep project recovery.** Read README §§1–2, 41–44 and all of progress.md; verified every
   material claim against the repo rather than trusting the handoff. Found progress.md one session
   stale (§6.15).
2. **Found and fixed the clip-ordering flaw** in `eval/run_baselines.py::balanced_ids` (§6.16, D15).
   Killed PID 67741, patched, relaunched as PID 88180. Verified `resuming: 140/667`.
3. **Corrected the sweep ETA docstring** in `eval/run_baselines.py` — the ~97 s/clip / ~18 h figures
   were measured on a 6-clip smoke run and are too optimistic. Measured over 140 clips of the live
   run: **~167 s/clip**; measured over session 5's own 90 clips after restart: **~117 s/clip**.
4. **Wrote `eval/peek.py` (55 lines)** — reads the live `scores.jsonl` and prints metrics on whatever
   has landed, clearly marked PARTIAL. Needed because `benchmark.run()` deliberately writes
   `metrics.json` only on completion.
5. **Resolved U-B7** (the `original_fps` question) and wrote `scripts/badas_fps_probe.py` (§6.18).
6. **Resolved U6** and wrote `scripts/u6_compare_weights.py` (§6.19). Phase 0 acceptance item.
7. **Wrote `runs/legacy-colab/`** — `training_log.txt`, `metrics_val.txt` (both extracted verbatim
   from `data/ccd/Untitled0.ipynb`) and a `README.md` explaining the artefact/metric disconnect.
   Phase 0 acceptance item.
8. **Deleted the unreachable fault-reporting block** in `code/crash_detection_enhanced.py` — the
   block, the `fault_info` parameter, both call-site arguments and both dead `first_fault = None`
   locals. Phase 3's own grep criterion now passes. Both tests re-run and still pass.
9. **Fixed documentation contradictions §18 items 1–5** — a dated delta note in README §2, and a
   superseded banner + three in-place corrections in `docs/HANDOFF.md`.
10. **Re-verified the environment:** `eval/benchmark.py` self-check PASSES (reproduces T3 exactly);
    `tests/test_weights_load.py` PASSES; `tests/test_score_regression.py` PASSES (safe.mp4 → 0.7914).

**Nothing was committed.** HEAD is still `cc13966`.

---

### 4.13 Everything completed in SESSION 6 (2026-09-13, ~09:30–14:45 IST) — THIS SESSION

1. **Deep project recovery** (Auto Mode, no user prompt beyond "plan the next step" / "check the
   sweep"). Read progress.md's true tail state (found it structured non-chronologically — session 5's
   summary sits ABOVE session 3's in the file, which is correct: newest-first — verified this was not
   a discrepancy, just the file's own convention). Verified session 5's committed claims against git
   and found progress.md's own top banner one commit behind its own HEAD (cosmetic, self-corrected).
2. **Watched PID 88180 (the sweep session 5 launched) to completion** via `eval/peek.py` at four
   checkpoints (543→571→583→616→667/667), never touching the process. Confirmed via `ps` it exited
   cleanly and `runs/baselines/baseline_table.json` existed.
3. **Explained the whole project in plain language, twice**, at the user's explicit request (once a
   full walkthrough of history/architecture/BADAS/moat, once specifically "what happens after the
   second sweep"). No files touched by these turns.
4. **Judged the Phase 5 gate**: read the finished `metrics.json`/`baseline_table.json`, compared
   against README's gate ladder (§41 Phase 5), reported AP 0.8349/AUC 0.8498 vs the disputed published
   figures. Verdict: PASS, harness independently re-validated.
5. **`plan the next step` — full investigation before touching code:**
   - Two parallel `Explore` subagents: one traced the BADAS predictor forward path end-to-end
     (exact tensor shapes, confirmed token-axis concat is a 2-line no-shape-change edit, confirmed the
     feature-axis alternative would crash), one inventoried `eval/benchmark.py` + `eval/plots.py` +
     matplotlib env locations + `sweep.log` (no errors, one clean resume) so nothing was rebuilt that
     already existed.
   - Direct inspection (not agent-delegated) of `scores.jsonl`: **discovered the 18h sweep discarded
     ~300 per-frame scores per clip, keeping only one `nanmax`.** Computed a threshold sweep by hand
     from the saved scores to show there is currently no usable FP/hour operating point below ~1.1/h
     given only 0.90 h of negative footage — independent of model quality.
   - `AskUserQuestion`: asked how much more compute to spend (no more / cheap probe / full instrumented
     re-sweep) and how to record the ego-row gap. User chose **full instrumented re-sweep** + **strike
     the ego row like mTTA**.
6. **Implementation (Sonnet, after the user's `/model` switch mid-task):**
   - `vendor/badas-open/badas/utils/video.py` + `.../models/vjepa.py`: threaded `skip_predictor`
     through as a model **attribute** (not a `forward()` arg — the sliding-window closure calls
     `self.model(processed_frames)` positionally, so a kwarg couldn't reach it). Ponytail comment
     explains why an attribute, not a signature change.
   - `eval/adapters.py`: `BadasOpen` gained `skip_predictor` and `save_frames_dir` params; `score()`
     now optionally writes `<save_frames_dir>/<clip_id>.npz` with the full per-frame array plus
     `target_fps`/`stride`/`frame_count` before reducing to the scalar.
   - `eval/run_baselines.py`: wired `--skip-predictor` (previously silently dropped) and added
     `--save-frames-dir`.
   - **Smoke-tested before trusting**, twice: a 4-clip `--limit` run with both new flags (verified
     `.npz` files land with the right shape and leading NaNs), then a direct one-clip A/B
     (`skip_predictor=False` vs `True`) proving the score is bit-identical (0.995916 both ways) with
     real wall-clock savings (116.0s vs 99.9s). **Caught and fixed my own typo mid-launch** — first
     attempt passed a nonexistent `--no-badas-skip` flag, argparse rejected it before anything ran,
     no time or compute lost.
   - Launched the real second sweep into `runs/baselines2/` (PID 75682) — see the ⏳ banner.
7. **Closed out Phase 4 while the sweep runs, CPU-only, no MPS contact:**
   - `git add -f` the entire finished `runs/baselines/` tree (was untracked, blanket `*.log` gitignore
     rule was silently excluding `sweep.log` too — force-added as deliberate evidence, same treatment
     as `runs/falsification/`).
   - Ran `eval/plots.py` against real `metrics.json` for the first time ever; visually inspected the
     PR-curve PNG (opened via Read) rather than trusting the exit code alone.
   - Added `benchmark.threshold_sweep()`; regenerated all three `metrics.json` via the existing resume
     cache (0.0s runtime — proof nothing was re-scored) to backfill the new field.
   - Wrote and ran `tests/test_leakage.py` — real ids (1500 train vs 667 test-public, zero overlap)
     plus an injected 3-id violation that the function is proven to catch.
   - Wrote `scripts/badas_gate_provenance.py`, ran it once against the committed `metrics.json`.
   - Struck the ego/non-ego row from README §41 Phase 4 (task 3 + its acceptance criterion), verified
     directly against `data/nexar/{test-public,train}/*/metadata.csv` column headers, not inferred.
   - Re-ran `eval/benchmark.py`'s self-check after every change — still PASSES throughout.
8. **Committed `e7ac3e6`** — 18 files, +15,896/−7. Full message in `git log`; do not re-read it here,
   read it in git.
9. **Answered "what happen about the 17hr sweep??" and "what will happen after the second sweep"** —
   both in plain language, no files touched.

**Nothing else was committed. `runs/baselines2/` (the second sweep) is deliberately untracked — it is
incomplete and a running process is writing to it.**

---

## 5. CURRENT TECHNICAL TRUTH

### What works (verified this session)

| Thing | Evidence |
|---|---|
| `~/envs/crashdet` can load and run both shipped artefacts | `tests/test_weights_load.py` **PASSES** |
| Shipped weights are readable and architecturally intact | **578,689** non-optimizer parameters — matches README §11 and the Colab `model.summary()` exactly |
| Feature extractor runs | output shape `(2, 1280)` on a zeros batch |
| NumPy LSTM head forward pass | zeros input → `0.388557`, finite, in `[0, 1]` |
| The T3/T124 scoring path | Reproduces prior results **exactly** (see §6.2) |

### What does NOT work / is NOT verified

- **`code/crash_detection_enhanced.py` has NEVER been run end-to-end, and still cannot be.** It
  passes `py_compile`, but `import crash_detection_enhanced` fails with
  `ModuleNotFoundError: No module named 'torch'`. It also needs `ultralytics`, which was
  deliberately removed from `requirements.txt` (AGPL). **Its runtime correctness after the dead-code
  deletion is UNVERIFIED beyond `py_compile` plus a grep for dangling references.** Do not claim it runs.
- **The old model does not detect crashes.** Confirmed at chance on a corpus-controlled benchmark (§6.1).
- No evaluation harness exists. No BADAS-Open work has been started. No UK data exists.
- Calibration (ECE), FP/hour on real driving, mTTA, time-to-detection: **never computed.**

### Current architecture (as shipped, and now retired)

```
video → YOLOv8n vehicle detection → MobileNetV2 (frozen, ImageNet) + GlobalAveragePooling
      → 10×1280 sequence → LSTM head (578,689 params) → sigmoid
      → is_crash = (cnn >= 0.80) AND (n_vehicles >= 2)
```

The surrounding Kalman filter, pinhole ground projection and TTC are computed and **discarded** —
they reach the verdict only during the 9-frame CNN warm-up. MiDaS, the BEV renderer, `EgoZone` and
`FaultDetector` have been deleted from the tree.

### Current datasets on disk

| Dataset | Location | State |
|---|---|---|
| **Nexar test-public** | `data/nexar/test-public/{positive,negative}/*.mp4` | **667 clips, 2.7 GB, complete and verified.** Labels in `data/nexar/solution.csv`. Gitignored. |
| Nexar metadata | `data/nexar/test-public/*/metadata.csv` | Columns: `file_name, time_of_event, time_of_alert, light_conditions, weather, scene, time_to_accident` |
| CCD annotations | `data/Crash-1500.txt`, `data/train.txt`, `data/test.txt` | Committed |
| Colab notebook | `data/ccd/Untitled0.ipynb` | Committed, outputs included |
| Local videos | `videos/` | `crash1.mov`, `crash2.mov`, `safe.mp4` |
| **Nexar train split** | — | **NOT downloaded** (1,500 clips, 25.5 GB). **Now worth fetching** for probe training on the M4 Max (revised D2), at step 5 of §14 — not before. Fetch only this split; test-private is not needed. |

### Known limitations that must be carried forward

- **Nexar test-public contains only 0.90 hours of negative footage** (333 clips × 9.7 s mean).
  **You cannot credibly claim an FP/hour below ~1/hour from 54 minutes of negatives.** Any FP/hour
  figure must be reported with its denominator. This is precisely the gap README §34's UK-HN-500 fills.
- We hold **667 of the 1,344** clips in Nexar's full test set (test-public only; test-private not
  downloaded). This may matter for comparing against published BADAS-Open numbers (§10, U-B2).

---

## 6. EXPERIMENTS AND EVIDENCE

> Chronological. **Nothing here is to be deleted when newer results arrive.** Negative results are
> the project's primary asset.

### 6.1 T3 — Corpus control (THE headline experiment of this session)

- **Why:** The last outstanding falsification test (README §30 test 3, bug B4). CCD's positives are
  YouTube crash compilations and its negatives are BDD100K — two different corpora. A model can score
  0.9977 AUC by recognising *which corpus a clip came from*. T3 removes that shortcut by evaluating on
  a benchmark where positives and negatives share one corpus and one anonymisation pipeline.
- **Command:** `~/envs/crashdet/bin/python scripts/t3_corpus_control.py`
- **Input:** Nexar test-public, all **667** clips (334 positive / 333 negative); labels from
  `data/nexar/solution.csv` filtered to `Usage == "Public"`. Model: `models/crash_model_weights.weights.h5`.
- **Runtime:** 527.6 s.

**Result:**

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.5339** (chance = 0.50) |
| **Average Precision** | **0.5218** |
| Confusion matrix @ threshold 0.80 | TP **332** · FP **325** · FN **2** · TN **8** |
| TPR / FPR @ 0.80 | **0.9940** / **0.9760** |

Score distributions:

| Class | mean | median | min | max | fraction ≥ 0.80 |
|---|---|---|---|---|---|
| Positive (n=334) | 0.9978 | **0.9998** | 0.7110 | 0.9999 | 0.9940 |
| Negative (n=333) | 0.9766 | **0.9998** | 0.0016 | 0.9999 | 0.9760 |

- **What this proves (CONFIRMED):** on a corpus-controlled benchmark the shipped model performs at
  chance. It is worse than "does not generalise" — it emits near-1.0 on essentially everything from a
  corpus it was not trained on, firing on 97.6% of negatives. The CCD val AUC of 0.9977 was measuring
  the corpus boundary, exactly as bug B4 predicted.
- **What this does NOT prove:** it does not measure how a *correctly designed* model would do on
  Nexar (BADAS-Open reports AP 0.86 — see §10). It does not establish an FP/hour figure for real
  driving; the negative footage is only 0.90 h. It says nothing about the *architecture's* ceiling
  independent of the CCD training data — though README §28 argues that separately from the GAP bottleneck.
- **Consequence:** the old model is **retired with evidence, not assertion** (README §28's requirement).
  Bug B4 closed as "diagnosed, not fixable on this model or dataset." The Phase 2 gate is satisfied.
- **Raw output:** `runs/falsification/T3_corpus_control.json` (all 667 per-clip records),
  `runs/falsification/T3_corpus_control.md` (summary).

### 6.2 T3-validation — self-falsification of the T3 harness (CONFIRMED)

- **Why:** T3's most striking feature is that *both* classes have a median of 0.9998. That pattern is
  equally consistent with a broken preprocessing path. **The result had to be falsified before being
  accepted** — the alternative was building the whole project on a measurement bug.
- **Method:** ran the identical `score_clip()` function over the three local videos, whose scores were
  independently measured in a previous session by `scripts/t124_model_falsification.py` (and, for
  `safe.mp4`, by the original Keras pipeline before that).
- **Result:**

| Video | T3 scorer (this session) | T124 reference | Original README |
|---|---|---|---|
| `videos/safe.mp4` | **0.7914** (714 frames, 24.0 fps) | 0.7914 | "0.79" |
| `videos/crash1.mov` | **0.9998** (257 frames, 35.2 fps) | 0.9998 | — |

- **What this proves:** the scoring path discriminates strongly (0.79 vs 0.9998) when discrimination
  is present, and reproduces two independent prior measurements **exactly**. The Nexar collapse is
  therefore a property of the model under corpus shift, **not** an artefact of the harness. Three-way
  agreement across different environments also validates the NumPy forward pass against the real
  Keras pipeline.
- **Consequence:** T3 accepted as sound. This check should be kept as the harness's regression test (§13).

### 6.3 Numerical-sanity investigation (CONFIRMED cosmetic)

Covered in §4.6. Investigated unexpected `RuntimeWarning`s rather than suppressing them blind;
confirmed no NaN/inf anywhere in features, activations, or weights. Cosmetic.

### 6.4 Footage-duration measurement (CONFIRMED)

Computed from the T3 per-clip records (`decoded_frames / fps`):

| Class | Clips | Total | Mean clip |
|---|---|---|---|
| Negative | 333 | 54.0 min = **0.90 h** | 9.7 s |
| Positive | 334 | 55.3 min = 0.92 h | 9.9 s |

- **Consequence:** hard ceiling on FP/hour claims, recorded in §5. Note this contradicts README §49's
  description of Nexar clips as "~40 s" — for **test-public** the measured mean is ~9.8 s. Flagged as
  NEEDS VERIFICATION in §10 (may differ for the train split, or the README may be imprecise).

### 6.5 `tests/test_weights_load.py` (PASSED)

- **Command:** `~/envs/crashdet/bin/python tests/test_weights_load.py`
- **Output:**
  ```
  ok  weights readable, 578,689 non-optimizer parameters
  ok  feature extractor runs, output (2, 1280)
  ok  head forward pass on zeros -> 0.388557
  PASS: this environment can load and run the shipped model artefacts.
  ```
- **Consequence:** bug R2 is now detectable by a test instead of by surprise.

### 6.9 ⛔ SUPERSEDED — DO NOT ACT ON THIS SECTION. READ §6.11 FIRST.

> **Session 3 disproved this section's diagnosis by direct measurement.** The symptom it describes
> (199 "unexpected" keys) is real; the conclusion — that the predictor is unimplemented, that weights
> are silently lost, and that Phase 5 requires reimplementing the module — is **FALSE**. The predictor
> is V-JEPA2's own, already implemented; the checkpoint merely stores it twice; `load_state_dict`
> reports **missing 0**; all 199 duplicate pairs are **bitwise identical**. See **§6.11** for the
> evidence and **D9** for the resulting decision. This text is retained only as a record of how the
> wrong conclusion was reached.

#### (original session-2 text follows) BADAS-Open's published inference code CANNOT reproduce its own model (session 2)

**This is the most important finding of session 2 and it reshapes Phase 5's effort estimate.**

- **How found:** ran `scripts/badas_smoke.py` (U-B6 timing). Model load emitted
  **"unexpected keys in checkpoint (199 keys)"** — investigated rather than ignored, per D7.
- **Method:** loaded `models/badas/weights/badas_open.pth` directly with `torch.load` and enumerated
  every tensor. The checkpoint is the "internal format": top-level keys
  `epoch, model, optimizer, scheduler, val_acc, config, model_info`.
  **`epoch: 3`, `val_acc: 87.067`.**

**The checkpoint contains FOUR modules (802 tensors):**

| Module | Tensors | Shape detail |
|---|---|---|
| `backbone` | 587 | V-JEPA2 ViT-L, hidden 1024 |
| **`predictor`** | **199** | **12 layers @ dim 384**, `embeddings.mask_tokens (10,1,1,384)`, in-proj `(384,1024)`, out-proj `(1024,384)` |
| `temporal_processor` | 6 | `MultiheadAttention` in_proj `(3072,1024)` + out_proj `(1024,1024)` + LayerNorm(1024) |
| `classifier` | 10 | Linear(1024,768) · LN · Linear(768,768) · LN · Linear(768,2) at indices 0,2,4,6,8 |

**The training config embedded in the checkpoint:**

```
use_future_prediction: true          future_prediction_seconds: 1.0
predictor_combination_method: concat temporal_method: attention
frame_count: 16                      img_size: 224
temperature: 2.0                     use_temperature_scaling: true
original_fps: 4                      data_root: .../balanced_dataset_2s
head_type: mlp  head_hidden_dim: 768  head_num_layers: 3  temporal_num_heads: 8
```

- **CONFIRMED: `EnhancedVideoClassifier` (`vendor/badas-open/badas/utils/video.py:132-180`) has no
  `predictor` attribute at all.** Its `forward()` is `backbone → temporal_processor → classifier`.
  The 199 `predictor.*` tensors are therefore **silently discarded** by
  `load_state_dict(..., strict=False)` (`utils/video.py:379`). The count matches exactly: 199.
- **CONFIRMED: the head weights DO match** — `temporal_processor` and `classifier` shapes line up with
  the vendored classes precisely. So **nothing errors.** The model loads, runs, and returns
  plausible-looking probabilities while missing its entire future-prediction pathway.
- **HYPOTHESIS (not verified):** the intended forward is backbone → predictor (1024→384, 12 layers,
  →1024) predicting latents **1.0 s ahead**, then `concat` with encoder tokens **along the token
  axis** (it cannot be the feature axis — `temporal_processor` takes 1024, not 2048), then temporal
  attention → mean-pool → MLP. Shapes are consistent with this. **It needs verification against
  V-JEPA2's own predictor API before being relied on.**
- **Consequences:**
  1. **A naive `pip install badas && run` produces a silently wrong model.** Any AP it yields is not
     BADAS-Open's AP. If a reproduction attempt misses the published figure, **this is the first
     thing to suspect** — ahead of preprocessing (U-B3).
  2. **Phase 5 is materially harder than "load and evaluate."** Reproducing the published number
     requires reimplementing the predictor + concat path. This is real work, not configuration.
  3. It also supplies a **plausible explanation for the disputed numbers (U-B2)** — the model card's
     AP 0.86 and `config.json`'s AP 83.2 may have been produced by different code paths. **UNVERIFIED.**
  4. Recorded in README §49's reference row. ⚠️ **Not labeled "B10"** — that ID is already taken by
     README §15's "training code is not in version control." This finding has no bug ID yet; if one
     is assigned, use the next free letter/number, not B10.
- ⚠️ **Also unresolved: `original_fps: 4` in the training config vs `target_fps: 8.0` in
  `badas_loader.py:47`.** 16 frames @ 8 fps = 2.0 s, which matches `balanced_dataset_2s`. But the
  config's `original_fps: 4` is unexplained. **UNKNOWN — do not assume 8 fps is right.**

### 6.10 Environment + smoke test (session 2)

- `~/envs/badas` now holds **torch 2.14.0 · transformers 5.17.0 · albumentations 2.0.8 · cv2 5.0.0**.
- **`torch.backends.mps.is_available()` → True**; model placed on `mps:0` successfully.
- **Model load takes ~544 s** (9 min) — 3.7 GB checkpoint plus first-time backbone download.
- **`AutoVideoProcessor` resolves to `VJEPA2VideoProcessor`** — so the **HF processor path runs, not**
  the ImageNet manual fallback. Resolves that half of U-B3: normalization constants come from the
  backbone's processor config, not from `utils/video.py:249`.
- ❌ **BLOCKED — API drift.** `EnhancedVideoClassifier.forward` calls
  `self.backbone(pixel_values=...)`, but transformers 5.17.0's `VJEPA2Model.forward` requires
  **`pixel_values_videos`**. Fails with `TypeError: VJEPA2Model.forward() missing 1 required
  positional argument: 'pixel_values_videos'`. Their `requirements.txt` says only
  `transformers>=4.40.0`; the code was written against 4.x. **Fix: pin transformers 4.x, or patch the
  call.** Patching is cleaner but must not be confused with the separate predictor problem (§6.9).
- **U-B6 is therefore STILL UNMEASURED** — no forward pass has completed yet.

### 6.7 Nexar test-public clips are truncated BEFORE the event (CONFIRMED, session 2)

- **Why measured:** to build mTTA into the Phase 4 harness, the event timestamp must be locatable on
  the clip's own timeline.
- **Method:** compared `time_of_event` from `metadata.csv` against clip duration
  (`decoded_frames / fps` from the T3 records). Durations independently cross-checked against
  `cv2.CAP_PROP_FRAME_COUNT` on three clips — **exact match** (e.g. `00002`: 302 frames both ways),
  so this is not a decode artefact.

| Quantity | min | median | max |
|---|---|---|---|
| `time_of_event` (positives, n=334) | 9.63 s | **20.00 s** | 26.85 s |
| clip duration | — | **9.93 s** | — |
| `time_of_event − clip_duration` | **+0.49 s** | +10.02 s | +16.95 s |
| `time_of_event − time_of_alert` | 0.03 s | 1.83 s | 4.30 s |

- **CONFIRMED:** `time_of_event` exceeds the distributed clip duration for **334 of 334** positives.
  `time_to_accident` takes only three values: **{0.5, 1.0, 1.5}**.
- **CONFIRMED:** the clip's offset into the original video is **not derivable** from the shipped
  metadata — `time_of_event − clip_duration` is not constant (spans 0.49 s to 16.95 s) and does not
  reduce to `time_to_accident`.
- **HYPOTHESIS (not verified):** clips were deliberately truncated shortly before the collision, which
  is what makes this a collision **prediction** benchmark rather than a detection one. Plausible but
  not confirmed from Nexar documentation.
- **Consequences:**
  1. **mTTA and time-to-detection cannot be computed on test-public.** §14 action 1 listed both as
     harness metrics; they are struck from the Phase 4 metric set for this split. Any mTTA claim needs
     either the train split (which may carry usable timestamps — **UNKNOWN, verify**) or UK footage.
  2. **U-B3 is partially answered by elimination.** BADAS's 16-frame selection **cannot** be anchored
     to `time_of_alert` when evaluating test-public, because that timestamp is not inside the clip. It
     must be clip-relative (uniform / fixed stride / last-N). Which one still needs the source.
  3. The published BADAS mTTA of 4.9 s was therefore **not** measured on clips shaped like these.
     Do not compare against it using test-public.
- **Supersedes:** U-N1 is resolved in part — test-public clips really are ~9.8 s, not README §49's
  "~40 s". The ~40 s figure may describe the untrimmed originals. Train split still unverified.

### 6.8 `eval/benchmark.py` self-check (PASSED, session 2)

- **Command:** `~/envs/crashdet/bin/python eval/benchmark.py`
- **Reproduces the committed T3 numbers through the new code path**, which is the regression test
  §14 action 1 called for:
  ```
  ok  T3 reproduced: AUC 0.5339  AP 0.5218  TP/FP/FN/TN 332/325/2/8
  ok  FP/hour 361.4 over 0.90 h of negatives
  ok  precision@recall0.80 0.5253   ECE 0.4880
  ok  by weather           {'Clear': 458, 'Cloudy': 177, 'Rain': 32}
  ok  by scene             {'Highway': 183, 'Industrial': 12, 'Other': 23, 'Rural': 16, 'Sub-urban': 118, 'Urban': 315}
  ok  by light_conditions  {'Bright': 3, 'Dark': 21, 'Normal': 608, 'Twilight': 35}
  ok  leakage check fires on overlap
  PASS
  ```
- **Two metrics newly computed for the old model, never measured before:**
  **precision @ recall 0.80 = 0.5253** and **ECE = 0.4880**. The ECE is catastrophic, as expected from
  a model that emits ~0.9998 on both classes. Both are consistent with D1 and change nothing.
- **The 361.4 FP/hour figure is a ceiling artefact, not a usable number** — 325 FP over 0.90 h. It is
  printed only to prove the denominator is carried. Do not quote it as a real-driving rate (§12).
- **Per-condition counts are badly unbalanced** for any per-condition claim: Rain n=32, Bright n=3,
  Dark n=21, Industrial n=12. Per-condition AP on these cells is not meaningful. Noted, not fixed.

### 6.11 ⚠️ §6.9 IS WRONG — the predictor needs no reimplementation (CONFIRMED, session 3)

**This supersedes §6.9's conclusion. §6.9 is kept below as history; its *finding* was real, its
*diagnosis* was wrong.** Evidence: `scripts/badas_predictor_probe.py`, plus a direct `torch.equal`
comparison.

- **CONFIRMED — V-JEPA2 has its own predictor.** `transformers.VJEPA2Config` for
  `facebook/vjepa2-vitl-fpc16-256-ssv2` gives `pred_hidden_size 384`, `pred_num_hidden_layers 12`.
  `VJEPA2Model` has `.predictor` with **exactly 199 parameters**, including
  `predictor.embeddings.mask_tokens`, `predictor.proj.weight (1024, 384)`. Names and shapes match the
  checkpoint's `predictor.*` **exactly**. `VJEPA2Model.forward` takes a `skip_predictor` argument
  whose **default is `False`** — so the predictor runs by default.
- **CONFIRMED — the checkpoint stores the predictor TWICE.** `obj["model"]` has **802 tensors**:
  `backbone` 587 · `predictor` 199 · `temporal_processor` 6 · `classifier` 10. The model
  (`EnhancedVideoClassifier`) has **603 tensors**, of which **199 are `backbone.predictor.*`**.
  802 − 199 = 603.
- **CONFIRMED — the as-shipped load loses nothing.** `load_state_dict(strict=False)` reports
  **`missing 0`, `unexpected 199`**. Missing 0 is the decisive number: the model's
  `backbone.predictor.*` keys were **already satisfied** by the checkpoint's `backbone.*` block. The
  199 "unexpected" keys are the **top-level duplicate**.
- **CONFIRMED — the two copies are bitwise identical.** All **199/199** pairs pass `torch.equal`
  (`sibling predictor.X` vs `backbone.predictor.X`). 0 differing.
- **Therefore: session 2's claim "any AP from a naive run is not BADAS-Open's AP" is FALSE as stated.**
  The predictor weights are loaded. Nothing is discarded at load time.
- **The REAL gap (CONFIRMED, and smaller):** `EnhancedVideoClassifier.forward()` uses only
  `outputs.last_hidden_state` and **discards `outputs.predictor_output`**. Training config says
  `use_future_prediction: true`, `future_prediction_seconds: 1.0`,
  `predictor_combination_method: "concat"`. So the future-prediction pathway is loaded, computed, paid
  for, and ignored. **Phase 5 task 3 = consume `predictor_output`. A forward-path change.**
- **Resolved in passing:** the 224-vs-256 confusion. The checkpoint config carries **both**
  `img_size: 224` **and** `vjepa2_crop_size: 256`. Also `original_fps: 4` vs `target_fps: 8.0` —
  **STILL UNRESOLVED**, still worth settling before trusting a number.
- **NEEDS VERIFICATION:** the concat axis for task 3. Token axis is the hypothesis; feature axis is
  ruled out because `temporal_processor` takes 1024, not 2048. **Do not guess it.**

### 6.12 U-B6 ANSWERED — timing, and a correction to a number this session itself produced

Evidence: `scripts/badas_smoke.py` (compute only), `eval/run_baselines.py --limit 6` (end-to-end).

| Measurement | Value | CONFIRMED by |
|---|---|---|
| Model load | **8.5 s** | smoke run. **Session 2's "~544 s / 9 min" was a one-time HF download, not load cost.** |
| Per-window, compute only, 16×224×224 MPS | **0.856 s** (min 0.850 / max 0.858) | smoke run |
| Per-window, `skip_predictor=True` | **0.629 s** — predictor costs **+0.213 s = 25%** of the pass | direct A/B |
| `last_hidden_state` with vs without predictor | **bit-identical** (`allclose`, atol 1e-4) | direct A/B |
| Windows per clip @ stride 1, 8 fps | ~64 → **42,688 passes** per 667-clip sweep | arithmetic |
| **End-to-end per clip** | **~97 s** (583.7 s for 6 clips) | `--limit 6` run |
| **Full 667-clip sweep** | **≈18 h at stride 1** (≈9 h at stride 2) | 667 × 97 s |

**⚠️ CORRECTION MADE WITHIN THIS SESSION.** An earlier README edit this session stated the sweep was
**≈10.1 h**, derived from the compute-only figure. `badas_smoke.py` explicitly feeds a synthetic
tensor and bypasses video IO. Decode + `VJEPA2VideoProcessor` add ~1.8×. **The honest end-to-end
figure is ≈18 h.** README was corrected. **Always quote end-to-end for planning; the per-window
number is only for comparing model configurations.**

### 6.13 Phase 4 AC #1 MET — three models, one code path (CONFIRMED, session 3)

`eval/run_baselines.py --no-badas`, all 667 clips, `~/envs/crashdet`:

```
mobilenetv2-lstm (CCD-trained, retired)   AP 0.5218  AUC 0.5339  FP/h 361.4 over 0.90h  ECE 0.4880
always-negative                           AP 0.5007  AUC 0.5000  FP/h   0.0 over 0.90h  ECE 0.5007
```

- **CONFIRMED — the new path is faithful.** The retired model reproduces its committed T3 numbers
  *through the adapter path*: AUC 0.5339, AP 0.5218, TP/FP/FN/TN **332/325/2/8**. Asserted in code.
- **CONFIRMED — identical denominator.** Both models: **0.8992 h** of negatives. This is deliberate —
  `benchmark.durations()` is ONE canonical table sourced from the committed T3 run (which actually
  decoded all 667 clips rather than trusting container headers). A per-adapter denominator would be a
  silent bug that makes rates non-comparable.
- **CONFIRMED — the trivial baseline wins on the metric that matters.** 0.0 vs **361.4** FP/hour. The
  retired model's AP edge over always-negative is **0.0211**. This is README §14 item 14, measured.
- **CONFIRMED — new metrics for the old model:** `precision@recall0.80 = 0.5253`, **ECE 0.4880**
  (target < 0.05 — so the model is almost maximally uncalibrated).
- `tests/test_score_regression.py` **PASSES**: `safe.mp4` = **0.7914** exactly, 714 frames, 24.0 fps.
  `CNN_THRESH = 0.80` clears this single negative by **0.0086**.

### 6.14 BADAS adapter works end-to-end — but produced NO usable number (session 3)

`eval/run_baselines.py --limit 6` (3 positive + 3 negative), `~/envs/badas`, MPS:

```
badas-open(stride=1,fps=8.0,img=224)   AP 1.0000  AUC 1.0000  FP/h 120.5 over 0.01h  ECE 0.2455  [583.7s]
```

- **CONFIRMED:** the adapter runs end-to-end — load, `VJEPA2VideoProcessor`, sliding window
  (window 16, stride 1), temperature-2.0 scaling, `np.nanmax` reduction, finite scores in [0,1].
- **⚠️ NOT A RESULT. DO NOT QUOTE AP 1.0000.** n=6. Three positives and three negatives. ECE 0.2455 at
  n=6 is noise. This run proves the *plumbing*, nothing about BADAS's performance.
- **UNKNOWN: BADAS-Open's actual AP/AUC on Nexar test-public.** Requires the full ~18 h sweep.
- Scoring contract CONFIRMED by reading the vendored source: `predict()` returns one probability per
  frame; `softmax(logits / 2.0, dim=1)[:, 1]` → **class index 1 = collision**; leading
  `frame_count` frames are NaN by design.
- **Latent upstream bug confirmed:** `badas/cli.py:120` and `examples/basic_inference.py:103` use
  builtin `max(predictions)`, which returns **NaN** whenever a NaN is seen first — and their own
  windowing guarantees leading NaNs. Our adapter uses **`np.nanmax`** (U-B5). Do not "simplify" it back.

### 6.15 progress.md was ONE SESSION STALE (CONFIRMED, session 5)

**Claim in §13 as written:** *"THE EXACT NEXT ACTION — run the full BADAS sweep"*, with a pending
user decision on stride 1 vs stride 2.

**Reality found at 20:05 session 5:** the sweep had been running since **13:48** (PID 67741,
`caffeinate -i`, stride 1, no `--skip-predictor`), 135 clips deep. Three commits from a session 4 —
`e488a05`, `4204bf2`, `cc13966` — postdate progress.md's last write at 13:14.

**Consequences, both acted on:**
- **The stride decision is moot.** Stride 1 is running and ~230 clips are banked at that setting.
  Switching now would invalidate the whole cache (`_resume` raises on an adapter-name mismatch, by
  design). Stride 1 is also the faithful comparison. **Treat stride 1 as settled.**
- **The ~18 h ETA was wrong.** Measured over the first 140 clips: **~167 s/clip → ≈31 h**, not the
  ~97 s/clip → ≈18 h projected from a 6-clip smoke run. After the session-5 restart the observed rate
  improved to **~117 s/clip** over 90 clips. `eval/run_baselines.py`'s module docstring now carries
  the measured figure.

**Lesson for the next session: verify the running-process state before trusting §13.** A handoff
cannot record what happened after it was written.

---

### 6.16 🔴 THE SWEEP WAS SCORING ALL POSITIVES FIRST — found and fixed (CONFIRMED, session 5)

**The defect.** `eval/run_baselines.py::balanced_ids(None)` returned `sorted(labels)`. In Nexar
test-public **every one of the 334 positive clip ids sorts below every one of the 333 negatives** —
verified: the first 140 sorted ids are 140/140 positive. So the sweep scored positives exclusively
for its first ~9 hours.

**Why that is not cosmetic.** AP, ROC-AUC, precision@recall and FP/hour are **all undefined with one
class present** — `benchmark.evaluate()` cannot produce a number. So a ~31 h run would have yielded
**zero diagnostic signal until hour ~9**. README §41 Phase 5's own gate note lists five unresolved
ambiguities in this setup (disputed published figure, split size, predictor path, score reduction,
fps). Any of them could make the number wrong. Discovering that at hour 31 instead of hour 2 is a
pure, avoidable loss — and if the laptop had needed a reboot first, the run would have produced
nothing usable at all.

**The fix.** `balanced_ids` now interleaves `pos[0], neg[0], pos[1], neg[1], …` via
`itertools.zip_longest` (334 vs 333 handled). `--limit N` still yields a balanced subset.

**Safety of the restart, verified before acting:**
- `scores.jsonl` ended with a newline and all 140 lines parsed cleanly — no torn record.
- Killed PID 67743 (caffeinate) + 67741 (python); relaunched; log printed
  **`resuming: 140/667 already scored`**. Nothing was re-scored.
- **Metrics are order-independent** (AP/AUC/ECE operate on arrays, not visit order), so the completed
  numbers are identical to what the original ordering would have produced. This is a **visiting-order
  change only** — it is not a change to the measurement.

**Immediate payoff:** the first negative landed within 2 minutes of the restart instead of ~9 hours
later, and a partial AP was readable within 3 hours (§6.17).

---

### 6.17 FIRST REAL BADAS-OPEN SIGNAL — ⚠️ PARTIAL, NOT A RESULT (session 5)

**Two readings, both partial. The later one is far more informative because the classes are
balanced by then — this is exactly the payoff of the interleaving fix in §6.16.**

| Metric | @ 230/667 (140 pos / 90 neg) | **@ 535/667 (268 pos / 267 neg)** |
|---|---|---|
| Average precision | 0.9274 | **0.8464** |
| ROC-AUC | 0.8988 | **0.8579** |
| ECE | 0.2794 | **0.3425** |
| FP/hour @ thr 0.80 | 236.8 over 0.24 h | 226.9 over **0.72 h** |
| TP / FP / FN / TN @ 0.80 | 138 / 58 / 2 / 32 | 258 / 164 / 10 / 103 |
| precision @ recall 0.80 | 0.8984 | **0.7883** |

**The drop from 0.9274 to 0.8464 is expected and healthy**, not a regression: the first reading had
all 140 positives against only 90 negatives, so it was biased upward exactly as §6.17 predicted.

**⚠️ Where the 535-clip reading sits against the disputed published figures — this is the Phase 5
gate evidence:**

| Source | AP | AUC |
|---|---|---|
| Model card | 0.86 | 0.88 |
| Vendored `badas/config.json` (Nexar) | 0.832 | 0.85 |
| **Ours, partial @ 535, 667-clip public half** | **0.8464** | **0.8579** |

**It lands between the two published numbers on both metrics.** That is the outcome that says the
harness is correct — and it makes README §41 Phase 5's "AP ≈ 0.8+ → harness trustworthy" branch the
live one. **Still partial. Confirm against the full 667 before recording anything.**

**Classification: NEEDS VERIFICATION. Do NOT commit, quote, or put this in `metrics.json`.** The
clips scored so far are a **prefix of the visiting order, not a random sample** — all 140 positives
are in, but only 90 of 333 negatives, so the negative set is heavily under-represented and AP is
biased upward. The final number will move, most likely down.

**What it does legitimately tell us — README §41 Phase 5's own interpretation ladder:**
- *AP ≈ 0.5 → "something in our setup is broken, fix before anything else"* — **decisively cleared.**
- *AP ≈ 0.6 → "stop, either the harness is wrong or the published figures do not transfer"* —
  **cleared.**
- *AP ≈ 0.8+ → harness trustworthy, Phase 5 passes* — **on track, pending the full run.**

**Also observable and worth carrying forward:** positive scores are saturated near 0.995 (median
0.9949 over the first 140), which is why **ECE is poor (0.2794)**. Calibration is a named Phase 6
deliverable (temperature scaling, target ECE < 0.05) and this is early evidence it will be needed.
**FP/hour 236.8 is a ceiling artefact printed to prove the denominator is carried, not a rate** —
0.24 h of negatives cannot evidence anything near the < 0.1/h product target.

---

### 6.18 U-B7 RESOLVED — `original_fps: 4` vs `target_fps: 8.0` is NOT a contradiction (CONFIRMED)

README §41 Phase 5's gate note, reason 5, states this is *"unresolved"*. **It is now resolved, and it
was never a conflict.** The two keys count different things:

```
16 frames @ target_fps 8.0          = 2.0 s        <- matches data_root "balanced_dataset_2s"
V-JEPA2 tubelet_size 2              -> 8 tokens
8 tokens / 2.0 s                    = 4 tokens/s   <- THIS is original_fps
future_prediction_seconds 1.0 x 4   = 4 token steps ahead
```

`target_fps` is the **video** sampling rate; `original_fps` is the post-tubelet **token** rate, used
only to convert `future_prediction_seconds` into a token offset. Every number in the BADAS training
config reconciles simultaneously, with no contradiction left over.

**Evidence** (`scripts/badas_fps_probe.py`, which *asserts* the reconciliation rather than asserting
a reading of it):
- Checkpoint `config`: `frame_count 16`, `original_fps 4`, `future_prediction_seconds 1.0`,
  `data_root ".../balanced_dataset_2s"`, `predictor_combination_method "concat"`, `temperature 2.0`,
  `epochs 15` (stopped at **epoch 3**, `val_acc 87.07`).
- `AutoConfig` for `facebook/vjepa2-vitl-fpc16-256-ssv2`: **`tubelet_size: 2`**, `frames_per_clip: 16`.
- Upstream's own `badas/badas_loader.py` defaults, README and example all use
  **`target_fps=8.0, num_frames=16, window_stride=1`** — exactly what `eval/adapters.py` runs.

**Why this was checked before anything else:** a genuine 2× mismatch here would be this project's own
**B1 bug** (README §15 — the old model trained on 5 s spans and run on 0.33 s ones) repeated on a new
backbone, and it would have invalidated the entire multi-hour run. **It is not. The sweep's
configuration is faithful to training.**

---

### 6.19 U6 ANSWERED — `crash_model_cpu/` IS the shipped artefact (CONFIRMED, session 5)

README §41 Phase 0 task 7, open since the audit. **Answer: SAME RUN.** All **12 of 12** weight arrays
are identical (`np.allclose`, atol 1e-6; r = +1.0000 on every array with size > 1), 578,689 parameters
on both sides.

**Method** (`scripts/u6_compare_weights.py`): matched by an **explicit name-pair table**, not by
shape. A first attempt matched on shape and its own guard correctly refused to run — `(256,)` is both
the `dense` bias and the `lstm_1` bias, so shape is not a unique key. The script asserts every pair's
shapes agree, so a wrong pairing fails loudly instead of printing a false correlation.

**The mechanism, now established.** Notebook cell 11 (CPU re-save) and cell 12 (`save_weights`) both
**`load_model()` the same `best_crash_model.keras` file from Drive** rather than using the in-memory
model — cell 12 is literally `load_model(...)` → `save_weights(...)`. `ModelCheckpoint` writes that
path and **overwrites in place**, with no run id and no artefact hash.

**What this changes:** combined with README §1 finding 3 (the `best_crash_model.keras` committed at
`c63e307` differs from the shipped weights, `feat_reduce` kernel r = 0.0073), the evidence now shows
**exactly two distinct runs, not three**. U1/U2 — *which* run produced the shipped weights and what
its metrics were — remain **permanently UNRESOLVABLE** retrospectively, as the README already states.

---

### 6.6 Prior-session results (preserved — do NOT re-run, do NOT delete)

From `runs/falsification/RESULTS.md`, `T124_local_videos.json`, `T5_source_leakage.txt`:

| Test | Result | Verdict |
|---|---|---|
| **T1** single frame tiled ×10 | reproduces deployed score within 0.02 | **FAILED** — it is a per-frame image classifier |
| **T2** temporal shuffle | mean score moves **≤ 0.0013** on all three videos | **FAILED** — the two LSTM layers (246,528 params, 43% of the model) are inert |
| **T5** source leakage | 1,500 crash clips come from **133** YouTube videos; a random 80/20 split puts **113 of 133** sources on both sides → **91.4%** of clips leaked | **FAILED, severely** |
| **T5b** official CCD split | **107 of 133** sources on both sides | the official split does not fix it either |
| **T5c** label quality | accident onset at frame **37.2 of 50** on average → **~72%** of a positive clip's sampled frames contain no accident but are labelled 1 | labels wrong for most frames |
| **T6** always-negative baseline | 0 FP/hour vs the model's derived ≈23 FP/hour | **the trivial baseline wins** |
| **B1** stride | `safe.mp4` 0.0241 → 0.0003 at the training-matched stride | real, but second-order given T2 |
| **T4** crash excision | **NOT RUN — dropped** (decision D3) | — |

**The falsification suite is now closed. Every test that was run, failed.**

---

## 7. REPOSITORY CHANGES

**As of end of session 2 (2026-09-12). This supersedes any "this session" labels below the line.**

| Path | Purpose | Status |
|---|---|---|
| `README.md` | **Master plan** (3,201 lines) | **MODIFIED session 1 + session 2** — see §16 for both |
| `progress.md` | This execution-state file | **UNTRACKED**, session 1 + 2 content, not yet committed |
| `.gitignore` | Ignore rules | **MODIFIED session 2** — added `models/badas/` (was about to let 3.7 GB of gated weights into git; caught and fixed, see §4.9) |
| `eval/benchmark.py` | Session 2. Phase 4 harness core (see §6.8) | **UNTRACKED**, passing |
| `scripts/badas_smoke.py` | Session 2. Times one BADAS forward pass on MPS (U-B6) | **UNTRACKED**, written, **not yet run successfully** — blocked on §13 step 1 |
| `vendor/badas-open/` | Session 2. BADAS-Open source, vendored from the ungated GitHub mirror `getnexar/BADAS-Open` (Apache-2.0). 228 KB, `.git`/`assets/` stripped. | **UNTRACKED** |
| `models/badas/weights/badas_open.pth` | Session 2. BADAS-Open checkpoint, downloaded from HF after user accepted the gate. 3.7 GB. | **Present on disk, gitignored** (fixed this session) |
| `requirements.txt` | Python deps | **REWRITTEN session 1** — TF 2.19.1/Keras 3.15.1; ultralytics/torch/scipy removed with documented reasons |
| `.python-version` | Pins Python 3.11 | **CREATED session 1** |
| `tests/test_weights_load.py` | Artefact-load smoke test (bug R2). First test in the repo. | **CREATED session 1**, passing |
| `scripts/t3_corpus_control.py` | T3 corpus-control experiment | **CREATED session 1** |
| `runs/falsification/T3_corpus_control.json`, `.md` | T3 raw output + summary | **CREATED session 1** |
| `scripts/t124_model_falsification.py`, `scripts/t5_source_leakage.py` | T1/T2/B1/T5 experiments | Untouched — deliberately, to keep results reproducible (D5) |
| `runs/falsification/RESULTS.md` | Prior falsification results | Untouched |
| `code/crash_detection_enhanced.py` | The only surviving pipeline, 977 lines | Untouched. Compiles; **cannot import** (no torch) |
| `models/crash_model_weights.weights.h5` | The shipped LSTM head, 578,689 params | Untouched. **Retired by evidence (D1)** |
| `models/feature_extractor_saved/` | MobileNetV2 SavedModel used at inference | Untouched. Provenance still UNKNOWN (U5) |
| `data/ccd/Untitled0.ipynb` | Recovered Colab training notebook | Untouched. Historical record only |

---

### 7.1 Files changed in SESSIONS 4 and 5 (verified against `git status` / `git diff`)

| Path | Session | State | What it now does |
|---|---|---|---|
| `eval/benchmark.py` | 4 | **COMMITTED** `e488a05` | Sweep is resumable: per-clip append+fsync to `scores.jsonl`, `_resume()` on restart, adapter-name mismatch raises. Self-check still reproduces T3 exactly. |
| `eval/plots.py` (119 ln) | 4 | **COMMITTED** `4204bf2` | PR curves + reliability diagrams on shared axes. Reads a finished `metrics.json`. Run from `~/envs/crashdet` (matplotlib is not in `~/envs/badas`). |
| `eval/run_baselines.py` | 4 + 5 | **MODIFIED, uncommitted** | Session 4 wired in resume. **Session 5: `balanced_ids` now interleaves pos/neg** (§6.16) via `itertools.zip_longest`, and the module docstring carries the **measured** ~167 s/clip / ~31 h instead of the smoke-run ~97 s / ~18 h. |
| `eval/peek.py` (55 ln) | 5 | **NEW, untracked** | Reads the live `scores.jsonl` and prints metrics on whatever has landed, marked PARTIAL. The safe way to check an in-flight sweep. `~/envs/crashdet/bin/python eval/peek.py` |
| `scripts/u6_compare_weights.py` (85 ln) | 5 | **NEW, untracked** | Answers U6 by explicit name-pair comparison of the SavedModel checkpoint against the `.h5` (§6.19). |
| `scripts/badas_fps_probe.py` (68 ln) | 5 | **NEW, untracked** | Answers U-B7; asserts the fps/token reconciliation against the checkpoint + `AutoConfig` + upstream defaults (§6.18). |
| `runs/legacy-colab/README.md` | 5 | **NEW, untracked** | Phase 0 acceptance item: states plainly why `Best val AUC : 0.9977` does not describe the shipped weights — wrong run, selection-contaminated, leaking split, corpus-aligned classes — and gives the T3 counter-number. |
| `runs/legacy-colab/training_log.txt` · `metrics_val.txt` | 5 | **NEW, untracked** | Notebook cell 8 + cell 9 outputs, verbatim, extracted from the committed `data/ccd/Untitled0.ipynb`. |
| `runs/baselines/` (40 KB) | 4 + 5 | **NEW, untracked, LIVE** | `sweep.log` + `badas-open/scores.jsonl`. **A RUNNING PROCESS IS APPENDING TO THIS.** Do not edit, move or delete. |
| `code/crash_detection_enhanced.py` | 5 | **MODIFIED, uncommitted** | 977 → 965 lines. Removed the unreachable fault-reporting block, the `fault_info` parameter, both call-site arguments and both dead `first_fault = None` locals. `grep -rn "at_fault\|FaultDetector\|EgoZone" code/` is now **clean** — Phase 3's own acceptance criterion, previously failing. Still cannot import (no torch/ultralytics/scipy); `ast.parse` clean. |
| `docs/HANDOFF.md` | 5 | **MODIFIED, uncommitted** | Superseded banner at the top pointing to progress.md, plus three in-place corrections where it claimed "T3 — Corpus control: NOT YET RUN". That file is committed and was contradicting the project's headline result. |
| `README.md` | 5 | **MODIFIED, uncommitted** | **One added note only** — a dated delta table under the §2 `b539d6e` inventory. See §16. |
| `progress.md` | 4 + 5 | `cc13966` then **MODIFIED** | Committed unchanged by session 4; rewritten by session 5, then this handoff (session 6). |

---

### 7.2 Files changed in SESSION 6 (committed `e7ac3e6`, verified against `git show --stat`)

| Path | State | What it now does |
|---|---|---|
| `runs/baselines/` (whole tree: 3 models × `metrics.json`+`scores.jsonl`, `baseline_table.json`, `sweep.log`, `plots/*.png`) | **NEW, COMMITTED** | The finished 18h BADAS-Open sweep. Force-added despite `*.log` catching `sweep.log` (deliberate evidence, like `runs/falsification/`). |
| `eval/benchmark.py` | **MODIFIED, COMMITTED** | Added `threshold_sweep()` (9-point threshold→recall/precision/FP-h curve, persisted into every `metrics.json`). Self-check still passes unchanged. |
| `eval/adapters.py` | **MODIFIED, COMMITTED** | `BadasOpen` gained `skip_predictor` (wired to `VJEPAModel`) and `save_frames_dir` (writes `<id>.npz` per frame before reducing to a scalar). |
| `eval/run_baselines.py` | **MODIFIED, COMMITTED** | `--skip-predictor` now actually reaches the adapter (previously dead). New `--save-frames-dir` flag. |
| `vendor/badas-open/badas/utils/video.py` | **MODIFIED, COMMITTED** | `EnhancedVideoClassifier` gained a `self.skip_predictor` attribute (default False), read in `forward()`. Ponytail comment explains why an attribute, not a `forward()` arg. |
| `vendor/badas-open/badas/models/vjepa.py` | **MODIFIED, COMMITTED** | `VJEPAModel.__init__` accepts `skip_predictor`, sets it on `self.model` after `load()`. |
| `tests/test_leakage.py` (59 ln) | **NEW, COMMITTED** | Phase 4 AC #3. Real 1500 train ids vs 667 test-public ids (zero overlap) + an injected 3-id violation that must raise. |
| `scripts/badas_gate_provenance.py` (83 ln) | **NEW, COMMITTED** | Run once; writes the Phase 5 `gate` block into `runs/baselines/badas-open/metrics.json`. Re-running it is idempotent (overwrites the same key). |
| `README.md` | **MODIFIED, COMMITTED** | 2 lines: struck the ego/non-ego breakdown from Phase 4 task 3 + its acceptance criterion. See §16. |

**Untracked at handoff, deliberately:** `runs/baselines2/` — the second sweep, still running. Do not
add or commit until it finishes (§12).

---

## 8. GIT STATE

**Verified at 2026-09-13 14:45 IST, not remembered.**

- **Branch:** `main`
> **🔴 SUPERSEDED 2026-09-16 (session 8). The block immediately below is session 6's git state and is
> STALE. Current git state:**
>
> - **HEAD: `252d282`** — "Confirm the reduction finding survives held-out selection (R1 gate 2)"
> - **Branch `main`, IN SYNC with `origin/main`. Pushed, user-approved.**
> - **Working tree CLEAN** apart from this handoff edit to `progress.md`.
> - **`runs/baselines2/` is COMMITTED and STATIC.** No process is writing to it. PID 75682 is gone.
> - Session 8 added: `eval/heldout_half.py`, and committed `eval/reduction_study.py` +
>   `eval/calibration.py` (both written by the lost session, previously untracked).
>
> Session 6's text follows unaltered as the record of what was true then.

- **HEAD:** `e7ac3e6` — "Close Phase 4 and judge the Phase 5 gate: commit the 667-clip sweep, plots,
  threshold sweep, leakage test" (Sun 2026-09-13)
- **SESSION 6 COMMITTED.** One commit, 18 files, +15,896/−7. See §7.2 for the file-by-file breakdown.
- **Working tree:**
  ```
   M progress.md                      (this handoff)
  ?? runs/baselines2/                 <- LIVE, a running process (PID 75682) is writing here
  ```
- **Nothing else is dirty.** `README.md`, `eval/*`, `tests/*`, `scripts/*`, `vendor/*` are all clean
  against HEAD — everything from this session's implementation work is already committed.
- **Not pushed.** Do not push to the remote without asking — no approval was given or requested this
  session.

**Commit history (newest first):**

```
e7ac3e6  Close Phase 4 and judge the Phase 5 gate: sweep, plots, threshold sweep, leakage test  <- HEAD  (session 6)
18d18c7  Delete the last fault-attribution code; correct stale docs; handoff                    (session 5)
2db2b9f  Close Phase 0: answer U6 and U-B7, record the legacy Colab run                          (session 5)
03ad988  Read an in-flight sweep, and stop it scoring one class first                            (session 5)
cc13966  Track progress.md — the session handoff state                                           (session 4)
4204bf2  Add PR curves and reliability diagrams (Phase 4 task 4)                                 (session 4)
e488a05  Make the sweep resumable: append each clip to scores.jsonl as it lands                  (session 4)
869934c  Record T3 in RESULTS.md, restate the Phase 5 gate, ignore the BADAS weights
bbea10f  Add BADAS probes and the safe.mp4 score-regression guard
be82313  Add eval/ harness: three models through one code path (Phase 4 AC #1)
836c856  Vendor BADAS-Open source (Apache-2.0) with one transformers 5.x patch
420ef17  T3 corpus control: the shipped model performs at chance (AUC 0.5339)
08d9453  Relocate project out of iCloud-synced Desktop to ~/dev/crash_detection
7640950  Fix crash_detection_enhanced.py: remove dangling depth_est/bev refs
e58885d  Finish dead-code deletion in crash_detection_enhanced.py; remove dead forks
ad45389  Consolidate project into crash_detection_v2; reclaim 5.8 GB
c9a6fda  Audit rev2: recover training pipeline, run falsification tests
c995fd7  update readme
```

**Be careful not to overwrite:**
- **`runs/baselines2/` — ~~A RUNNING PROCESS IS WRITING TO THESE~~ 🔴 SUPERSEDED 2026-09-16.** The
  sweep **finished** and the whole directory is **committed at `252d282`**. PID 75682 is gone. It is
  now static, like `runs/baselines/`. **Still do not edit, move, truncate or delete it** — it is ~20 h
  of MPS time and the 667 `.npz` traces are the input for `t_start`/`t_peak`/`t_end` (§13) — but the
  reason is value, not a live writer. **Do not relaunch `run_baselines.py --out runs/baselines2`;**
  it would append to a complete log.
- Anything in `runs/falsification/` — this is the project's evidence base, and README §17 and the
  user's standing instruction both forbid destroying it.
- `scripts/t124_model_falsification.py` and `scripts/t5_source_leakage.py` — refactoring these would
  break reproducibility of committed results. Duplication between them and
  `scripts/t3_corpus_control.py` (the `Head` class and NumPy `lstm`) is **intentional and accepted**.
- A previous parent repo had a **GitHub PAT in plaintext** in its remote URL. That repo is deleted and
  the user confirmed the token is revoked. The current remote is SSH. Do not reintroduce HTTPS-with-token.

---

## 9. DATASET / LICENSING STATE

No licence status changed this session. Current position, carried from README §23:

| Dataset / dep | Position | Usable commercially? |
|---|---|---|
| **CCD (Car Crash Dataset)** | Positives are YouTube-derived (per-clip `youtubeID`); negatives are sampled from **BDD100K**, whose basic licence is limited to personal use. Repo labelled MIT but the README has no licence section. | **NO — 🔴 research only.** The current model is trained on this. |
| **BDD100K** (via CCD negatives) | Basic licence limited to personal use | **NO — needs verification with Berkeley DeepDrive** |
| **Nexar Collision Prediction** | Licence text at `data/nexar/LICENSE`. Grants "use, copy, modify, and distribute" free of charge. **No non-commercial restriction.** Conditions: attribution with the specified citation, retain the notice on redistribution, **No Resale** of the dataset itself, and ethical-use limits (no deepfakes, re-identification, weaponisation, "exploitative practices … such as unethical insurance practices"). | **YES for training**, per a reading of the text. Counsel should confirm that No-Resale restricts redistributing the dataset, not derived models. |
| **`ultralytics` (YOLOv8)** | **AGPL-3.0** (bug C6). Its network clause obliges a hosted service to offer complete corresponding source. | **NO.** Removed from `requirements.txt` this session. |
| **BADAS-Open** | Apache-2.0, commercial use with attribution; disclaimed for safety-critical use | YES (unverified first-hand — see §10) |
| **V-JEPA 2** | MIT (majority) | YES (unverified first-hand) |

**Unresolved licensing actions (nobody has done these):**
- **U7** — email the CCD authors (Wentao Bao, RIT) for the operative scope of the MIT label. **NOT SENT.**
- Verify BDD100K's current terms directly with Berkeley DeepDrive. **NOT DONE.**
- **C6 decision** — migrate to RT-DETR from `lyuwenyu/RT-DETR` (Apache-2.0) or buy the Ultralytics
  Enterprise Licence. **NOT DECIDED.** ⚠️ Trap recorded in README §15: do **not** take RT-DETR via the
  `ultralytics` package — its RT-DETR wrapper is also AGPL-3.0 and reintroduces the obligation.
- **L4** — no `LICENSE` file exists in the repository. README has claimed MIT.

**Do not make legal claims beyond the above.** These are engineering readings, not legal advice.

---

## 10. BLOCKERS AND UNKNOWNS

### BLOCKED

- **Fine-tuning the V-JEPA2 ViT-L backbone** — no CUDA device exists (M1 + M4 Max are both Apple
  Silicon). **Probe/head training on cached frozen features is NOT blocked** — see the revised D2.
  Resolution if full fine-tuning is ever needed: ~$200–500/month of spot cloud GPU.
- **Running `code/crash_detection_enhanced.py` end-to-end** — needs `torch` + `ultralytics`, and
  `ultralytics` is AGPL. Resolution: install both temporarily *only* to verify, and say plainly that
  it was a temporary verification install; or accept that this pipeline is being retired and never run it.

### BADAS-Open unknowns — PARTIALLY RESOLVED 2026-09-11

Research was started at the end of the session and **interrupted part-way**. Source: the
BADAS-Open model card, `https://huggingface.co/nexar-ai/BADAS-Open`.

- **B-GATE — PARTIALLY BYPASSED (session 2). Code obtained; WEIGHTS still gated.**
  The same source is published **ungated on GitHub at `getnexar/BADAS-Open`** (Apache-2.0).
  Cloned and vendored to **`vendor/badas-open/`** (228 KB, `.git` and `assets/` stripped).
  **U-B3 and U-B4 are now RESOLVED from that source without HF access** — see below.
  **Still blocked:** `weights/badas_open.pth`, which `badas_loader.py` fetches via
  `hf_hub_download(repo_id="nexar-ai/badas-open", filename="weights/badas_open.pth")`.
  **No inference is possible until the user completes the HF gate.** The user has no company email;
  `gated:"auto"` means no human reviews the form, and "Independent" + a personal address is a truthful
  answer for an independent researcher.

- **B-GATE detail. `nexar-ai/BADAS-Open` is a GATED repository.**
  `hf download` returns **"Access denied. This repository requires approval."** and every file —
  including `README.md` — returns **HTTP 401** unauthenticated. The HF API reports
  `gated: "auto"` with `extra_gated_fields` = First Name, Last Name, Title, **Company Email**,
  **Company**, Topic of Interest.
  - `gated: "auto"` means approval is **automatic on submitting the form**, not a human review — so
    this should unblock in minutes, not days.
  - **Claude cannot do this step.** It requires the user's own identity and a form submission under
    their HF account.
  - **Resolution (user action):** log in at `https://huggingface.co/nexar-ai/BADAS-Open`, accept the
    terms, then create a read token at `https://huggingface.co/settings/tokens` and either run
    `~/envs/badas/bin/hf auth login` or export `HF_TOKEN`.
  - **Everything in U-B3, U-B4 and the U-B1 correction below is blocked on this.**
  - Env `~/envs/badas` (Python 3.11 + `huggingface_hub`) **already exists** — do not rebuild it.
    Note the CLI is `hf`, not the deprecated `huggingface-cli`.

- **U-B1 — RESOLVED, BUT THE PRIOR SESSION'S FILE LIST WAS WRONG.** It does ship runnable code, not
  just weights. **Actual** repo contents, read from the HF API `siblings` field (which is public even
  though the files are gated):
  ```
  README.md  config.json  requirements.txt  badas_loader.py
  src/core/base.py  src/models/vjepa.py  src/train/video_training.py
  src/utils/sliding_window.py  src/utils/video.py
  weights/badas_open.pth
  assets/{arch.png, example.png, performance.png, BADAS_Demo.gif, temp.txt}
  ```
  **There is no `preprocessing.py` and no `model.safetensors`** — the prior session's §10 entry named
  both. Weights are a **`.pth`**, and there is a **`src/` tree** it did not know about.
  **The U-B3 answer therefore lives in `src/utils/video.py` and `src/utils/sliding_window.py`**, not
  in a `preprocessing.py`. `src/utils/sliding_window.py` in particular suggests the clip is scored by
  a sliding window rather than one 16-frame draw — which, if true, is a different inference contract
  than assumed and directly shapes the Phase 4 adapter.
  **The forward pass still does not need reimplementing by hand** — that part of U-B1 stands.

- **U-B2 — RESOLVED, AND IT CHANGES THE PHASE 5 GATE.** The published Nexar figures are measured on
  a test set of **1,344 clips**. **We hold only test-public: 667 clips.** So the headline
  **AP 0.86 / AUC 0.88 / mTTA 4.9 s is on the full public+private test set**, and a run against our
  667 clips **cannot be expected to match it exactly**.
  **Action required before applying the gate:** either (a) restate the gate as "within ~0.02 AP of a
  *test-public subset* figure" and accept it is not a like-for-like comparison, or (b) download
  test-private (677 clips, 2.97 GB — modest) to reconstruct the full 1,344-clip set and make the
  comparison genuinely like-for-like. **(b) is strongly preferred** — it is under 3 GB and it is the
  difference between a real reproduction and an approximate one, which is the entire point of the gate.
  Other published figures for reference: DoTA AP 0.94 (n=367), DADA-2000 AP 0.87 (n=113), DAD AP 0.66 (n=116).

  **SESSION 6 UPDATE — (b) was NOT done. The gate was judged on the 667-clip subset anyway**, per
  README's session-3 restated gate (a judgement call on named deviations, not a hard ±0.02 stop).
  Measured AP 0.8349 vs config.json's 83.2 (within 0.003) on a smaller N than the published figure's
  1,344. `data/nexar/test-private/*/metadata.csv` exists (metadata only, no video downloaded) —
  downloading those ~677 clips and re-running would be the genuinely like-for-like comparison this
  note originally asked for. **Still open, not done, not blocking** — Phase 5's gate already passed
  on the evidence available; doing (b) would only tighten it, not reverse it.

- **U-B3 — RESOLVED (session 2), read from `vendor/badas-open/`. Sliding window, NOT a single draw.**

  The exact contract, from `badas/badas_loader.py:41-50` (`load_badas_model`'s own defaults):

  | Parameter | Value | Note |
  |---|---|---|
  | `model_name` | `facebook/vjepa2-vitl-fpc16-256-ssv2` | backbone, fetched from HF (ungated) |
  | `frame_count` | **16** | window size |
  | `img_size` | **224** | ⚠️ **NOT 256.** Prior session recorded "16 frames @ 256×256" — wrong. The `-256` in the backbone name is its pretraining crop, not this model's input. |
  | `target_fps` | **8.0** | clip is resampled to 8 fps first |
  | `window_stride` | **1** | stride of 1 frame **at 8 fps** = 0.125 s |
  | `use_sliding_window` | **True** | |
  | temperature | **2.0** | `apply_temperature_scaling(logits, 2.0)` before softmax — an explicit calibration step, applied at inference (`models/vjepa.py:195,259`) |

  - **Frame selection is clip-relative.** `load_full_video_frames()` resamples the whole clip to 8 fps
    and 224×224, then `create_windows()` emits every contiguous 16-frame window at stride 1
    (`utils/sliding_window.py:53-71`). Confirms §6.7's elimination: nothing is anchored to
    `time_of_alert`.
  - **Semantics: each window predicts what happens AFTER its last frame** (`sliding_window.py:19-24`).
    The first 16 frames get **NaN**. This is a *prediction* model, matching §6.7's finding that the
    clips are truncated before the event.
  - **Resize is a plain `cv2.resize` to 224×224 — no aspect-preserving crop** (`utils/video.py:556`),
    so 16:9 dashcam footage is squashed. Deliberate, matches training.
  - **Normalization:** the preferred path is `AutoVideoProcessor.from_pretrained(model_name)`, whose
    constants come from the HF backbone, **not** from this repo. The manual fallback uses ImageNet
    mean `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]` (`utils/video.py:249`). **These two paths
    may not agree — verify which one actually runs**, because a silent fallback changes the numbers.
  - ⚠️ **`albumentations` is imported by `utils/video.py:11` but is MISSING from `requirements.txt`.**
    Install it explicitly or the import fails.

- **U-B5 — NEW UNKNOWN (session 2). How does a per-frame curve become one clip score?**
  This directly determines AP, and the code is **internally inconsistent**:
  - `VJEPAModel.predict()` returns `results["per_frame"]` — a per-frame array (`models/vjepa.py:274`).
    It **discards** `results["per_video"]`.
  - `SlidingWindowPredictor` computes `per_video = np.mean(window_predictions)`
    (`utils/sliding_window.py:150`) — **mean**.
  - But both user-facing entry points use **max**: `cli.py:120` `max(predictions)` and
    `examples/basic_inference.py:103` `max_risk = max(predictions)`.
  - **HYPOTHESIS (not verified): the published AP uses max.** It is the user-facing convention and it
    matches what T3 already does (max over windows), which keeps the two rows comparable.
  - ⚠️ **Latent bug to avoid inheriting:** the first 16 frames are NaN, and builtin `max()` over an
    array containing NaN is order-dependent and can return NaN. **Use `np.nanmax`.**
  - **Resolve empirically once weights are available:** score the clips once, then compute AP under
    both max and mean and report which reproduces the published figure. Cheap — one inference pass,
    two reductions.

- **U-B2 — REOPENED (session 2). The two published number sets disagree.**
  `vendor/badas-open/badas/config.json` reports **Nexar AP 83.2 / AUC 0.85**. The prior session read
  **AP 0.86 / AUC 0.88 / mTTA 4.9 s** off the HF model card. Other benchmarks disagree too:

  | Benchmark | config.json | prior session's model-card reading |
  |---|---|---|
  | Nexar | AP **83.2**, AUC 0.85 | AP 0.86, AUC 0.88 |
  | DoTA | AP **95.9**, AUC 0.79 | AP 0.94 |
  | DADA | AP **92.9**, AUC 0.88 | AP 0.87 |
  | DAD | AP **60.9**, AUC 0.96 | AP 0.66 |

  **UNKNOWN which is authoritative.** Possibly different checkpoints, different splits, or a
  transcription error in one of them. **Decide the target number before running the gate**, and state
  which source it came from. Combined with §6.7 (mTTA unreproducible) and U-B2's 667-vs-1,344 split
  gap, **the Phase 5 gate needs restating on AP alone against one named source.**

- **U-B4 — RESOLVED (session 2). MPS is usable, but only if passed explicitly. D2 SURVIVES.**
  - Auto-detect is **CUDA-or-CPU only**: `get_device()` returns
    `torch.device("cuda" if torch.cuda.is_available() else "cpu")` (`utils/video.py:188-190`), and
    `badas_loader.py:29,75` repeat the same expression. **No MPS branch anywhere** — so on the M4 Max
    the default path silently runs on **CPU**.
  - **But `device` is a parameter throughout.** `VJEPAModel.__init__` does
    `self.device = torch.device(device) if device else get_device()` (`models/vjepa.py:43`), and
    `load_badas_model(device=...)` / `BADASModel(device=...)` both forward it. **Passing
    `device="mps"` works** — nothing is hard-wired to `.cuda()`, and there is no `flash_attn`,
    `autocast`, `device_map` or `bfloat16` anywhere in the inference path.
  - **Consequence for D2: the frozen-backbone/probe plan survives.** Still **UNVERIFIED empirically** —
    whether every V-JEPA2 op has an MPS kernel is a separate question. Test with
    `PYTORCH_ENABLE_MPS_FALLBACK=1` and watch for CPU fallbacks.

- **U-B6 — NEW: Phase 5 compute cost is much larger than assumed (session 2).**
  Stride 1 at 8 fps means a ~9.93 s clip → ~79 resampled frames → **~64 windows**, each a **full
  V-JEPA2 ViT-L forward pass** on 16×224×224. Over 667 clips that is **~42,700 ViT-L forward passes**
  for one evaluation run — not the ~667 the prior session's planning implicitly assumed.
  - **This is the main feasibility risk for Phase 5 on Apple Silicon.** Time one forward pass on the
    M4 Max **before** committing to a full sweep, and extrapolate.
  - Raising `window_stride` cuts cost linearly but **deviates from the published configuration** and
    would invalidate a like-for-like reproduction. If you do it, say so explicitly and report both.
  - Feature caching (D2) does not help here: the sliding window re-encodes overlapping frames, so
    there is no per-clip feature to cache without changing the contract.

**Architecture — CORRECTED (session 2), read from `vendor/badas-open/badas/utils/video.py:117-180`.**
The prior session's description ("attentive probe with **12 learned queries**", "2048 patches × 1024
dim, reduced to 12 queries × 64 dim") is **not what the code implements.** Actual:

```
VJEPA2Model (ViT-L, hidden 1024)              # backbone, from facebook/vjepa2-vitl-fpc16-256-ssv2
  -> last_hidden_state (B, N, 1024)           # patch tokens
_TemporalAttentionProcessor                    # NOT learned queries:
  nn.MultiheadAttention(1024, num_heads=8)     #   plain self-attention over the token sequence
  nn.LayerNorm(1024)
  .mean(dim=1)                                 #   then mean-pool -> (B, 1024)
_MLPClassifierHead                             # head_num_layers=3, head_hidden_dim=768, dropout 0.1
  Linear(1024,768) GELU LayerNorm Dropout
  Linear(768,768)  GELU LayerNorm Dropout
  Linear(768, 2)                               # num_classes=2
-> logits / 2.0 -> softmax -> P(class 1)
```

**There are no learned queries and no 64-dim bottleneck.** The head is ~2.0M params
(1024·768 + 768·768 + 768·2, plus attention 4·1024² ≈ 4.2M) — small, consistent with D2's claim that
probe training is tractable, but the **disk estimate below was derived from the wrong architecture
and should be recomputed** if feature caching is revisited. See also U-B6: the sliding-window contract
makes per-clip feature caching ill-defined anyway.

**Revised feature-cache disk estimate** (supersedes the rough ~16 MB/clip guess): caching full patch
tokens costs 2048 × 1024 × 4 bytes ≈ **8.4 MB/clip in fp32**, ≈ **4.2 MB/clip in fp16**.
For the 1,500-clip train split: **≈12.6 GB fp32 / ≈6.3 GB fp16.** Alongside the 25.5 GB of raw video
that is ~38 GB in fp32 — at the very edge of the ~40 GB free on the college machine.
**Therefore: cache in fp16, and/or delete the raw mp4s once features are cached.** Verify free disk
on that machine before starting the download.
- **U-N1** — README §49 describes Nexar clips as "~40 s"; measured mean for test-public is **~9.8 s**
  (§6.4). Determine whether the train split differs or the README is imprecise. Affects any
  duration-based metric (FP/hour, mTTA).

### SESSION 5 STATE — blocked vs merely unfinished (read this before the older lists)

**NOTHING IS BLOCKED.** Distinguishing carefully, because the older subsections below predate this:

| Item | Class | Note |
|---|---|---|
| BADAS full-run AP | **UNFINISHED, not blocked** | 230/667 at handoff. Just needs wall-clock. Partial AP 0.9274 (§6.17) |
| U-B2 — which published BADAS figure is authoritative | **UNKNOWN, open** | Model card AP 0.86 / AUC 0.88 vs vendored `config.json` Nexar AP 83.2 / AUC 0.85. Must be named in `metrics.json` (Phase 5 task 4) |
| Phase 5 task 3 — consume `predictor_output` | **NOT STARTED** | Needs the GPU; cannot start until the sweep ends. Not blocked, queued |
| U1 / U2 — which run made the shipped weights | **UNRESOLVABLE** | Confirmed again by U6 (§6.19): ModelCheckpoint overwrote in place, no run ids. Two runs in evidence, neither attributable |
| U7 — CCD licence text · BDD100K terms | **WAITING ON EXTERNAL** | Emails never sent. Not blocking any code |
| mTTA / time-to-detection on Nexar test-public | **BLOCKED PERMANENTLY on this data** | `time_of_event` lies beyond the clip for all 334 positives (§6.7) |
| Tracks B and C | **NOT STARTED** | Zero dependencies, zero compute, zero code. README §40/§45 call them the critical path |
| U5 — how `feature_extractor_saved/` was produced | **UNKNOWN, low priority** | README says regenerate deterministically rather than preserve |

**Non-blocking environment note:** claude-mem's observer allowance is exhausted (since
2026-09-11T22:17Z). No session memories are being captured. **Do not restart the worker.**

---

### UNKNOWN (carried from README §18, unchanged this session)

- **U1/U2** — which training run produced the shipped weights, and its metrics. Unresolvable
  retrospectively (`ModelCheckpoint` used a fixed path). 37 of 38 weight arrays differ from the
  archived checkpoint; the first Dense kernel correlates at **r = 0.0073** — two independent runs.
- **U4** — what `crash_detection_model.h5` (296 MB) and `crash_detection_model/` (197 MB) are. They are
  gitignored; **their presence in the current working tree was not verified this session.**
- **U5** — how `models/feature_extractor_saved/` was produced. Regenerate deterministically.
- **U6** — whether `models/crash_model_cpu/` shares weights with the shipped `.h5`. **Now cheaply
  answerable** — a working TF 2.19 env exists; load both and compare `get_weights()`.
- **U7** — CCD's operative licence scope.

### NON-BLOCKING

- Spurious arm64 matmul warnings (§4.6) — cosmetic, diagnosed.
- No `LICENSE` file (L4); `.DS_Store` hygiene (L1); no CI.

### WAITING

Nothing. No external request has been sent, so nothing is pending a reply.

---

## 11. DECISIONS MADE

Do not reverse these without new evidence.

- **D1 — The old MobileNetV2+LSTM model is retired, with evidence.** Basis: T1, T2, T3 and T5 all
  failed (§6). T3 (AUC 0.5339 on a corpus-controlled benchmark) is the decisive measurement. README
  §28 required it be "measured honestly once before retiring" — that has now been done.
- **D2 — Frozen backbone, trainable probe. No cloud GPU. (REVISED 2026-09-11, same session.)**

  **Original decision:** no model would be trained at all, because compute was "local M1 only, no
  cloud spend."

  **Revision:** the user has access to an **M4 Max machine at college**, with permission to install
  software and ~40 GB free. This is Apple Silicon, so PyTorch MPS is available and reasonably fast.
  The decision splits in two:

  | Task | Feasible? | Why |
  |---|---|---|
  | BADAS-Open **inference** over 667+ clips | **Yes** | Forward passes only; MPS handles this |
  | **Caching** frozen V-JEPA2 features to disk | **Yes** | One-off cost, embarrassingly parallel per clip |
  | Training an **attentive probe + multi-head** on cached features | **Yes** | Small model. The old LSTM head was 578k params and trained in 5.5 min on a T4. This is that class of job. |
  | **Fine-tuning the V-JEPA2 ViT-L backbone** | **No** | Apple Silicon has **no CUDA**. V-JEPA2/BADAS training code near-certainly assumes CUDA; PyTorch MPS still has op gaps and silent CPU fallbacks. Do not plan around this working. |

  **This is exactly the pattern README §30 Step 2 prescribes first** — *"freeze the V-JEPA2 backbone
  initially; train only the probe and heads — the same frozen-backbone pattern your current pipeline
  already uses well"* — and it is also the one genuinely sound piece of engineering inherited from the
  old pipeline (README §6.3 credits precomputing frozen features to disk as "exactly right").

  **Consequences:**
  - README §41 Phase 6 is **partially revived**: the multi-head split (collision / near-miss /
    ego-involvement) and calibration are now reachable. Only the backbone-unfreezing step is not.
  - The **Nexar train split (1,500 clips, 25.5 GB) is now worth downloading** — you need it to train a
    probe against. It was previously pointless. Disk on the college machine (~40 GB) is sufficient but
    not generous: cache features, then consider deleting the raw mp4s.
  - Phase 6's acceptance criterion *"new model beats BADAS-Open zero-shot"* is still **unlikely** to be
    met by probe training alone, since BADAS-Open's own probe was trained on more data with more
    compute. The honest target remains: **match** Channel A, then beat it **as a system** via Channel B
    (detector + tracker + calibrated geometry + TTC) and a calibrated logistic fusion layer, all CPU-cheap.
  - **This reframing is recorded here and has NOT been written into README.md.** See §16.

  **Unverified assumption to check on first contact with the machine:** access pattern was not
  specified. Assume jobs may be interrupted — **make feature caching checkpoint per clip** so it is
  resumable, rather than one long unrecoverable run.
- **D3 — Five README tasks dropped as dead work, because T3 retired the model they serve.**
  Recorded in README §41 Phase 2 (see §16). They are:
  1. Porting the Colab notebook to `train/` as a maintained pipeline — never training this architecture again.
  2. Fixing **B1** (train/inference temporal stride mismatch) — correcting the stride of a chance-level model.
  3. Deriving **`CNN_THRESH`** from a PR curve — there is no operating point on a chance-level ranker.
  4. Building a source-grouped frozen **CCD** test split — existed only to fairly evaluate a
     CCD-trained model; T3 did that differently and better.
  5. **T4** (crash excision) — would confirm via a fourth route a conclusion already established three ways.
  Net saving: roughly 8–10 days of work.
- **D4 — `ultralytics` removed from the default install.** AGPL-3.0 (C6), and required only by the
  retired YOLO-gated path. Reinstate only with a resolved licence, and never via the `ultralytics`
  RT-DETR wrapper.
- **D5 — Evidence scripts are not refactored, and duplication between them is accepted.** Preserving
  byte-reproducibility of committed results outranks DRY here.
- **D6 — CCD is a research corpus only, never training data for anything customer-facing.** Licence
  chain (§9) plus the corpus confound.
- **D7 — Every measurement gets falsified before it is accepted.** Applied to T3 itself (§6.2). Adopt
  this for future results.
- **D8 — All 6 pending commits were pushed to GitHub**, with explicit user approval, including the
  negative results. README §30 argues that publishing your own falsified result is a credibility asset.

---

### Decisions made in SESSION 3 (D9–D14)

- **D9 — The BADAS predictor will NOT be reimplemented; its output will be consumed.** Because the
  weights are already loaded and the module already exists (§6.11). Reverses session 2's plan. Saves an
  unscoped rebuild and removes the main Phase 5 risk.
- **D10 — Timing is quoted end-to-end, not compute-only.** ~97 s/clip, ~18 h/sweep. The compute-only
  0.856 s/window understates real cost by ~1.8× because it bypasses video IO (§6.12).
- **D11 — The old model's adapter REPLAYS committed scores; it does not re-score.** `CachedScores`
  reads `runs/falsification/T3_corpus_control.json`. Re-deriving 527.6 s of committed evidence through
  a reimplementation risks changing a published result to gain nothing, and D5/§12 protect those
  scripts. "One code path" is satisfied where it matters: identical metrics, splits, denominators.
- **D12 — Clip reduction is `np.nanmax`, never builtin `max`.** Upstream's `max()` is a latent
  NaN bug on every clip (§6.14).
- **D13 — ONE canonical FP/hour denominator for all models**, via `benchmark.durations()`. Per-adapter
  durations would make rates silently non-comparable.
- **D14 — `skip_predictor=True` is a legitimate cheaper baseline**, because `last_hidden_state` is
  bit-identical and upstream discards the predictor output anyway. It must be reported as a named
  configuration, not silently substituted.

### Decisions made in SESSION 5 (D15–D18)

- **D15 — Clip visiting order is pos/neg INTERLEAVED, never `sorted(labels)`.** Because in Nexar
  test-public every positive id sorts below every negative, so a plain sort makes AP/AUC/FP-h
  undefined for the first half of any run (§6.16). This is a **visiting-order** decision only;
  metrics are order-independent, so no measurement changes. Do not "simplify" `balanced_ids` back to
  a sort — the docstring says why.
- **D16 — Stride 1 is settled; the stride question is closed.** Session 4 launched at stride 1 and
  ~230 clips are banked at that setting. `_resume` raises on an adapter-name mismatch by design, so
  switching to stride 2 would discard every banked clip. Stride 1 is also the faithful comparison.
  **Do not re-open this with the user.**
- **D17 — Partial sweep numbers are diagnostic, never results.** `eval/peek.py` exists so an
  in-flight run can be sanity-checked, and it prints `PARTIAL` for that reason. `benchmark.run()`
  deliberately writes `metrics.json` only on completion, so a partial file can never be mistaken for
  a result. Do not commit or quote a partial AP.
- **D18 — The README audit body is NOT rewritten to match the current tree; it gets a dated delta
  note instead.** The §2 inventory is explicitly labelled "verified at `b539d6e`", which makes it a
  dated record rather than an error. Rewriting 49 stale `enhanced.py:NNNN` references and the file
  inventory would destroy what an audit is for. The note carries the corrected pointers
  (`CNN_THRESH` → `:97`, decision gate → `:754`/`:849`) in one place. See §16.

### Decisions made in SESSION 6 (D19–D24)

- **D19 — `skip_predictor` is threaded as a model ATTRIBUTE, not a `forward()` keyword argument.**
  The sliding-window closure in `vjepa.py` calls `self.model(processed_frames)` positionally — a
  kwarg added to `forward()`'s signature could never be reached from there. Setting
  `self.model.skip_predictor` after `VJEPAModel.load()` is the only point in the call chain where the
  flag can land. A/B-verified: score is bit-identical (0.995916 both ways on one clip), ~14–25%
  faster. Do not "clean this up" into a `forward()` parameter — it would silently stop working.
- **D20 — Per-frame BADAS scores are now persisted (`.npz` per clip) alongside the reduced scalar,
  going forward.** The first 18h sweep kept only `np.nanmax` per clip and discarded ~300 per-frame
  values per clip for no reason — free data (~1.6 MB for all 667 clips) that is the only way to
  answer the mean-vs-max reduction ambiguity or extract `t_start`/`t_peak`/`t_end` without a re-run.
  Any future BADAS sweep should pass `--save-frames-dir`.
- **D21 — `threshold_sweep()` was added to `eval/benchmark.py` as a shared, generic function**, not a
  one-off script, because every model that goes through `run()` benefits from it and it costs nothing
  (reuses arrays already in memory). Regenerating all three `metrics.json` from the existing resume
  cache cost 0.0s — confirms this is additive instrumentation, not a re-measurement, and validates
  that the resume-cache mechanism (D-session-4-era) genuinely skips re-scoring.
- **D22 — The predictor-concat experiment (Phase 5 task 3, D9) was investigated but NOT run this
  session, and this is a decision, not a gap.** The 2-line concat change was verified feasible
  (exact tensor shapes traced, token-axis works, feature-axis would crash). But with the checkpoint's
  default context/target masks, V-JEPA2's predictor reconstructs the SAME tokens it was just given —
  `future_prediction_seconds: 1.0` needs training-time mask offsets not present anywhere in this repo
  (training code was never open-sourced). Spending a second 17.5h sweep to test a hypothesis the
  checkpoint's own mask config already refutes was judged not worth the compute. Documented as a
  named deviation in the `gate` block (`scripts/badas_gate_provenance.py`) instead of measured.
  **If training code or real mask offsets ever surface, this decision should be revisited** — until
  then, do not re-derive this reasoning from scratch, and do not spend a sweep on the naive concat
  expecting it to move AP; it won't (the concatenated half is one-second-old information dressed as
  new).
- **D23 — The ego-involved/non-ego breakdown is struck from Phase 4, not deferred silently.** Same
  treatment as mTTA (D-session-2-era): verified directly against `data/nexar/{test-public,train}/
  */metadata.csv` column headers (no ego field exists on either split), then the README was edited to
  say so explicitly rather than just skipping the task quietly. See §16.
- **D24 — `sweep.log` files are force-added past the blanket `*.log` `.gitignore` rule when they are
  evidence of a specific, otherwise-irreproducible run** (18h of GPU time, timing, warnings, the one
  resume event). Same treatment as `runs/falsification/`. Do not add a blanket exception to
  `.gitignore` for `*.log` — most logs really should stay ignored; force-add case by case.

### Session 7 decisions (2026-09-14)

- **D25 — One plan document, not two.** The two parallel red-team revisions were merged into
  `NEW_PLAN.md`; the other was archived rather than deleted. **Why:** two competing documents with
  misleading names (`NEW_PLAN_v2.md` was older than `NEW_PLAN.md`) is a guaranteed source of a future
  session acting on the wrong one. Nothing was lost — the archived file is intact and its one
  additive idea was carried across as R9.
- **D26 — Paired bootstrap is now the mandatory instrument for every model comparison.**
  **Why:** the unpaired 95% CI on AP is [0.7910, 0.8734], i.e. ±0.04. Unpaired testing cannot resolve
  any improvement this project is realistically going to produce. Several past claims are smaller
  than their own error bar. Do not reverse this by quoting a bare point estimate.
- **D27 — No performance claim without a measurement on the user's own hardware.** **Why:** every
  timing on record (0.856 s/window; ~97 s/clip) is MPS-only, some from synthetic tensors, some from a
  6-clip sample, and none was ever run on the M4 Air or M4 Max the product would ship against.
- **D28 — Calibration is demoted from headline result to credibility infrastructure.** **Why:**
  monotone calibration provably cannot change AP or AUC. It makes probabilities usable and the
  operating point principled; it is not a detection improvement and must not be presented as one.
  This corrects the framing in the earlier plan.

**Added session 8 (2026-09-16):**

- **D29 — R1 stays UNPROMOTED at 2 of 3 gates; `np.nanmax` remains the shipped reduction.**
  **Why:** gates 1 and 2 both passed convincingly (+0.0556 full-667; +0.0551 held-out median,
  positive in 100% of 1000 splits), but both live entirely inside Nexar, whose truncation is the very
  thing the finding exploits. `NEW_PLAN.md` §11 point 8 is explicit that success here may not transfer.
  Switching the reduction now would alter every downstream number before the one test that could
  still kill the finding has been run.
- **D30 — The held-out null control permutes labels per split, not traces.** **Why:** the
  noise-trace version was measured and is biased — one lucky `nanmax` draw (AP 0.5630 vs 0.5007
  chance) persists across all splits and drags the null to −0.0252. Label permutation preserves each
  reduction's distribution and puts the baseline on equal footing (+0.0025). Recorded because the
  broken version looks more obviously "null" and someone will be tempted to revert it.
- **D31 — `runs/baselines2/` is committed in full, including all 667 `.npz` traces and `sweep.log`.**
  **Why:** ~20 h of MPS time existed only as untracked files. 3.1 MB total, so the cost is trivial
  against the risk. `sweep.log` force-added past `*.log`, matching `runs/baselines/` (D24). The
  traces are also the input for `t_start`/`t_peak`/`t_end` (§13), so they are a product asset, not
  just an experiment artefact.

**Added session 9 (2026-09-16):**

- **D32 — Incident timing uses a TWO-STAGE threshold: a global derived gate, then per-clip
  half-prominence.** **Why:** "did it happen" and "when did it happen" are different questions. One
  threshold serving both collapses `t_start` and `t_end` to the same instant on any clip that only
  just clears the gate. The gate is read off the PR curve at a stated target recall (README §41
  Phase 2 task 5), never hard-coded (B5). User chose this over two simpler alternatives.
- **D33 — `eval/timing.py` reports BOTH reductions and promotes neither.** `confidence.shipped` =
  `nanmax` (what `adapters.py` actually does), `confidence.candidate_r1_last_window` = the R1
  candidate, labelled pending gate 3. **Why:** D29 forbids switching before gate 3, but discarding a
  value already computed from the same trace would make the eventual promotion a wider edit. This
  keeps it a one-line change.
- **D34 — Unproducible incident-record fields are OMITTED, not null-filled.** **Why:** a `null`
  implies a field that could be populated and merely isn't. `ego_involved` is not "missing", it is
  *not derivable from Nexar* (D23). Omission is the honest encoding.
- **D35 — 8 fps stays. The question is closed for the third time.** **Why:** upstream's own default;
  16 frames @ 8 fps = the 2.0 s window the head was trained on; changing it invalidates every
  committed number; a mismatch here would be this project's own B1 bug repeated. Other frame rates
  belong in `NEW_PLAN.md` R2 as an *ensemble*, never as a replacement. Full argument in §21.4 item 9.

**SESSION 10 decisions:**

- **D36 — "Real-time on Apple Silicon GPU" is now permitted language; "CPU-capable" is not.**
  **Why:** `NEW_PLAN.md` §7.3 banned both until measured. Measured (§21.5 item 4): MPS 0.871 s/window
  against a 1.00 s budget at a 1 Hz alert cadence (**keeps up**), CPU 1.917 s (**does not**), CPU
  2.20× slower. **The permitted sentence must carry "compute only"** — decode dominates end-to-end
  (~97 s/clip vs ~55 s compute) and is excluded from the measurement. `scripts/hw_bench.py`.
- **D37 — Beta remains the calibration map of record; isotonic is rejected despite a better ECE.**
  **Why:** Tier 1 shows isotonic reaching ECE 0.0560 but dropping AP to 0.8342, while beta reaches
  0.0809 with AP **bit-identical** at 0.8596. Isotonic is only weakly monotone, so it collapses ties
  and destroys ranking. Paying 0.025 AP for 0.025 ECE is a bad trade for a detection product.
- **D38 — The deployable calibration number is Tier 1's, not Tier 3's.** **Why:** Tier 3
  (cross-fitted, beta ECE 0.0503) uses all the data and produces **no shippable map**; §6 forbids
  reporting it as a deployed result. Tier 1 (ECE **0.0809** [.062,.127]) fits on a calibration half
  and reports on an evaluation half that had no part in the fit. **Quoting Tier 3 as a product
  figure overstates deployed calibration by ~60%.** Score source is now a `--source` parameter so
  R1 promotion means a re-run, not a rewrite.
- **D39 — R1's gate 3 splits into 3a (mechanism, DADA-2000) and 3b (AP, DAD's 466-clip test split),
  and R1 must survive both.** **Why:** the original AP design was thought uncomputable but is not
  (DAD has 301 negatives); and DAD is degenerate for the mechanism test (collision-time IQR 0.16 s,
  structurally frame 90 ≈ normalised 0.90, near the clip end like Nexar). §21.5 item 2. **This is a
  decision of record — the user signed it off** and it is committed in `NEW_PLAN.md` at `d0bc237`.

**NOT decisions of record:** `NEW_PLAN.md`'s hybrid keep/rebuild verdict and its ranked R1–R9 plan.
Those are **proposals** pending the user's explicit acceptance. **Exception, session 10:** R1's
gate-3 design was explicitly signed off and IS a decision of record (D39) — the rest of R1–R9 is not.
~~Also not decided: the §21.3 item 2 redesign of R1 gate 3.~~ **RESOLVED — see D39.**

---

## 12. THINGS THE NEXT CLAUDE MUST NOT DO

**Added after SESSION 10 — do NOT redo these, and do NOT undo them:**

- **🔴 Do not run gate 3 on DAD as the MECHANISM test.** Measured: DAD's `Time-of-collision` has
  IQR **0.16 s** (84% within 0.25 s of exactly 3.00 s) and is structurally frame 90 of 100 at 30 fps
  = normalised ~0.90, near the clip end like Nexar. There is nothing for the peak to track. **DADA-2000
  is the mechanism target** (IQR 4.63 s); DAD is the **AP** target (§21.5 item 2, `NEW_PLAN.md` R1).
- **Do not repeat the claim that the external sets have no negatives.** The vendored
  `*_concensus.csv` files are positives-only, but they are *timing* annotations. **DAD's test split
  is 466 clips: 165 pos + 301 neg.** Session 8's conclusion, and session 10's first plan change
  built on it, were both too strong and are corrected in `d0bc237`.
- **Do not re-measure CPU vs MPS.** Answered (D36): **MPS 0.871 s/window, CPU 1.917 s/window**, CPU
  2.20× slower, MPS keeps up with a 1 Hz cadence and CPU does not. `scripts/hw_bench.py` reproduces
  it. Do not write a second timing script — `scripts/badas_smoke.py` already takes `BADAS_DEVICE`.
- **Do not claim "CPU-capable" or bare "real-time".** Only *"real-time at a 1 Hz alert cadence on
  Apple Silicon GPU (MPS), compute only"* is earned. Decode dominates end-to-end and is excluded.
- **Do not quote ECE 0.0498 or 0.0503 as a deployed/product calibration figure.** Those are Tier 3
  (cross-fitted, no shippable map). **The deployable number is Tier 1 beta ECE 0.0809** [.062,.127].
  §6 forbids reporting Tier 3 as a deployed result.
- **Do not rewrite `eval/calibration.py`'s Tier 3 path.** Verified bit-identical to the pre-session
  version after the Tier 1/2 additions. Tiers 1 and 2 are new; Tier 3 is unchanged.
- **Do not build a second reliability plotter.** `eval/plots.py` already renders it from
  `--emit`-produced JSON: `~/envs/crashdet/bin/python eval/plots.py runs/calibration/*.json`.
- **Do not download DoTA.** ~55 GB against ~30 GB free on the Air. Also do not assume the Studio has
  room — its ~40 GB figure is from session 1 (2026-09-11) and is **stale and unverified**.

**Added after SESSION 9 — still current:**

- ~~**Do not discard `progress.md`** — it holds uncommitted handoffs.~~ **RESOLVED in session 10:**
  committed at `fbf127c` and pushed. The general warning still stands for any *new* uncommitted work.
- **Do not rebuild `eval/timing.py`.** It exists, it is 335 lines, its 7 self-checks pass, and it is
  **committed at `024bec9`**. Run `~/envs/badas/bin/python -m eval.timing --self-check` to confirm.
- **Do not "fix" `eval/reduction_study.load_traces` to keep the NaN offset.** It strips leading NaNs
  deliberately and correctly for ranking; `heldout_half.py` and `reduction_study.py` both depend on
  that. `eval/timing.py` has its own `load_traces_abs()` for the time base. §21.4 item 5.
- **Do not re-open the 8 fps / `target_fps` question. Answered three times now** (§6.18, §12, §21.4
  item 9). 8 fps is upstream's own default and is the rate the head was trained at. Other rates
  belong in `NEW_PLAN.md` R2 as an ensemble, never as a replacement. D35.
- **Do not re-derive the time base.** `t = index / target_fps`, fps read per-clip from the `.npz`.
  Confirmed three independent ways (§21.4 item 6), including a ±0.0623 s cross-check against
  `benchmark.durations()` on all 667 clips.
- **Do not switch `eval/adapters.py` to `last_window`.** D29 stands; gate 3 is still open. Session 9
  deliberately left line 129 as `np.nanmax`.
- **Do not quote 92.3 FP/hour without its 0.899 h denominator**, and do not expect calibration to
  improve it — a monotone map cannot change FP/hour at matched recall (D28).
- **Do not claim any timing accuracy, mTTA, or time-to-detection from Nexar test-public.** There is
  no usable ground-truth event time. §21.4 item 11.
- **Do not re-run the §3.2 peak-position measurement.** Reproduced at full n=667 this session:
  positives 0.975 / negatives 0.802, 73.7% vs 38.4% in the final 10% (§21.4 item 4).

- **Do not put progress updates, session history, or experiment results into `README.md`.** They go
  here. Modify the README only for a genuine plan-level change (and see §16).
- **Do not claim the old model works.** It is at chance on a corpus-controlled benchmark. Do not
  quote the 0.9977 val AUC as a performance figure without stating it is leakage- and
  selection-contaminated.
- **Do not retrain, tune, threshold-fit, or stride-fix the MobileNetV2+LSTM.** README §44 forbids it;
  D1/D3 close it. Its ceiling is architectural (GAP destroys spatial layout), not a tuning problem.
- **Do not train on CCD or BDD100K for anything customer-facing.** Licence chain is 🔴.
- **Do not re-download the Nexar test-public split** — it is complete (667/667, verified). Do not
  start a blind 31.4 GB full-dataset download either. The **train** split (1,500 clips, 25.5 GB) *is*
  now worth fetching for probe training (revised D2), but only at step 5 of §14, and fetch **only**
  that split — not test-private, which nothing needs.
- **Do not attempt to fine-tune the V-JEPA2 backbone.** No CUDA device exists. Probe/head training on
  cached frozen features is the supported path (D2).
- **Do not delete, overwrite, or "clean up" anything in `runs/falsification/`.**
- **Do not refactor `scripts/t124_model_falsification.py` or `scripts/t5_source_leakage.py`** (D5).
- **Do not report raw accuracy as a headline metric.** README §31 bans it. Report AP, ROC-AUC and
  **FP/hour with its denominator**.
- **Do not quote an FP/hour figure derived from Nexar test-public as though it described real driving** —
  there are only 0.90 hours of negatives. The old model's **361.4 FP/hour** (§6.8) is a ceiling
  artefact printed to prove the denominator is carried, not a rate.
- **Do not compute or claim mTTA / time-to-detection on Nexar test-public**, and do not compare against
  BADAS's published **mTTA 4.9 s** using it. The clips are truncated before the event and the offset is
  unrecoverable (§6.7). Restate the Phase 5 gate on **AP only**.
- **Do not look for `preprocessing.py` in BADAS-Open — it does not exist.** The prior session's file
  list was wrong; the real files are in `src/` (§10 U-B1).
- **Do not rebuild `~/envs/badas`** — it exists. And use `hf`, not the deprecated `huggingface-cli`.
- **Do not skip README Phase 5's reproduction gate** — but do restate it first if U-B2 shows the
  published number is on a different split.
- **Do not resurrect** `EgoZone`, `FaultDetector`, MiDaS/`depth_estimator.py`, `bev_renderer.py`, or
  any of the four deleted pipeline forks. Fault attribution in particular is a product-liability
  hazard (README C7).
- **Do not take RT-DETR via the `ultralytics` package** — that wrapper is AGPL-3.0.
- **Do not create Python environments or store datasets under `~/Desktop`** (iCloud eviction).
- **Do not push to the remote without asking the user.** Session 1's push was explicitly approved;
  that approval does not carry forward. **Session 3 committed nothing at all.**

**Added after SESSION 8 (2026-09-16) — do NOT redo these:**

- **Do not look for the background sweep. It finished.** PID 75682 is gone, 667/667 scored, all of
  `runs/baselines2/` is committed at `252d282`. **Do not relaunch `run_baselines.py --out
  runs/baselines2`** — it would append to a complete log. Session 7's "job still running" banner has
  been deleted from the top of this file; if you see that text anywhere else, it is stale.
- **Do not re-run R1 gate 1 or gate 2 to "check".** Both are committed with exact numbers (top banner)
  and both scripts assert their own preconditions. `python -m eval.reduction_study` takes ~43 s and
  `python -m eval.heldout_half` ~10 s if you genuinely need to reproduce them, but nothing depends on
  re-running them.
- **Do not switch `eval/adapters.py::BadasOpen.score` from `np.nanmax` to last-window.** R1 has passed
  2 of 3 gates; promotion needs gate 3. Changing the reduction now would silently alter every
  downstream number and pre-empt the one test that could still kill the finding.
- **Do not try to run R1 gate 3 as `NEW_PLAN.md` R1 words it.** The AP comparison is impossible —
  DAD/DADA/DoTA are positives-only (§21.3 item 2). Redesign first, with user sign-off.
- **Do not plan DoTA.** ~55 GB against 33 GB free on the Air and a stale ~40 GB on the Studio. **DAD
  (165 clips) is the first target and needs only the Air, ~4.5 h** (§21.3 items 4–5).
- **Do not "fix" `eval/heldout_half.py`'s null control back to random traces.** That version was
  measured and is biased (−0.0252 from one lucky baseline draw). Per-split label permutation is
  correct and gives +0.0025 (§21.3 item 8). **Do not loosen the assertion threshold to make a failing
  null pass** — the failure is the check working.
- **Do not run `python eval/calibration.py`** — it raises `ModuleNotFoundError`. Use
  **`python -m eval.calibration`**. Same for `reduction_study` and `heldout_half`: all need `-m`.
- **Do not quote `eval/calibration.py`'s ECE numbers as a deployed result.** It is `NEW_PLAN.md` §6
  **Tier 3 only** — "achievable calibration". Tier 1 is unwritten (§21.3 item 6).
- **Do not describe the product as "crash detection".** `README.md` §27: *"The product is not crash
  detection... The product is the structured incident record."* (§21.3 item 9).

**Added after SESSION 7 (2026-09-14) — do NOT redo these:**

- **Do not re-derive §21.1's measurements.** The bootstrap bar, the reduction comparison, the
  length-leakage check, the split-half replication and the calibration table are all recorded there
  with exact numbers. **Re-running the reduction comparison on the FULL 667 is required and is the
  next action — that is not a redo, it is the confirmation step.**
- **Do not go looking for the user's "10 points" red-team document.** It was never a file; it was an
  adversarial self-review produced in conversation in the parallel window. Its ten points are closed
  in `NEW_PLAN.md` §2. A previous session burned time searching for it and recorded it as missing.
- **Do not un-archive or resurrect `archive/NEW_PLAN_v2_parallel_session.md` as a competing plan.**
  It was merged into `NEW_PLAN.md`; its C2 became R9; C4/C6/C7 were rejected with written reasons.
  Two competing plan documents is the exact confusion that was just cleaned up.
- **Do not quote the ECE 0.3286 → 0.0498 calibration result externally until `eval/calibration.py`
  exists.** It is currently unreproducible. And **do not claim beta beats Platt** — 0.0026 apart,
  well inside the CI.
- **Do not claim "real-time" or "CPU-capable" anywhere.** Every timing on record is MPS, most from a
  6-clip sample, none measured on the user's actual M4 Air or M4 Max.
- **Do not plan to train a calibrator or probe on the Nexar train split.** It is BADAS-Open's own
  training data *and* it is not downloaded (metadata only, zero `.mp4`).
- **Do not treat the last-window reduction finding as settled.** n=268, non-random order, and it was
  selected by looking at test-public. See `NEW_PLAN.md` R1 for the three promotion conditions.

**Added after SESSION 6 — do NOT redo these, and do NOT undo them:**

- **🔴 Do not kill PID 75682 (the SECOND sweep, `runs/baselines2/`), and do not run any other model on
  MPS while it lives.** PID 88180 (referenced in the "SESSION 5" block below) is finished, committed,
  and gone — do not go looking for it. Check the live one with
  `eval/peek.py runs/baselines2/badas-open/scores.jsonl` (the bare `eval/peek.py` checks the OLD
  finished run and will misleadingly say `667/667`).
- **Do not re-litigate whether to run a second sweep at all.** The user was asked explicitly
  (`AskUserQuestion`) and chose the full instrumented re-sweep over "no more compute" and "cheap
  probe". Session 6 already had this conversation.
- **Do not re-run `scripts/badas_gate_provenance.py` expecting new numbers** — it is a documentation
  step over the already-frozen `runs/baselines/badas-open/metrics.json` from the FIRST (finished)
  sweep. It does not touch or need the second sweep at all. Re-running it is harmless (idempotent) but
  pointless unless the gate reasoning itself changes.
- **Do not redo the `skip_predictor` A/B test** — bit-identical, already proven (D19). Don't spend a
  sweep validating this again.
- **Do not attempt the "trained" predictor-concat (real future-prediction masks)** — D22. The mask
  offsets are not in this repo and are not derivable from the checkpoint. The naive concat (default
  masks) was verified to add no information; do not run it expecting a different AP.
- **Do not re-open the ego/non-ego breakdown as an oversight** — it's a documented, deliberate strike
  (D23), not something forgotten. Nexar's metadata genuinely has no ego field.
- **Do not re-verify `tests/test_leakage.py` "just to be sure" without a reason** — it passed, it's
  committed, it's exercised by re-running `python tests/test_leakage.py` if genuinely needed.

**Added after SESSION 5 — do NOT redo these, and do NOT undo them (mostly historical: PID 88180 is
now finished and committed, but the underlying decisions below still hold):**

- **🔴 [SUPERSEDED — PID 88180 finished and is committed at `e7ac3e6`.]** ~~Do not kill PID 88180, and
  do not run any other model on MPS while it lives.~~ It is the
  667-clip BADAS sweep, ~3 h in at handoff, ETA Sunday early afternoon. Check it with
  `eval/peek.py`, never by attaching to it.
- **Do not delete or rewrite `runs/baselines/badas-open/scores.jsonl`.** It is the resume log and it
  represents hours of GPU time.
- **Do not change `balanced_ids` back to `sorted(labels)`** (D15) and **do not switch stride** (D16).
- **Do not quote the partial AP 0.9274 / AUC 0.8988 as a result** (§6.17, D17). It is 140 positives
  against only 90 of 333 negatives and is biased upward.
- **Do not re-measure U6.** Answered: 12/12 arrays identical, `crash_model_cpu/` IS the shipped
  artefact (§6.19). **Do not conclude there are three runs** — there are two.
- **Do not re-open the `original_fps: 4` vs `target_fps: 8.0` question.** Resolved: they are the
  token rate and the video rate respectively, and they agree (§6.18). The sweep config is faithful.
- **Do not re-run `scripts/u6_compare_weights.py` or `scripts/badas_fps_probe.py` to "check"** —
  both assert their own conclusions and both passed.
- **Do not rewrite the README audit body to fix stale line numbers** (D18). The delta note in §2
  covers it.
- **Do not re-derive the ~18 h sweep estimate.** Measured: **~167 s/clip** over the first 140 clips,
  **~117 s/clip** over session 5's 90. Both are in `eval/run_baselines.py`'s docstring.
- **Do not restart the claude-mem worker.** Its provider allowance is exhausted (since
  2026-09-11T22:17Z) and restarting clears the backoff that stops it hammering the provider.

**Added after SESSION 3 — do NOT redo these:**

- **Do not reimplement the BADAS `predictor` module.** It exists, the weights load, nothing is lost
  (§6.11, D9). Session 2's §6.9 diagnosis was wrong. Consume `predictor_output` in `forward()` instead.
- **Do not re-patch `vendor/badas-open/badas/utils/video.py:175`** — already `pixel_values_videos`.
- **Do not re-measure U-B6.** Answered: 0.856 s/window compute, ~97 s/clip end-to-end, ~18 h sweep.
  **Do not repeat the ≈10 h figure** — it was compute-only and is wrong for planning (§6.12).
- **Do not believe "model load takes ~9 minutes."** It is **8.5 s**. The 9 min was a one-time download.
- **Do not rewrite `eval/benchmark.py`'s self-check or `durations()`** into per-adapter durations (D13).
- **Do not replace `np.nanmax` with `max`** anywhere in the scoring path (D12).
- **Do not re-score the retired model to "do it properly"** — `CachedScores` is deliberate (D11).
- **Do not quote the BADAS `--limit 6` numbers (AP 1.0000).** n=6. Plumbing proof only (§6.14).
- **Do not treat `scripts/badas_predictor_probe.py`'s "WITH REMAP → missing 0 unexpected 0" line as a
  needed fix.** The remap is unnecessary; the as-shipped load already works. The script says so.
- **Do not assume `code/crash_detection_enhanced.py` runs.** It needs `torch`, `ultralytics`, `scipy`,
  none installed. It has only ever been syntax-checked since the dead-code deletion.

---

## 13. EXACT NEXT ACTION  ·  **rewritten 2026-09-17, end of session 10**

### ══ THE ONE EXACT NEXT ACTION ══
###
### **Establish whether DADA-2000's Google Drive mirror allows per-clip download of the 221 test
### clips. This single fact decides whether gate 3a is a 6-hour overnight run or is blocked on disk.**
###
### ```bash
### df -h /                       # Air free space; ~30 GB as of session 8's check
### open https://github.com/JWFangit/LOTVS-DADA
### ```
###
### **Why this first:** gate 3 is the test that decides R1, the only measured detection improvement
### this project has (+0.0556 AP). It is blocked on **data acquisition, not code**. DADA-2000's
### primary distribution is a single **~53 GB** Baidu Pan archive (extraction code `9pab`) against
### **~30 GB free** — it does not fit. But **we need only 221 of its 2,000 clips**, and the repo
### mentions "DADA-2000 in the google drive for worldwide utilization". **If that mirror is a
### per-file folder, we download ~221 clips instead of 53 GB and the blocker evaporates.** If it is
### one archive too, gate 3a is genuinely disk-blocked and the user must choose: free disk on the
### Air, check the Studio's real free space (the ~40 GB figure is from session 1, 2026-09-11, and is
### **stale**), or attach an external drive.
###
### The clip ids to match are the 221 rows of
### `vendor/badas-open/annotation/dada2000_small_test_concensus.csv`.
###
### **Do NOT start any download without telling the user the size and the terms first.** DADA-2000's
### repo carries **no licence statement at all** (verified session 10) — §22/§23 licence discipline
### applies and this project has already excluded BDD100K on licence grounds.
###
### **In parallel, and needing the USER not Claude:** DAD (gate 3b, 466 clips incl. 301 negatives)
### is distributed by a **Google Form request to the authors** at
### http://aliensunmin.github.io/project/dashcam/ with **no publicly posted terms**. Claude cannot
### submit it. Ask the user to send the request, and record the terms in writing when they arrive.
###
### **Before gate 3a's correlation is believed, run the time-base check (§21.5 item 3):** decode one
### DADA clip, compare its true duration and fps against its `Time-of-collision`. DAD's own two
### readings disagree by 1.5 s, and a systematic offset would corrupt 3a silently.
###
### ---
###
### ### 13-S9. Session 9's next action — ✅ **DONE in session 10.** Kept for history.
###
### **Commit and push the two outstanding files. Ask the user first — they were asked twice in
### session 9 and said "wait" both times, so this is PENDING THEIR GO-AHEAD, not agreed.**
###
### **Status: COMPLETE.** The user approved in session 10. `fbf127c` (progress.md) and `024bec9`
### (eval/timing.py), both pushed. All three guards verified reproducing before committing. The
### original specification follows.
###
### ```bash
### cd /Users/khushpalsinghchouhan/dev/crash_detection/crash_detection_v2
### git status --short          # expect:  M progress.md   ?? eval/timing.py
### ```
###
### **Why this first:** `progress.md` holds session 8's AND session 9's entire handoffs and is
### uncommitted. One `git checkout` destroys both. `eval/timing.py` is 331 lines of working,
### self-checked deliverable that exists only in the working tree. Nothing else should be built on
### top of an unsaved base.
###
### **Verify before committing** — if any number moves, STOP and do not commit:
### ```bash
### ~/envs/badas/bin/python -m eval.timing --self-check   # 7/7, PASS
### ~/envs/badas/bin/python eval/benchmark.py             # T3: AUC 0.5339 / AP 0.5218 / 332,325,2,8
### ~/envs/badas/bin/python -m eval.reduction_study       # max 0.8349, last_window 0.8905 +0.0556
### ```
###
### **Two atomic commits, not one** — they are unrelated changes by different sessions, and merging
### them buries the fact that session 8's handoff was nearly lost:
### 1. `progress.md` — recovers session 8's stranded handoff plus session 9's.
### 2. `eval/timing.py` — session 9's deliverable. Message should carry the evidence in the style of
###    `252d282`: gate 0.9733 @ recall 0.80, 92.3 FP/h over 0.899 h, 351 records, §3.2 reproduced at
###    n=667, capability-not-metric, `adapters.py` untouched.
###
### Both messages end with:
### ```
### Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
### ```
###
### Then `git push origin main`. **Never force-push** — `252d282` is on the remote and the 667
### `.npz` traces are ~20 h of MPS time. If the push is rejected, STOP and report.
###
### **After the push:** `git log origin/main..HEAD --oneline` must be empty and `git status --short`
### must be clean.

---

## 13-S8. Session 8's next action (DONE in session 9 — kept for its specification)

**Status: COMPLETE.** `eval/timing.py` was built exactly as specified below. See §21.4.

## 13-S8-ORIGINAL. **written 2026-09-16, end of session 8**

### ══ THE ONE EXACT NEXT ACTION ══
###
### **Build `eval/timing.py`: turn the 667 committed per-frame traces into `t_start` / `t_peak` /
### `t_end` in seconds — the incident record's second field.**
###
### **Why this and not gate 3:** gate 3 is blocked three ways (its design is broken — §21.3 item 2;
### the videos are not downloaded; DAD needs ~4.5 h of compute). This needs **zero compute and zero
### new data**, the traces are already committed, and `README.md` §27 makes it a *product* deliverable
### rather than another benchmark number. `progress.md` §13 has listed it as step 2 since session 6
### and it was blocked on data that now exists.
###
### ```bash
### cd /Users/khushpalsinghchouhan/dev/crash_detection/crash_detection_v2
### ls runs/baselines2/badas-open/frames/*.npz | wc -l    # 667, committed at 252d282
### ```
###
### - **Reuse, do not re-implement:** `eval.reduction_study.load_traces` (strips leading NaNs),
###   `eval.benchmark.load_labels`. Each `.npz` carries `scores`, `target_fps`, `stride`,
###   `frame_count` — **use those to convert frame index → seconds; do not hard-code 8 fps.**
### - `t_peak` = argmax of the trace. `t_start`/`t_end` = first/last crossing of a threshold derived
###   from the trace (e.g. a fraction of peak height). **Do not hard-code 0.80** — B5/§12 forbid
###   hard-coded thresholds, and 0.80 is the retired model's number.
### - Emit JSON shaped like `README.md` §27's incident record.
### - Self-check asserting `t_start <= t_peak <= t_end` and all three inside the clip duration.
###
### **🔴 REPORT IT AS A CAPABILITY, NOT A METRIC.** Nexar has no usable ground-truth event time —
### `time_of_event` is corrupt for all 334 positives (§6.7, settled). There is nothing to validate
### timing against here, so **no accuracy claim of any kind.** §12 forbids mTTA/time-to-detection on
### test-public and that prohibition still stands; this is `README.md` §27's product field, not a
### resurrection of mTTA.
###
### **Still required before executing anything else from `NEW_PLAN.md`:** an explicit
### accept/reject/revise decision from the user. It remains a PROPOSAL. **And §21.3 item 2 is a
### genuine plan defect in R1 gate 3 that needs the user's sign-off to fix.**
###
### Everything below this line is session 6's still-valid guidance for the ORIGINAL master-plan track
### (Tracks B/C) and remains the fallback. **Note its "STEP 0 — is the sweep alive" block is now
### DEAD: the sweep finished, PID 75682 is gone, and `runs/baselines2/` is committed.**

### Phase 4 is COMPLETE and Phase 5's gate has PASSED. Nothing about the master plan is blocked. A
### second, non-required sweep is in flight. Session 5's §13 is superseded — it is below as §13-S5.

### STEP 0 — BEFORE ANYTHING ELSE, establish whether the SECOND sweep is still alive

**This is a different sweep from anything session 5 described.** PID 88180 is gone — it finished and
its output is committed. The live one now is PID 75682, in `runs/baselines2/`.

```bash
cd /Users/khushpalsinghchouhan/dev/crash_detection/crash_detection_v2
ps -p 75682 -o pid,etime,comm                                    # is it still running?
~/envs/crashdet/bin/python eval/peek.py runs/baselines2/badas-open/scores.jsonl   # how far has it got?
```

- **If `peek.py` says `667/667`** → the second sweep finished. Go to THE EXACT NEXT ACTION below.
- **If it says fewer and PID 75682 is alive** → still working. **Leave it alone.** Do CPU-only work
  meanwhile (Track B/C, below). Re-check periodically. Also check `ls runs/baselines2/badas-open/frames/
  | wc -l` to see how many `.npz` files have landed (should track the scores.jsonl count).
- **If it says fewer and PID 75682 is GONE** → it died. **Relaunch the identical command**; it will
  print `resuming: N/667` and continue from the log. Nothing is lost — per-frame `.npz` files already
  written stay on disk regardless.

  ```bash
  PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -i \
      ~/envs/badas/bin/python eval/run_baselines.py --out runs/baselines2 --skip-predictor \
      --save-frames-dir runs/baselines2/badas-open/frames
  ```

### THE EXACT NEXT ACTION — decide the reduction method, then commit the second sweep

Once `runs/baselines2/badas-open/metrics.json` exists and 667 `.npz` files sit in
`runs/baselines2/badas-open/frames/`:

1. **Load every clip's `.npz`** (`scores`, `target_fps`, `stride`, `frame_count` keys — see
   `eval/adapters.py::BadasOpen.score()` for the exact write format) and recompute AP/AUC/FP-hour
   using **mean** instead of **max** as the clip-level reduction (both are used inconsistently by
   upstream — see `gate.deviations` in `runs/baselines/badas-open/metrics.json` for the exact
   citation). Compare against the committed `nanmax` result (AP 0.8349 / AUC 0.8498).
   - If mean does meaningfully better on FP/hour at comparable recall → this is worth reporting as a
     finding and possibly changing the default reduction in `eval/adapters.py::BadasOpen.score()`.
   - If it doesn't → record that it was tried, keep `nanmax` (matches what `cli.py`/the example do,
     the more "faithful to what a user actually runs" choice per the existing gate deviation note).
2. **Extract a naive `t_start`/`t_peak`/`t_end` per positive clip** from the per-frame trace (e.g.
   first/last frame crossing some threshold, argmax for peak) using the `target_fps`/`stride` stored
   in each `.npz` to convert frame index → seconds. This is exploratory — there is no ground truth to
   validate against on Nexar (§10, mTTA is unrecoverable here), so treat it as a capability
   demonstration for the product (README §27's `t_start`/`t_peak`/`t_end` requirement), not a metric.
3. **Commit `runs/baselines2/`** once the sweep is finished and reviewed (same treatment as
   `runs/baselines/` — force-add `sweep.log` past the `*.log` gitignore rule if you want it kept,
   per D24).
4. **Do NOT treat this as reopening Phase 5's gate.** That already passed on the first sweep's
   evidence (§3, §17). This second sweep is groundwork for Phase 6 / the product's timing needs, not
   a re-judgment.

### Safe to do in parallel while the second sweep runs (CPU-only, no MPS)

- **Track B and Track C — README §40/§45 call these the real critical path, and they need none of
  the above.** Track B: commission the UK consent form, buy a dashcam, arrange a paid driver. Track C:
  30 UK fleet-operator calls, one question each. **Still at literally zero progress** across six
  sessions now — see §14.
- The two Phase 0 licence emails: the CCD authors (U7, Wentao Bao / RIT) and Berkeley DeepDrive.
- Downloading `data/nexar/test-private` video (see §10's SESSION 6 update on U-B2) if a tighter
  like-for-like Phase 5 comparison is wanted — optional, gate already passed without it.

---

## 13-S5. Session 5's next action (SUPERSEDED — kept for history)

### Nothing is blocked. A long job is in flight. Session 3's §13 is superseded — it is below as §13-S3.

### STEP 0 — BEFORE ANYTHING ELSE, establish whether the sweep is still alive

```bash
cd /Users/khushpalsinghchouhan/dev/crash_detection/crash_detection_v2
ps -p 88180 -o pid,etime,comm          # is it still running?
~/envs/crashdet/bin/python eval/peek.py   # how far has it got?
```

- **If `peek.py` says `667/667`** → the sweep finished. Go to THE EXACT NEXT ACTION below.
- **If it says fewer and PID 88180 is alive** → it is still working. **Leave it alone.** Do CPU-only
  work meanwhile (the "safe to do in parallel" list below). Re-check periodically.
- **If it says fewer and PID 88180 is GONE** → it died. **Relaunch the identical command**; it will
  print `resuming: N/667` and continue from the log. Nothing is lost.

  ```bash
  PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -i \
      ~/envs/badas/bin/python eval/run_baselines.py --out runs/baselines
  ```

### THE EXACT NEXT ACTION — close Phase 4, then judge Phase 5's gate

Once `runs/baselines/baseline_table.json` and the per-model `metrics.json` exist:

```bash
~/envs/crashdet/bin/python eval/plots.py runs/baselines/*/metrics.json --out runs/baselines/plots
```

**That single command closes README §41 Phase 4's last outstanding acceptance criterion** (task 4 —
reliability diagrams and PR curves, on shared axes, for all three models). `eval/plots.py` already
exists and is committed; it has **never been run against real data**, because until now no finished
`metrics.json` existed. Note it must run from `~/envs/crashdet` — matplotlib is not in `~/envs/badas`.

**Then judge the Phase 5 gate, using README §41 Phase 5's own ladder (NOT a ±0.02 numeric stop —
that hard gate was removed in session 3 because it is unpassable by construction):**

| Full-run BADAS AP | Verdict |
|---|---|
| **≈ 0.8+** | Harness trustworthy. Phase 5 passes. Proceed. |
| **≈ 0.6** | Stop. Either the harness is wrong or the published figures do not transfer. Investigate; do not build on it. |
| **≈ 0.5** | Something in our setup is broken. Fix before anything else. |

**Partial evidence at handoff (230/667) put AP at 0.9274 — so the ≈0.5 and ≈0.6 branches are already
effectively excluded. Expect the final AP to land lower than 0.9274** (the negative class is
under-represented in the partial), but the gate should pass.

**Record in `metrics.json`, per Phase 5 task 4 — this is an acceptance criterion, not optional:**
- **Which published figure is authoritative** (U-B2, still open): model card **AP 0.86 / AUC 0.88**
  vs the vendored `badas/config.json` **Nexar AP 83.2 / AUC 0.85**. Name one; record both.
- **Every deviation** from that published setup: split size **667 (public half) vs their 1,344**,
  **stride 1**, score reduction **`np.nanmax`**, predictor **present (faithful)**,
  `target_fps 8.0 / frame_count 16` (**faithful — confirmed §6.18**).

### Safe to do in parallel while the sweep runs (CPU-only, no MPS)

- Commit the session-4/5 work if the user agrees (§8) — **ask first, and do not push**.
- The two Phase 0 licence emails: the CCD authors (U7, Wentao Bao / RIT) and Berkeley DeepDrive.
- Track B and Track C (below) — zero code, zero compute, zero dependency on any of this.

**[Everything above this line ACTUALLY HAPPENED as described — the sweep finished, Phase 4 closed,
Phase 5's gate passed. This is preserved verbatim as session 5's plan, not rewritten with hindsight.]**

---

## 13-S3. Session 3's next action (SUPERSEDED — kept for history)

### Nothing is blocked. Everything needed is on disk. Session 2's §13 is superseded — ignore it.

### ⚠️ ONE USER DECISION IS PENDING BEFORE THE LONG RUN

The user was asked at the end of session 3 and **did not answer**: **stride 1 (~18 h) or stride 2
(~9 h)?** Stride 1 is the faithful comparison; stride 2 is legitimate only if the deviation is recorded
in `metrics.json` (README §41 Phase 5 task 2). Also unanswered: **how to commit** the large
uncommitted tree (§8). **Ask once, briefly, then proceed — do not stall on it.**

### THE EXACT NEXT ACTION — run the full BADAS sweep (Phase 4's last row / Phase 5's input)

```bash
cd /Users/khushpalsinghchouhan/dev/crash_detection/crash_detection_v2
PYTORCH_ENABLE_MPS_FALLBACK=1 ~/envs/badas/bin/python eval/run_baselines.py \
    --out runs/baselines            # add --stride 2 to halve the time, if the user chose that
```

- **Run it in the background** — ~18 h at stride 1. Do not hold the session on it.
- Writes `runs/baselines/baseline_table.json` plus a `metrics.json` per model.
- **Why this is the correct next action:** it is the only missing row of the baseline table, it is
  README §41 Phase 4's remaining substance and Phase 5's entire input, and the Phase 2 gate that
  guards it is already satisfied. Nothing downstream can be planned honestly without this number.
- **Interpreting the result (README §41 Phase 5 acceptance criteria, as revised):**
  - **AP ≈ 0.8+** → harness trustworthy, Phase 5 passes, proceed to Phase 6.
  - **AP ≈ 0.6** → stop. Either the harness is wrong or the published figures do not transfer.
    Investigate; do not build on it.
  - **AP ≈ 0.5** → something in our setup is broken. Fix before anything else.
  - The gate is **AP only**, against a **named** published source, judged against the deviation list —
    **not** a hard ±0.02 numeric stop. The old hard gate was removed this session because it is
    unpassable by construction (see the gate note in README §41 Phase 5).

### AFTER THAT, in order

1. **Finish Phase 4.** `pip install matplotlib` into `~/envs/crashdet` (absent from both envs), then
   write reliability diagrams + PR curves to `runs/<id>/plots/`. That plus the committed `metrics.json`
   closes Phase 4's remaining acceptance criteria.
2. **Phase 5 task 3 — consume `predictor_output`** in `EnhancedVideoClassifier.forward()` (D9).
   Evaluate **with and without**; report both. Do not guess the concat axis (§6.11).
3. **Settle `original_fps: 4` vs `target_fps: 8.0`** before trusting any published comparison (§6.11).
4. **Resolve U-B2** — name which published BADAS figure is authoritative (`config.json` AP 83.2 vs
   model card AP 0.86) and record it in `metrics.json`. Optionally download test-private (677 clips,
   ~3 GB) for a like-for-like 1,344-clip comparison.
5. **Resolve U6** — load `models/crash_model_cpu/` and `models/crash_model_weights.weights.h5`,
   compare `get_weights()`, write the answer down. One command; a Phase 0 acceptance item.

### Not blocked, zero code, and more important than all of the above for a startup outcome

- **Track B (README §41 Phase 7, now P0/week 1):** commission the consent form, buy a dashcam, arrange
  a paid UK driver. **Zero progress so far.** It is the moat (§40) and the only fix for the 0.90-hour
  negative-footage ceiling that caps every FP/hour claim.
- **Track C (30 UK fleet calls, README §42 item 18, now P0):** one question —
  *"What happened the last time you trialled an AI dashcam?"* **Do not pitch. Zero progress so far.**
  The user deferred this in session 2 ("skip that for now, work on the token"); it is **still open, not
  declined.** Raise it again.

**Environments (both exist — do NOT rebuild):** `~/envs/crashdet` (TF 2.19.1 / Keras 3.15.1, for the
old model, `eval/benchmark.py`, and the regression test) and `~/envs/badas` (torch 2.14.0 /
transformers 5.17.0 / **scikit-learn 1.9.1 added session 3**, for BADAS). Keep them separate.

---

## 13-OLD. Session 2's next action (SUPERSEDED — kept for history only)

### ✅ B-GATE IS CLEARED. Weights are on disk. Nothing is blocked on the user.

- HF auth done: `~/envs/badas/bin/hf auth whoami` → `user=khushpal`. Token at `~/.cache/huggingface/token`.
- **Weights: `models/badas/weights/badas_open.pth`, 3.7 GB.** Gitignored. Do NOT re-download.
- `~/envs/badas`: torch 2.14.0 · transformers 5.17.0 · albumentations 2.0.8 · cv2 5.0.0. MPS available.
- **U-B3 and U-B4 are RESOLVED** (§10). Do not re-derive them.
- Source vendored at `vendor/badas-open/` — ungated mirror of `github.com/getnexar/BADAS-Open`.

### THE ONE THING THAT MATTERS NOW

**§6.9: BADAS-Open's published inference code cannot reproduce its own model.** The checkpoint has a
199-tensor `predictor` module (future prediction, 1.0 s ahead, `concat`); `EnhancedVideoClassifier`
has no predictor and silently drops all 199 via `strict=False`. It runs, returns plausible numbers,
and is wrong. **Read §6.9 in full before touching Phase 5.**

### EXACT NEXT ACTION — patch the API drift, then time it (~20 min)

1. **Patch one line** in `vendor/badas-open/badas/utils/video.py:175`:
   `self.backbone(pixel_values=pixel_values)` → `pixel_values_videos=...`.
   transformers 5.17 renamed the argument; the code was written against 4.x.
   Mark it with a `# ponytail:` comment noting it is a vendored-upstream patch.
2. **Run `scripts/badas_smoke.py`** (already written) to get the U-B6 per-window time on MPS.
   It prints the projected full-sweep hours at stride 1, 2, 4 and 8.
   ```bash
   PYTORCH_ENABLE_MPS_FALLBACK=1 ~/envs/badas/bin/python scripts/badas_smoke.py
   ```
   ⚠️ Model load alone is **~9 minutes** — budget for it, and do not re-load per experiment.
3. **Decide on the U-B6 number, and only then plan Phase 5's scope.** If stride 1 is intractable on
   this hardware, a larger stride is legitimate **only if the deviation is reported**.

### AFTER THAT (do not start before step 3 gives a number)

4. **Reimplement the predictor + concat path** (§6.9). This is the real Phase 5 work. Verify against
   V-JEPA2's own predictor API; do not guess the concat axis (token axis is the hypothesis — feature
   axis is ruled out because `temporal_processor` takes 1024, not 2048).
5. **Write the Phase 4 adapter** on `eval/benchmark.py` (already built and passing, §6.8).
   Contract: video path in → per-frame array out → clip score via **`np.nanmax`** (NOT builtin `max`;
   the first 16 frames are NaN).
6. **Settle U-B5 empirically** — compute AP under both max and mean from one scoring pass.
7. **Settle U-B2** — (a) name which published figure is authoritative (`config.json` 83.2 vs model
   card 0.86); (b) decide on downloading Nexar test-private (677 clips, 2.97 GB) for a like-for-like
   1,344-clip comparison. **Recommended: yes on (b).** Gate is **AP only** (README §41 Phase 5, edited).
8. **Resolve `original_fps: 4` vs `target_fps: 8.0`** (§6.9) before trusting any number.

**Dependencies for step 1–3:** none new — everything needed is already on disk (weights, torch stack,
vendored source). ~20 minutes: 1-line patch, 9-minute model load, then the timed passes.
**Dependencies for step 7(b):** ~3 GB disk if test-private is downloaded.

### Not blocked — can proceed in parallel right now

- **UK fleet customer discovery** (README §42 item 18 / §14 action 4). Zero code, zero dependencies,
  and README §40/§45 both argue it is the actual critical path for a startup outcome. The user has
  explicitly deferred this in session 2 ("skip that for now, work on the token") — it is still open,
  not declined; raise it again once the BADAS investigation reaches a stopping point.

**Environments (both exist, do not rebuild):** `~/envs/crashdet` (TF 2.19.1/Keras 3.15.1, for the old
model and `eval/benchmark.py`) and `~/envs/badas` (torch 2.14.0/transformers 5.17.0, for BADAS-Open).
Keep them separate — do not add torch to `~/envs/crashdet` or TF to `~/envs/badas`.

---

## 14-S10. NEXT 3–5 ACTIONS  ·  **written end of SESSION 10. This supersedes §14-S9 below.**

After §13's DADA-mirror check:

1. **Resolve gate 3's data acquisition** — whichever route §13 establishes. If the Drive mirror is
   per-file, fetch the 221 DADA test clips and run the tail-scoring pass (~6 h, Air overnight). If
   not, put the disk decision to the user before anything else. Ask the user to submit DAD's Google
   Form in parallel — it is the only path to gate 3b and Claude cannot do it.
   → **Model: sonnet · Effort: medium** for the acquisition mechanics.
2. **🔴 Run gate 3a (mechanism) on DADA-2000, then gate 3b (AP) on DAD's 466 clips.**
   **This is the test that decides R1.** Run the §21.5 item 3 time-base check first. Reuse
   `eval/timing.py::load_traces_abs` (keeps the absolute NaN offset — the 2-second trap) and its
   `t_peak` argmax; do not rebuild them. Report the correlation with its CI **and the annotation's
   own IQR beside it**, so a reader can see whether there was variance to explain.
   → **🔴 Model: opus · Effort: high** — `NEW_PLAN.md` R1 marks this opus/high itself, and session
   10 just demonstrated why: the first two attempts at specifying this gate were both wrong.
3. **Decide R1 on gate 3's result.** If both pass, promote: `eval/adapters.py:129` `np.nanmax` →
   last-window, **then refit calibration on the new scores** (`--source last_window`, which is
   already supported) and re-run every committed metric. If either fails, record the kill and keep
   `nanmax`. → **Model: opus · Effort: high.**
4. **Settle the 0.0498 vs 0.0503 discrepancy** (§21.5 item 11). The committed script gives beta
   Tier 3 ECE 0.0503; `progress.md` §21.1 and `NEW_PLAN.md` §3.4 quote 0.0498 from session 7's
   in-conversation script. The committed script is reproducible and should win; correcting the two
   documents is then a small plan-document edit. → **Model: sonnet · Effort: low.**
5. **Track B and Track C.** Buy a dashcam; send 10 fleet messages. Both are **P0 in `README.md`
   §41**, both need no code and no compute, and both are at **zero across TEN sessions** while the
   engineering track keeps advancing. `NEW_PLAN.md` §9 sets a kill condition: **< 3 replies by end of
   week 2 → formally close Track C and stop listing it.** README §40/§45 both call Track B the moat.
   → **Model: haiku · Effort: low** (drafting only; sending is the user's).

**Also available, unblocked, if gate 3 stalls on data:** `NEW_PLAN.md` R4 (flip TTA, 100-clip pilot
then full tail pass) and R2 (multi-scale at 4/16 fps) are the next Week-2 items and need no new
datasets — only compute on clips already downloaded.

---

## 14-S9. NEXT 3–5 ACTIONS  ·  **written end of SESSION 9 (SUPERSEDED by §14-S10 above).**

After §13's commit + push:

1. **`scripts/hw_bench.py`** — time ONE window on M4 Air **CPU** vs **MPS** and report both.
   *Why this next:* small, zero new data, and it unblocks language the project is currently
   **forbidden** from using — `NEW_PLAN.md` §7.3 bans any "real-time" or "CPU-capable" claim until
   this measurement exists (D27). Week-1 item, still unwritten after nine sessions.
   → **Model: sonnet · Effort: medium.**
2. **Get user sign-off on the §21.3 item 2 gate-3 redesign**, then download **DAD only** (165 clips,
   ~4.5 h on the Air, **no Studio needed**) and run the peak-position test against the annotated
   `Time-of-collision`. **This is the test that decides R1.** Session 9's full-n reproduction of §3.2
   (§21.4 item 4) strengthens the mechanism but does **not** substitute for this — it is still inside
   Nexar. → **Model: opus · Effort: high** — the leakage reasoning is the whole ballgame.
3. **Calibration Tier 1** — the deployable stratified calibration/eval half split (`NEW_PLAN.md` §6).
   `eval/calibration.py` currently implements **Tier 3 only** (5-fold cross-fitted), which §6 says
   must be reported as *"achievable calibration"* and never as a deployed result. Note it reads
   `runs/baselines/` (nanmax scores); if R1 is ever promoted it must be refit on last-window.
   → **Model: sonnet · Effort: medium.**
4. **Decide R1** on gate 3's result. If it passes, promote: `eval/adapters.py:129` `np.nanmax` →
   last-window, then refit calibration and re-run every committed metric. If it fails, record the
   kill and keep `nanmax`. → **Model: opus · Effort: high.**
5. **Track B and Track C.** Buy a dashcam; send 10 fleet messages. Both are **P0 in `README.md`
   §41**, both need no code and no compute, and both are at **zero across nine sessions** while the
   engineering track keeps advancing. `NEW_PLAN.md` §9 sets a kill condition: **< 3 replies by end of
   week 2 → formally close Track C and stop listing it.** README §40/§45 both call Track B the moat.
   → **Model: haiku · Effort: low** (drafting only; sending is the user's).

---

## 14. NEXT 3–5 ACTIONS  ·  **updated end of SESSION 6, session 7 note added (SUPERSEDED by §14-S9)**

**SESSION 7 NOTE (corrected 2026-09-14):** the relevant proposal is **`NEW_PLAN.md`**, not
`NEW_PLAN_v2.md` (archived). **If and only if the user accepts it**, its §7.4 timeline and §10 ranking
supersede the ordering below for what it covers (reduction, calibration, ensembling, external corpora,
negatives) — but it does **not** cover Tracks B/C, which remain governed by `README.md` and item 2
below. If it is not accepted, the list below is unchanged and still current.

**The next 3–5 actions, if the proposal is accepted, are:**
1. Confirm or kill the last-window reduction on the full 667 (paired bootstrap) — §13.
2. Write `eval/calibration.py` so the ECE 0.33 → 0.05 result becomes reproducible. Zero compute.
3. Benchmark M4 Air CPU vs MPS throughput. **No "real-time" or "CPU-capable" claim may be made
   anywhere until this exists** — every timing on record is MPS-only and from a 6-clip sample.
4. Flip-TTA pilot on ~100 clips, then multi-temporal-scale ensembling (4/8/16 fps).
5. Acquire DAD (smallest external corpus, 165 annotated clips) and run the **falsification test** for
   the reduction finding — on untruncated external data, last-window should perform *worse* than max.

Straight from README §41's roadmap. No new roadmap is invented here.

1. **Check the second sweep (§13 STEP 0)** and, once it's done, decide the reduction method
   (mean vs `nanmax`) and extract naive `t_start`/`t_peak`/`t_end` from the per-frame `.npz` files.
   This is groundwork, not a gate — **Phase 4 is already complete and Phase 5's gate already passed**
   (session 6). Do not treat finishing this sweep as "still working on Phase 5."
2. **Track B + Track C, in parallel, still at ZERO progress across SIX sessions now.** README §41
   re-sequenced both to **P0/week 1** and §40/§45 call them the real critical path. Track B:
   commission the consent form, buy a dashcam, arrange a paid UK driver — it is the only fix for the
   **0.90-hour** negative-footage ceiling that caps every FP/hour claim this project can make. Track
   C: 30 UK fleet calls, one question — *"What happened the last time you trialled an AI dashcam?"*
   **Do not pitch.** The user deferred Track C in session 2 ("skip that for now"); it is **still
   open, not declined.** Raise it again — six sessions of pure model/eval work with zero customer
   contact is itself a signal worth naming to the user.
3. **Phase 6** — multi-head split (collision / near-miss / ego-involvement — note: ego-involvement
   has the SAME "no label in Nexar's metadata" problem session 6 found for the eval breakdown, D23 —
   this will need the UK data too, not just Nexar), temperature scaling on a dedicated calibration
   split (the committed BADAS ECE is **0.3286** — worse than session 5's partial 0.2794, calibration
   is genuinely uncalibrated, not just under-measured), Channel B revival for FP suppression. Per D2
   and README Phase 6: **the target is a system number, not a backbone number**, because no CUDA
   device exists.
4. **The two Phase 0 licence emails** (U7 / Wentao Bao / RIT, and Berkeley DeepDrive) — still not
   sent as of session 6, still external and non-blocking, but now the longest-standing open item.
5. **Optional tightening of Phase 5** (not required, gate already passed): download
   `data/nexar/test-private` video (~677 clips) to reconstruct the full 1,344-clip published test set
   and get a genuinely like-for-like AP comparison (§10, U-B2 session 6 update).

> **Sequencing note, unchanged and reinforced across three sessions now:** if forced to choose between
> engineering (1, 3, 5) and data/customers (2), README §40 and §45 both argue for **2**. Probe
> training will not beat BADAS-Open, which had more data and more compute. The value is the
> multi-head split and the UK data, not a better headline AP. **Session 6 spent its entire budget on
> engineering/eval (Phase 4/5) and the user's own choice to extend it further (the second sweep) —
> this is not wrong, the gate work genuinely needed doing, but Track C's zero-progress streak is now
> long enough that the next session should say so plainly before defaulting back into more model work.**

---

## 14-OLD. Session 2's action list (SUPERSEDED — kept for history)

1. **Build the evaluation harness — README §41 Phase 4. ~HALF DONE (session 2).**
   - **DONE:** `eval/benchmark.py` — loader (labels from `solution.csv`, per-condition fields from
     `metadata.csv`), metrics AP · ROC-AUC · precision@recall0.80 · ECE · **FP/hour with denominator**,
     per-condition breakdown, leakage assertion. Self-check reproduces **ROC-AUC 0.5339 / AP 0.5218**
     exactly (§6.8).
   - **STRUCK — do not attempt on test-public:** **mTTA and time-to-detection.** §6.7 proves
     `time_of_event` is outside the clip for all 334 positives and the offset is not recoverable.
   - **REMAINING, blocked on B-GATE:** the model-agnostic **adapter interface**. Its shape depends on
     BADAS's actual input contract — in particular whether scoring is one 16-frame draw or a sliding
     window (`src/utils/sliding_window.py`). **Do not guess it.**
   - **Still to add when the adapter lands:** the `videos/safe.mp4` → **0.7914** regression check.
2. **Reproduce BADAS-Open — README §41 Phase 5** (~2–3 days). Create `~/envs/badas` with PyTorch,
   evaluate through the harness, and compare against the published figure. **Gate: within ~0.02 AP,
   or stop and fix the harness** (restated first if U-B2 requires it).
3. **Publish the baseline table — README §41 Phase 5** (~0.5 day). One table, one code path, identical
   split: BADAS-Open zero-shot · MobileNetV2+LSTM (**already measured: AUC 0.5339 / AP 0.5218**) ·
   always-negative. Two of three rows exist. This is the project's first defensible claim.
4. **Start UK fleet customer discovery — README §42 item 18** (ongoing, £0, no code). 30 calls, one
   question: *"What happened the last time you trialled an AI dashcam?"* This is still **the actual
   critical path for a startup outcome**, not the engineering, and it runs fully in parallel with 1–3.
5. **Multi-head probe training on cached features — README §41 Phase 6, partial** (newly viable under
   the revised D2; do **not** start before step 2's gate passes). Order of operations:
   a. Download the Nexar **train** split (1,500 clips, 25.5 GB) to the M4 Max — now worth doing.
   b. Cache frozen V-JEPA2 features to disk, **checkpointing per clip** so the job is resumable.
   c. Train the attentive probe + **multi-head** (collision / near-miss / ego-involvement) on the
      cached features. Seed everything (README B9); checkpoint to `runs/<ts>-<sha>/` with
      `metrics.json` (README B8) — the fixed-path overwriting that made the old artefact untraceable
      must not recur.
   d. Temperature-scale on a **dedicated calibration split**, distinct from val and test; report ECE.
   **Do not attempt to unfreeze the backbone** — no CUDA (D2).
6. **UK data + UK-HN-500 — README §41 Phases 7–8.** 100 h of consented UK footage and 500 curated hard
   negatives. This is the moat (README §40) and the only fix for the 0.90-hour negative-footage
   ceiling. Gated on ~£500 and a UK driver arrangement, plus a consent/data agreement in place
   **before** the first recording.

> **Sequencing note:** if forced to choose between 5 and 6, README §40 and §45 both argue for **6**
> (data and the benchmark are the moat; the model is not). Probe training is unlikely to beat
> BADAS-Open, which was trained with more data and more compute. Step 5's real value is the
> **multi-head split** — collision vs near-miss is what the product needs and what Nexar collapses —
> not a better headline AP.

---

## 14B. GIT STATE AND FILE STATE AT END OF SESSION 3 (verified, not remembered)

**Branch `main`. HEAD `420ef17`. 0 unpushed commits. SESSION 3 COMMITTED NOTHING.**

```
 M .gitignore                      (session 2: + models/badas/)
 M README.md                       (session 3: +126/-36 lines — see §16)
 M runs/falsification/RESULTS.md   (session 3: +34 lines — T3 section corrected)
?? eval/                           (session 2: benchmark.py · session 3: adapters.py, run_baselines.py)
?? progress.md                     (this file — never committed, by design)
?? scripts/badas_predictor_probe.py  (session 3)
?? scripts/badas_smoke.py            (session 2, first run session 3)
?? tests/test_score_regression.py    (session 3)
?? vendor/                           (session 2, patched session 3)
```

⚠️ **MUST NOT BE LOST — none of it is in git:** the entire `eval/` harness (474 lines across three
files), the vendored BADAS source including the session-3 one-line patch, both new scripts, the new
test, and this file. **`models/badas/` (3.7 GB) is correctly gitignored — verified with
`git check-ignore -v` → `.gitignore:28`. Never commit it.**

**Key file purposes now:**

| Path | What it does |
|---|---|
| `eval/benchmark.py` (256 ln) | Metrics: AP, ROC-AUC, precision@recall0.80, ECE, FP/hour **with denominator**, per-condition breakdown, leakage assertion. Plus `durations()`/`hours()` (canonical denominator), `clip_paths()`, `run()`, `summarise()`, `metrics.json` writer. Self-check reproduces T3. **PASSES.** |
| `eval/adapters.py` (115 ln) | The model-agnostic interface. `.name` + `.score(path)`. `AlwaysNegative`, `CachedScores`, `BadasOpen`. |
| `eval/run_baselines.py` (103 ln) | Baseline-table CLI. `--limit`, `--no-badas`, `--stride`, `--skip-predictor`, `--out`. |
| `tests/test_weights_load.py` (112 ln) | R2's verification. 578,689 params. **PASSES.** |
| `tests/test_score_regression.py` (58 ln) | `safe.mp4` → **0.7914**. **PASSES.** |
| `scripts/badas_smoke.py` (81 ln) | U-B6 timing, compute only. |
| `scripts/badas_predictor_probe.py` (124 ln) | Settles the predictor question (§6.11). |
| `code/crash_detection_enhanced.py` (977 ln) | The original pipeline. **Cannot run** (missing torch/ultralytics/scipy). Still contains the dead `CNN_THRESH = 0.80` at `:97` and the decision at `:754`/`:849`. |

---

## 18. DOCUMENTATION CONTRADICTIONS — FOUND SESSION 3, **ITEMS 1–5 AND 7 FIXED IN SESSION 5**

> **Status at end of session 5:**
> - **Items 1–4 (README staleness): FIXED**, by a single dated delta note under the §2 inventory
>   rather than by rewriting the audit body (D18). The note carries the corrected pointers.
> - **Item 5 (`docs/HANDOFF.md` "T3 NOT YET RUN"): FIXED** — superseded banner at the top plus three
>   in-place corrections. This was the dangerous one; that file is committed.
> - **Item 6 (§6.9 is wrong): unchanged and correct as written** — §6.9 is retained as history and
>   §6.11 corrects it. Read §6.11 first.
> - **Item 7 (`at_fault` remnant): FIXED** — the unreachable block, the parameter, both call-site
>   arguments and both dead locals are deleted. `grep -rn "at_fault\|FaultDetector\|EgoZone" code/`
>   now returns nothing, satisfying Phase 3's own acceptance criterion.
>
> The original session-3 text follows, unaltered, as the record of what was found.

Verified against the filesystem. **These will mislead the next session if not handled.** The user was
offered the fixes at the end of session 3 and did not answer.

1. **README §2's repository inventory is stale.** It lists `camera_detect.py`,
   `code/crash_detection.py`, `code/crash_detection_linux.py`, `code/depth_estimator.py`,
   `code/bev_renderer.py` — **all five confirmed GONE.**
2. **README says `enhanced.py` is 1,334 lines.** It is **977**.
3. **README §2 says "Zero tests."** There are **2**, both passing.
4. **Every `enhanced.py:NNNN` line reference in README §15/§19/§21/§47 is wrong** — the file shrank.
   `CNN_THRESH` is `:97` not `:109`; the decision gate `:754`/`:849` not `:1050`.
5. **`docs/HANDOFF.md:182` still says "T3 — Corpus control: NOT YET RUN".** T3 is the project's
   headline result. **This file IS committed** and is the most dangerous of these.
6. **§6.9 of this file is wrong** — corrected by §6.11, but §6.9's text is retained as history. Read
   §6.11 first.
7. **Phase 3's own acceptance criterion is unmet**: `grep -rn "at_fault"` returns a hit at
   `code/crash_detection_enhanced.py:934-942`. Dead code (`fault_info` always `None` at `:898`).

**These are audit-body staleness, not plan defects** — which is why they were not auto-fixed. A single
mechanical pass would clear 1–5.

---

## 21.1 SESSION 7's MEASURED FINDINGS — added 2026-09-14, missing from the block below

**These were produced by the parallel session and are the most important evidence in this file after
T3. They are read-only measurements over already-committed scores — no model was run, the sweep was
never touched. All are reproducible from `runs/baselines/badas-open/scores.jsonl` and
`runs/baselines2/badas-open/frames/*.npz`.**

**1. The statistical bar. CONFIRMED.** Bootstrap on the committed 667-clip run:
AP 0.8349, 95% CI **[0.7910, 0.8734]** (width 0.082); AUC 0.8498, CI [0.8195, 0.8774].
**Consequence: any unpaired improvement below ~0.04 AP is invisible at n=667.** Every future model
comparison must use a *paired* bootstrap on the same clips. This bar did not exist before and several
past claims in this file are smaller than it.

**2. A reduction change worth ~+0.033 AP. PROVISIONAL (n=268 of 667).** Paired bootstrap vs `nanmax`:

| reduction | AP | ΔAP vs max | paired 95% CI |
|---|---|---|---|
| max (current) | 0.8809 | — | — |
| **last window** | 0.9140 | **+0.0330** | **[+0.0008, +0.0744]** excludes zero |
| **max×last (geometric)** | 0.9158 | **+0.0349** | **[+0.0027, +0.0729]** excludes zero |
| top-3 mean | 0.8585 | −0.0224 | [−0.0383, −0.0073] worse |
| p90 | 0.8154 | −0.0656 | [−0.1041, −0.0293] worse |

**3. The mechanism, CONFIRMED.** Position of each clip's score peak (0 = start, 1 = end):
**positives median 0.984**, negatives median 0.782. Peak in the final 10% of clip: **83.0% of
positives vs 42.5% of negatives.** Nexar test clips are truncated 500–1500 ms *before* the event, so
for a positive the last window is the aligned one; for a negative `max` wanders and grabs a spurious
mid-clip peak. **Last-window wins by suppressing false peaks on negatives, not by boosting positives.**

**4. Adversarial checks on finding 2.**
- **Clip length does NOT leak the label. CONFIRMED.** AP(length) = 0.5330, AUC 0.5483 — chance.
  Correlation of length with the last-window score is +0.109. Not a duration artefact.
- **Split-half replication: 5 of 6 halves positive, spread −0.009 to +0.075.** Direction consistent,
  **magnitude NOT established.** → stays PROVISIONAL until 667/667.
- **NEEDS VERIFICATION / known risk:** the reduction was selected by looking at test-public. Promotion
  requires (a) the full 667, (b) a held-out-half confirmation, and (c) the falsification test in
  `NEW_PLAN.md` R1 — *on untruncated external data (DoTA/DADA) last-window should perform WORSE than
  max.* If it helps there too, the mechanism is wrong and the Nexar gain is suspect.

**5. Calibration. CONFIRMED, but see the caveat.** 5-fold cross-fitted on the 667 committed scores:
beta calibration takes **ECE 0.3286 → 0.0498** (bootstrap CI [0.040, 0.087]), Brier 0.3069 → 0.1601,
NLL 1.1348 → 0.4903, **AP unchanged within 0.002**. Platt 0.0524, isotonic 0.0528 *but loses 0.02 AP*,
temperature only 0.2002. **Caveat the parallel session correctly raised: beta-vs-Platt (0.0026) is
well inside the CI and is NOT a meaningful difference — do not claim beta "wins".** Also **CONFIRMED:
`eval/calibration.py` does not exist**; these numbers came from an in-conversation script and must be
committed before being quoted anywhere.

**6. F3 (temporal smoothing hurts) — reconfirmed and extended, still PROVISIONAL.** mean 0.7066,
persistence k=4/8/16 all below max. Every averaging form is worse, for the reason in finding 3.

---

## 21.5 SESSION 10's FINDINGS — added 2026-09-17. Read after the top banner.

**1. Sessions 8 and 9's handoff is now COMMITTED AND PUSHED. CONFIRMED.** `fbf127c` (813 insertions
to `progress.md`) and `024bec9` (`eval/timing.py`, 335 lines). Session 8's `§17-S8` had claimed a
clean tree while its own handoff sat uncommitted; that claim was false for two sessions and one
`git checkout` would have destroyed both. **All three regression guards were re-run and reproduced
exactly BEFORE committing** — the stop condition ("if any number moves, do not commit") did not fire.
The user approved the commit and push explicitly.

**2. 🔴 R1 GATE 3's REDESIGN WAS ITSELF WRONG, TWICE, AND WAS CORRECTED THE SAME SESSION.
CONFIRMED — this is the most important finding of the session.**

Sequence: `c005757` committed session 8's proposed redesign on the user's sign-off. Then, checking
DAD's construction *before* downloading it, two errors surfaced, fixed in `d0bc237`.

**Error 1 — "the AP test cannot be computed" was too strong.** Session 8 verified the vendored
`vendor/badas-open/annotation/*_concensus.csv` files are positives-only. That is TRUE and was
re-verified independently in session 10:

| Set | n | Event-type |
|---|---|---|
| `dad_test_concensus.csv` | 165 | 151 Collision + 14 Near-collision |
| `dada2000_small_test_concensus.csv` | 221 | 198 + 23 |
| `dota_test_concensus.csv` | 598 | 562 + 36 |

**But those files are collision-TIMING annotations, and a negative clip has no collision to time.**
Their composition says nothing about the underlying datasets. From the authors' own project page
(http://aliensunmin.github.io/project/dashcam/): DAD is **1,750 clips — 620 positive / 1,130
negative**; train 1,284 = 455/829; **test 466 = 165 positive / 301 NEGATIVE.** Corroborated by BADAS
publishing **AP 0.66 / AUC 0.87 on DAD** — both metrics require two classes. **The AP comparison
`NEW_PLAN.md` R1 originally specified is computable after all**, from the full DAD download.

**Error 2 — DAD is the wrong dataset for the mechanism test regardless. CONFIRMED by measurement:**

| Set | n | `Time-of-collision` median | **IQR** | usable? |
|---|---|---|---|---|
| DAD | 165 | 2.96 s | **0.16 s** | ❌ 84% within 0.25 s of exactly 3.00 s |
| DADA-2000 | 221 | 5.33 s | **4.63 s** | ✅ range 0.37–14.43 s |
| DoTA | 598 | 4.65 s | 1.95 s | ✅ range 0.90–14.40 s, but disk-blocked |

The mechanism test asks whether the score peak **tracks** the annotated collision time. **Against a
near-constant there is nothing to track** — a null result would be uninterpretable, neither pass nor
fail. And the constant is **structural, not observed**: DAD's page states the accident occurs "at the
last 10 frames" of a 100-frame clip — frame 90, which at 30 fps is **exactly 3.00 s**, matching the
clustering. So DAD's collision sits at normalised **~0.90 — near the clip end, much like Nexar's
0.975.** DAD is not an untruncated mid-clip contrast at all, on either count.

**Resulting design, committed in `NEW_PLAN.md` R1's gate-3 block. R1 must survive BOTH:**

| Test | Dataset | Why | Cost (~97 s/clip, D10) |
|---|---|---|---|
| **3a — mechanism** (peak tracks annotation) | **DADA-2000**, 221 clips | only set with real variance | ~6.0 h, Air overnight |
| **3b — AP** (last-window should LOSE untruncated) | **DAD full test**, 466 clips | restores the original design | ~7.5 h, Air overnight |

**3. A time-base trap that would corrupt gate 3a silently. NEEDS VERIFICATION before 3a is run.**
Our trace timestamps are real seconds (`t = index / target_fps`, upstream resampling to 8 fps). The
consensus annotation's seconds are only the same seconds if the annotators used each clip's true
frame rate. **DAD's own two readings disagree by 1.5 s**: its page describes 100-frame clips as 5 s
(⇒ 20 fps, frame 90 = 4.50 s), while the annotation clusters at 3.00 s (⇒ 30 fps). **Before trusting
any offset on DADA, decode one clip and check its true duration and fps against its annotation.**
Recorded in the plan.

**4. CPU vs MPS MEASURED. CONFIRMED — `NEW_PLAN.md` §7.3's ban is now partially lifted (D36).**
`scripts/hw_bench.py`, M4 Air, one window (16 frames @ 8 fps = 2.0 s of video), budget 1.00 s per
window at a 1 Hz alert cadence:

```
mps   0.871 s/window   0.87x budget   KEEPS UP
cpu   1.917 s/window   1.92x budget   TOO SLOW
CPU is 2.20x slower than MPS
```

§7.3 predicted *"plausibly real-time on MPS, probably not on CPU"* — **confirmed.** MPS 0.871 s also
independently re-confirms **D10**'s 0.856 s/window. **Permitted now:** *"real-time at a 1 Hz alert
cadence on Apple Silicon GPU (MPS), compute only."* **Still forbidden:** "CPU-capable", and any
"real-time" claim that does not carry the compute-only caveat or names other hardware.

Two limitations recorded in the module rather than smoothed over:
- **COMPUTE ONLY.** End-to-end is ~97 s/clip against ~64 windows × 0.87 s ≈ 55 s of compute, so
  **decode already dominates** — `NEW_PLAN.md` §11 point 4's least-verified assumption, now with
  supporting arithmetic. A deployed system decodes a live stream, not a file, so its decode cost is
  not ours to extrapolate.
- **RUN-ORDER SPREAD.** Timing both devices in one process slows the second. CPU measured **1.680 s**
  in a fresh process and **1.917 s** after MPS; MPS **0.860 s** and **0.871 s**. Every CPU figure is
  above budget and every MPS figure below, so the verdict is stable across the spread.

**No new measurement code was written for this.** `scripts/badas_smoke.py` already timed one window
and already took a `BADAS_DEVICE` env var; `hw_bench.py` adds the two-device comparison, the cadence
arithmetic, and the derived claim sentence.

**5. CALIBRATION TIERS 1 AND 2 EXIST AND ARE COMMITTED. CONFIRMED.** `eval/calibration.py` extended
(not duplicated — it already owned `CALIBRATORS`). Tier 1 = stratified 333/334 split, fixed seed, map
fit on the calibration half, every metric reported on the evaluation half only.

```
TIER 1, evaluation half (n=334, pos=167)
map             ECE                  adaptive ECE   Brier    NLL      AP
uncalibrated    0.3273 [.282,.374]   0.3273         0.2927   1.0793   0.8596
beta            0.0809 [.062,.127]   0.0841         0.1426   0.4468   0.8596
isotonic        0.0560 [.049,.107]   0.0648         0.1391   0.4343   0.8342
platt           0.0908 [.068,.134]   0.0898         0.1447   0.4494   0.8596
temperature     0.2209 [.177,.263]   0.2344         0.2109   0.6102   0.8596

TIER 2, evaluation-half ECE over 100 random stratified splits
beta         median 0.0596   5-95% [0.0405, 0.0850]
platt        median 0.0662   5-95% [0.0484, 0.0890]
isotonic     median 0.0592   5-95% [0.0361, 0.0898]
temperature  median 0.2014   5-95% [0.1655, 0.2387]
uncalibrated median 0.3299   5-95% [0.3147, 0.3461]
```

**🔴 THE DEPLOYABLE NUMBER IS WORSE THAN THE ACHIEVABLE ONE, and that is the whole point of the tier
split.** Beta's Tier 1 ECE is **0.0809** against Tier 3's **0.0503** — half the fit data plus an
honest held-out report. **Quoting 0.0498/0.0503 as a product figure overstates deployed calibration
by roughly 60%.** Tier 2 also shows seed 0 is a slightly **unlucky** split (0.0809 sits near the upper
end of [0.0405, 0.0850]), so the headline is not flattered by split luck.

**Beta remains the right map** (D37): near-best ECE at **zero** AP cost. Isotonic wins on ECE but
drops AP 0.025 — it is only weakly monotone, and the Tier 1 AP column now shows that directly instead
of asserting it.

**6. The score source is now a parameter, so R1 promotion is a re-run not a rewrite (D38).**
`--source` replaces the hard-coded `runs/baselines/` path. Tier 3 always read nanmax scores; if R1 is
promoted the map **must** be refit, and `python -m eval.calibration --tier1 --source last_window`
does it. The source is printed in every report and written into every emitted file, because a
calibration map fit on one reduction is meaningless against another.

**7. Adaptive (equal-mass) ECE added, as §6 requires — and it is not decoration.** Equal-width ECE
can look excellent purely because scores pile into one or two bins, which is exactly this model's
shape (see `runs/calibration/reliability.png`: the uncalibrated bottom panel shows >100 of 334 clips
in the top bin). The self-check builds a case where **equal-width reports 0.0000 — "perfectly
calibrated" — while adaptive reports 0.0990.**

**8. A self-check of mine was wrong and was REPLACED, not loosened** — third session running that
this has happened and been recorded. The first adaptive-ECE case compared 0.46 against 0.45: both
large, no contrast, demonstrating nothing. The replacement requires equal-width to look *clean*
(< 0.01) **while** adaptive exposes real error (> 0.05) — strictly harder to satisfy. Same discipline
as session 8's D30 null-control fix and session 9's item 8.

**9. The existing reliability plotter was reused with zero new plotting code.**
`python -m eval.calibration --tier1 --emit runs/calibration` writes one `plots.py`-shaped JSON per
map, so `~/envs/crashdet/bin/python eval/plots.py runs/calibration/*.json` renders §6's reliability
diagram unchanged. `runs/calibration/reliability.png` and `pr_curve.png` are committed. The diagram
reads correctly: uncalibrated sits far below the diagonal (it says 0.95 and 68% are collisions; it
says 0.55–0.85 and almost none are), beta and isotonic track it.

**10. Tier 3 output verified BIT-IDENTICAL to the pre-change version. CONFIRMED.** The pre-edit
`eval/calibration.py` was checked out to a scratch copy and run: every figure matches to 14 decimal
places. The Tier 1/2 additions changed no existing behaviour.

**11. A pre-existing number discrepancy, NOT introduced this session. NEEDS VERIFICATION.**
The committed script gives beta Tier 3 ECE **0.0503**; `progress.md` §21.1 item 3 and `NEW_PLAN.md`
§3.4 both quote **0.0498**, which came from session 7's in-conversation script. The committed script
is the reproducible one and should be treated as authoritative. Neither document was edited for this
— it is recorded here for the next session to settle.

**12. All four regression guards reproduce exactly, run after every change.** `eval/benchmark.py`
(T3: AUC 0.5339 / AP 0.5218 / 332,325,2,8), `eval.reduction_study` (max 0.8349, last_window 0.8905
+0.0556 [+0.0263,+0.0876]), `eval.timing --self-check` (7/7), `eval.heldout_half --self-check` (null
median ΔAP +0.0025). **No committed number moved this session.**

### 21.5b — FILE / REPOSITORY CHANGES (session 10)

| Path | Change | What it is and why |
|---|---|---|
| `eval/timing.py` | **committed** (`024bec9`, 335 lines) | Session 9's build; per-frame trace → `t_start`/`t_peak`/`t_end`, README §27's incident-record field 2. Two-stage threshold (D32). `--self-check` 7/7, `--emit DIR` writes 351 JSON records. Owns `load_traces_abs()` — the NaN-offset-preserving loader that avoids the 2-second trap. |
| `eval/calibration.py` | **modified** (+292 lines) | Was Tier 3 only. Now Tiers 1+2 as well: `tier1()`, `tier2()`, `adaptive_ece()`, `bootstrap_ci()`, `load_scores(source)`, `_emit()`, plus a `--tier1 / --source / --emit` CLI. Self-check 1 → 6 checks. **Tier 3 path verified bit-identical.** |
| `scripts/hw_bench.py` | **new** (151 lines) | CPU-vs-MPS throughput; states the 1 Hz cadence budget *before* measuring so the bar cannot move, derives the permitted claim sentence from the measurement, and self-checks the arithmetic (including that an unmeasurable device is never called real-time). |
| `NEW_PLAN.md` | **modified** (+108 lines) | R1 gate-3 block (3a/3b), its correction, the kill condition, §11 point 1's residual-risk update. **The only plan-document change; user-approved twice.** |
| `runs/calibration/` | **new** | 5 × `tier1_nanmax_*.json` (evaluation half, `plots.py`-shaped) + `reliability.png` + `pr_curve.png`. |
| `progress.md` | **modified** | `fbf127c` recovered sessions 8+9; this handoff adds session 10. |

**Not touched, deliberately:** `eval/adapters.py` (still `np.nanmax` — R1 unpromoted), `eval/benchmark.py`,
`eval/reduction_study.py`, `eval/heldout_half.py`, `eval/plots.py`, `README.md`, all of `vendor/`.

### 21.5c — GIT STATE at end of session 10

```
branch:        main
HEAD:          37f02b0
origin/main:   37f02b0        (git log origin/main..HEAD is EMPTY)
git status:    clean          (no modified, no untracked)
```

Session 10's commits, oldest first: `fbf127c` · `024bec9` · `c005757` · `faa823e` · `d0bc237` ·
`37f02b0`. The previous session's HEAD was `252d282`.

**Nothing to avoid overwriting — everything is pushed.** The standing caution still applies to the
667 `.npz` traces in `runs/baselines2/` (~20 h of MPS time, committed at `252d282`): **never
force-push**, and if a push is rejected, STOP and report.

### 21.5d — TECHNICAL STATE (what actually works, end of session 10)

**Works, committed, reproducible:**
- **Scoring pipeline.** `eval/adapters.py::BadasOpen` wraps vendored BADAS-Open (V-JEPA2 ViT-L),
  sliding window 16 frames @ 8 fps stride 1, clip reduction `np.nanmax`. 667 Nexar test-public clips
  scored; 667 per-frame traces committed.
- **Metrics harness.** `eval/benchmark.py` — one code path for three models, shared duration table
  so FP/hour denominators are comparable, leakage check, self-checking.
- **R1 study.** `eval/reduction_study.py` (gate 1) and `eval/heldout_half.py` (gate 2), both with
  paired bootstrap and a null control.
- **Temporal localisation.** `eval/timing.py` — incident-record field 2.
- **Calibration.** `eval/calibration.py` Tiers 1, 2, 3 + reliability diagrams via `eval/plots.py`.
- **Hardware.** `scripts/hw_bench.py`.

**Does not work / not built:**
- **R1 is not promoted** — `adapters.py:129` is still `np.nanmax`, deliberately, pending gate 3.
- **Gate 3** — designed, not run; blocked on data.
- **Calibration Tier 4** (external transfer) — not written; needs external data, same blocker.
- **R2 / R3 / R4 / R5 / R6 / R7 / R9** — none started.
- **Incident record** — 2 of ~10 fields. `ego_involved`, severity, closing speed, GPS are omitted
  (not null-filled) by design (D34).
- **`code/crash_detection_enhanced.py`** — legacy, retired model, only ever syntax-checked; its deps
  are not installed. Do not assume it runs.
- **Tracks B and C** — zero across ten sessions.

**Known limitations that bound every number here:** 0.899 h (54 minutes) of negative footage is the
denominator ceiling, so **92.3 FP/hour** at recall 0.80 is the honest state and calibration cannot
change it (D28). Nexar's `time_of_event` is corrupt for all 334 positives, so timing is a
**capability, not a metric** — no mTTA, ever, from this benchmark.

---

## 21.4 SESSION 9's FINDINGS — added 2026-09-16 16:45 IST. Read after the top banner.

**1. `eval/timing.py` EXISTS AND WORKS. CONFIRMED.** 331 lines, untracked. Executes §13 exactly as
session 8 specified it. Per-frame trace → `t_start`/`t_peak`/`t_end` in seconds, emitted as a
`README.md` §27-shaped incident record. Zero compute, zero new data — it reads the 667 `.npz` traces
committed at `252d282`.

Invocations:
```bash
~/envs/badas/bin/python -m eval.timing --self-check   # 7/7, prints PASS
~/envs/badas/bin/python -m eval.timing                # the report
~/envs/badas/bin/python -m eval.timing --emit DIR     # one JSON record per gated clip (351 files)
```

**2. The two-stage threshold, and why it is two stages. DECISION (D32).** "Did an incident happen"
and "when did it happen" are different questions; one threshold answering both is degenerate on
marginal clips (`t_start` and `t_end` collapse to the same instant).

- **Stage 1 — gate (global, derived).** Lowest threshold still reaching a stated target recall, read
  off the PR curve over all 667 clips. Satisfies `README.md` §41 Phase 2 task 5. **Not hard-coded** —
  B5/§12 forbid it and 0.80 is the *retired* model's number.
- **Stage 2 — localisation (per-clip, relative).** `cut = median(trace) + 0.5*(max - median)`;
  `t_start`/`t_end` are first/last crossings, `t_peak` is the argmax. Half-prominence above the
  clip's *own* quiet level, so a uniformly-high clip does not report a 10-second incident.

Measured at target recall 0.80:
```
threshold      0.9733   (derived, not chosen)
recall         0.8024   precision 0.7635
83 FP over 0.899 h of negatives = 92.3 FP/hour
351 records emitted: 268 positive + 83 negative (the gate's false alarms)

gated POSITIVES n=268: window median 4.62 s, t_peak at 0.974 of clip, 77.6% in final 10%
gated NEGATIVES n=83 : window median 4.50 s, t_peak at 0.899 of clip, 49.4% in final 10%
```
Threshold is data-derived, proven behaviourally: recall 0.70/0.80/0.95 → 0.9830/0.9733/0.8098,
FP 60/83/194.

**3. 🔴 92.3 FP/hour is the honest state, and it must never be quoted without its denominator.**
CONFIRMED. 0.899 h is **54 minutes** of negative footage — the known data ceiling. Also note
**calibration cannot fix this**: a monotone map cannot change AP, AUC, or FP/hour at matched recall
(D28). HYPOTHESIS, explicitly not a claim: Nexar's negatives are *curated hard* negatives, not random
driving, so the rate on ordinary footage is probably lower — **by an unknown factor**. That unknown
is exactly why `NEW_PLAN.md` §8.2 wants comma2k19 and ZOD. Do not use it to soften 92.3.

**4. §3.2's mechanism reproduced at FULL n=667 through an independent code path. CONFIRMED — new
evidence, and it strengthens R1.** Session 7 measured peak position at n=268 only. Session 9's
absolute-offset loader (a different code path from `reduction_study.load_traces`) gives:

| | median peak position | peak in final 10% |
|---|---|---|
| positives (n=334) | **0.975** | 73.7% |
| negatives (n=333) | 0.802 | 38.4% |
| *session 7, n=268* | *0.984 / 0.782* | *83.0% / 42.5%* |

Same direction and magnitude, mild softening at full n — exactly what a partial prefix → full set
should do. **This was a stop condition** ("if `t_peak` lands at the clip end on negatives too, the
mechanism is wrong and R1 is suspect"). It did not trigger. R1's mechanistic story survives its first
full-n check. It is **not** a substitute for gate 3, which remains the test that decides R1.

**5. THE 2-SECOND TRAP, caught by an assertion rather than by luck. CONFIRMED.**
`eval/reduction_study.load_traces()` **strips the 16 leading NaN frames** — correct for its own job,
since ranking a clip needs no time base. Reuse it naively for timing and **every timestamp lands
exactly 2 s early**. Nothing crashes; the numbers are just silently, plausibly wrong forever.
`eval/timing.py` has its own `load_traces_abs()` that keeps the absolute offset, and self-check 4
asserts the difference (`5.000s with, 3.000s without`). **`reduction_study.load_traces` was left
unchanged** — `heldout_half.py` and `reduction_study.py` both depend on its current behaviour.

**6. The time base is confirmed THREE independent ways. CONFIRMED.** `t = index / target_fps`, with
fps read from each `.npz`, never hard-coded:
- (a) traced to source — `vendor/badas-open/badas/utils/sliding_window.py` resamples to `target_fps`
  *before* windowing, so a trace index is an index in the **resampled** timeline; the source video's
  fps is irrelevant. A window `(start, start+16)` files its prediction against `end_idx`, the first
  frame the model has **not** seen, so index *i* reads "having watched to (i−1)/fps, this is
  P(collision next)".
- (b) **cross-checked against `benchmark.durations()`** (the T3 run, which decoded all 667 clips
  months ago for an unrelated purpose): agreement within **0.0623 s — under one 8 fps frame — on all
  667 clips**. This is self-check 2.
- (c) upstream's own `badas_loader.py:112` does `return i / fps` at `fps=8.0` — identical arithmetic.

**7. Upstream silently DISCARDS the final window's prediction. CONFIRMED — new.**
`_create_predictive_frame_array` guards with `if 0 <= target_frame < total_frames`. At stride 1 the
last window targets `total_frames` exactly, which fails that test. An 81-frame clip runs **66 windows
and keeps 65**. Consequences: (a) R1's "last window" is really the *second*-to-last — harmless for
R1, which applies the same reduction to every clip, but the naming is misleading; (b) the latest
representable timestamp is `(len-1)/fps`, so **`t_end` must never be read as "end of clip"**.

**8. A self-check of mine was wrong and was replaced, not loosened.** The first version of check 7
grepped the module source for the string `"0.80"` to prove the threshold wasn't hard-coded. It failed
— on the *docstring sentence explaining why 0.80 is forbidden*. It was testing prose, not behaviour.
Replaced with a behavioural check: asking for more recall must move the threshold **down** and cost
more FP, which a hard-coded constant could not do. **The assertion was made stronger, not weaker**
(same discipline as session 8's D30 null-control fix).

**9. THE 8 FPS QUESTION — ASKED BY THE USER, ANSWERED: DO NOT CHANGE IT. CONFIRMED.**
This is already closed in §12 and §6.18; restating because the user asked directly.
- 8 fps is **upstream BADAS's own default**, not ours — `vendor/badas-open/badas/badas_loader.py`
  hard-codes `target_fps=8.0` at lines 47, 89 and 121, and upstream's README and example use
  `target_fps=8.0, num_frames=16, window_stride=1`, exactly what `eval/adapters.py` runs.
- It is **what the classifier head was trained at**: 16 frames @ 8 fps = 2.0 s, matching the
  checkpoint config's `data_root ".../balanced_dataset_2s"`. At 4 fps that window is 4 s, at 16 fps
  1 s — neither is what the head ever saw. (`original_fps: 4` is the post-tubelet **token** rate, not
  a video rate. §6.18. Not a contradiction.)
- Changing it **invalidates every committed number**: the ~18 h sweep, AP 0.8349, R1's +0.0556, T3,
  the calibration table.
- A 2× mismatch here would be **this project's own B1 bug repeated** (README §15: the old model
  trained on 5 s spans, run on 0.33 s ones). `scripts/badas_fps_probe.py` exists to assert it hasn't.
- **Where other frame rates DO belong:** `NEW_PLAN.md` **R2 (multi-temporal-scale ensemble, TIER 1)**
  — score at 4 / 8 / 16 fps and **ensemble**. That *adds* decorrelated channels; it never replaces 8.
  Week-2 item, ~4–5 h per scale tail-only, expected +0.01–0.03.

**10. Fields the incident record cannot produce are OMITTED, not null-filled.** DECISION (D34).
`ego_involved` (Nexar has no ego field, D23), severity band, closing speed, GPS. A `null` implies a
field that *could* be populated and merely isn't, which misrepresents what the system knows. The
record also carries `"timing_validated": false` explicitly.

**11. 🔴 THIS IS A CAPABILITY, NOT A METRIC — unchanged from session 8's §13 instruction.**
Nexar's `time_of_event` is corrupt for all 334 positives (median 20.0 s against a 9.93 s clip — the
event is *outside* the distributed video). There is **nothing to validate a timestamp against**. No
mTTA, no time-to-detection, no "±0.3 s accurate", ever, from this benchmark. The honest claim is:
*the pipeline now emits these three fields; their accuracy is unmeasured on this benchmark because
this benchmark cannot measure it.* Stated in the module docstring and printed by the report itself.

---

## 21.3 SESSION 8's FINDINGS — added 2026-09-16. Read after the top banner.

**1. R1 gate 2 PASSED. CONFIRMED.** Numbers in the top banner. Method: `eval/heldout_half.py`.
Stratified 50/50 split; the winning reduction is chosen on half A by AP; its ΔAP vs `nanmax` is
reported on half B only, with the paired bootstrap; repeated over 1000 random stratified splits as a
split-luck control (the same discipline `NEW_PLAN.md` §6 Tier 2 mandates for calibration).
Session 7's weak version (6 halves at n=268, spread −0.009 to +0.075) is superseded — at n=667 the
held-out ΔAP is **positive in 100% of 1000 splits** and the instability is gone.

**2. 🔴 GATE 3 CANNOT BE RUN AS `NEW_PLAN.md` R1 SPECIFIES IT. CONFIRMED — this is a plan defect.**
R1's falsification test reads: *"on DoTA/DADA, where clips are NOT truncated, last-window should
perform WORSE than max."* That is an **AP comparison, and AP requires both classes.** Verified by
reading the vendored annotation CSVs directly:

| Set | n | Composition |
|---|---|---|
| `dad_test_concensus.csv` | 165 | 151 Collision + 14 Near-collision — **all positive** |
| `dada2000_small_test_concensus.csv` | 221 | 198 + 23 — **all positive** |
| `dota_test_concensus.csv` | 598 | 562 + 36 — **all positive** |

There are **no negatives in any of them**. AP is undefined on one class, so the test as written
cannot be computed at all, on any of the three.

**PROPOSED REDESIGN (not implemented, needs user sign-off as a plan change):** all three sets ship a
`Time-of-collision` column, so test the **mechanism** directly instead of the AP proxy — on
untruncated clips the score peak should sit **at the annotated collision time, mid-clip**, not at the
clip end. Nexar positives peak at normalised position **0.984** (§3.2 / `NEW_PLAN.md` §3.2). If
external positives *also* peak at ~0.98 despite the collision being seconds earlier, then the model
merely drifts upward with watch-time, the §3.2 mechanism is wrong, and R1 is suspect. This works on
positives-only data and is a sharper test than the AP comparison it replaces.

**3. The external clips genuinely do continue past the collision. CONFIRMED.** The user challenged
whether post-crash footage can exist ("when crash happen car stops"). Measured from the same CSVs:

| Set | Camera car crashed (`pos-ego`) | **Witnessed** (`pos-not-ego`) | Time-of-collision |
|---|---|---|---|
| DAD | 13 (8%) | **150 (91%)** | min 1.6s · median **3.0s** · max 3.7s |
| DADA-2000 | 75 (34%) | 51 (23%) | min 0.4s · median 5.3s · max 14.4s |
| DoTA | 327 (55%) | 255 (43%) | min 0.9s · median 4.7s · max 14.4s |

**In DAD, 91% of clips are collisions the camera vehicle was not involved in** — it keeps driving, so
the footage keeps rolling. Contrast with Nexar, where `time_of_event` (median **20.0 s**) lies
**beyond** the clip (median **9.93 s**): the event is outside the video entirely. That structural
contrast is exactly what gate 3 needs, and it is confirmed from files already in the repo.

**4. DAD does NOT need the Mac Studio. CONFIRMED by arithmetic.** 165 clips × ~97 s/clip end-to-end
(D10's measured rate) ≈ **4.5 h** — one overnight Air run, the same shape as every sweep so far. The
Studio is only required for DoTA-scale work (598 clips) or the 25.5 GB Nexar train split. **Earlier
in this session this was wrongly stated as needing the Studio; that was corrected.**

**5. Disk is a hard constraint on DoTA. CONFIRMED.** `df -h` on the Air: **33 GB free**. `NEW_PLAN.md`
§7.4 puts DoTA at ~55 GB — it does not fit. The Studio's recorded **~40 GB free** is from session 1
(2026-09-11), is stale, and is **still below 55 GB**. Re-check `df -h` on the Studio before any
download. **DAD is the only realistic first target.**

**6. `eval/calibration.py` now exists and is committed** — closes the §21.1 item 5 P0 gap where the
ECE 0.3286 → 0.0498 table had no reproducible script. **Caveat: it implements `NEW_PLAN.md` §6
Tier 3 ONLY** (5-fold cross-fitted), which §6 says must be reported as *"achievable calibration"* and
**never** as a deployed result. **Tier 1 — the primary deployable calibration/eval half split — is
still unwritten.** It also reads `runs/baselines/` (nanmax scores); if R1 is ever promoted the
calibration must be refit on last-window scores.

**7. Bug found and fixed in `eval/calibration.py`'s docstring.** It documented `python
eval/calibration.py`, which fails with `ModuleNotFoundError: No module named 'eval'` — the
`from eval.benchmark import ...` needs the repo root on `sys.path`. Correct invocation is
**`python -m eval.calibration --self-check`**. Docstring corrected; no code change.

**8. A self-check caught its own bias — worth knowing, because the fix is subtle.**
`eval/heldout_half.py`'s null control initially replaced traces with random noise. It **failed**
(median ΔAP −0.0252). Diagnosis: with one fixed random draw, `nanmax` happened to score AP **0.5630**
against a chance level of 0.5007, and that single lucky draw persists across every split, dragging the
whole null negative. It was measuring "how lucky was the baseline" not "does selection inflate the
estimate". **Fixed by permuting the labels per split instead** — this preserves each reduction's own
distribution and puts the baseline on equal footing. Corrected null: **+0.0025**. The assertion
threshold was *not* loosened to make it pass.

**9. Product framing correction, from the user.** This session initially described the product as
"software that watches dashcam video and flags crashes". **That is wrong and `README.md` §27 says so
explicitly:** *"The product is not crash detection... The product is the structured incident record."*
Detection is a commodity input (BADAS is free and Apache-2.0). Against §27's ~10-field record,
**exactly one field currently works** (collision detected), and it is the commodity one. **The 667
committed per-frame traces newly unblock `t_start`/`t_peak`/`t_end`** — §13 step 2, zero compute.
That would be the second working field and the first non-commodity one.

---

## 21.2b DATA / LICENSING — SESSION 10 ADDITIONS (gate-3 datasets). Read with §21.2 below.

**All from primary sources this session. Both are UNRESOLVED on licence and both are needed for
R1's gate 3, so neither can be used in a commercial claim until its terms are in writing.**

| Asset | Composition | Distribution | Size | Licence | Status |
|---|---|---|---|---|---|
| **DAD** (gate 3b) | 1,750 clips, 620 pos / 1,130 neg, six Taiwanese cities. **Test split 466 = 165 pos + 301 neg.** 100 frames/clip, 720p, accident nominally in "the last 10 frames" | **Google Form request to the authors** — http://aliensunmin.github.io/project/dashcam/ | not stated | 🔴 **NO terms posted publicly** | **NOT downloaded. The USER must submit the form.** |
| **DADA-2000** (gate 3a) | 2,000 videos, 658,476 frames, 1584×660, ~6.1 h @ 30 fps. We need only the **221 test clips** in `dada2000_small_test_concensus.csv` | Baidu Pan, code `9pab` (train/test ~53 GB) or `ahyz` (full ~116 GB); **a Google Drive mirror is mentioned** — https://github.com/JWFangit/LOTVS-DADA | **~53 GB compressed** | 🔴 **NO licence statement in the repo at all** | **NOT downloaded. ~53 GB vs ~30 GB free — does not fit unless the Drive mirror is per-file.** |

**Three things this changes:**
1. **DAD's 301 test negatives are the fact that unblocked gate 3b.** Corroborated by BADAS
   publishing AP 0.66 / AUC 0.87 on DAD — impossible on one class. `README.md` §49 does not record
   this; adding it there would be a reasonable future edit.
2. **Both are YouTube/web-derived** (DADA explicitly: YouTube, Youku, Bilibili, iQiyi, Tencent),
   which is the same provenance problem already flagged for CCD and DoTA in §21.2 and README §23.
3. **Neither may support a commercial claim** until terms are obtained. They can still serve as
   **internal falsification tests** — gate 3's only purpose is to try to kill R1, not to ship.
   Keep that distinction explicit if either is ever cited externally. Precedent: BDD100K was
   **excluded** from this project on exactly these grounds (`NEW_PLAN.md` §8.2).

**DoTA remains excluded** — ~55 GB against ~30 GB free on the Air, and the Studio's ~40 GB figure is
from session 1 (2026-09-11), **stale and still below 55 GB**. Re-check `df -h` before any claim
about Studio capacity.

---

## 21.2 DATA / LICENSING STATE — verified from primary sources, session 7

| Asset | Licence | Train | Eval | Commercial product | Status on disk |
|---|---|---|---|---|---|
| Nexar collision dataset | `nexar-open-data-license` | ✅ | ✅ | ✅ **with attribution** | test-public only (667 clips, 2.7 GB). train + test-private are **metadata-only** |
| BADAS-Open weights + code | **Apache-2.0** | — | ✅ | ✅ | on disk (3.7 GB) + vendored source |
| comma2k19 | **MIT** | ✅ | ✅ | ✅ | **not downloaded** — 33 h, California highway, ships CAN+IMU (verify per release) |
| Zenseact ZOD | **CC BY-SA 4.0** | ✅ | ✅ | ✅ | **not downloaded** — 1,473 × 20 s + 29 drives, European, diverse weather |
| DAD / DADA-2000 / DoTA | varies; DoTA repo says MIT but videos are YouTube-sourced | verify | ✅ | verify | **videos not downloaded**; consensus annotations ARE vendored (984 clips) |
| BDD100K | research / non-profit only | ❌ | grey | ❌ needs UC Berkeley OTL | not downloaded — **excluded on licence grounds** |

**⚠️ CORRECTION of an error made and then caught within session 7:** it was briefly stated that
Nexar's licence forbids commercial use. **That is wrong.** `data/nexar/LICENSE` grants the right to
"use, copy, modify, and distribute the Dataset"; the No-Resale clause forbids selling or
redistributing **the dataset itself**, not products built from it. README §43's "commercial use
permitted" was correct. The ethical-use restrictions (no weaponisation, no reidentification, no
predatory insurance) are all compatible with this product.

**The practical shape of it:** the *model* is commercially usable (Apache-2.0) and the Nexar *data*
permits commercial model training with attribution. comma2k19 and ZOD are the two permissively
licensed sources that could underpin a shipped product. DAD/DADA/DoTA/BDD100K are evaluation-side
only. **comma2k19 is highway-only and would flatter any FP/hour number — it is a stress test, not a
general rate.**

---

## 21. `NEW_PLAN_v2.md` — SESSION 7's OUTPUT, UNREVIEWED PROPOSAL, NOT YET ACCEPTED

> **🔴 SUPERSEDED 2026-09-14.** This section describes `NEW_PLAN_v2.md`, which has been **merged into
> `NEW_PLAN.md` and moved to `archive/NEW_PLAN_v2_parallel_session.md`.** `NEW_PLAN.md` is now the
> single authoritative proposal. The section is kept below because its per-candidate reasoning
> (C1–C7) is still useful, and because its C2 became `NEW_PLAN.md`'s R9. **Where this section and
> `NEW_PLAN.md` disagree, `NEW_PLAN.md` is correct.** Both remain PROPOSALS — neither is accepted.

**File: `NEW_PLAN_v2.md`, repo root, ~450 lines, untracked. Supersedes `NEW_PLAN.md` (session 6's
output, also still just a proposal — never merged into README or this file) as the active proposal.
Neither `NEW_PLAN.md` nor `NEW_PLAN_v2.md` have been accepted, started, or reflected in README.md.**

**Why it exists:** the user gave a 10-point red-team brief (see SESSION 7 SUMMARY above) demanding
`NEW_PLAN.md` be revised for leakage/hardware/provisional-finding rigor and for genuine
detection-improvement candidates beyond calibration, scoped against the user's REAL compute (M4
MacBook Air 24/7, Mac Studio M4 Max ~2–4h weekdays / ~10h weekends — **NOT** the richer, unstated
budget this file's own compute language elsewhere implicitly assumes; see below, this is a real
correction the next session should propagate if `NEW_PLAN_v2.md` is accepted).

**Contents summary (read the file directly for full detail — this is a pointer, not a replacement):**

- **§0–1**: notes the user's referenced "10 points" prior artifact was not found in this session
  (open item — supply it and re-diff if it exists); recomputes the realistic Studio budget at
  ~25–30h/week (a ~4–5× reduction from prior implicit assumptions) and states its consequence for the
  in-flight sweep and any future full-corpus sweep (default to stride 2 unless stride 1 required).
- **§4**: seven detection-improvement candidates (C1–C7), each with hypothesis/mechanism/data/compute/
  cost/expected-gain/failure-condition/leakage-risk/validation fields. Standouts: **C2** (comma2k19
  IMU/CAN-informed hard-negative rejection classifier — the only candidate adding a signal BADAS's
  video-only backbone structurally cannot use) ranked highest expected value; **C1** (supervised head
  on DAD/DADA/DoTA external annotations, 984 clips, none downloaded yet) ranked second but flagged as
  the plan's highest leakage risk since BADAS's own authors report published figures on those exact
  benchmarks; **C5** (probe trained on Nexar train) re-examined and **kept killed** — Nexar train is
  literally BADAS-Open's own training data (confirmed on disk this session: train/test-private are
  metadata-only, zero `.mp4`, consistent with `NEW_PLAN.md`'s F1).
- **§5**: formal calibration protocol — 5-fold nested cross-fitting **within `test-public` only**
  (the only labeled video data on disk), with an explicit statement of what it does/doesn't prove;
  external-transfer (DAD/DADA/DoTA) calibration is scoped as a separate, weaker generalization claim.
  **P0 action: `eval/calibration.py` does not exist yet** — the committed F2 calibration numbers in
  `NEW_PLAN.md` came from an uncommitted, unreproducible script. Write it before quoting that table
  again.
- **§6**: comma2k19 reclassified as hard-negative/stress-test data only, never a general FP/hour
  claim; Zenseact ZOD proposed as a second, geographically distinct source (not yet scoped/acquired).
- **§7**: every "real-time"/"CPU-capable" claim in the repo audited — most rest on a synthetic-tensor
  timing (`badas_smoke.py`) or a 6-clip sample, neither confirmed on either of the user's actual two
  machines. **New required script, not yet written: `scripts/hw_bench.py`** — benchmarks real
  end-to-end throughput on M4 Air (MPS + forced-CPU) and M4 Max Studio (MPS) separately, with
  bootstrap CIs. This gates every timeline estimate downstream of it.
- **§9**: week-by-week timeline built off the real budget; explicitly states DoTA acquisition is a
  two-weekend item and that comma2k19/DAD-DADA scoring will contend for the same weekend Studio slots.
- **§10**: F3 (temporal smoothing hurts — `max` beats mean/persistence) re-examined and **kept
  provisional** (trace sweep was 222/667 at time of writing, now 261/667 per the banner above) with
  three explicit, written promotion criteria (full 667/667, bootstrap CI excluding zero, reproduced
  through the committed eval path) — corrects `NEW_PLAN.md`'s own language, which had called this
  "Settled: max."
- **§11**: Track C (30 UK fleet calls) — recommends executing with a forcing weekly quota, but
  explicitly flags that it has sat at zero for **six-plus sessions now** despite zero cost, which the
  plan states is evidence of an unnamed blocker (contact access? confidence? scheduling?) that only
  the user can diagnose — not resolved, an open question.
- **§12**: uncertainty audit of existing deltas — notably reclassifies `NEW_PLAN.md`'s beta-vs-Platt
  calibration "win" (ECE 0.0498 vs 0.0524, a 0.0026 delta) as **not meaningful**, well within beta's
  own stated bootstrap CI [0.040, 0.087].
- **§13–15**: ranked plan (by expected gain ÷ Studio-hours, not ease — C2 ranked #1 by value despite
  not being the cheapest item), a **hybrid** keep/rebuild verdict (keep BADAS-Open frozen as
  backbone/ranker, no fine-tuning — no CUDA anywhere, MobileNetV2+LSTM confirmed chance-level per T3 —
  but explicitly reject "calibration is the ceiling" and commit to C2 and, more cautiously, C1 as
  real evidence-gated additions), and a decisive one-month answer: commit calibration protocol +
  `hw_bench.py` first (zero Studio cost, week 1), freeze F3, run the free C3 temporal head, then spend
  the bulk of the month's Studio budget on C2.
- **§16**: a self-adversarial second pass — found and fixed one real timeline overcommitment (an
  earlier draft had DAD+DADA+DoTA all landing by week 3, which doesn't fit the recomputed budget;
  fixed by deferring DoTA to weeks 5–6), added a missing cross-reference (C3's temporal-head training
  must reuse calibration's clip-level split, not create a new ad hoc one), and softened one
  overconfident claim to match the document's own CI-or-caveat discipline.

**What the next session must do with this:** get an explicit accept/reject/revise decision from the
user before starting any of its ranked items (§13's `eval/calibration.py` and `scripts/hw_bench.py`
are the two zero-cost, zero-risk items if accepted — reasonable to propose starting there). Do **not**
assume acceptance and do not silently merge its content into `README.md` — the user's own rule (§16
below) is that README changes require either an explicit README instruction or a master-plan-level
discovery, and "a proposal document exists" is neither.

---

## 15. CONTEXT-WINDOW HANDOFF

### SESSION 7 FINAL END STATE — corrected/expanded 2026-09-14 00:55 IST (the block immediately
### below was written by the parallel window earlier the same evening and understates what happened)

**Exactly what was happening when this session stopped:** the sweep (PID 75682) was **alive at
10h24m elapsed, 377/667 clips, 377 `.npz` frame files** — measured, not assumed. Nothing else was
running. The last actions taken were documentation edits to this file; no code, no commits.

**Git:** branch `main`, HEAD **`1b8d5c9`** (session 6's handoff commit — unchanged all session).
Working tree: `M progress.md`, `?? NEW_PLAN.md`, `?? archive/NEW_PLAN_v2_parallel_session.md`,
`?? runs/baselines2/`. **Nothing was committed this session.** `README.md` is byte-identical to
HEAD (verified with `git diff --stat README.md`).

**Must not be accidentally overwritten:** `runs/baselines2/` (10+ hours of GPU work in progress, and
the `frames/*.npz` are the only per-frame data that exists); `runs/baselines/` and
`runs/falsification/` (committed evidence).

**Partially complete — do NOT record any of these as done:**
- **The sweep is ~57% done (377/667), ~8 h remaining.** No final number for it yet.
- **The last-window reduction finding is measured but NOT confirmed** (n=268 of 667).
- **`NEW_PLAN.md` is complete as a document but UNREVIEWED and UNACCEPTED by the user.** Nothing in
  it has been started.
- **`eval/calibration.py` does not exist.** Zero lines written. Same for any hardware benchmark.
- **Tracks B and C remain at zero, now seven sessions running.** No UK footage, no fleet calls, and
  the two Phase 0 licence emails are still unsent.

---

### SESSION 7 END STATE (earlier, parallel window) — superseded by the block above

**Exactly what was happening when session 7 stopped:** the same background sweep from session 6
(PID 75682, `runs/baselines2/`) was still running, now at **261/667** (confirmed by direct process
check + `wc -l scores.jsonl`, not assumed). No code was written or executed this session. The only
artifact produced was `NEW_PLAN_v2.md` (see §21), a planning document.

**Why the session ended:** the user said the context window was about to run out and asked to end the
session; this handoff was written on that explicit instruction.

**Partially completed, stated plainly — do NOT record any of these as done:**
- **The second sweep is still in flight, now ~39% complete (261/667).** Still no final number for it.
  §13 STEP 0's check commands are unchanged and still correct.
- **`NEW_PLAN_v2.md` is a complete, self-consistent document (it includes its own adversarial
  self-review pass, §16 of that file) — but it is UNREVIEWED BY THE USER and UNACCEPTED.** Do not
  start executing any of its ranked items until the user has explicitly said so.
- **`scripts/hw_bench.py` and `eval/calibration.py`, both specified in `NEW_PLAN_v2.md` as the correct
  zero-cost first steps if the plan is accepted, do NOT exist yet.** Nobody has written them.
- **The "10 points" artifact the user referenced when giving the red-team brief was never located.**
  This is a standing open question, not resolved this session (§10 below should carry this forward).
- **Nothing was committed this session** (no code changed, so nothing to commit besides the two new
  untracked planning files — see §16B below for exact git state).

**Environment reminder, unchanged from session 5/6:** two envs, keep them separate — `~/envs/crashdet`
(TF 2.19.1 / Keras 3.15.1 / matplotlib / sklearn) and `~/envs/badas` (torch 2.14.0 / transformers
5.17.0 / sklearn). **Do NOT rebuild either.**

---

### SESSION 5 END STATE — 🔴 HISTORY, SUPERSEDED. Read §17-S8's "SESSION END STATE" instead.

> Everything below is true as of **session 5** and is kept as the record. It is **wrong as a
> description of today**: the sweep it calls "34% complete" finished long ago, Phase 4 and the Phase 5
> gate both closed in session 6, and `runs/baselines2/` is committed at `252d282`. Do not act on this
> block.

**Exactly what was happening when session 5 stopped:** **a GPU job was still running.** The
667-clip BADAS-Open sweep, PID 88180, 3 h 07 m elapsed, **235/667 clips visited** at 23:24 IST.
It was left running deliberately. The last technical acts were writing this handoff and confirming
the sweep was still healthy.

**Why the session ended:** the user's own rule is to start a fresh session at ~200k context. The
session was at 199.2k of a 1M window (20%). This was explained, the user reaffirmed the rule, and the
handoff was written on their instruction. **Nothing was left half-edited.**

**Partially completed, stated plainly — do NOT record any of these as done:**
- **The BADAS sweep is 34% complete.** No final number exists. §13 STEP 0 tells you how to check
  whether it survived.
- **Phase 4 is ~90% done, NOT done.** Its last acceptance criterion needs `eval/plots.py` run against
  a finished `metrics.json`, which has never existed.
- **Phase 5 is IN FLIGHT, not passed.** The partial AP 0.9274 is not a result (D17).
- **Phase 5 task 3 (consume `predictor_output`) has not been started.**
- **Nothing is committed.** 4 modified files, 5 untracked paths (§8). The user was asked about
  committing and the session ended before an answer.
- **One live offer to the user is unanswered:** whether to edit README §41 Phase 5's gate note,
  reason 5, which still calls the fps question "unresolved" after session 5 resolved it (§16).

**Environment reminder:** two envs, keep them separate — `~/envs/crashdet` (TF 2.19.1 / Keras 3.15.1
/ matplotlib / sklearn — for the old model, `eval/benchmark.py`, `eval/peek.py`, `eval/plots.py`,
both tests) and `~/envs/badas` (torch 2.14.0 / transformers 5.17.0 / sklearn — for BADAS only).
**Do NOT rebuild either.**

---

### SESSION 3 END STATE (history)

**Exactly what was happening when session 3 stopped:** nothing was running. The last technical act was
verifying that `eval/benchmark.py`'s self-check still passes after the extension (it does). The user
then asked for a full teaching explanation of the project, which was delivered in conversation only —
**no files were changed by that request.** Then this handoff was written.

**Partially completed, stated plainly:**
- **Phase 4 is ~60% done, NOT done.** AC #1 (three models, one path) is met. Plots and a committed
  `metrics.json` are not. Do not record Phase 4 as complete.
- **Phase 5 is NOT started.** Prerequisites are all assembled and the risk is gone, but **no BADAS
  performance number exists.** The only BADAS run was 6 clips of plumbing verification.
- **Nothing is committed.** ~850 lines of new Python, a patched vendored dependency, and two modified
  tracked files sit in the working tree only.

**Two questions the user was asked and did not answer:** stride 1 (~18 h) vs stride 2 (~9 h); and how
to structure the commits. Ask once, then proceed.

---

### Session 1 handoff notes (history)

**What was happening when session 1 ended:** T3 had been run, validated, committed (`420ef17`) and
pushed. The README had been updated with the result. A plan for what follows had been presented to the
user and they had answered three framing questions (push: yes; GPU: none; goal: startup). The user was
choosing between "start Phase 4 now" and "settle the four BADAS unknowns first" — **that choice was
never made**, and §13 records the recommendation (unknowns first, with the reasoning).

**What the next session needs to know immediately:**

1. **Read `README.md` first.** It is the master plan and it is long (3,191 lines — page through it with
   offset/limit, ~400 lines per read; it exceeds single-read limits).
2. **The falsification phase is over.** Do not re-run T1/T2/T3/T5, and do not re-litigate whether the
   old model works. It does not. The numbers are in `runs/falsification/`.
3. **Use `~/envs/crashdet/bin/python`** for anything touching the model. The system Python is 3.14 and
   has neither TF nor cv2. `torch` is nowhere yet.
4. **There is no GPU budget.** Any plan whose critical path requires training is invalid (D2).
5. The user prefers primary sources read in full over summaries, and holds a strict evidence standard:
   falsify before accepting, keep raw outputs, never soften an unfavourable number.

---

## 15.10 PLAN POSITION — where the repo sits against BOTH planning documents (SESSION 10, current)

### `README.md` POSITION (master plan)

- **Current phase:** Phase 4 **COMPLETE**, Phase 5 gate **PASSED**. Three tracks run concurrently
  from week 1 (§41): **A** model & measurement, **B** UK data/benchmark, **C** customer discovery.
- **What README says should happen:** Track A phases 4→5→6; Track B the hard-negative benchmark
  (§34, the moat per §40/§45); Track C 30 UK fleet calls. §27 defines the product as the
  **structured incident record**, not detection.
- **Completed:** Track A is well advanced — falsification suite closed, BADAS baseline measured
  (AP 0.8349 / AUC 0.8498), evaluation harness built, R1 at 2 of 3 gates, calibration **Tiers 1, 2
  and 3**, incident-record field 2 (`t_start`/`t_peak`/`t_end`), operating point derived from the PR
  curve (§41 Phase 2 task 5), and now a **measured CPU/MPS throughput figure**.
- **Remains:** Phase 6. **Tracks B and C are at ZERO after TEN sessions** — still the single largest
  gap between the master plan and reality, and still not a technical blocker.

### `NEW_PLAN.md` POSITION (detailed/current plan)

- **Current task:** Week 1, Track A. `NEW_PLAN.md` is still labelled a **PROPOSAL** — see the
  alignment note below.
- **What it says should happen:** R1 → R4 → R2 → R3 → fuse → calibrate. Every comparison by
  **paired** bootstrap (§1). Week 1 = confirm R1, calibration §6 Tiers 1–3, operating-point policy,
  CPU/MPS benchmark (§7.3).
- **Completed:** R1 gates 1 ✅ and 2 ✅. §3.2's mechanism confirmed at full n ✅. Calibration
  **Tier 3 ✅, Tier 1 ✅, Tier 2 ✅**. Operating-point policy ✅. **CPU/MPS benchmark ✅ (§7.3 closed,
  session 10).** **Week 1 is now essentially complete apart from gate 3.**
- **Remains:** **R1 gate 3 (3a on DADA + 3b on DAD) — blocked on data acquisition**, then R4 / R2 /
  R3 (Week 2), calibration Tier 4 (transfer), fuse.

### ALIGNMENT

**No conflict. They are aligned and operate at different altitudes** — README gives the roadmap and
the product definition; `NEW_PLAN.md` gives the detailed experiment sequence inside README's Track A.

**The gate-3 defect that stood across sessions 8 and 9 is now CLOSED** — redesigned and then
corrected in session 10 (`c005757`, `d0bc237`), both on explicit user sign-off. `NEW_PLAN.md` R1 now
carries a gate-3 block specifying 3a (DADA mechanism) and 3b (DAD AP). **There is no longer a known
defect in either planning document.**

**Two stale statements in `NEW_PLAN.md`, deliberately NOT edited** (they record progress, not plan,
and §13 of the session brief forbids editing plans merely to record progress — flagged here for the
user to decide):
1. Its header still reads **"Status: PROPOSAL — nothing implemented."** That is now false: R1 gates
   1–2, calibration Tiers 1–3, the operating point and the hardware benchmark are all implemented.
   The user was told and did not ask for a change.
2. §3.4 quotes beta ECE **0.0498**; the committed script gives **0.0503** (§21.5 item 11).

---

## 15.9 PLAN POSITION — session 9 (HISTORY, superseded by §15.10 above)

### `README.md` POSITION (master plan)

- **Current phase:** Phase 4 **COMPLETE**, Phase 5 gate **PASSED**. Three tracks run concurrently
  from week 1 (§41): **A** model & measurement, **B** UK data/benchmark, **C** customer discovery.
- **What README says should happen:** Track A phases 4→5→6; Track B the hard-negative benchmark
  (§34, the moat per §40/§45); Track C 30 UK fleet calls. §27 defines the product as the **structured
  incident record**, not detection.
- **Completed:** Track A is well advanced — falsification suite closed, BADAS baseline measured
  (AP 0.8349 / AUC 0.8498), evaluation harness built, R1 at 2 of 3 gates, calibration Tier 3,
  and now **incident-record field 2** (`t_start`/`t_peak`/`t_end`). Phase 2 task 5's "derive the
  operating point from the PR curve" is now satisfied inside `eval/timing.py`.
- **Remains:** Phase 6. **Tracks B and C are at ZERO after nine sessions** — this is the single
  largest gap between the master plan and reality, and it is not a technical blocker.

### `NEW_PLAN.md` POSITION (detailed/current plan)

- **Current task:** Week 1, Track A. `NEW_PLAN.md` is still a **PROPOSAL** — never formally
  accepted, so its R1–R9 are not decisions of record (see the note at the end of §11).
- **What it says should happen:** R1 → R4 → R2 → R3 → fuse → calibrate. Every comparison by
  **paired** bootstrap (§1). Week 1 = confirm R1, calibration §6 Tiers 1–3, operating-point policy,
  CPU/MPS benchmark (§7.3).
- **Completed:** R1 gates 1 and 2 ✅. Calibration Tier 3 ✅. Operating-point policy ✅ (session 9).
  §3.2's mechanism now confirmed at full n ✅.
- **Remains:** R1 gate 3 (design broken, needs sign-off), calibration Tier 1, `hw_bench.py`,
  then R4/R2/R3.

### ALIGNMENT

**No conflict. They are aligned and operate at different altitudes** — README gives the roadmap and
the product definition; `NEW_PLAN.md` gives the detailed experiment sequence inside README's Track A.
Session 9's work sits in both: it is `progress.md` §13's step 2 *and* README §27's field 2 *and*
README §41 Phase 2 task 5's operating-point requirement.

**One known DEFECT, not a conflict:** `NEW_PLAN.md` R1's gate-3 falsification test **cannot be
computed as written** — it specifies an AP comparison on DAD/DADA/DoTA, all three of which are
**positives-only** (165/221/598, all Collision or Near-collision). AP is undefined on one class. A
redesign is proposed in §21.3 item 2 (peak position vs annotated `Time-of-collision`) but **has not
been applied** — it needs the user's sign-off as a plan change. This is the one place a planning
document is known to be wrong.

---

## 16. README MODIFICATION STATUS

## SESSION 10 — README CHANGED: **NO.** NEW_PLAN CHANGED: **YES, twice — both user-approved.**

### `README.md` — **NOT CHANGED.**
The master plan did not change. Nothing discovered this session alters the roadmap, the phases, the
product definition (§27) or the track structure (§41). Every session-10 result is *progress against*
the existing plan, and progress belongs here, not in the README.

**One README-adjacent fact worth noting without editing it:** §49's dataset table describes DAD as
"1,750 clips, six Taiwanese cities", which is correct, but nothing in the README records that DAD's
**test split carries 301 negatives** — the fact that unblocked gate 3b. It is recorded in
`NEW_PLAN.md` R1 and in §21.5 item 2 here. Adding it to §49 would be a reasonable small edit for a
future session; it was not made because the master plan itself did not change.

### `NEW_PLAN.md` — **CHANGED, twice, both on explicit user sign-off.**

**Change 1 (`c005757`) — R1's gate 3 redesigned.** *Why it was a genuine plan change, not progress:*
the gate as written specified an AP comparison on DoTA/DADA and **could not be executed**. It had
blocked R1's last gate across sessions 8 and 9 awaiting a decision. A plan step that cannot be run is
a defect in the plan, so fixing it is a plan change. The user approved the redesign (peak position
vs annotated `Time-of-collision`) before it was written.

**Change 2 (`d0bc237`) — that redesign corrected.** *Why:* checking DAD's construction before
downloading revealed two errors in change 1 — the AP test **is** computable (DAD's test split has
301 negatives) and DAD is **degenerate** for the mechanism test (collision-time IQR 0.16 s). Gate 3
now splits into 3a (DADA mechanism) and 3b (DAD AP), with R1 required to survive both. The user
approved this before it was written.

Also updated in the same two commits, as dependent text: R1's kill condition, and §11 point 1's
residual-risk note (mitigations (a) and (b) marked discharged against the measured gate 1/2 results).

**Nothing else in either document was touched.** In particular the two stale statements in
`NEW_PLAN.md` (§15.10) were deliberately left alone, because correcting them would be recording
progress rather than changing the plan.

---

## SESSION 9 — README CHANGED THIS SESSION: **NO.** NEW_PLAN CHANGED: **NO.**

**`README.md`:** unchanged. Nothing discovered this session changes the master plan. Session 9 built
a deliverable README §27 already called for and satisfied a §41 Phase 2 acceptance item; that is
execution, not a plan change.

**`NEW_PLAN.md`:** unchanged. The known gate-3 defect (§21.3 item 2) is still **proposed, not
applied** — it remains awaiting the user's sign-off, now across two sessions. Session 9's finding
that the §3.2 mechanism reproduces at full n=667 *supports* R1 but changes none of its wording.

**Also unchanged this session:** `eval/adapters.py`, `eval/reduction_study.py`, `eval/benchmark.py`,
`eval/calibration.py`, `eval/heldout_half.py`, and everything under `runs/`.



## SESSION 8 — README CHANGED THIS SESSION: **NO.**

The master plan did not change. Everything session 8 did executes `README.md` §41 Phase 5/6 and
`NEW_PLAN.md` R1 as already written; confirming a finding is progress, not a plan change, so it
belongs here rather than in the README (§12, first bullet).

**One genuine plan-level defect WAS found but deliberately NOT edited:** `NEW_PLAN.md` R1's gate 3
specifies an AP comparison that is impossible on positives-only data (§21.3 item 2). That is a defect
in `NEW_PLAN.md`, not `README.md`, and `NEW_PLAN.md` is still an unaccepted PROPOSAL — so the fix was
recorded here for the user's sign-off rather than applied unilaterally.

**Still open from earlier sessions:** the session-5 live offer to edit the `original_fps` gate-note
line in the README remains unanswered, now four sessions running.

## SESSION 7 — README CHANGED THIS SESSION: **NO.**

**Verified directly (`git status`, `git diff README.md`): zero changes to `README.md` this session.**
This session's output (`NEW_PLAN_v2.md`, §21) is, by design and by the user's own explicit rule, a
proposal document that does not touch the master plan until reviewed and accepted — nothing found
this session rose to the "master-plan-level discovery" bar the user's rule requires for a README edit.
The master plan remains exactly as session 6 left it.

---

## SESSION 6 — README CHANGED THIS SESSION: **YES, two lines. Not a plan change.**

**Exactly what changed:** README §41 Phase 4, task 3 and its matching acceptance criterion. Task 3
("Ego-involved vs non-ego reporting (B7)") was struck through and replaced with a note stating it is
**not computable on Nexar test-public** — verified directly against
`data/nexar/{test-public,train}/*/metadata.csv` column headers (`file_name, time_of_event,
time_of_alert, light_conditions, weather, scene, time_to_accident` — no ego field on either split).
The acceptance criterion line was edited to drop "ego/non-ego rows" from what `metrics.json` must
contain, with a pointer to the struck task explaining why.

**Why this was necessary, and why it is not a plan change:** same shape as session 2's mTTA strike —
a labelling gap in the dataset, not a code gap (`by_condition()` already handles any manifest field
generically). No phase was re-sequenced, no gate moved, no priority changed. This is the plan
acknowledging a data limitation, which is exactly the treatment mTTA already got one revision earlier.

**Deliberately NOT changed, still an open offer from session 5:** README §41 Phase 5's gate note,
reason 5, still says *"`original_fps: 4` contradicts `target_fps: 8.0` in the config, and is
unresolved."* This was resolved as evidence back in session 5 (§6.18) and is not re-verified or
re-touched this session — the offer to edit that line is **still live and still unanswered.** If
you're the next session: either edit it (it's a one-line factual update, not a plan change — the
resolution has been sitting confirmed for two sessions now) or ask the user directly.

---

## SESSION 5 (history — superseded by the entry above for what's current, but its own edit stands)

## SESSION 5 — README CHANGED THIS SESSION: **YES, one note. Not a plan change.**

**Exactly what changed:** one blockquote inserted in **§2 Project Status**, immediately below the
repository-inventory code block (+13 lines, the only diff to README.md this session). It is a dated
delta table: *"The inventory above is a snapshot of `b539d6e` and is retained as the dated audit
record. The working tree has since moved. Verified 2026-09-12"* — then five deleted files, the
1,334 → **977** line-count correction with the corrected pointers (`CNN_THRESH` → `:97`, decision gate
→ `:754`/`:849`), the 493 MB deletion, "zero tests" → **two, both passing**, "zero training code" →
the notebook is committed, and 16 → **81** tracked files. It closes with: *"The audit body is **not**
rewritten to match."*

**Why this was necessary, and why it is not a plan change:** progress.md §18 items 1–4 recorded that
README §2 describes a tree that no longer exists, and that every `enhanced.py:NNNN` reference is off
by the deleted blocks. That is a **factual** defect that actively misleads — but the roadmap, the
phases, the gates and the acceptance criteria are **untouched**. No priority moved, no criterion was
added or removed, no phase was re-sequenced. Per the README rule, this is a correction to the audit
record, not progress logging and not a roadmap edit.

**Deliberately NOT changed, and flagged to the user instead:** README §41 Phase 5's gate note,
reason 5, still says *"`original_fps: 4` contradicts `target_fps: 8.0` in the config, and is
unresolved."* Session 5 **resolved it** (§6.18) — but the gate itself is unchanged by that, so
editing the plan's reasoning was offered to the user rather than done unilaterally. **The user did
not answer before the session ended. This is a live, one-line offer for the next session.**

**Also changed (not the README):** `docs/HANDOFF.md` — superseded banner + three T3 corrections (§18).

---

## 16-OLD. Sessions 1–3 README modification status (history)

## README CHANGED THIS SESSION: **YES.**

**Session 3 (2026-09-12): README.md WAS changed in two passes — (A) 4 plan-level edits + stale-status
fixes on explicit user instruction ("okay then edit the plan"), and (B) 4 further Phase 5 corrections
forced by measurements taken later the same session.** `runs/falsification/RESULTS.md` was also
corrected. **Why changing the master plan was necessary in each case is stated per row.**

### Pass B — Phase 5 corrections forced by session-3 measurements (later, same session)

These were NOT progress logging. Each one removed a **factually false premise** from the plan that
would have sent the next session to do unnecessary or mis-sized work.

| Location | Change | Why the plan itself had to change |
|---|---|---|
| §41 Phase 5, gate note reason 3 | Rewritten. Was: "the published code does not implement the published model … reproducing their AP means reimplementing an undocumented module." Now: the predictor **is** implemented (V-JEPA2's own), the checkpoint **duplicates** it, `missing 0`, all 199 pairs bitwise identical, **no weights lost** — and the real gap is the discarded `predictor_output`. | The old text was **disproved** (§6.11). Leaving it would have sent the next session to rebuild a module that already exists. |
| §41 Phase 5, task 2 | Struck as **ANSWERED**, with both figures recorded and the distinction made explicit: compute-only 0.856 s/window vs **end-to-end ~97 s/clip ⇒ ≈18 h sweep**. | The task was a go/no-go gate and it has been answered. Recording only the compute figure had already caused one error (below). |
| §41 Phase 5, task 3 | "**Reimplement** the predictor + concat pathway … effort unscoped" → "**Consume `predictor_output`** … a forward-path change, not a reimplementation." | Same disproof. Also removes an "unscoped" item from a P0 phase. |
| §41 Phase 5, effort line | "plus an unscoped spike for task 3" → "plus ~18 h of unattended sweep compute (measured end-to-end)". | Scope is now known. |

> **⚠️ Self-correction inside the same session, recorded deliberately:** pass B initially wrote
> **"≈10.1 h"** for the sweep, taken from the compute-only measurement. That was wrong by ~1.8× because
> `badas_smoke.py` bypasses video IO. It was corrected to **≈18 h** after the 6-clip end-to-end run.
> **The next session should trust ≈18 h.**

### Pass A — the four plan-level edits + stale-status fixes (earlier, same session)

| # | Location | Change | Kind |
|---|---|---|---|
| 1 | §41 head | **New three-track block.** The governing rule now orders each track, not the whole document. Track A (model, phases 4→5→6), Track B (data/benchmark, 7→8), Track C (30 fleet calls) run concurrently from week 1. Track C given its own gate, since no phase owns it any more. | **Sequencing — the biggest change** |
| 2 | §41 Phase 5 | Retitled "BADAS-Open reference baseline". **Gate restated** from "within ~0.02 AP of published, do not proceed" to a judgement call on a documented deviation list — the old gate is unpassable by construction (disputed number, half the split, missing predictor, ambiguous reduction, fps contradiction) and its purpose is already served by the T3 + `eval/benchmark.py` double validation. **Predictor reimplementation added as task 3** (it previously had no home in any phase). U-B6 timing made the phase's explicit go/no-go (task 2). Discovery calls moved out to Track C. | **Gate + scope** |
| 3 | §41 Phase 6 | Reframed per **D2** — "beat the baselines" → "match Channel A, beat it as a system". Records that no CUDA device exists, that backbone fine-tuning is out of reach, and that the win is the multi-head split + calibration + Channel B suppression. AC split into a "match" clause and a fused-system FP/hour clause. **This closes the D2 question that had been awaiting the user's agreement since session 1.** | **Scope** |
| 4 | §41 Phases 7–8 | Promoted to **Track B, P0, week 1**. Phase 7's dependency corrected from "Phase 3" to "a consent form and a data agreement; NOT Phases 4–6". Phase 8 now starts on Phase 7's **first ~20 hours**, not its full 100. Rationale recorded: 0.90 h of Nexar negatives makes < 0.1 FP/hour undemonstrable, so Track B is the only route to the plan's own headline metric. | **Sequencing** |
| 5 | §2 Project Status | Two rows were factually false. "Training code in version control ❌ No" → ✅ (`data/ccd/Untitled0.ipynb`, `c9a6fda`). "Any test in the repository ❌ No" → 🟡 one (`tests/test_weights_load.py`). | Stale status |
| 6 | §41 Phase 0 | "Outstanding: commit the notebook to git" removed — it was committed at `c9a6fda`. U4 marked resolved by deletion (`ad45389`). ACs 1–2 marked `[x]` with the real paths (`data/ccd/`, `data/`, not `train/`). | Stale status |
| 7 | §42 rows 16–18 · §43 head | Matrix rows 16/17/18 → **P0** with track labels, for consistency with edit 1. §43's ten immediate actions marked superseded with a per-item status table (1–8 done or dropped) and a pointer to the three tracks. | Stale status |
| 8 | `runs/falsification/RESULTS.md` | **Not the README.** Its T3 section still read "**NOT YET RUN** … a formality rather than a question" — a committed artefact contradicting the headline result the README cites. Replaced with the measured figures, the self-falsification note, and the B4 closure. Header verdict updated. | Stale evidence |

**What was deliberately NOT touched:** §2's repository inventory (still lists deleted files and a
1,334-line `enhanced.py`; actual is 977), and every `enhanced.py:NNNN` line reference in §15/§19/§21/§47
(all stale — `CNN_THRESH` is at :97, the decision gate at :754/:849). These are **audit-body staleness,
not plan defects**, and correcting them is a mechanical pass worth doing in one go rather than
piecemeal. The fault-reporting remnant at `code/crash_detection_enhanced.py:934-942` also still fails
Phase 3's grep criterion — dead code (`fault_info` is always `None` at :898), but it fails the AC verbatim.

---

**Session 2: README.md WAS changed — 5 edits, all user-approved or factual corrections of the plan's
own reference data.** The user was asked and chose "AP only, edit README".

| Location | Change |
|---|---|
| §41 Phase 4, task 1 | **mTTA and time-to-detection struck** from the Phase 4 metric set, with the §6.7 evidence inline. Added "FP/hour **always with its denominator**". |
| §41 Phase 5, task 3 | Gate restated on **AP alone**; recorded that the two published sources disagree (model card 0.86 vs `config.json` 83.2) and that one must be named in `metrics.json`. |
| §41 Phase 5, acceptance criteria | "within ~0.02 AP of the **named** published figure … **AP only** — no mTTA or AUC clause." |
| §31 Evaluation Framework, mTTA row | "Never computed" → **"Not computable on Nexar test-public"**, with the reason. |
| §49 BADAS-Open reference row | Architecture corrected from "attentive probe (12 queries), 16 frames @ 256×256" to what the vendored GitHub source implements (temporal-attention + mean-pool + 3-layer MLP); added the sliding-window contract, the disputed-numbers note, and the predictor-module finding (§6.9 — no bug ID assigned, do not call it B10, that ID is taken) |

**Why these qualify as plan-level:** the first three change a **gate** — what Phase 5 must achieve to
pass. The last two correct **factual claims the README makes about an external dependency** that are
now known wrong from primary source; leaving them would send the next session to build against a
non-existent 256×256 / 12-query contract.

**Still NOT written into README, deliberately:** D2's Phase 6 reframing ("match Channel A, then beat
it as a system") — carried from session 1, still needs the user's explicit agreement.

---

**Was `README.md` changed during session 1? — YES.**

**Important:** these edits were made and committed (`420ef17`, pushed) **earlier in the session,
before the instruction to keep progress out of the README was given.** They were not made in order to
record progress for this handoff. No further README edits were made after that instruction, and none
were made while creating this file.

**Why the change was necessary (plan-level, qualifying under "an important project decision/discovery"):**
the README asserted in at least nine places that T3 was **unrun** and described it as "blocked on the
31.4 GB Nexar download", "a formality rather than a question", and listed the model's corpus-controlled
performance as "never computed". After T3, all of those statements were **factually false**, and the
README is the project's designated source of truth — README §1 itself records that the previous audit's
central error came from a conclusion drawn without accessing available evidence. Leaving it stale
risked exactly that failure again.

**What plan-level information changed:**

| Location | Change |
|---|---|
| Header update block (§ intro) | Added the T3 result and that the falsification suite is closed |
| §2 Project Status | "Falsification tests run: ❌ No" → ✅ Yes; added a corpus-controlled-measurement row |
| §5 Current Model | Replaced "Test 3 — not yet run" with the full result, distributions, and the self-falsification |
| §12.3 Falsification table | T3 row: "not yet run" → the measured figures |
| §13 What Has Been Proven | Added item 17 — chance-level performance on a corpus-controlled benchmark |
| §14 What Has NOT Been Proven | Item 1: struck the "how bad, not whether" caveat — now measured |
| §15 bug B4 | Added a 2026-09-11 status; closed as "diagnosed, not fixable"; resolution = retire the model |
| §16 Leakage table row 2 | "exploitation unproven" → **proven exploited**, with the 0.9977 → 0.5339 drop |
| §31 Evaluation Framework | AP and ROC-AUC "current value" cells now carry the real corpus-controlled numbers |
| §41 Phase 2 | Marked the gate satisfied; recorded the five dropped dead-work items with reasons (D3) |

**Not written into the README, recorded only here:** decision **D2**'s consequence — that Phase 6's
"beat BADAS-Open zero-shot" criterion is unachievable under a no-GPU constraint and should be reframed
to "match Channel A, then beat it as a system." **If the next session and the user agree that
reframing is settled project direction, it is a legitimate plan-level README edit — but ask first.**

**Going forward: progress, session history, experiment results and task status belong in this file,
not in `README.md`.**

---

## 17-S10. FINAL HANDOFF CHECK  ·  **SESSION 10, 2026-09-17. This supersedes §17-S9 below.**

| Question | Answer |
|---|---|
| 1. What are we building? | §1. **Not "crash detection"** — `README.md` §27's structured incident record. Detection is a commodity input (BADAS-Open is free, Apache-2.0). **Two of ~10 fields now work.** |
| 2. What does README say the plan is? | §41, via §15.10. Three parallel tracks: A model/measurement, B UK data, C customer calls. **Unchanged this session.** |
| 3. What does NEW_PLAN say? | §15.10. Week 1, Track A: R1 → R4 → R2 → R3 → fuse → calibrate, paired bootstrap mandatory. **Week 1 now complete except gate 3.** |
| 4. Which phase are we in? | Phase 4 COMPLETE, Phase 5 gate PASSED, inside `NEW_PLAN.md` Week 1. **Tracks B and C at zero across TEN sessions.** |
| 5. What did the PREVIOUS session complete? | Session 9: `eval/timing.py` built (uncommitted), §3.2 reproduced at full n=667, the 8 fps question answered. §21.4. |
| 6. What did THIS session complete? | **Six commits, all pushed.** Rescued sessions 8+9's handoff; committed `timing.py`; redesigned **and then corrected** R1 gate 3; measured CPU vs MPS; built calibration Tiers 1+2. §21.5. |
| 7. What evidence/results exist? | §21.5. MPS 0.871 s/window KEEPS UP at 1 Hz, CPU 1.917 s TOO SLOW. Tier 1 beta ECE **0.0809** [.062,.127] with AP unmoved at 0.8596. Tier 2 beta median 0.0596. DAD collision-time IQR **0.16 s** vs DADA **4.63 s**. All four regression guards reproduce exactly. |
| 8. What failed / negative results? | **A plan change I committed was wrong and I corrected it the same session** (§21.5 item 2) — DAD has 301 negatives and is degenerate for the mechanism test. **The deployable ECE (0.0809) is ~60% worse than the figure the project has been quoting (0.0503).** CPU is not real-time. One of my own self-checks was wrong and was **replaced with a stronger one, not loosened** (item 8). |
| 9. What is broken or uncertain? | **Gate 3 is blocked on DATA ACQUISITION** — DADA's 53 GB archive vs ~30 GB free, DAD behind a Google Form. The time-base compatibility of DADA's annotation is **unverified** and would corrupt 3a silently (item 3). The 0.0498-vs-0.0503 discrepancy is unsettled (item 11). Studio free disk still unknown. |
| 10. What decisions are made? | §11 — D1–D35, **plus D36–D39 (session 10)**. D39 settles gate 3's design (3a DADA + 3b DAD) and, unlike the rest of R1–R9, **is** a decision of record — the user signed it off. |
| 11. What files changed? | `eval/calibration.py` (+292), `eval/timing.py` (new, committed), `scripts/hw_bench.py` (new), `NEW_PLAN.md` (+108), `runs/calibration/*` (new), `progress.md`. **All committed and pushed. HEAD `37f02b0`.** |
| 12. What is the exact next action? | **§13 — check whether DADA-2000's Google Drive mirror allows per-clip download of the 221 test clips.** That one fact decides whether gate 3a runs overnight or is disk-blocked. |
| 13. What must NOT be redone? | §12, especially the **new session-10 block at its top** — above all: do not run the mechanism test on DAD, do not repeat "the external sets have no negatives", do not re-measure CPU/MPS, do not quote 0.0503 as deployable, do not download DoTA. |
| 14. Did either plan change, and why? | §16 — **README NO. NEW_PLAN YES, twice, both user-approved**: gate 3 was unexecutable as written, then the fix itself needed correcting. Both are plan defects, not progress. |
| 15. BLOCKED vs UNKNOWN? | **BLOCKED:** gate 3, on data acquisition only (DADA disk/mirror; DAD's Google Form needs the *user*). **UNKNOWN, not blocked:** whether the DADA Drive mirror is per-file; Studio's real free disk; DADA's annotation time base; DAD's and DADA's licence terms. |

### What session 10 did, precisely

1. **Full project recovery from zero memory** — read `NEW_PLAN.md` in full, `progress.md`'s banners
   and §§12/13/21.3/21.4/15.9/17-S9, `README.md` §27/§41/§49, then **verified the claims against the
   repo** (667 traces, git state, all guards re-run) rather than trusting the handoff.
2. **Committed and pushed sessions 8+9's stranded work** after the user's explicit go-ahead, running
   all three guards first as §13 required.
3. **Redesigned R1's gate 3** on the user's sign-off, then **found two errors in that redesign** by
   checking DAD's actual construction before downloading it, and corrected it — also on sign-off.
4. **Measured CPU vs MPS**, reusing `badas_smoke.py` rather than writing new timing code, and
   recorded the run-order spread rather than reporting a single flattering number.
5. **Built calibration Tiers 1 and 2**, extending the existing module, parameterising the score
   source so R1 promotion is a re-run, and **verifying Tier 3 stayed bit-identical**.
6. **Replaced one of its own self-checks** after it failed to demonstrate the property it claimed to.
7. **Did not touch `progress.md` during the work** — only now, on the handoff prompt.

### SESSION END STATE

**Nothing is running. Nothing is partially implemented. Everything is committed and pushed.**
`git status` clean, `git log origin/main..HEAD` empty, HEAD `37f02b0` on `main`.

The session ended on this handoff request. The one thing left mid-flight is **gate 3's data
acquisition**, which stopped at a decision point rather than a half-finished state: the DADA Drive
mirror has not been checked, and DAD's Google Form has not been submitted (the user must do that).
**No download was started**, deliberately — neither dataset's terms have been confirmed, and this
project has already excluded BDD100K on licence grounds.

### MODEL / EFFORT HANDOFF

- **Immediate next action (check the DADA mirror, resolve acquisition): sonnet · medium.** Mechanics
  and a web/disk check, fully specified in §13.
- **🔴 Switch to opus · high BEFORE running or interpreting gate 3.** `NEW_PLAN.md` R1 marks it
  opus/high, and session 10 is the evidence for why: **two successive attempts to specify this gate
  were both wrong**, and each was only caught by checking a primary source instead of a summary. The
  leakage and selection reasoning is the most consequential judgment left in the project.
- **Then R1's promotion decision: opus · high.** It invalidates or revalidates every committed
  number.
- **Tier-4 calibration, R4/R2 implementation, the 0.0498 fix: sonnet · medium/low.**
- **Switch before next task? NO** for the acquisition check; **YES** before gate 3 itself.

---

## 17-S9. FINAL HANDOFF CHECK  ·  **SESSION 9, 2026-09-16 16:45 IST (history — superseded by §17-S10).**

| Question | Answer |
|---|---|
| 1. What are we building? | §1. **Not "crash detection"** — `README.md` §27's structured incident record. Detection is a commodity input (BADAS-Open is free, Apache-2.0). |
| 2. What does README say the plan is? | §41, via §15.9. Three parallel tracks: A model/measurement, B UK data, C customer calls. **Unchanged this session (§16).** |
| 3. What does NEW_PLAN say? | §15.9. Week 1, Track A: R1 → R4 → R2 → R3 → fuse → calibrate, paired bootstrap mandatory. **Still a PROPOSAL. Unchanged this session.** |
| 4. Which phase are we in? | Phase 4 COMPLETE, Phase 5 gate PASSED, inside `NEW_PLAN.md` Week 1. **Tracks B and C at zero across NINE sessions.** |
| 5. What did the PREVIOUS session complete? | Session 8: R1 gate 2 (`eval/heldout_half.py`), commit `252d282`, found the gate-3 plan defect. §21.3. |
| 6. What did THIS session complete? | **`eval/timing.py`** — incident-record field 2. §21.4. Plus: §3.2 reproduced at full n=667; the 8 fps question answered; two new upstream findings (items 5 and 7). |
| 7. What evidence/results exist? | §21.4. Gate 0.9733 @ recall 0.80; 92.3 FP/h over 0.899 h; 351 records; positives peak 0.975 vs negatives 0.802. All three regression guards reproduce exactly. |
| 8. What failed / negative results? | **92.3 FP/hour is bad** and is the honest number (§21.4 item 3). One of session 9's own self-checks was wrong and was **replaced with a stronger one, not loosened** (item 8). Upstream silently discards the final window (item 7). |
| 9. What is broken or uncertain? | **R1 gate 3's design is still broken and still unsigned-off** (§21.3 item 2) — now across two sessions. Calibration Tier 1 unwritten. No hardware timing ever measured on the user's machines. |
| 10. What decisions are made? | §11 — D1–D31, **plus D32–D35 (session 9)**. `NEW_PLAN.md` R1–R9 remains a PROPOSAL. |
| 11. What files changed? | `eval/timing.py` **created** (untracked). `progress.md` updated. **NOTHING COMMITTED.** HEAD is still `252d282`. |
| 12. What is the exact next action? | **§13 — commit and push the two outstanding files, after asking the user.** They said "wait" twice in session 9. |
| 13. What must NOT be redone? | §12, especially the **new session-9 block at its top** — above all, **do not discard `progress.md`**, do not rebuild `timing.py`, do not re-open 8 fps, do not switch `adapters.py`. |
| 14. Did either plan change, and why? | §16 — **README NO, NEW_PLAN NO.** Both left unchanged; session 9 was execution, not planning. |
| 15. BLOCKED vs UNKNOWN? | **BLOCKED:** R1 gate 3 (broken design awaiting sign-off + videos not downloaded + DoTA's ~55 GB exceeds both machines' free disk). **UNKNOWN, not blocked:** Studio's real free disk (the ~40 GB figure is from session 1 and stale); DAD's download size/terms; CPU-vs-MPS throughput. |

### What session 9 did, precisely

1. **Full project recovery from zero memory** — read `README.md` §27/§41, `NEW_PLAN.md` in full,
   `progress.md` §§11/12/13/21.x, then verified the claims against the repo rather than trusting them.
2. **Caught that session 8's handoff was never committed** despite §17-S8 claiming a clean tree.
3. **Traced the index→seconds mapping to vendored source** instead of assuming it, and cross-checked
   it against `benchmark.durations()` (±0.0623 s on all 667 clips).
4. **Wrote `eval/timing.py`** (331 lines) — two-stage threshold, §27-shaped JSON, 7 self-checks.
5. **Replaced one of its own self-checks** after it failed for the wrong reason — made it stronger.
6. **Ran all three regression guards.** All reproduce exactly. No committed number moved.
7. **Reproduced §3.2's peak-position mechanism at full n=667** through an independent code path — a
   stop condition that did not trigger, and new supporting evidence for R1.
8. **Answered the user's 8 fps question** from primary sources (upstream defaults + checkpoint config).
9. **Did NOT commit** — the user was asked twice and said "wait" both times.

### SESSION END STATE

**Nothing is running. Nothing is partially implemented.** `eval/timing.py` is complete and working.
The session ended on this handoff request, with **the commit + push still pending the user's
go-ahead** — that is the only unfinished item, and it is a decision, not work. Two plan files exist
outside the repo (`~/.claude/plans/`); they are scratch, **this file is authoritative**.

### MODEL / EFFORT HANDOFF

- **Recommended model for the immediate next action (commit + push): sonnet · effort low.** It is
  mechanical git work against a fully specified §13.
- **Then `scripts/hw_bench.py`: sonnet · medium.** Well-specified implementation.
- **🔴 Switch to opus · high BEFORE touching R1 gate 3 (action 2).** The leakage and
  selection-risk reasoning is the single most consequential judgment left in the plan, and
  `NEW_PLAN.md` R1 itself marks it "opus · high".
- **Switch before next task? NO** for the commit; **YES** before gate 3.

---

## 17-S8. FINAL HANDOFF CHECK  ·  **SESSION 8, 2026-09-16 (history — superseded by §17-S9 above).**

| Question | Answer |
|---|---|
| 1. What are we building? | §1. **Not "crash detection"** — `README.md` §27's structured incident record. Detection is a commodity input (§21.3 item 9). |
| 2. What does README say the plan is? | §2 (pointer). Three parallel tracks: A model/measurement, B UK data, C customer calls. **Unchanged this session (§16).** |
| 3. Which phase are we in? | Phase 4 COMPLETE, Phase 5 gate PASSED. Working inside `NEW_PLAN.md` Week 1 (Track A). Tracks B and C remain at **zero across eight sessions**. |
| 4. What has actually been completed? | §4 (s1–s6), §21.1 (s7), **§21.3 (s8)**. Session 8 wrote `eval/heldout_half.py` and made the first commit since `fd5d97f`. |
| 5. What evidence/results do we have? | Top banner (gate 1 + gate 2 numbers), §21.3, §6. BADAS baseline AP 0.8349 / AUC 0.8498. Old model retired at AUC 0.5339. |
| 6. What is broken or uncertain? | **R1 gate 3's design is broken (§21.3 item 2).** Calibration Tier 1 unwritten. No hardware timing ever measured on the user's machines. No Docker/lockfile/Makefile. |
| 7. What decisions are made? | §11 — D1–D28, **plus D29–D31 (session 8)**. `NEW_PLAN.md` R1–R9 is still a PROPOSAL. |
| 8. What files changed? | Commit `252d282`: 678 files added (667 `.npz` + `runs/baselines2/` metrics + 3 eval scripts). **Working tree clean, pushed, in sync with origin/main.** |
| 9. What is the exact next action? | **§13 — build `eval/timing.py` for `t_start`/`t_peak`/`t_end` from the committed traces.** Zero compute. |
| 10. What must NOT be redone? | §12, especially the **new session-8 block at its top** (the sweep is finished; don't switch the reduction; don't run gate 3 as written; don't revert the null control). |
| 11. Did README change, and why? | §16 — **NO.** Master plan unchanged. One `NEW_PLAN.md` defect found and left for user sign-off. |
| 12. What failed / negative results? | The `heldout_half` null control failed on first run and was **correctly** failing (§21.3 item 8). Gate 3 found to be uncomputable as specified. `eval/calibration.py`'s documented invocation was broken. |
| 13. UNKNOWN vs BLOCKED? | **BLOCKED:** R1 gate 3 (broken design + videos not downloaded + DoTA exceeds disk). **UNKNOWN, not blocked:** Studio's real free disk; DAD's actual download size and access terms; CPU-vs-MPS throughput. |

### What session 8 did, precisely

1. **Recovered the lost session's result.** A previous session finished R1 gate 1 but exited before
   writing anything; the result existed nowhere on disk or in git. **Re-ran `eval/reduction_study.py`
   and reproduced it exactly** rather than trusting the user's pasted numbers.
2. **Verified the sweep finished** — 667/667 scores, 667 `.npz`, PID gone.
3. **Wrote `eval/heldout_half.py`** (180 lines) and ran R1 gate 2. **PASSED.**
4. **Found and fixed a bias in its own null control** before trusting any result from it.
5. **Ran all three existing self-checks** as regression guards: `eval/benchmark.py` (T3 reproduced
   exactly), `eval.calibration --self-check`, `eval.reduction_study`. All pass.
6. **Fixed `eval/calibration.py`'s docstring** — it documented an invocation that raises.
7. **Committed and pushed `252d282`** — user explicitly approved the push.
8. **Verified the gate-3 annotation sets are positives-only** and measured the ego/not-ego and
   collision-timing breakdowns (§21.3 items 2–3).
9. **Corrected the product framing** after the user pushed back (§21.3 item 9).

### NEXT 3–5 ACTIONS after §13

1. **`eval/timing.py`** — `t_start`/`t_peak`/`t_end` (§13). Zero compute. Second incident-record field.
2. **Get user sign-off on the §21.3 item 2 gate-3 redesign**, then download **DAD only** (~4.5 h on
   the Air, no Studio) and run the peak-position test. This is the test that decides R1.
3. **`scripts/hw_bench.py`** — one window on M4 Air CPU vs MPS. `NEW_PLAN.md` §7.3 forbids any
   "real-time" or "CPU-capable" language until this exists. Week 1 item, still unwritten.
4. **Calibration Tier 1** — the deployable stratified calibration/eval split (`NEW_PLAN.md` §6).
5. **Track B and Track C.** Buy a dashcam; send 10 fleet messages. Both are P0 in `README.md` §41,
   both need no code, both are at zero after eight sessions. `NEW_PLAN.md` §9 sets a kill condition:
   **< 3 replies by end of week 2 → formally close Track C and stop listing it.**

### SESSION END STATE

**Nothing is running. Nothing is partially done.** Working tree clean, `main` in sync with
`origin/main` at `252d282`. The session ended on a handoff request, not mid-task. `eval/timing.py`
(§13) is **planned but not started — zero lines written.** A plan file also exists outside the repo at
`~/.claude/plans/yes-it-was-not-splendid-llama.md`; it is scratch, not authoritative — **this file is.**

---

## 17. FINAL HANDOFF CHECK  ·  **SESSION 7 (history — superseded by §17-S8 above)**

| Question | Answered where |
|---|---|
| 1. What are we building? | §1 · §3 header |
| 2. What does README.md say the plan is? | §2 (pointer; not copied). Three tracks: §3 header. **Unchanged this session (§16).** |
| 3. Which phase are we in? | §3 — **Phase 4 COMPLETE. Phase 5 gate PASSED. Master plan unchanged. In PARALLEL, an unreviewed revised proposal (`NEW_PLAN_v2.md`, §21) exists but is NOT part of the active plan until accepted.** |
| 4. What has actually been completed? | §4 (…4.12 s5 · 4.13 s6), §7/§7.1/§7.2. **Session 7 added NO code/experiments — see §21 for its one deliverable, a planning document.** |
| 5. What evidence/results do we have? | §6 + sessions 1–6 evidence, unchanged. **Session 7's contribution was verification, not new evidence**: confirmed `eval/calibration.py` doesn't exist (F2 table uncommitted), confirmed train/test-private are metadata-only (re-confirms F1), confirmed the sweep's progress (222→261/667) — all folded into §21. |
| 6. What is broken or uncertain? | §5, §10, §18, **§21 (F3 still provisional, calibration script missing, hardware timing unconfirmed on real machines — all named explicitly in the new plan rather than left implicit)** |
| 7. What decisions are already made? | §11 — D1–D24 (s1–s6) **plus D25–D28 (session 7: one plan document; paired bootstrap mandatory; no perf claim without hardware measurement; calibration demoted to infrastructure)**. `NEW_PLAN.md`'s ranked R1–R9 plan is a **proposal**, not a decision — it becomes D29+ only on acceptance. |
| 8. What files changed? | **NONE in the tracked repo.** Two new untracked files: `NEW_PLAN.md` (session 6) and `NEW_PLAN_v2.md` (session 7, new this session) — see §21. |
| 9. What is the exact next action? | **§13 — check whether the sweep hit 667/667; if so, re-run §21.1's reduction comparison on all 667 with a PAIRED bootstrap and decide whether the +0.033 AP last-window finding survives.** Separately, get an accept/reject decision from the user on `NEW_PLAN.md` before executing any of it. |
| 10. What must NOT be redone? | §12, **plus: do not re-derive the "10 points" the user referenced — it was searched for and not found; ask the user directly rather than guessing again (§21)** |
| 11. Did README change, and why? | §16 — **NO this session.** Still two lines from session 6 (ego-row strike); the session-5 `original_fps` live offer remains unanswered, now three sessions running |
| 12. What failed / what is a negative result? | Unchanged from session 6 (§6.1, §6.13, §6.16, ECE 0.3286). **New this session: `NEW_PLAN_v2.md`'s own uncertainty audit (its §12) reclassifies the beta-vs-Platt calibration "win" as NOT statistically meaningful** — a negative result about a prior positive-sounding claim. |
| 13. What is genuinely UNKNOWN vs BLOCKED? | Unchanged from session 6, **plus: the "10 points" artifact's existence/location is now an explicit open question (§21)**, and Track C's zero-progress streak is now flagged by `NEW_PLAN_v2.md` itself as evidence of an unnamed, undiagnosed blocker rather than a scheduling oversight. |

### What a brand-new Claude should read, in order

1. **The ⏳ running-job block at the very top of this file.** Same PID 75682 job as sessions 6–7,
   now further along (261/667 at session 7's end) — check current progress, don't trust that number.
2. **The SESSION 7 SUMMARY block**, immediately below the running-job banner.
3. **README.md §41** — the roadmap and the three-track block, unchanged. Then §30–§31 for metric rules.
4. **`NEW_PLAN_v2.md` in full** if continuing the planning thread — §21 here is a pointer/summary only.
5. **This file: §13 (now with its session-7 step 0.5) → §14 → §21.**
6. **§12** before touching anything.
7. `runs/falsification/` and `runs/baselines/` — committed evidence. Never delete or "tidy" either.

### Honest statement of what is NOT finished

- **The second BADAS sweep (`runs/baselines2/`) is still in flight**, now 261/667 (~39%), started
  session 6, still running unattended through session 7. Not required for any passed gate.
- **`NEW_PLAN_v2.md` is a complete document but is UNREVIEWED AND UNACCEPTED.** Nothing in it has been
  started. Treating it as "the plan" without checking with the user first would be a mistake.
- **`eval/calibration.py` and `scripts/hw_bench.py`, both specified as the correct first steps if
  `NEW_PLAN_v2.md` is accepted, do not exist.** Zero lines written.
- **Phase 5 task 3 and Tracks B/C are exactly where session 6 left them** — no session-7 progress,
  because session 7 was planning-only. Track B/C now at zero across **seven** sessions.
- **The "10 points" prior artifact referenced by the user's brief was never located.** This is now a
  standing open question that should be asked of the user directly, not silently re-guessed again.
- **README's session-5 live offer** (edit the `original_fps` gate-note line) is still open, unanswered,
  now three sessions running.

---

## 17-OLD. Session 2's handoff check (history)

| Question | Answered where |
|---|---|
| 1. What are we building? | §1 |
| 2. What does README.md say the plan is? | §2 (pointer + relevant sections; not copied) |
| 3. Which phase are we in? | §3 — Phase 2 complete, Phase 4 next |
| 4. What has actually been completed? | §4, §7 |
| 5. What evidence/results do we have? | §6 (exact numbers preserved, negatives included) |
| 6. What is broken or uncertain? | §5, §10 |
| 7. What decisions are already made? | §11 (D1–D8) |
| 8. What files changed? | §7, §8 |
| 9. What is the exact next action? | §13 |
| 10. What must NOT be redone? | §12 |
| 11. Does README need a plan-level change? | §16 — one candidate (D2's Phase 6 reframing); **ask the user first** |

---

## SESSION TIMELINE

```text
2026-09-16 (SESSION 9, ~16:20-16:45 IST) — eval/timing.py; incident record field 2. NO COMMIT.
- Full recovery from zero memory. Read README SS27/S41, NEW_PLAN.md in full, progress.md
  SS11/12/13/17-S8/21.x. Verified claims against the repo rather than trusting the handoff.
- FOUND: session 8's handoff was NEVER COMMITTED. progress.md modified 16:21, after its own
  13:05 commit 252d282. S17-S8's "working tree clean, in sync" is false about progress.md.
- Traced the index->seconds mapping to vendor source instead of assuming: upstream resamples to
  target_fps BEFORE windowing, so trace index is in the RESAMPLED timeline; t = index/target_fps.
  A window (start, start+16) files its prediction against end_idx, the first UNSEEN frame.
  Cross-checked against benchmark.durations() (the T3 decode): agree to 0.0623s = under one
  8fps frame, on all 667 clips. Third confirmation: upstream badas_loader.py:112 does i/fps.
- FOUND: upstream SILENTLY DISCARDS the final window. The guard
  `if 0 <= target_frame < total_frames` fails when target == total_frames at stride 1. An
  81-frame clip runs 66 windows, keeps 65. So R1's "last window" is the SECOND-to-last, and
  t_end is not "end of clip". Harmless to R1 (applied uniformly) but the naming misleads.
- Wrote eval/timing.py (331 lines, UNTRACKED). Two-stage threshold, user-chosen:
  stage 1 global gate derived from the PR curve at a stated recall (README S41 Phase 2 task 5,
  never hard-coded per B5); stage 2 per-clip half-prominence above the clip's own median.
  Reports BOTH reductions (shipped=nanmax, candidate=last_window) and promotes neither (D29/D33).
  Unproducible fields OMITTED not null-filled (D34).
- TRAP AVOIDED BY ASSERTION: reduction_study.load_traces STRIPS the 16 leading NaNs. Reusing it
  for timing puts every timestamp EXACTLY 2s early, silently. timing.py has its own
  load_traces_abs(); self-check 4 asserts the delta (5.000s with offset, 3.000s without).
  reduction_study.load_traces left UNCHANGED - heldout_half and reduction_study depend on it.
- One of my own self-checks was WRONG: it grepped the source for "0.80" and failed on the
  DOCSTRING sentence explaining why 0.80 is forbidden. It tested prose, not behaviour. Replaced
  with a behavioural check (more recall must move the threshold DOWN and cost more FP).
  Made STRONGER, not loosened - same discipline as session 8's D30.
- RESULTS: gate @ recall 0.80 -> threshold 0.9733, recall 0.8024, precision 0.7635,
  83 FP over 0.899 h = 92.3 FP/HOUR (bad, and honest; 0.899h is 54 minutes).
  351 records emitted: 268 positive + 83 negative (the gate's false alarms).
  Gate is data-derived: recall 0.70/0.80/0.95 -> 0.9830/0.9733/0.8098, FP 60/83/194.
- STOP CONDITION DID NOT TRIGGER, and gave new evidence: S3.2's mechanism REPRODUCED AT FULL
  n=667 through an independent code path. Positives peak at 0.975 of clip vs negatives 0.802;
  73.7% vs 38.4% in the final 10%. Session 7 had 0.984/0.782 and 83.0%/42.5% at n=268. Same
  direction, mild softening at full n. Supports R1's mechanism; does NOT replace gate 3.
- Regression guards ALL PASS unchanged: eval/benchmark.py (T3 AUC 0.5339 / AP 0.5218 /
  332,325,2,8), eval.calibration --self-check, eval.reduction_study (max 0.8349,
  last_window 0.8905 +0.0556 [+0.0263,+0.0876]).
- USER ASKED: "do we need to change the 8fps?" ANSWER: NO. Upstream's own default (badas_loader
  lines 47/89/121 + their README + example); 16 frames @ 8fps = the 2.0s window the head was
  trained on (data_root "balanced_dataset_2s"); changing it invalidates every committed number;
  a mismatch here would repeat this project's own B1 bug. Other rates belong in NEW_PLAN R2 as
  an ENSEMBLE, never a replacement. D35. Third time this question has been closed.
- NOTHING COMMITTED. User asked to commit+push, then said "wait" twice. HEAD still 252d282,
  0 unpushed. eval/adapters.py:129 still np.nanmax - R1 deliberately unpromoted (D29).
- README.md NOT changed. NEW_PLAN.md NOT changed. The gate-3 defect redesign is still only
  PROPOSED, now awaiting user sign-off across two sessions.

2026-09-16 (SESSION 8, ~09:00-14:15 IST) — R1 gate 2; first commit since fd5d97f
- Recovered a LOST session's work: a prior session finished R1 gate 1 (full-667 paired bootstrap)
  but exited via a misclicked `exit` before writing progress.md. The result was in no file and no
  commit. Re-ran eval/reduction_study.py and reproduced it EXACTLY rather than trusting the paste:
  last_window AP 0.8905, dAP +0.0556, paired CI [+0.0263, +0.0876].
  Cross-check: max on sweep 2 = 0.8349 = committed sweep 1 AP exactly -> re-confirms D19.
- Verified the second sweep finished: 667/667 scores.jsonl, 667 .npz, PID 75682 gone.
  Found eval/reduction_study.py and eval/calibration.py untracked -- the lost session's output.
- Read NEW_PLAN.md in full, README.md SS41-45 + S27, progress.md SS11/12/13/17/21.x.
- Wrote eval/heldout_half.py (180 lines) for R1 gate 2. Reuses reduction_study's bootstrap and
  reductions rather than re-implementing them (a second copy of the stats is where a fake result
  would enter -- NEW_PLAN.md R5's own warning).
- Its null control FAILED on first run (median dAP -0.0252). Diagnosed instead of loosening the
  threshold: the noise-trace null gave nanmax one fixed lucky draw (AP 0.5630 vs 0.5007 chance)
  that persists across every split. Fixed by permuting labels per split -> +0.0025. D30.
- R1 GATE 2 PASSED. Primary split: A selects last_window, held-out B dAP +0.0598 CI
  [+0.0169, +0.1085]. 1000 splits: A picks last_window 52.4% / max_x_last 47.6%, never a loser;
  held-out dAP median +0.0551, 5-95% [+0.0291, +0.0796], POSITIVE IN 100% OF SPLITS.
  Held-out median (+0.0551) ~= full-667 (+0.0556): no winner's-curse shrinkage.
- Regression guards all pass: eval/benchmark.py (T3 reproduced: AUC 0.5339 / AP 0.5218 /
  332,325,2,8), eval.calibration --self-check, eval.reduction_study.
- Fixed eval/calibration.py docstring: documented `python eval/calibration.py` raises
  ModuleNotFoundError; correct form is `python -m eval.calibration`.
- Committed 252d282 (678 files: 667 .npz + runs/baselines2 metrics + 3 eval scripts; sweep.log
  force-added per D24) and PUSHED to origin/main, user-approved. First commit since fd5d97f.
- FOUND A PLAN DEFECT: R1 gate 3 cannot be run as NEW_PLAN.md words it. DAD/DADA/DoTA consensus
  sets are POSITIVES-ONLY (165/221/598, all Collision or Near-collision), so the specified AP
  comparison is undefined. Proposed redesign: peak-position vs annotated Time-of-collision.
  Left for user sign-off, NOT applied -- NEW_PLAN.md is still an unaccepted proposal. S21.3 item 2.
- Measured, answering a user challenge ("when crash happen car stops"): DAD is 91% pos-not-ego
  (camera car witnessed, kept driving), Time-of-collision median 3.0s max 3.7s -- footage continues
  past every collision. Contrast Nexar: time_of_event median 20.0s vs clip 9.93s, event is OUTSIDE
  the clip. That contrast is exactly what gate 3 needs.
- Corrected DAD compute: 165 clips x ~97 s = ~4.5 h, one overnight Air run. The Studio is NOT
  needed for DAD (an earlier statement this session said it was; wrong, corrected). Disk: 33 GB
  free on the Air, Studio's ~40 GB figure is from session 1 and stale -- DoTA (~55 GB) fits neither.
- User corrected the product framing: this session had called it "software that flags crashes".
  README S27 says "The product is not crash detection... it is the structured incident record."
  Against S27's ~10 fields exactly ONE works, and it is the commodity one. The 667 committed traces
  newly unblock t_start/t_peak/t_end -> that became the S13 next action.
- README.md NOT modified (S16). progress.md updated once, at session end, on the handoff prompt.


2026-09-12 (SESSION 5, ~19:50-23:15 IST)
- Deep recovery: read README SS1-2/41-44 + all of progress.md; verified claims against the repo.
  Result: progress.md was ONE SESSION STALE. A session 4 had already launched the BADAS sweep at
  13:48 and committed e488a05 / 4204bf2 / cc13966 without updating the handoff.
- Verified live state: benchmark.py self-check PASSES (reproduces T3 exactly); both tests PASS;
  sweep running as PID 67741, 135/667, ~167 s/clip => ~31 h, not the planned ~18 h.
- FOUND A REAL DEFECT: balanced_ids() returned sorted(labels), and every Nexar positive id sorts
  below every negative -> all 140 scored clips were positives -> AP/AUC/FP-h undefined for ~9 more
  hours. Fixed by interleaving; killed 67741, relaunched as 88180.
  Result: "resuming: 140/667" - nothing re-scored. First negative landed within 2 minutes.
- Wrote eval/peek.py to read an in-flight sweep without touching it.
- Resolved U-B7 (original_fps 4 vs target_fps 8.0). Result: NOT a contradiction - token rate vs
  video rate. 16 frames @ 8 fps = 2.0 s -> tubelet 2 -> 8 tokens -> 4 tokens/s. Sweep config is
  faithful to training; no B1-style mismatch. Evidence: scripts/badas_fps_probe.py.
- Resolved U6. Result: 12/12 weight arrays IDENTICAL - crash_model_cpu/ IS the shipped artefact.
  Two runs in evidence, not three. Evidence: scripts/u6_compare_weights.py. Phase 0 item closed.
- Wrote runs/legacy-colab/ (training_log.txt, metrics_val.txt extracted verbatim from the committed
  notebook, plus a README on the artefact/metric disconnect). Last Phase 0 acceptance item closed.
- Deleted the unreachable fault-reporting block in crash_detection_enhanced.py (block, parameter,
  both call sites, both dead locals). Result: Phase 3's own grep criterion now passes; tests still pass.
- Fixed doc contradictions 1-5: dated delta note in README S2; superseded banner + three T3
  corrections in docs/HANDOFF.md.
- 23:15 handoff written. Sweep at 230/667, partial AP 0.9274 / AUC 0.8988 (PARTIAL, not a result).
  Nothing committed.

2026-09-11
- Ran docs/HANDOFF.md §1 verification block.
  Result: no discrepancies. 667/667 Nexar clips; tree clean; dead-code deletion confirmed finished.
- Built ~/envs/crashdet (Python 3.11, TF 2.19.1, Keras 3.15.1, cv2 4.10.0, numpy 2.1.3).
  Result: both shipped artefacts load. Fixes the R2 blocker that stopped the previous session.
- Wrote scripts/t3_corpus_control.py; smoke-tested on 4 clips.
  Result: all four scored ~0.999 regardless of class. Investigated before trusting it.
- Numerical sanity investigation (features, activations, all weight arrays).
  Result: no NaN/inf anywhere; the RuntimeWarnings are cosmetic arm64 BLAS noise.
- Ran the full T3 sweep: 667 clips, 527.6 s.
  Result: ROC-AUC 0.5339, AP 0.5218, FPR 0.9760 @ threshold 0.80. Chance-level.
- Self-falsified the T3 harness against the three local videos.
  Result: safe.mp4 -> 0.7914 and crash1.mov -> 0.9998, reproducing T124 EXACTLY.
  Decision: T3 accepted as sound; the collapse is the model's, not the harness's.
- Measured negative footage duration: 0.90 hours across 333 clips.
  Decision: FP/hour must always be quoted with its denominator; UK-HN-500 is the real fix.
- Read all 3,191 lines of README.md.
- Updated README.md (9 locations) with the T3 result; dropped 5 dead-work tasks with reasons.
- Rewrote requirements.txt (TF 2.19.1/Keras 3.15.1; dropped ultralytics/torch/scipy),
  added .python-version, added tests/test_weights_load.py.
  Result: test PASSES — 578,689 params, FE output (2,1280), head on zeros -> 0.388557.
- Committed 420ef17 and pushed c995fd7..420ef17 (all 6 commits) to origin/main, user-approved.
- User confirmed: startup goal retained; NO GPU budget (local M1 only).
  Decision D2: no training; BADAS-Open zero-shot becomes Channel A; differentiate via
  Channel B + calibrated fusion (CPU-only) and via evaluation rigour + UK data.
- Created progress.md (this file). Left untracked/uncommitted deliberately.
- User disclosed access to an M4 Max machine at college (installs allowed, ~40 GB free).
  Decision D2 REVISED: probe/head training on cached frozen features is now feasible;
  full ViT-L backbone fine-tuning remains out of reach (Apple Silicon has no CUDA).
  Consequence: the Nexar train split (25.5 GB) is now worth downloading, and README Phase 6
  is partially revived. Added as step 5 of §14. Access pattern unspecified -> feature caching
  must checkpoint per clip so it is resumable.
- Open question at session end: settle the four BADAS-Open unknowns first, or start Phase 4?
  Recommendation recorded in §13: unknowns first. U-B4 (MPS support) now also gates D2.

2026-09-11 (session 2)
- Verified state against progress.md: HEAD 420ef17, only progress.md untracked, env loads
  TF 2.19.1 / Keras 3.15.1. No discrepancies.
- Created ~/envs/badas (Python 3.11 + huggingface_hub). CLI is `hf`; `huggingface-cli` is deprecated.
- Attempted the §13 download. FAILED: "Access denied. This repository requires approval."
  Confirmed all files 401 unauthenticated, including README.md. HF API: gated="auto",
  extra_gated_fields = name / title / Company Email / Company / topic.
  -> NEW BLOCKER B-GATE. Requires user action; Claude cannot submit the form.
- Read the repo file list from the public HF API `siblings` field (public even while files are gated).
  Result: prior session's U-B1 list was WRONG. No preprocessing.py, no model.safetensors.
  Actual: badas_loader.py, config.json, requirements.txt, weights/badas_open.pth, and a src/ tree
  (core/base.py, models/vjepa.py, train/video_training.py, utils/video.py, utils/sliding_window.py).
  U-B3's answer lives in src/utils/{video,sliding_window}.py. Corrected §10.
- Investigated metadata to build mTTA. Found time_of_event (median 20.0 s) exceeds clip duration
  (median 9.93 s) for ALL 334 positives. Cross-checked durations with cv2.CAP_PROP_FRAME_COUNT on
  three clips -> exact match, so not a decode artefact. Clip offset into the original is not
  derivable from the shipped metadata.
  Decision: mTTA and time-to-detection STRUCK from the Phase 4 metric set for test-public.
  Consequence: the published BADAS mTTA 4.9 s is not reproducible here; restate the Phase 5 gate
  on AP only. Also narrows U-B3 -- frame selection cannot be time_of_alert-anchored.
- Built eval/benchmark.py (the BADAS-independent half of Phase 4) and ran its self-check.
  Result: PASS. Reproduces T3 exactly (AUC 0.5339, AP 0.5218, 332/325/2/8).
  Newly measured for the old model: precision@recall0.80 = 0.5253, ECE = 0.4880.
  One self-check was deleted as worthless: it grepped its own source for "accuracy_score"
  and matched its own assert message.
- User: "i dont have company email" -> searched for an ungated mirror instead of pressing on the form.
  Found github.com/getnexar/BADAS-Open (Apache-2.0), cloned it, vendored to vendor/badas-open/.
  B-GATE BYPASSED for source. Weights (badas_open.pth) remain HF-gated.
- Read the vendored source in full: utils/video.py, utils/sliding_window.py, models/vjepa.py,
  badas_loader.py, config.json, cli.py, examples/basic_inference.py.
  U-B3 RESOLVED: sliding window, 16 frames, img_size 224 (NOT 256), target_fps 8.0, stride 1,
    plain cv2.resize (no aspect crop), temperature scaling 2.0 at inference.
  U-B4 RESOLVED: no MPS in auto-detect (cuda-or-cpu), but `device` is a parameter everywhere and
    nothing is hard-wired to .cuda(). device="mps" works. D2 SURVIVES, pending empirical check.
  Architecture CORRECTED: no 12 learned queries; it is MultiheadAttention(1024, 8 heads) +
    LayerNorm + mean-pool, then a 3-layer MLP (768 hidden, 2 classes).
  U-B2 REOPENED: config.json says Nexar AP 83.2 / AUC 0.85; the model card said AP 0.86 / AUC 0.88.
    All four benchmark numbers disagree between the two sources.
  U-B5 NEW: clip-score reduction is ambiguous -- per_video uses mean, but cli.py and the example
    both use max. Latent NaN bug: use np.nanmax, not builtin max.
  U-B6 NEW: stride 1 @ 8fps => ~64 windows/clip => ~42,700 ViT-L forward passes per eval run.
    Main Phase 5 feasibility risk. Time one forward pass before committing to a sweep.
  Also: albumentations is imported by utils/video.py but missing from their requirements.txt.
- README.md edited (5 locations): mTTA/time-to-detection struck from Phase 4 metrics, Phase 5 gate
  restated as AP-only against a named source, BADAS architecture description corrected.
- User accepted the HF gate terms and ran `hf auth login` themselves (token never pasted into chat,
  by design). Verified: `hf auth whoami` -> user=khushpal. Gated download test succeeded.
  B-GATE CLEARED for both source and weights.
- Downloaded weights/badas_open.pth (3.7 GB) to models/badas/weights/. In parallel, built the torch
  stack in ~/envs/badas (torch 2.14.0, transformers 5.17.0, albumentations 2.0.8, cv2 5.0.0).
  Confirmed torch.backends.mps.is_available() == True.
- Wrote scripts/badas_smoke.py to time one forward pass (U-B6) and ran it.
  Model load: ~544s (9 min) plus "unexpected keys in checkpoint (199 keys)" warning.
  Investigated rather than ignored (D7): loaded the .pth directly with torch.load and enumerated
  every tensor. FOUND (S6.9, the biggest finding of the session): the checkpoint has a 199-tensor
  `predictor` module (12 layers, dim 384, future-prediction, trained with
  use_future_prediction=true, future_prediction_seconds=1.0) that EnhancedVideoClassifier does not
  implement at all -- load_state_dict(strict=False) silently drops all 199 keys. The count matches
  exactly. Consequence: the published inference code, run naively, produces a plausible-looking but
  WRONG model missing its entire future-prediction pathway. Phase 5 now requires reimplementing the
  predictor + concat path, not just running the loader.
  Also confirmed: AutoVideoProcessor resolves to VJEPA2VideoProcessor (not the ImageNet fallback) --
  resolves the normalization half of U-B3.
- Smoke test then hit a second, separate bug: transformers 5.17.0's VJEPA2Model.forward() requires
  `pixel_values_videos`, but the vendored code passes `pixel_values` (written against transformers
  4.x per their own requirements.txt, which only pins >=4.40.0). TypeError, forward pass did not
  complete. U-B6 (per-window timing) is therefore STILL UNMEASURED. One-line fix identified,
  NOT YET APPLIED -- this is the exact next action (S13 step 1).
- Caught and fixed a real bug: models/badas/ (3.7 GB) was not in .gitignore and was about to become
  committable. Added `models/badas/` to .gitignore; verified with `git check-ignore -v`.
- Ran the handoff-update prompt (this entry). Rewrote S3, S4 header, S7, S8 to distinguish
  session-1-only content from current (session 2 end) state, since those sections had gone stale
  after S6.9/S6.10/S10 were written live. No code changed during the handoff pass itself.
- Nothing committed this session. Git HEAD still 420ef17. Working tree: M .gitignore, M README.md,
  ?? eval/, ?? progress.md, ?? scripts/badas_smoke.py, ?? vendor/ (models/badas/ correctly excluded
  after the gitignore fix).
- Session ends here: B-GATE fully cleared, weights on disk, but U-B6 unmeasured because of the
  1-line API-drift bug. Next session starts by fixing that one line.

2026-09-12 (session 3)
- Verified state rather than trusting the handoff: full git history, tracked file list, the blocking
  line, weights on disk (3,979,436,545 B), ~/envs/badas (torch 2.14.0 / transformers 5.17.0 / MPS True).
- User asked whether the README plan was "best and accurate" for the startup goal. Audited it and
  reported three structural defects: (a) Phase 5's ±0.02 AP gate is unpassable by construction,
  (b) the moat (Tracks B/C) was sequenced behind the most expensive and least certain track,
  (c) customer discovery was buried as a subtask. User said "edit the plan".
  -> README pass A: three-track block, Phase 5 gate restated, Phase 6 reframed per D2, Phases 7-8
     promoted to P0/week 1, plus stale-status fixes (§2 rows, Phase 0, §42/§43).
  -> Also corrected runs/falsification/RESULTS.md, whose T3 section still said "NOT YET RUN".
- Verified the transformers arg name via inspect.signature BEFORE patching. Patched
  vendor/.../video.py:175 -> pixel_values_videos, with a ponytail comment.
- Ran scripts/badas_smoke.py successfully. U-B6 ANSWERED: load 8.5s (NOT 9 min), 0.856 s/window
  compute, stride-1 sweep projected 10.1 h *on compute alone*.
- Noticed `skip_predictor` in VJEPA2Model.forward -> investigated the §6.9 predictor claim instead of
  accepting it. Built scripts/badas_predictor_probe.py.
  FOUND: V-JEPA2's native predictor has exactly 199 params, names+shapes matching the checkpoint.
  As-shipped load reports missing 0 / unexpected 199 -> the checkpoint stores the predictor TWICE.
  Compared both copies with torch.equal: 199/199 BITWISE IDENTICAL. No weights lost.
  => SESSION 2's §6.9 DIAGNOSIS WAS WRONG. Corrected the script's own printed verdict too.
  Real gap: forward() discards predictor_output (computed by default, 25% of every pass).
  Measured skip_predictor A/B: 0.842s vs 0.629s, last_hidden_state bit-identical.
  -> README pass B: gate-note reason 3 rewritten, task 2 marked answered, task 3 rescoped from
     "reimplement" to "consume predictor_output", effort line updated. D9 recorded.
- Built Phase 4's missing adapter: eval/adapters.py (AlwaysNegative, CachedScores, BadasOpen),
  eval/run_baselines.py, and extended eval/benchmark.py with durations/hours/clip_paths/run/summarise
  + metrics.json. Decisions D11 (replay committed scores), D12 (np.nanmax), D13 (one denominator).
- Read the vendored source to get the scoring contract right rather than guessing: predict() ->
  per-frame probs, softmax(logits/2.0)[:,1] => class 1 = collision, leading NaNs by design.
  Confirmed upstream's builtin max() is a latent NaN bug (U-B5).
- Ran --no-badas over all 667 clips: old model AP 0.5218 / AUC 0.5339 / FP-h 361.4 / ECE 0.4880
  (reproducing committed T3 through the NEW path, asserted); always-negative AP 0.5007 / FP-h 0.0.
  => Phase 4 acceptance criterion #1 MET: three models, one code path.
- Wrote tests/test_score_regression.py; ran it: safe.mp4 = 0.7914 exactly (714 frames, 24.0 fps).
- Installed scikit-learn 1.9.1 into ~/envs/badas (needed by benchmark.py; not a rebuild).
- Ran --limit 6 to verify the BADAS adapter end-to-end: works. 583.7s for 6 clips.
  => CORRECTED MY OWN EARLIER NUMBER: ~97 s/clip end-to-end, so the sweep is ~18 h, not ~10 h.
     The 10 h figure was compute-only. README corrected again. D10 recorded.
  => AP 1.0000 at n=6 is NOT a result and must not be quoted.
- Confirmed matplotlib absent from both envs (blocks Phase 4's plots).
- User asked for a full teaching explanation of the whole project; delivered in conversation only,
  no files touched. Flagged 7 documentation contradictions (now §18) and offered to fix them;
  user did not answer.
- Two questions asked and unanswered: stride 1 (~18 h) vs stride 2 (~9 h); commit structure.
- NOTHING COMMITTED. HEAD still 420ef17. Working tree: M .gitignore, M README.md,
  M runs/falsification/RESULTS.md, ?? eval/, ?? progress.md, ?? scripts/badas_predictor_probe.py,
  ?? scripts/badas_smoke.py, ?? tests/test_score_regression.py, ?? vendor/
- Session ends here: Phase 4 ~60% (AC#1 met), Phase 5 unblocked and de-risked but NOT started.
  Next session starts by running the full BADAS sweep (§13).
```
