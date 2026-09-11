# HANDOFF — AI Crash Detection project (paste this into the new Claude Code session)

## 0. READ THIS FIRST — identity correction

The handoff request that generated this document described a project called
**"ULPF Solo Build" / "Universal Log Pre-processing Framework" (SIH 2026, NTRO)**.

**That is not this project.** Nothing in this repository or the prior session relates to log
pre-processing, NTRO, or SIH. The prior session was entirely about an **AI dashcam crash-detection
system**. The user appears to have mixed up two different projects when writing the handoff request.

**Act on the crash-detection project described below.** If the user actually wants ULPF work, say so
and ask — there is zero ULPF context to inherit.

Everything else from the handoff request (the falsification numbers, licence findings, environment
bugs, evidence standards) is accurate and is preserved here, corrected where the filesystem disagrees.

---

## 1. Working directory & how to verify state before touching anything

```
/Users/khushpalsinghchouhan/Desktop/crash_detection/crash_detection_v2
```

Run this verification block FIRST, before changing a single file:

```bash
cd ~/Desktop/crash_detection/crash_detection_v2
git log --oneline | head -5
git status --short | grep -v '^?? data/nexar'   # filter the ~500 untracked mp4s
git diff --stat
sed -n '1,60p' README.md
ls runs/falsification/ scripts/ data/ docs/
grep -n "self\.depth_est\|self\.bev\|EgoZone\|FaultDetector" code/crash_detection_enhanced.py
find data/nexar/test-public -name '*.mp4' | wc -l
pgrep -fl dl_nexar.sh
```

Report any discrepancy between this document and what you find. Do not assume this document is
correct where the filesystem disagrees — the filesystem wins.

---

## 2. Project objective and scope

Independent solo product/research project. Goal: a **technically defensible** dashcam crash-detection
system — reproducible experiments, honest evaluation, evidence-backed claims. Explicitly NOT
"make the accuracy number look good".

The repository currently contains a prototype, a ~3,200-line audit (`README.md`), and a set of
falsification experiments that **disproved the prototype's headline result**.

---

## 3. CORRECTIONS to the incoming handoff (verified against the filesystem)

| Handoff claim | Reality | Evidence |
|---|---|---|
| "Nothing from the falsification work has been committed. This is intentional." | **FALSE.** Two commits were made with the user's explicit approval. | `git log`: `c9a6fda` "Audit rev2: recover training pipeline, run falsification tests", `ad45389` "Consolidate project into crash_detection_v2; reclaim 5.8 GB" |
| "Tests were run against commit b539d6e" | True at run time, but **HEAD has since moved**. `b539d6e` is now 5 commits back. Current HEAD = `ad45389`. | `git log` |
| "A 31.4 GB Nexar download was started and was interrupted by closing the Mac." | **Two errors.** (a) The download is only the **test-public split: 667 clips / 2.88 GB**, not 31.4 GB. (b) It was **NOT interrupted — it is still running** and is resumable regardless. | `pgrep -fl dl_nexar.sh` → PID 22313 alive; 491/667 files, 2.0 GB on disk, 0 corrupt |
| Dataset repo "https://github.com/Cogito" | Truncated. Real URL: **https://github.com/Cogito2012/CarCrashDataset** | README §49 |
| Implied immediate task = "download Nexar" | **No.** The real immediate task is finishing a half-completed code deletion that has left the main pipeline broken and uncommitted. See §10. | `git status` shows `M code/crash_detection_enhanced.py`; 8 dangling attribute refs |

---

## 4. Repository structure (446 MB total, everything consolidated here)

```
crash_detection_v2/
├── README.md              277K  the full audit, 50 numbered sections — THE source of truth
├── code/
│   ├── crash_detection_enhanced.py   THE main pipeline (MODIFIED, UNCOMMITTED, BROKEN — see §10)
│   ├── crash_detection.py            fork, slated for deletion
│   ├── crash_detection_linux.py      fork, slated for deletion
│   ├── depth_estimator.py            MiDaS wrapper, slated for deletion
│   ├── bev_renderer.py               BEV minimap, slated for deletion
│   ├── ORBSTACK_SETUP.md, setup_orbstack.sh
├── camera_detect.py       4th fork (Raspberry Pi), slated for deletion
├── models/
│   ├── crash_model_weights.weights.h5   6.7 MB — THE deployed model (Keras 3 format)
│   ├── feature_extractor_saved/         19 MB — MobileNetV2 SavedModel used at inference
│   ├── feature_extractor.keras          9.2 MB
│   └── crash_model_cpu/                 4.5 MB
├── videos/                182 MB  crash1.mov, crash2.mov, safe.mp4 (real dir, was a symlink)
├── data/
│   ├── Crash-1500.txt     CCD per-clip annotations (1500 rows) — THE grouping key lives here
│   ├── train.txt, test.txt  official CCD split (3600 / 900)
│   ├── ccd/Untitled0.ipynb  the recovered Colab training notebook, outputs included
│   └── nexar/             LICENSE, README.md, 9 metadata CSVs, test-public/*.mp4 (downloading)
├── runs/falsification/    RESULTS.md, T5_source_leakage.txt, T124_local_videos.json
├── scripts/               t5_source_leakage.py, t124_model_falsification.py
├── archive/parent_repo_v1/  older duplicate code preserved before the parent repo was deleted
├── docs/                  crash_detection_masterclass.md, HANDOFF.md (this file)
└── yolov8n.pt             6.5 MB
```

---

## 5. What the system actually is

```
video → YOLOv8n vehicle detection → MobileNetV2(frozen, ImageNet) + GlobalAveragePooling
      → 10×1280 sequence → LSTM head (578,689 params) → sigmoid
      → is_crash = (score >= 0.80) AND (n_vehicles >= 2)
```

Surrounding it: Kalman filter, pinhole ground projection, TTC, MiDaS depth, BEV renderer,
ego-zone, fault attribution — **none of which reach the verdict.** Verified at
`code/crash_detection_enhanced.py:1050` (pre-edit line numbering).

### Training pipeline (recovered from Colab, previously believed nonexistent)
Notebook: https://colab.research.google.com/drive/1Bmudju26-q7JQyI4FAsZDBXsJw7PRnnM
(also committed at `data/ccd/Untitled0.ipynb`)

| Item | Value |
|---|---|
| Dataset | Car Crash Dataset (CCD), 1,500 crash + 3,000 normal, 50 frames @ 10 fps |
| Frames/clip | 10, `np.linspace(0, 49, 10)` → ~0.54 s stride, ~4.9 s span |
| Input | 112×112, BGR→RGB, `mobilenet_v2.preprocess_input` |
| Backbone | MobileNetV2 ImageNet, **frozen**; features precomputed to `.npy` |
| Head | Dense256 relu → Drop.3 → LSTM128 → Drop.3 → LSTM64 → Drop.3 → Dense64 relu → Drop.2 → Dense1 sigmoid |
| Params | 578,689, all trainable |
| Split | `train_test_split(test_size=0.2, random_state=42, stratify)` → 3,600 / 900. **No test set.** |
| Optimizer / loss | Adam 1e-3 / binary_crossentropy |
| Epochs | budget 50, ran 17, early-stopped, best epoch 9 |
| Reported | `val AUC 0.9977`, `val acc 0.9822` — **both invalid, see §6** |
| Framework | TensorFlow 2.19.0 / Keras 3.13.2, Colab T4 |

---

## 6. FALSIFICATION RESULTS — preserve exactly, do not soften

Run 2026-09-10 against artifact `models/crash_model_weights.weights.h5`.
Scripts: `scripts/t5_source_leakage.py`, `scripts/t124_model_falsification.py`.
Raw: `runs/falsification/RESULTS.md`, `T5_source_leakage.txt`, `T124_local_videos.json`.

### T2 — Temporal shuffle: **FAILED**

| Video | Deployed | Shuffled | 1 frame ×10 |
|---|---|---|---|
| crash1.mov | 0.9997 | 0.9998 | 0.9798 |
| crash2.mov | 0.9640 | 0.9649 | 0.9460 |
| safe.mp4   | 0.0241 | 0.0228 | 0.0512 |

Permuting the 10 frames changes the score by **≤ 0.0013**. The two LSTM layers hold
**246,528 params (~43% of the model)** and are **effectively inert**. Repeating a single frame ten
times reproduces the answer. **The system behaves as an image classifier, not a temporal model.**

### T5 — Source leakage: **FAILED SEVERELY**
- 1,500 crash clips come from only **133 YouTube videos** (mean 11.3/source, max 34)
- A random 80/20 split puts **113 of 133 sources on both sides** → **91.4% of clips source-leaked**
- ~**274 of 300** validation crash clips have a sibling from the same source in training
- **The official CCD split does NOT fix it: 107 of 133 sources appear on both sides there too**
- Therefore evaluation is not independent at source/video level, and prior validation performance
  is not evidence of real-world generalisation.

### Label quality
- Accident begins at **frame 37.2 of 50** on average (median 36, min 30, max 49)
- **~72%** of the frames shown for a "crash" clip contain **no accident**, yet are labelled 1
- Discarded CCD metadata that was available: `binlabels` (per-frame), `egoinvolve` (Yes 801/No 699),
  `timing` (Day 1325/Night 175), `weather` (Normal 1141/Snowy 235/Rainy 124), `youtubeID`

### Threshold
- `safe.mp4` max = **0.7914** exactly; `CNN_THRESH = 0.80` clears the only negative by **0.0086**
- Under single-frame repetition `safe.mp4` reaches **0.9695** — confirms per-frame sensitivity,
  not temporal understanding
- The threshold 0.80 appears **nowhere** in training; Colab evaluated at 0.50, suggested 0.55

### Derived false-alarm rate
Val FPR 19/600 = **3.17%** → as 5-second windows over continuous driving (720/hour) ≈
**23 false alarms per driving hour**. Target for a shippable system: **< 0.1/hour**.
An always-negative baseline scores 0. **On the metric that matters, the trivial baseline wins.**

### T3 — Corpus control: **NOT YET RUN** (the one remaining test; needs the Nexar download)
### T4 — Crash excision: **NOT RUN** (needs the cached `.npy` features, which live in Google Drive)

---

## 7. Conclusions JUSTIFIED by the evidence

1. The LSTM contributes nothing measurable (T2, direct measurement).
2. The model is a per-frame appearance classifier.
3. The training/validation split leaks by source at 91.4%; the official split does too.
4. ~72% of positive-clip frames are mislabelled as accident frames.
5. `CNN_THRESH = 0.80` has no derivation and is fitted to one negative video.
6. The reported `val AUC 0.9977` is both leakage-inflated and model-selection-contaminated
   (EarlyStopping and ModelCheckpoint both selected on the same `val_auc` that was then reported).
7. The physics stack (Kalman/TTC/MiDaS/BEV/ego-zone/fault) does not affect the verdict.
8. Nexar's licence permits commercial training.

## 8. Conclusions NOT justified — do not claim these

- That the model detects collisions at all (never measured on a corpus-controlled benchmark).
- Any generalisation claim from CCD validation numbers.
- That fixing the frame stride, the threshold, or the epochs will fix the model.
- That the 3D/physics outputs are numerically valid (focal length is wrong ~4× at test resolution).
- That any metric describes the shipped weights: the deployed `.weights.h5` is from a **different
  training run** than the archived checkpoint — 37 of 38 weight arrays differ, and the first Dense
  kernel correlates at **r = 0.0073**. The recorded metrics cannot be attributed to it.

---

## 9. ARCHITECTURAL DIRECTION — preserve this reasoning

**Do NOT retrain MobileNetV2 + LSTM as the primary solution.** The reason is not that newer models
exist. It is that MobileNetV2's **GlobalAveragePooling2D discards spatial layout**, so the relative
spatial/motion relationship between two vehicles is not representable in the 1,280-d vector the LSTM
receives. A recurrent layer cannot recover information already destroyed by pooling. **T2 is the
empirical measurement supporting this.** This is an architectural ceiling, not a tuning problem.

The replacement must retain spatial structure into the temporal stage, or model motion explicitly.
Candidate directions (evaluate against real constraints, do not pick one on novelty):
CNN features without premature spatial collapse · 3D CNNs · video transformers ·
optical-flow / motion-aware · two-stream · V-JEPA2 + attentive probe warm-started from BADAS-Open.

**Keep from the current pipeline:** the frozen-backbone + cached-features pattern (efficient and
correct for a small GPU budget), the consistent train/inference preprocessing, the callback config.

**Measure the old model honestly once on a corpus-controlled benchmark before retiring it** — it has
never had a fair evaluation. Retire it with evidence, not assertion.

---

## 10. THE IMMEDIATE TASK — finish the interrupted dead-code deletion

**Status: IN PROGRESS, INCOMPLETE, UNCOMMITTED, AND THE MAIN PIPELINE IS CURRENTLY BROKEN.**

`code/crash_detection_enhanced.py` is modified in the working tree. It **compiles** (`py_compile`
passes) but will raise `AttributeError` at runtime: the `self.depth_est` and `self.bev`
initialisation blocks were removed, but **8 usage sites remain**:

```
810:  if self.depth_est is not None:
811:      depth_map = self.depth_est.estimate_metric(
847:      if self.bev is not None:
848:          bev_img = self.bev.render(
931:  if self.depth_est is not None:
932:      depth_map = self.depth_est.estimate_metric(
960:  if self.bev is not None:
961:      bev_img = self.bev.render(
```

### Already done in this file (uncommitted)
- Removed `FaultDetector` and `EgoZone` classes (163 lines)
- Removed `depth_estimator` / `bev_renderer` imports
- Removed `FaultDetector()` from `_make_pipeline()` and both unpack sites
- Collapsed both dashcam branches (video + webcam) to the single CNN-gated path
- Removed the display ego-zone overlay (18 lines)
- Removed the depth-estimator and BEV-renderer `__init__` blocks
- Line count 1,334 → 1,088

### Remaining steps to finish
1. Remove the 8 dangling `self.depth_est` / `self.bev` blocks and the `depth_map` / `bev_img`
   arguments threaded into `Display.draw_with_overlays(...)`.
2. Remove the now-unused `Display` parameters: `dashcam`, `ego_hits`, `depth_map`, `bev_img`,
   `fault_info`, and the `show_depth` / `show_bev` toggles and their `B` / `D` keybindings.
3. Remove the `--dashcam` CLI flag and `self.dashcam` (its only implementation was `EgoZone`).
4. Delete the dead files:
   `code/crash_detection.py`, `code/crash_detection_linux.py`, `camera_detect.py`,
   `code/depth_estimator.py`, `code/bev_renderer.py`, and
   `code/ORBSTACK_SETUP.md` + `code/setup_orbstack.sh` (they exist only for the linux fork).
5. Remove now-dead `Config` constants: `EGO_ZONE_*`, and consider `TTC_WARN`, `MIN_VEHICLE_Z`,
   `CONF_CONTINUE` (already documented as dead in README §47).
6. ~~Add a `.gitignore` rule for the Nexar mp4s~~ — **DONE.** `.gitignore` now ignores
   `data/nexar/**/*.mp4`. This shows as `M .gitignore` in the working tree; commit it with the rest.
7. Verify, then commit.

### Definition of DONE for the immediate task
- [ ] `python -m py_compile code/crash_detection_enhanced.py` passes
- [ ] `grep -n "EgoZone\|FaultDetector\|fault_det\|DepthEstimator\|BEVRenderer\|self\.depth_est\|self\.bev" code/crash_detection_enhanced.py` returns **nothing**
- [ ] The pipeline **runs end-to-end**: `python code/crash_detection_enhanced.py --video crash1 --no-display --max-frames 100`
- [ ] The crash decision expression is **unchanged**: `(cnn >= Config.CNN_THRESH) and (len(vehs) >= 2)`
- [ ] Only the 4 intended files remain under `code/` + none at repo root
- [ ] `git status` is clean apart from intended changes; no `.mp4` staged (verify with `git status --short | grep '\.mp4'` returning nothing)
- [ ] Committed with a message naming what was deleted and why (README bugs C2, C3, C7, H1, H5, M1)

### Caveat about verification
Running the pipeline needs `ultralytics` + `torch`, which are **not installed** in the working
environment. `ultralytics` is **AGPL-3.0** (README bug C6) and the audit recommends removing it.
Options: (a) install it temporarily just to verify, (b) verify by import + static analysis and state
plainly that an end-to-end run was not performed. **Do not claim an end-to-end run you did not do.**

---

## 11. Git state

```
ad45389  Consolidate project into crash_detection_v2; reclaim 5.8 GB   <- HEAD
c9a6fda  Audit rev2: recover training pipeline, run falsification tests
c995fd7  update readme            (user's commit)
09302c0  update-readme            (user's commit)
b539d6e  update readme file       (the commit the falsification tests were run against)
```

Uncommitted at handoff time:
```
 M .gitignore                          <- adds the Nexar mp4 ignore rule
 M code/crash_detection_enhanced.py    <- the broken in-progress edit (the immediate task)
?? docs/HANDOFF.md                     <- this document
```
The 667 Nexar `.mp4` files are now correctly ignored and no longer appear in `git status`.

Remote: `git@github.com:khushpal-cipher/crash-detection-system.git` (SSH — good).
A previous parent repo had a **GitHub PAT embedded in plaintext** in its remote URL; that repo has
been deleted and the user has confirmed the token is revoked.

**Do not commit blindly.** Inspect `git diff`, explain what should be committed and why, then commit.
Never destroy or overwrite existing evidence in `runs/`.

---

## 12. Nexar dataset — status and licence

**Licence: GREEN (resolved, no email needed).** Text at `data/nexar/LICENSE`:
grants "free of charge … to use, copy, modify, and distribute". **No non-commercial restriction.**
Conditions: attribution (specified citation), retain notice on redistribution, **No Resale** of the
dataset itself, and ethical-use limits (no malicious systems, deepfakes, re-identification,
weaponisation, **"exploitative practices … such as unethical insurance practices"**, comply with law).
Commercial model training is permitted. Have counsel confirm the No-Resale scope re derived models.

**Repo:** `nexar-ai/nexar_collision_prediction` on Hugging Face — **public, not gated, no token needed.**
Full dataset: 2,844 mp4 + 9 CSV, **31.38 GB**. Splits: train 750+750 (25.5 GB),
test-public 334+333 (2.88 GB), test-private 338+339 (2.97 GB). Labels in `solution.csv`.

**Download COMPLETE — do not re-download, and do NOT start a 31.4 GB download.**
Only the **test-public split** was fetched, deliberately: it is the official benchmark split, it ships
with labels (`solution.csv`), and it is sufficient for the corpus-control test.

- Location: `data/nexar/test-public/{positive,negative}/*.mp4`
- **Final state: 667 / 667 files, 2.7 GB, 0 corrupt, 0 truncated** (verified: all begin with a valid
  `ftypisom` MP4 header; none under 100 KB)
- Counts match the metadata exactly: 334 positive + 333 negative
- Completed over four passes; ~43 clips failed on transient network errors and were recovered by
  simply re-running the script, which skips any file already present and non-empty (`if [ -s "$out" ]`)
- Script retained at `/tmp/dl_nexar.sh`, file list at `/tmp/nexar_files.txt` (both may be cleared by
  the OS; regenerate the list from the HF API if needed — existing files are reused, nothing refetches)
- **`.gitignore` now contains `data/nexar/**/*.mp4`** — these must never be committed

Not downloaded (not currently needed): `train/` 1,500 clips 25.5 GB, `test-private/` 677 clips 2.97 GB.

Disk: ~44 GB free. Fine for test-public; the full 31.4 GB would also fit but is not currently needed.

---

## 13. Environment — two confirmed bugs

**R2 — the old env cannot load the model.** `crash_env` has **TensorFlow 2.15 / Keras 2.15**. The
shipped weights are **Keras 3** format. Keras 2 physically cannot read them. `requirements.txt` still
pins `tensorflow==2.13.1`, which is worse. **Never claim `crash_env` can load the current weights.**
The falsification run worked around this with an exact **NumPy forward pass** of the LSTM head read
straight from the HDF5 file (see `scripts/t124_model_falsification.py` — reusable).

**iCloud eviction.** `~/Desktop` is iCloud-synced; cloud-evicted files caused `TimeoutError: [Errno 60]`
during TensorFlow import. **Never create heavy Python environments under Desktop.**

**Working environment built for the falsification run** (local disk, outside iCloud):
```
/private/tmp/claude-501/-Users-khushpalsinghchouhan-Desktop-crash-detection-crash-detection-v2/<session-id>/scratchpad/tfenv
tensorflow 2.19.1 · keras 3.15.1 · opencv-python-headless 5.0.0 · scikit-learn 1.9.0 · numpy 2.1.3
```
**This path is session-scoped and is probably gone.** Recreate it under a stable non-iCloud path,
e.g. `~/envs/crashdet` (`python3.11 -m venv`), with the same pinned versions. Doing this properly is
README Phase 1 work and is worth doing once rather than repeatedly.

---

## 14. Dataset provenance / licensing

- **CCD** — https://github.com/Cogito2012/CarCrashDataset. Repo labelled MIT, but the README has no
  licence section. **Positives are YouTube-derived** (each clip has a `youtubeID` + `startframe`);
  **negatives are sampled from BDD100K**, whose basic licence is limited to personal use.
  → **RED for commercial use.** Research/benchmark only. The current model is trained on this.
- **The annotations-vs-pixels problem**: a repo's MIT licence covers the researchers' annotations,
  never the underlying video copyright. This applies to CCD, DoTA, DADA-2000, MM-AU, DAD, CADP.
- **Nexar** — GREEN, see §12.
- **`ultralytics` (YOLOv8) is AGPL-3.0** — README bug C6, unresolved. Migrate to RT-DETR from
  `lyuwenyu/RT-DETR` (Apache-2.0), **not** the `ultralytics` RT-DETR wrapper, or buy a licence.

---

## 15. What has NOT been done

- T3 corpus control (blocked on the download finishing) · T4 crash excision (needs Drive features)
- Source-grouped, frozen test split · calibration split · threshold derivation from a PR curve
- Reproducing BADAS-Open · any Nexar evaluation
- Fixing camera calibration (C1) · replacing the tracker (H2) · resolving AGPL (C6)
- Any reproducible training script in-repo (the pipeline still exists only as a Colab notebook)
- Seeding anything but the split; run-isolated checkpointing; experiment tracking

---

## 16. Planned sequence AFTER the immediate task

1. Finish + verify + commit the dead-code deletion (§10).
2. Reconcile README with what actually shipped.
3. Rebuild the environment at a stable non-iCloud path; pin TF 2.19 / Keras 3.
4. ~~Finish the Nexar test-public download~~ — **DONE, 667/667, verified.**
5. **T3 corpus control** — evaluate the existing CCD-trained model on Nexar test-public (already
   downloaded, 667 clips), where positives and negatives share a corpus. This is the last
   falsification test, and it is now fully unblocked. Reuse the NumPy forward-pass approach in
   `scripts/t124_model_falsification.py` so no Keras-3 runtime is strictly required. Expect a large
   AUC drop; report whatever comes out.
6. Build the evaluation harness: source-grouped splits (group key = `youtubeID` from
   `data/Crash-1500.txt`), leakage test that fails CI, metrics AP / ROC-AUC / **FP-per-hour** / ECE /
   per-condition breakdown. **Ban raw accuracy from all reports.**
7. Reproduce BADAS-Open (`nexar-ai/BADAS-Open`, Apache-2.0, V-JEPA2 ViT-L + attentive probe,
   published Nexar AP 0.86 / AUC 0.88 / mTTA 4.9 s). **If you cannot reproduce a published result,
   you cannot claim to beat one.** Gate: within ~0.02 AP.
8. Baselines on identical splits: BADAS-Open zero-shot · the old MobileNetV2+LSTM · always-negative.
9. Revisit label quality / temporal localisation using CCD `binlabels`.
10. Choose an architecture that can represent the evidence (§9). Train only after the protocol is sound.
11. Re-run falsification and ablation on the new model. Document what it proves and what it does not.

**Evaluation design and data integrity come before training. Do not skip to model training.**

---

## 17. Working constraints the user has set

- Bias toward acting; ask only when genuinely blocked or when a decision is the user's to make.
- Prefer Bash (`cat`, `sed`, `grep`, heredocs) over dedicated file tools where it can do the job.
- Minimal output style: read files fully first; complete copy-pasteable solutions; no placeholders;
  test before declaring completion; no over-engineering.
- **Evidence standard:** prefer falsification over confirmation. Report failures. Quantify leakage.
  Keep raw outputs. Never delete inconvenient results. Never cherry-pick. Never make a claim stronger
  than the evidence. Record commands, commit hashes, environment versions, artefacts, provenance.
- Confirm before destructive or outward-facing actions.

---

## 18. Unresolved unknowns

| # | Unknown | How to resolve |
|---|---|---|
| U1 | Which training run produced `crash_model_weights.weights.h5`, and its metrics | Unresolvable retrospectively (`ModelCheckpoint` used a fixed path). Retrain with run isolation. |
| U2 | Whether the Colab's saved outputs describe the shipped weights | Same. `model.summary()` printed `functional_5`, implying ≥6 builds in one session. |
| U3 | Whether `crash_model_cpu/` shares weights with the shipped `.h5` | Load both in a TF 2.19 env, compare `get_weights()`. |
| U4 | CCD's operative licence scope | Email the authors (Wentao Bao, RIT). |
| U5 | How `models/feature_extractor_saved/` was produced | Notebook saved `.keras`; the SavedModel dir appeared a month later. Regenerate deterministically. |
| U6 | Whether BDD100K's current terms permit any commercial use | Ask Berkeley DeepDrive directly. |

---

## 19. Required output before you execute anything

Give the user:
**A.** Verified current state · **B.** What is proven · **C.** What is broken · **D.** What is unknown ·
**E.** The exact immediate task · **F.** Step-by-step plan · **G.** Expected artifact ·
**H.** What comes next · **I.** Anything needing their approval.

Then execute. Mark anything you could not verify as **UNKNOWN** rather than guessing.
