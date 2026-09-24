# September 30 demo plan

**Written 2026-09-21 (session 17). Status: PLAN — nothing implemented.**
`README.md` is the master plan · `NEW_PLAN.md` is the research plan · `progress.md` is
execution state. **This file is none of those.** It is a nine-day demo-prep plan and it
does not change either planning document.

---

## A. Objective

By **2026-09-30**, be able to sit in front of a fleet manager, type one command, and have
them understand what the product does without a technical explanation.

**The demo is a terminal command that opens the video and shows the model judging it.**

🔴 **There is no UI, no web app, no dashboard, no database, and no server.** The user ruled
this out explicitly and it is the scope boundary of this document. A request for a screen,
a page, a dashboard or a stored record belongs in §G, not here.

---

## B. The demo, exactly

```bash
~/envs/badas/bin/python scripts/demo.py videos/crash1.mov
```

1. The command prints what it is about to do: the video, its length, the model, the alert
   threshold and **where that threshold came from**.
2. It scores the video with BADAS-Open, printing progress so the viewer sees real work.
3. A window opens and plays the video back with the detector's score drawn on it, and the
   alert region marked.
4. The terminal prints a plain-English incident summary and writes the JSON record.

**What the fleet manager should take away:** *"It watched the footage, it found the moment,
and it produced a record I could send to my insurer."*

**What the presenter says out loud, every time:**
> "The detector is real and it is running now. The fleet, the drivers and the vehicles in
> the record are demo data — I have no customers yet."

That sentence is not a weakness. It is the reason the rest of the demo is believable.

---

## C. What exists, and what has to be built

| Piece | State |
|---|---|
| `eval/adapters.py::BadasOpen.score()` — the detector | ✅ works, swept 667 clips |
| `eval/timing.py` — threshold + t_start/t_peak/t_end | ✅ 7/7 self-check |
| `eval/calibration.py` — usable probability | ✅ ECE 0.33 → ~0.05 |
| `scripts/detect.py` — video → JSON incident record | ✅ 372 lines, 8 self-checks |
| `runs/incidents/` — three records already produced | ✅ crash1, crash2, safe |
| **`scripts/demo.py` — the live viewer** | ❌ **DOES NOT EXIST. This is the only new code.** |

🔴 `code/crash_detection_enhanced.py` has a `cv2.imshow` loop at line 654, but it drives the
**retired** MobileNetV2+LSTM model (chance-level, AUC 0.5339) and cannot run — it imports
`ultralytics`, which is not installed. **Do not resurrect it.** Take the playback idea, not
the code.

**`scripts/demo.py` is glue only.** It imports `eval/adapters.py`, `eval/timing.py` and
`eval/calibration.py` unchanged, exactly as `scripts/detect.py` does. It adds no model, no
metric and no threshold of its own. Four of the five regression guards live in `eval/`;
nothing in this demo is worth risking them for.

---

## D. 🔴 THE TIMING PROBLEM — read this before designing anything

**BADAS cannot score video in real time on this Mac.** Measured: **~1.7 s per window** at
stride 1, where a window advances one frame at 8 fps.

> 🟢 **SUPERSEDED IN PART, 2026-09-24 — see §E for the real measurement.** The conclusion of
> this section STANDS, but the input number was pessimistic: measured end to end it is
> **1.33–1.58 s/window**, not 1.7. `crash1.mov` scores in **57 s** (not ~70 s) and
> `safe.mp4` in **351 s** (~6 min, as predicted). **Two passes is still required, and the
> P0 clip needs no speed lever.** The estimates below are kept as written for the record.

```
crash1.mov   7 s of video  ->  ~40 windows  ->  ~70 s to score
safe.mp4    30 s of video  -> ~224 windows  -> ~6 MINUTES to score
```

**So the video cannot play at normal speed while being scored.** Any design that assumes it
can will either stutter for minutes or quietly fake the detection.

**The design that solves this honestly — two passes in one command:**

1. **Pass 1, score.** Print a progress line per window. The viewer watches real computation
   happen. Takes as long as it takes.
2. **Pass 2, play back.** Replay the video at normal speed with the score overlaid and the
   alert region marked. Instant, because the scores already exist.

This is honest — nothing is pre-baked, the command does the work in front of them — and it
is watchable. **Never ship a version that replays a score computed in an earlier run while
implying it is live.** That is the "fake functionality presented as real" failure.

**Two levers if pass 1 is still too slow to hold attention:**
- `--stride 8` scores at 1 Hz instead of 8 Hz. Roughly **8× faster**. The score changes
  slightly, so **it must be printed on screen** and never compared against the committed
  stride-1 numbers.
- `skip_predictor=True` drops a model pathway that is computed and discarded. ~25% cheaper,
  and verified score-identical on MPS (D67/D71). Not yet plumbed into `detect.py`.

**Measure this on day 1.** Every figure above is arithmetic from 1.7 s/window, not a
measurement of `demo.py`, which does not exist yet.

---

## E. Demo footage and its MEASURED scores

**MEASURED 2026-09-24 (session 18), day 1.** Command, on MPS, machine under ordinary desktop
load (`uptime` 2.06, Chrome + Claude running — deliberately *not* a quiet benchmark rig,
because demo day will look like this):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -i ~/envs/badas/bin/python scripts/detect.py \
    videos/crash1.mov videos/crash2.mov videos/safe.mp4
```

Three videos, all with usable rights. `data/nexar/LICENSE` permits use; the three local
files predate the project's dataset work.

| Video | Length | Windows | **Wall clock** | **s/window** | Score | Fires at 0.9733? |
|---|---|---|---|---|---|---|
| `videos/crash1.mov` | 7.38 s | 43 | **57.3 s** | 1.333 | **0.9968** | ✅ yes — a real detection |
| `videos/crash2.mov` | 5.75 s | 30 | **40.9 s** | 1.363 | **0.9959** | ✅ yes — a real detection |
| `videos/safe.mp4` | 29.75 s | 222 | **351.2 s** | 1.582 | **0.9758** | 🔴 **YES — and it is a FALSE ALARM** |

Total for all three: **449.4 s** of scoring, **463 s** wall clock including the ~14 s model
load. All three cleared the gate; three incident records written. `detect.py` exited 0 with
no traceback.

### What the measurement settles

**1. The ~1.7 s/window planning figure was PESSIMISTIC. It is 1.33–1.58 s/window.**
`s/window` rises with clip length (1.333 → 1.363 → 1.582) rather than staying flat — longer
clips are slightly *dearer* per window, so do not extrapolate a short clip's rate to a long
one. The first clip in a batch also absorbs MPS warm-up, which pushes its rate the other way;
the two effects are small and partly cancel.

**2. §D's arithmetic was right about the thing that matters.** `safe.mp4` really is ~6
minutes (351 s measured against ~6 min predicted). **Scoring during playback remains
impossible and the two-pass design stands.**

**3. 🟢 The P0 demo clip needs NO speed lever.** §B demos `crash1.mov`: **57 s** of pass 1.
That is a long pause but it is a watchable one with per-window progress on screen, and it is
the honest cost of real computation. `--stride 8` and `skip_predictor` are **P2, not P0** —
neither is needed to ship §B. Reach for them only if rehearsal says 57 s kills the room.

**4. `safe.mp4` cannot be demoed live at stride 1.** 351 s is unwatchable. If the §E decision
below is "show it", it needs `--stride 8` (~44 s projected) and the changed score must be on
screen — or it must be shown as a pre-recorded terminal capture, clearly labelled as one.

### 🔴 The scores MOVED, and `crash2` was already measured

Two corrections to what `progress.md` §21.15 item 7 and the previous version of this table
said:

**(a) `crash2.mov` was NOT "never measured".** `runs/incidents/crash2.json` and
`runs/incidents/frames/crash2.npz` are tracked and were committed at `90e8a58` ("Ship the
MVP") carrying **0.9961**. The claim arose because `runs/incidents/summary.jsonl` holds only
two rows (safe, crash1) — the last pre-session-18 run processed only those two, so crash2's
record survived from an earlier run and the docs lost track of it.

**(b) The two `.mov` scores CHANGED, because they are variable-frame-rate files with
inaccurate metadata.**

| Video | Codec | fps | Metadata claims | cv2 actually reads | Trace: was → now | Score: was → now |
|---|---|---|---|---|---|---|
| `crash1.mov` | h264 | 35.158 | 259 frames | **257** | 56 → 59 | 0.9966185 → **0.9968072** |
| `crash2.mov` | h264 | 31.175 | 187 frames | **177** | 44 → 46 | 0.9960537 → **0.9958688** |
| `safe.mp4` | h264 | 24.000 | 714 frames | 714 ✅ | 238 → 238 | 0.9758111 → **0.9758111** (identical) |

**Cause:** the sequential-decode patch (`6a705b3`) changed frame extraction from seek-per-frame
to straight-through reading. On a constant-frame-rate file with honest metadata the two agree
exactly — `safe.mp4` is bit-identical. On a VFR file whose header overstates its own length,
they land on different frames, so the 8 fps resample differs and the score shifts by ~2e-4.

🔴 **This does NOT retract `6a705b3`'s 333/333 byte-identical proof.** That was run over the
Nexar corpus, which is uniformly well-formed CFR H.264 — the `safe.mp4` category, which still
reproduces exactly. The proof was sound; it simply never covered VFR input. **The correct
scope for the claim is "byte-identical on constant-frame-rate H.264", and it should be
qualified that way wherever it is quoted.**

**Demo impact: NONE.** All three videos cleared the gate before and after. The deltas are
~2e-4 against a gate margin of 0.0025 (`safe.mp4`, the tightest). **User decision, session 18:
keep the files, record the new numbers.** Re-encoding to CFR would remove the ambiguity but
is blocked — ffmpeg is not installed.

🟡 **Consequence for comma2k19:** comma2k19 is raw HEVC with no container, a *third* decode
category, neither CFR-mp4 nor VFR-mov. This is direct evidence that GATE D (frame rate read
back **through cv2**) is load-bearing and must not be skipped.

### The `safe.mp4` false alarm

🔴 **`safe.mp4` is 30 seconds of ordinary driving and the system calls it a crash**, clearing
the gate by 0.0025. This is correct behaviour for a system measured at 92.3 false alarms per
hour. It is not a bug and it must not be "fixed" for the demo.

**Decide before the 30th how you use it.** Two defensible options:

- **Leave it out.** Demo only the true positives. Simple, and nothing shown is misleading.
- **Show it deliberately, as the closing point.** *"Here is the honest limitation. It fires
  on this normal clip. That is the problem I am measuring on 33 hours of real driving right
  now, and it is the number I will lead with when I have it."*

The second is stronger with a technical or safety-minded audience and weaker with a
non-technical one. **What is not acceptable is being surprised by it live.**
🔴 **If you show it, note §E item 4: at stride 1 it takes 351 s.**

**Score every video you might open, and write the number down, before demo day.** ✅ Done
2026-09-24 for all three.

---

## F. Definition of done — 2026-09-30

**Updated 2026-09-24 (session 18), after `scripts/demo.py` was built and committed (`7c96495`).**

- [~] `scripts/demo.py` exists, runs end to end on all three videos, and has a `--self-check`
      — **exists, 9/9 self-checks pass, verified end to end on `crash1.mov` only.**
      🔴 **Not yet run on `crash2.mov` (~41 s) or `safe.mp4` (~6 min).** Next action.
- [x] It imports `eval/` unchanged; the five regression guards still reproduce exactly
      — **`git status eval/ vendor/` empty; all five re-run after `demo.py` existed, all exact.**
- [x] Pass 1 prints visible progress; pass 2 plays back at watchable speed
      — **44 windows counted live with an ETA; all 257 frames replay, 8.5 s against 7.37 s
      of video (`waitKey` granularity, ~15% slow, smooth).**
- [x] The alert threshold is **derived at runtime**, never typed (B5 / `progress.md` §12)
      — **enforced by a `--self-check` that greps the executable body for threshold
      literals and fails if one ever appears.**
- [x] The terminal summary is readable by someone who has never seen the project
      — **asserted in `--self-check`: no field names, no ids, both the incident and the
      no-incident wording.**
- [x] Every demo video's score is measured and recorded in this file — **§E, day 1.**
- [ ] The `safe.mp4` decision is made and rehearsed — **OPEN. See §E. Decide at rehearsal.**
- [x] No traceback, no warning spam, no dead flags, nothing on screen that needs excusing
      — **the ~35-line transformers LOAD REPORT is suppressed by default; `--loud` restores
      it. Overlay sizes scale with frame height, because at 3408x1910 fixed sizes rendered
      as unreadable specks.**
- [ ] Rehearsed end to end at least twice, timed — **OPEN. Days 6 and 8.**
- [ ] Two answers rehearsed: *"how often does it false-alarm?"* and *"whose data is this?"*
      — **OPEN. Drafts do not exist yet.**

### What `demo.py` looks like when it runs

```
  crash1.mov  --  7.37s, 259 frames at 35.16 fps
  model      BADAS-Open (V-JEPA2 ViT-L), Apache-2.0, run locally on mps
  alert at   0.9733
  which is   the threshold that delivers 80% recall on nexar/test-public n=667 (nanmax)
             -- READ OFF THE DATA at startup, not typed into this file

  PASS 1 of 2 -- scoring. ~43 windows to do.
  window   44/~44 [############################]  55.2s elapsed, ~   0s left
  scored 44 windows in 56.5s (1.28s per window)

  score 0.9968  ->  INCIDENT
  [pass 2 opens, replays 7.37s with the score and alert region on screen]

  INCIDENT DETECTED in crash1.mov
  Something happened around 4.9 seconds in.
  The model was concerned from 3.8s to 6.1s -- about 2.4 seconds of footage.
  ...
```

🔴 **`demo.py` has NO cached-replay path and must never grow one.** Pass 2 replays only
what pass 1 just computed, in the same process. A mode that replays an earlier run's scores
while implying it is live is the one failure this demo cannot survive being caught doing.

---

## G. Deferred — post September 30, only if customers ask

Fleet dashboard · incident review screen · any web UI · database or persistence beyond JSON
files · API · authentication · multi-tenancy · customer/fleet/driver/vehicle models · billing
· real camera integrations · live streaming · cloud video storage · mobile or driver apps ·
production monitoring or hardening · evidence clip extraction (needs ffmpeg, not installed) ·
GPS and map context · severity banding (README §44 forbids deriving it from the score) ·
ego-involvement (D23 — no channel for it).

**None of this is started, and none of it should be until a real fleet asks for it.**

---

## H. Nine days

| Day | Work |
|---|---|
| 1 | Measure `detect.py` end to end on all three videos. Record the numbers in §E. Decide stride. |
| 2 | Build `scripts/demo.py` pass 1 — score with visible progress. |
| 3 | Build pass 2 — playback with the score overlay and the alert region. |
| 4 | The terminal summary: plain English, no jargon, no ids. |
| 5 | `--self-check`, then re-run the five regression guards. |
| 6 | Rehearse. Time it. Cut whatever is slow or confusing. |
| 7 | Fix what rehearsal exposed. Make the `safe.mp4` decision. |
| 8 | Rehearse again, cold, as if for the first time. |
| 9 | Buffer. Do not add features on day 9. |

**The comma2k19 measurement runs on Colab in the background throughout.** It is mostly
waiting, it does not compete for these nine days, and it must not block them. If it slips
past the 30th, that is fine — it changes no part of this demo.

---

## I. Priority

**P0** — `scripts/demo.py` working on `crash1.mov`, honest, no errors.
**P1** — playback overlay, plain-English summary, all three videos measured.
**P2** — stride/skip_predictor speedups, nicer overlay.
**POST** — everything in §G.

If something takes real time and does not change what the fleet manager sees on the 30th,
it is not P0. Push back and point at §G.
