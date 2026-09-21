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

## E. Demo footage and its known scores

Three videos, all with usable rights. `data/nexar/LICENSE` permits use; the three local
files predate the project's dataset work.

| Video | Length | Score | Fires at 0.9733? |
|---|---|---|---|
| `videos/crash1.mov` | 7.0 s | **0.9966** | ✅ yes — a real detection |
| `videos/crash2.mov` | — | measure it | — |
| `videos/safe.mp4` | 30 s | **0.9758** | 🔴 **YES — and it is a FALSE ALARM** |

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

**Score every video you might open, and write the number down, before demo day.**

---

## F. Definition of done — 2026-09-30

- [ ] `scripts/demo.py` exists, runs end to end on all three videos, and has a `--self-check`
- [ ] It imports `eval/` unchanged; the five regression guards still reproduce exactly
- [ ] Pass 1 prints visible progress; pass 2 plays back at watchable speed
- [ ] The alert threshold is **derived at runtime**, never typed (B5 / `progress.md` §12)
- [ ] The terminal summary is readable by someone who has never seen the project
- [ ] Every demo video's score is measured and recorded in this file
- [ ] The `safe.mp4` decision is made and rehearsed
- [ ] No traceback, no warning spam, no dead flags, nothing on screen that needs excusing
- [ ] Rehearsed end to end at least twice, timed
- [ ] Two answers rehearsed: *"how often does it false-alarm?"* and *"whose data is this?"*

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
