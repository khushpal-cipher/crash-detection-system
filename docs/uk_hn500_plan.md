# UK-HN-500 — starting the moat without the budget

**Status: PLAN, no footage collected.** Written 2026-09-18 (session 15).
README §34 calls this *"the single most valuable artefact you can build in 90 days"*; §40 lists it
as one of three moats actually buildable in 12 months. It has been at **zero for fourteen
sessions**, and the stated blocker has been money.

---

## The reframing that removes the budget dependency

UK-HN-500 is not one thing. It is three, and **only the third costs money**:

| Component | What it is | Cost |
|---|---|---|
| 1. **The definition** | the eleven categories, what qualifies, how each clip is labelled | **£0** |
| 2. **The harness** | scoring any clip set at the operating point that hits 80% recall on Nexar | **£0 — already built** |
| 3. **The clips** | 500 curated UK hard negatives | needs footage |

Component 2 already exists: `eval/fp_rate.py` implements the counting convention (D55) and
reproduces the committed 92.3 FP/hour; `eval/timing.py::gate_at_recall` derives the operating
point. **A new clip directory drops straight in.** Component 1 is a document.

**So the honest position is: two of the three parts can be finished now, for nothing, and the
project stops being blocked on £500.** What the money buys is the third part, later.

🔴 **Nothing in the technical plan depends on this budget.** Step 2 (comma2k19) and ZOD both
proceed regardless. UK-HN-500 is the *moat*, not the *measurement*.

---

## What costs money, precisely

| Item | Figure | Confidence |
|---|---|---|
| Consumer dashcam | README §33 says **£80–150** | 🔴 **not re-verified for 2026** — check before budgeting |
| Paid UK driver, 4 weeks | the balance of README's *"highest-value £500"* | 🔴 **not verified** — no quote obtained |
| Lawyer review of consent form + data agreement | — | 🔴 **unknown, deliberately not estimated.** README §35 says several items genuinely need a qualified lawyer. I will not invent a number |
| Storage/transfer for ~100 h of footage | — | 🔴 not costed |

**I have not verified any of these.** They come from `README.md`, written earlier in the project.
Treat them as the order of magnitude, not a quote.

## What costs nothing

1. **The category definition** — README §34's taxonomy, reproduced below.
2. **The curation protocol** — what makes a clip a hard negative, how it is labelled, what is
   recorded per clip.
3. **The consent form and data agreement drafts** — README §41 Phase 7 calls these *"the only
   genuine prerequisite… a day of work plus a lawyer's review"*. Drafting is free; only review costs.
4. **The scoring wiring** — done.
5. **ZOD** — CC BY-SA 4.0, commercial use permitted, access already requested.

---

## The eleven categories (README §34, verbatim targets)

| Category | Target | Why it is a hard negative |
|---|---|---|
| Emergency braking | 75 | the classic false positive |
| Speed bumps / potholes | 50 | camera shake without collision |
| Roundabout close-quarters | 50 | proximity without contact |
| Narrow rural passing | 40 | apparent near-miss geometry |
| Bus / lorry pull-out | 40 | large object fills frame suddenly |
| Heavy rain / spray / wipers | 50 | occlusion and motion artefacts |
| Low sun / tunnel transitions | 45 | exposure shock |
| Debris / stone strikes | 30 | impact sound and shake, no collision |
| Car park manoeuvring | 40 | close proximity at low speed |
| Pedestrian-crossing stops | 40 | sudden deceleration near VRUs |
| Motorcycle filtering | 40 | visually alarming, entirely normal |

**Headline metric:** false-positive rate at the operating point that achieves 80% recall on the
Nexar collision test split. Every one of these should score *low*.

---

## ZOD as the zero-cost bridge

`NEW_PLAN.md` §8.2 already selects ZOD as the second negative domain: European urban/rural,
diverse weather and light, **CC BY-SA 4.0 with commercial use permitted**. Access requested
2026-09-18 (`docs/zod_access_request.md`).

**ZOD is not UK and cannot *be* UK-HN-500.** But it is the closest free corpus to the target
distribution — European road furniture, roundabouts, narrow rural roads, genuine weather — and it
lets the entire definition-plus-harness be exercised end to end on real footage before a penny is
spent. If the categories cannot be populated from ZOD, they are probably not well defined.

🔴 **Flag, not assumed: CC BY-SA share-alike is viral.** Confirm **in writing** what licence a
published derivative benchmark must carry before building on ZOD clips. §22/§23 licence discipline
applies; the checklist is in `docs/zod_access_request.md`. A benchmark we cannot publish freely
would lose most of its strategic point (README §34: *"Publish it. Free."*).

---

## 🔴 Do NOT source this from YouTube

README §23 and §45 document that the CCD corpus's **YouTube provenance is a diligence liability**,
and §45 Q4 lists it as a thing that could kill the company. The corpus confound — positives from
YouTube, negatives from BDD100K, 0.9977 AUC that meant nothing — is what killed the original model
and cost this project its first year of work.

**Sourcing the intended moat that way would reproduce the project's original mistake in the one
artefact meant to be defensible.** Community footage is usable only with explicit **per-clip
written permission** (README §33 item 2), which is slower than it sounds.

---

## Minimum next step, costing £0

1. Finish this document's taxonomy into a `categories.json` the harness can read.
2. Draft the consent form and data agreement (unreviewed, marked as such).
3. When ZOD lands: populate what categories it can, run `eval/fp_rate.py`, publish the
   *methodology* even before the UK clips exist.
4. When funds exist: dashcam + driver per README §33, or revisit the driving-school route
   (README §26 ranks it "very high" ease of entry, 1–2 week cycle, and they already own cameras —
   **recorded as an option, not currently being pursued**).

**The order matters.** README §40: *"The benchmark buys credibility, which buys pilots, which
produce the corpus."* Nothing in steps 1–3 needs money.
