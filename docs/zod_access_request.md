# ZOD (Zenseact Open Dataset) — access request

**Status: NOT SENT.** Drafted 2026-09-18 (session 14). Update this line with the send date and
any reply, because §22/§23 licence discipline requires the terms to be confirmed **in writing
before use** — the same rule that governs DAD, and the reason DADA is marked internal-only (D43).

## Why we want it

`NEW_PLAN.md` §8.2 selects ZOD as the **second, distinct negative domain** beside comma2k19.
The reasoning matters and should survive into any write-up: comma2k19 is 33 h of California
**highway**, which is the easiest negative case there is — few pedestrians, few intersections,
few occlusions. Any FP/hour measured there is a **floor**, not a general rate, and reporting it
as one would be the mirror image of the 0.90-hour problem this project already has. ZOD's
European urban/rural mix, with diverse weather and lighting, is the corrective. §8.2 also
requires the two be reported **separately and never pooled**.

ZOD is also one of only two sources in `NEW_PLAN.md` §8.1's **"genuinely independent"** tier —
neither trained on nor annotated by Nexar — and it is licensed **CC BY-SA 4.0, commercial use
permitted**, which is why it is in the plan and BDD100K (research-only) is explicitly excluded.

## How to request

Email **opendataset@zenseact.com** with name, affiliation, the Dropbox email the download
should be shared with, and a description of intended use. Access is granted manually, so start
this early — it is the long pole, not the scoring.

## Draft

> **Subject:** ZOD access request — false-positive benchmarking for dashcam collision detection
>
> Hello,
>
> I would like to request access to the Zenseact Open Dataset.
>
> **Name:** Khushpal Singh Chouhan
> **Affiliation:** Independent researcher / early-stage startup (pre-incorporation), India
> **Dropbox email:** khushpalkun@gmail.com
>
> **Intended use.** I am evaluating a public, Apache-2.0 collision-detection model
> (BADAS-Open, V-JEPA2 based) for use in fleet dashcam review. The benchmark I have,
> Nexar's collision-prediction test split, contains only **0.9 hours** of negative footage,
> which makes a false-alarm rate below roughly 1 per hour impossible to measure at any
> confidence — my target is below 0.1 per hour.
>
> I would like to use ZOD as a source of **negative (non-collision) driving footage** to
> measure false positives per hour on European roads across varied weather and lighting. I
> plan to pair it with comma2k19 as a second negative domain and to report the two
> **separately**, since highway-only footage would understate the rate. No annotations are
> required for this — I need the footage itself, and its being collision-free by construction
> is the point.
>
> I intend to publish the methodology and the resulting false-positive numbers, with
> attribution to ZOD under CC BY-SA 4.0. If any additional terms apply to this use, or if
> the Sequences or Drives subsets would be more appropriate than Frames for continuous-footage
> evaluation, I would be glad to be told.
>
> Thank you,
> Khushpal Singh Chouhan

## On answering, record here

- [ ] Date sent
- [ ] Date access granted
- [ ] **Licence terms confirmed in writing** (required before any use — §22/§23)
- [ ] Which subset was granted (Frames / Sequences / Drives) and its size on disk
- [ ] Whether any term restricts commercial use or publication of derived metrics
