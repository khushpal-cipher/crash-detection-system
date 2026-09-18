# Track C — UK fleet discovery outreach

**Status: DRAFTED, NOT SENT.** Written 2026-09-18 (session 15). **You send these; I do not.**
Update the status line and `docs/fleet_replies.csv` as replies land.

**Track C has been at ZERO for fourteen sessions.** It is not technically blocked, not
compute-bound, and costs nothing. README §41 calls it *"the real critical path to a startup
outcome."* This file exists so that stops being true.

---

## The two documents disagreed, and this is how it was resolved

Named rather than silently picked, because they specify different things:

| Source | Volume | Questions |
|---|---|---|
| `NEW_PLAN.md` §9 | **10 messages**, week 1 | the **same 3** each |
| `README.md` §41 (Track C gate) | **30 calls** | **1** opener, *"do not pitch"* |
| `README.md` §43 item 10 | 10 calls | a "10-question script" |
| `README.md` §45 "If I were you" | thirty calls | 1 |

**Resolution (user-approved, session 15): they nest as a funnel, so neither document changes.**
§9's 10 messages are the **week-1 opener**; README's 30 calls are the **track's completion gate**.
The question counts differ because a cold message can carry three short questions and a call can
carry more — and README's single question is the *opening* of a call, not its entirety.

---

## Who to contact

README §26 Target #1, verbatim: **a UK commercial fleet of 30–300 vehicles that already runs
dashcams and currently reviews footage manually.**

- Below ~30 vehicles there is no budget. Above ~500 you hit procurement.
- At this size **one person can say yes** — an operations or fleet director.
- Roles to look for: Fleet Manager, Transport Manager, Head of Fleet, Operations Director,
  Health & Safety Manager, Compliance Manager.
- Sectors: last-mile delivery, regional haulage, waste/municipal contractors, plant hire,
  facilities and maintenance, food distribution, scaffolding/building supply.
- Channels: LinkedIn direct message, or published `info@`/`careers@`-adjacent contact addresses.

🔴 **Verify before sending:** LinkedIn rate-limits cold InMail and connection notes, and a burst
of near-identical messages can flag an account. Space them out. I have not verified current
limits — check before sending all ten in one sitting.

---

## The three questions (NEW_PLAN §9, verbatim intent)

Every message carries these three and nothing else:

1. **How are collisions currently reviewed?**
2. **How many hours of footage per vehicle per week?**
3. **What false-alarm rate would make an alert useless?**

**Question 3 is the one that matters.** It is the only place in this project where the target
number comes from a customer rather than from a document. README §31 asserts < 0.1 FP/hour; that
figure has never been checked against a person who would actually live with it. If ten fleet
managers say "one a day is fine", the entire technical bottleneck is reframed.

## The call opener (README §41, verbatim)

> *"What happened the last time you trialled an AI dashcam?"*

**Do not pitch.** README §41 is explicit. The answer is the go-to-market, and a pitch destroys it
— people stop describing a problem the moment they think they are being sold to.

---

## Message templates

Short on purpose. Cold messages that scroll are not read.

### A — LinkedIn connection note (300-character limit)

> Hi {Name} — I'm researching how UK fleets actually review dashcam footage after an incident.
> Not selling anything; I'm three questions deep into a study and would value 5 minutes of your
> experience. Would you be open to a short message?

### B — LinkedIn direct message / first reply

> Hi {Name},
>
> I'm doing independent research on collision detection for UK fleets — specifically on
> **false alarms**, which seem to be why most AI dashcam trials get switched off.
>
> Three questions, and I'd be glad of even one-line answers:
>
> 1. How do you currently review collisions — who watches the footage, and when?
> 2. Roughly how many hours of footage does one vehicle generate per week?
> 3. What false-alarm rate would make an alert useless to you? One a day? One an hour?
>
> I'm not selling anything and there's nothing to buy. I'm trying to make sure the thing I'm
> measuring is the thing that actually matters to you.
>
> Thanks either way,
> Khushpal

### C — Email (published contact address)

> **Subject:** Three questions about dashcam footage review — independent research, not a pitch
>
> Hello,
>
> I'm an independent researcher working on collision detection for commercial fleets. I'm trying
> to understand how fleets your size actually handle dashcam footage, and I'd be grateful if this
> reached whoever looks after your fleet or transport operation.
>
> Three questions:
>
> 1. How are collisions currently reviewed?
> 2. How many hours of footage does a vehicle generate per week?
> 3. What false-alarm rate would make an automated alert useless?
>
> There is nothing to buy and I won't follow up with a sales pitch. I'm building a public
> benchmark for false alarms on UK roads, and I would rather it measured what operators care
> about than what is convenient for me to measure.
>
> Happy to share the results with you when they exist.
>
> Khushpal Singh Chouhan

### D — If they reply and offer a call

Open with README §41's question and then be quiet:

> *"Thanks for making time. Can I start with the obvious one — what happened the last time you
> trialled an AI dashcam?"*

Follow-ups only if they stall: *Who watches the footage? · What did you do with the alerts? ·
What made you turn it off? · What would have had to be true for you to keep it?*

**Do not describe what you are building unless they ask twice.**

---

## Kill condition (NEW_PLAN §9, binding)

> **Fewer than 3 substantive replies by end of Week 2 → formally close Track C**, record it in
> `progress.md`, and stop listing it.

"Substantive" = answers at least one of the three questions with content. An out-of-office, a
"no thanks", or a referral that goes nowhere does not count.

**Week 2 ends: 2026-10-02** (counting from this file's creation, 2026-09-18).

A P0 that has not started in fourteen sessions is not a priority, and continuing to list it
without sending anything is the self-deception NEW_PLAN §9 was written to stop.

---

## Logging

Every message goes in `docs/fleet_replies.csv`, one row per organisation, filled at send time —
not reconstructed later. The kill condition is counted off that file, so a message that was sent
but not logged does not exist.
