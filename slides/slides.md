---
theme: default
title: SFT to RL — Training Models to Think Better
colorSchema: light
fonts:
  sans: 'IBM Plex Sans'
  serif: 'Kalam'
  mono: 'IBM Plex Mono'
css: unocss
transition: fade
mdc: true
---

<div class="eyebrow">AI Yatra &middot; Hands-On Session</div>

# Supervised Fine-Tuning to Reinforcement Learning

<div style="font-family:'Kalam',cursive; font-size:1.3em; color:var(--muted); margin-top:0.3em;">
Training Models to Think Better
</div>

<div class="stamp" style="margin-top:2em;">Sept 12 &middot; LSEG Hyderabad</div>

<!--
Opening: this is a hands-on session, not a lecture. Everything shown ran for real —
on a laptop CPU and on free-tier Colab GPUs — and every number on every slide is a
real measured result, including the ones where the experiment didn't work. That's
the point: we'll see genuine task improvement AND genuine reward-optimization
failure, side by side, so the difference is concrete instead of theoretical.
-->

---
layout: center
title: Scan for the repo
---

<div style="text-align:center;">
<img src="/repo-qr.png" style="width:400px; height:400px; border:2px solid var(--rule); border-radius:16px; padding:14px; background:white; margin:0 auto; display:block;" />
<div style="font-family:'IBM Plex Mono',monospace; font-size:1.2em; color:var(--navy); margin-top:0.8em; font-weight:600;">github.com/AI-Yatra/sft-to-rl</div>
<div style="color:var(--muted); margin-top:0.3em; font-size:0.95em;">Everything's in there — code, trained weights, results, this deck</div>
</div>

<!--
Hold on this slide for 30-60s at the very start — let the room actually scan it
before diving in, rather than mentioning the repo once and moving on. Self-generated
QR code (not a third-party shortener), points directly at
github.com/AI-Yatra/sft-to-rl, verified public and clonable.
-->

---
layout: center
title: Agenda
---

## Today's path

<div class="grid grid-cols-2 gap-6 mt-6">

<div class="panel-card">
<h3>Part 1 — One task, three training methods</h3>

Text → calendar JSON, one small model, taken through SFT → DPO → GRPO.
Real numbers at every stage. One field never gets fixed — we'll find out why.
</div>

<div class="panel-card">
<h3>Part 2 — Skip SFT entirely</h3>

A base model with zero training, taken straight to reinforcement learning
on a verifiable reward. Live demo on this laptop, no GPU required to watch.
</div>

</div>

<!--
Set expectations: Part 1 ends on an honest negative result (a field that never
gets fixed across three different methods). Part 2 is the payoff — RL succeeding
where it's supposed to succeed, once we understand why Part 1's failure happened.
-->

---
title: Two different questions
---

## Two questions people mix up

<div class="grid grid-cols-2 gap-5 mt-6">

<div class="panel-card">
<h3>How do you update weights?</h3>

<div style="font-size:1.1em; margin-top:0.5em;">
<b>LoRA</b> — freeze the original weights, train two small extra matrices
per layer instead of touching the model itself.
</div>

<div class="stamp" style="margin-top:1em;">an efficiency trick</div>
</div>

<div class="panel-card">
<h3>What signal do you train on?</h3>

<div style="font-size:1.1em; margin-top:0.5em;">
<b>SFT</b> — imitate labeled examples<br>
<b>DPO</b> — prefer one answer over another<br>
<b>GRPO</b> — trial and reward, no labels at all
</div>

<div class="stamp" style="margin-top:1em;">a training method</div>
</div>

</div>

<div class="mt-8" style="text-align:center; color:var(--muted); font-family:'IBM Plex Mono',monospace; font-size:0.9em;">
LoRA is orthogonal to all three — you can do SFT with LoRA, DPO with LoRA, GRPO with LoRA. We did.
</div>

<!--
This slide is the one to slow down on. The most common confusion in the room will
be "is LoRA a type of RL?" — no. LoRA answers HOW you update weights (efficiently).
SFT/DPO/GRPO answer WHAT SIGNAL trains those weights. Every experiment today used
LoRA underneath, varying only the training signal.
-->

---
title: LoRA in one picture
---

## LoRA: freeze the big thing, train two small things

<div class="grid grid-cols-2 gap-8 mt-4 items-center">

<div class="lora-diagram">
  <div class="lora-col">
    <div class="lora-box lora-w">W</div>
    <div class="lora-caption">frozen<br><span style="opacity:0.7;">(d_out &times; d_in)</span></div>
  </div>
  <div class="lora-op">+</div>
  <div class="lora-col">
    <div class="lora-ab-row">
      <div class="lora-box lora-a">A</div>
      <div class="lora-op" style="margin:0 0.3em;">&times;</div>
      <div class="lora-box lora-b">B</div>
    </div>
    <div class="lora-caption">trainable, rank r<br><span style="opacity:0.7;">(d_out&times;r) &middot; (r&times;d_in)</span></div>
  </div>
</div>

<div>
<div class="panel-card">
<div class="metric" style="font-size:1.4em; color:var(--navy);">1,638,400 / 363,459,520</div>
<div style="color:var(--muted); margin-top:0.3em;">trainable params in our SFT run</div>
<div class="metric" style="font-size:1.6em; color:var(--gold); margin-top:0.6em;">0.45%</div>
</div>

<div class="mt-4" style="color:var(--muted);">
The adapter file is a few MB, not gigabytes. That's what made "train on a laptop CPU
in 17 minutes" possible at all.
</div>
</div>

</div>

<div class="mt-6 panel-card" style="text-align:center;">
<div style="font-family:'IBM Plex Mono',monospace; font-size:1.05em;">
y = x&middot;W <span style="color:var(--muted);">+</span> x&middot;A&middot;B
</div>
<div class="mt-2" style="color:var(--muted); font-size:0.85em;">
A(d_out&times;r) &times; B(r&times;d_in) &rarr; the shared <b>r</b> cancels, leaving
(d_out&times;d_in) — <b>the same shape as W</b>, which is the only reason
"+" is even legal here. r=8 in our run vs. hundreds of real dimensions in W:
same shape, far fewer numbers to learn.
</div>
</div>

<!--
Point at the diagram: W is the original model's weight matrix, frozen entirely.
A and B are small — their product approximates a full-rank update without ever
materializing one. r=8 in our config means A and B's inner dimension is 8, tiny
compared to the hundreds of dimensions in W. If asked "why does A×B come out
the same shape as W": the inner dimension r cancels in the multiplication,
leaving the outer dimensions (d_out × d_in) -- same shape, which is required
for W + AB to even be valid matrix addition.
-->

---
layout: section
---

# Part 1
## One task, three training methods

<div style="color:var(--muted); margin-top:1em;">Text → calendar JSON &middot; SmolLM2-360M-Instruct</div>

---
title: The task
---

## The task: text → calendar JSON

<div class="panel-card mt-4">
<div style="font-family:'IBM Plex Mono',monospace; font-size:0.95em;">
<div style="color:var(--muted);">Input:</div>
<div style="margin:0.3em 0 0.8em 0;">"let's grab coffee tues at 3, zoom?"</div>
<div style="color:var(--muted); margin-bottom:0.3em;">Required output:</div>

```json
{
  "title": "Coffee",
  "date": "2026-09-08",
  "time": "15:00",
  "location": "Zoom",
  "attendees": []
}
```

</div>
</div>

<div class="mt-4" style="color:var(--muted);">
Deterministic synthetic dataset (seeded) — gold labels are exact by construction,
not hand-annotated. Two eval sets: <b>seen phrasing</b> (same style as training) and
<b>held-out phrasing</b> (phrasing types never shown during training — the real
generalization test).
</div>

<!--
Explain why field accuracy, not JSON validity, is the metric that matters: JSON
validity sits near 100% at every single stage of this whole talk, before any
training at all — it never discriminates a good model from a bad one. Field
accuracy is the real signal.
-->

---
title: SFT
---

## SFT — imitate labeled examples

<div class="fullform">Supervised Fine-Tuning</div>

<div class="grid grid-cols-2 gap-6 mt-4">
<div class="panel-card">
<h3>What it does</h3>
Show the model 150 (text, correct JSON) pairs. Mask the prompt tokens,
train only on producing the correct completion. Straightforward next-token
prediction, just on curated pairs.
</div>
<div class="panel-card">
<h3>What it can't do</h3>
Only as good as the labeled examples. Teaches the <i>shape</i> of a correct
answer — no guarantee the model can generalize the underlying logic
(we'll see this matters for dates specifically).
</div>
</div>

<div class="mt-6 panel-card good">
<div class="grid grid-cols-4 gap-4" style="text-align:center;">
<div><div class="metric good" style="font-size:1.3em;">150</div><div style="font-size:0.8em;color:var(--muted);">examples</div></div>
<div><div class="metric good" style="font-size:1.3em;">17 min</div><div style="font-size:0.8em;color:var(--muted);">CPU, laptop — $0</div></div>
<div><div class="metric good" style="font-size:1.3em;">38→56%</div><div style="font-size:0.8em;color:var(--muted);">field accuracy</div></div>
<div><div class="metric good" style="font-size:1.3em;">30→7</div><div style="font-size:0.8em;color:var(--muted);">hallucinations</div></div>
</div>
</div>

<!--
Run this live if time allows: from finetune-demo/, `uv run python sft/train_lora.py
--limit 150 --epochs 2`. While it trains (real ~17 min), walk the room through the
DPO section conceptually so the wait isn't dead air.
-->

---
title: SFT results in full
---

## SFT — full before/after, both eval sets

| Field | Seen before | Seen after | Held-out before | Held-out after |
|---|---|---|---|---|
| title | 23/30 | 26/30 | 25/30 | 27/30 |
| **date** | **0/30** | **1/30** | **0/30** | **5/30** |
| time | 11/30 | 13/30 | 11/30 | 15/30 |
| location | 22/30 | 21/30 | 17/30 | 19/30 |
| attendees | 7/30 | 23/30 | 4/30 | 23/30 |
| **Overall** | **42.0%** | **56.0%** | **38.0%** | **59.3%** |
| Hallucinated attendees | 30 | 7 | 36 | 6 |

<div class="mt-4 stamp">Held-out (phrasing never seen in training) improved MORE than seen — not memorization</div>

<!--
Point at attendees: biggest single-field jump (4→23), matches the hallucination
count dropping 36→6 — same underlying fix. Point at date: barely moved. Flag it
now, we come back to it three more times today.
-->

---
title: DPO
---

## DPO — prefer one answer over another

<div class="fullform">Direct Preference Optimization</div>

<div class="grid grid-cols-2 gap-6 mt-4">
<div class="panel-card">
<h3>How the data looks</h3>
<div style="font-family:'IBM Plex Mono',monospace; font-size:0.8em; margin-top:0.5em;">
<div style="color:var(--good);">chosen: gold JSON</div>
<div style="color:var(--bad); margin-top:0.3em;">rejected: gold JSON with one<br>targeted corruption</div>
</div>
</div>
<div class="panel-card">
<h3>Corruption types</h3>
<div style="font-size:0.85em; color:var(--muted);">
wrong_date &middot; wrong_time &middot; hallucinate_attendee<br>
drop_attendee &middot; wrong_location &middot; mis_nested
</div>
</div>
</div>

<div class="mt-6" style="color:var(--muted);">
No reward function, no sampling — just: given two candidate answers, learn to
assign higher probability to the better one. Good at stamping out
<i>specific known failure modes</i>, since you choose exactly what to corrupt.
</div>

<div class="mt-6 panel-card good">
<div class="grid grid-cols-3 gap-4" style="text-align:center;">
<div><div class="metric good" style="font-size:1.3em;">29 min</div><div style="font-size:0.8em;color:var(--muted);">Colab T4, free tier</div></div>
<div><div class="metric good" style="font-size:1.3em;">$0</div><div style="font-size:0.8em;color:var(--muted);">no paid compute</div></div>
<div><div class="metric good" style="font-size:1.3em;">150</div><div style="font-size:0.8em;color:var(--muted);">preference pairs</div></div>
</div>
</div>

<div class="mt-3" style="text-align:center; color:var(--muted); font-size:0.82em;">
Why GPU here and not the laptop CPU like SFT: DPOTrainer runs <b>two</b> forward
passes per step (policy + reference model) — double the throughput need.
</div>

---
title: The DPO gotcha
---

## The bug we actually hit

<div class="grid grid-cols-2 gap-6 mt-4">

<div class="panel-card bad">
<h3>lr = 5e-6 (default)</h3>
<div style="font-family:'IBM Plex Mono',monospace; font-size:0.85em; margin-top:0.5em;">
loss: flat at ln(2) &approx; 0.693<br>
final reward margin: <span class="metric bad">negative</span>
</div>
<div class="mt-3" style="color:var(--muted); font-size:0.9em;">
A completely dead run. No learning happened at all — and nothing crashed to tell us.
</div>
</div>

<div class="panel-card good">
<h3>lr = 5e-5 (10&times; higher)</h3>
<div style="font-family:'IBM Plex Mono',monospace; font-size:0.85em; margin-top:0.5em;">
loss: 0.69 &rarr; 0.55-0.60<br>
reward accuracy: <span class="metric good">0.80-0.95</span>
</div>
<div class="mt-3" style="color:var(--muted); font-size:0.9em;">
Real learning, immediately visible in the training curve.
</div>
</div>

</div>

<div class="mt-6 stamp">TRL's default learning rate is tuned for full fine-tuning, not LoRA-scale updates</div>

<!--
This is a genuinely useful "gotcha" for anyone doing this themselves: LoRA's
much smaller parameter count needs a correspondingly larger learning rate to
move at all, and the failure mode is silent — no error, no crash, just a flat
loss curve that looks like nothing happened, because nothing did.
-->

---
title: DPO results
---

## DPO — the hallucination fix

| Stage | Seen | Held-out | Hallucinations (seen) | Hallucinations (held-out) |
|---|---|---|---|---|
| +SFT | 56.0% | 58.7% | 4 | 6 |
| **+SFT+DPO** | **62.0%** | **66.0%** | **0** | **0** |

<div class="mt-6" style="color:var(--muted);">
Hallucinations go to <b>exactly zero on both eval sets</b>. Not a coincidence —
the <code>hallucinate_attendee</code> corruption type directly targeted this
failure mode, and DPO delivered on exactly what it was pointed at.
</div>

---
title: GRPO
---

## GRPO — trial and reward, no labels

<div class="fullform">Group Relative Policy Optimization</div>

<div class="mt-4" style="text-align:center;">
<div class="grid grid-cols-5 gap-2 items-center">
  <div class="pipeline-step">1 prompt</div>
  <div class="pipeline-arrow">&rarr;</div>
  <div class="pipeline-step">8 generations</div>
  <div class="pipeline-arrow">&rarr;</div>
  <div class="pipeline-step active">reward each,<br>compare to group</div>
</div>
</div>

<div class="mt-8 panel-card">
No gold completions in the dataset at all — just the prompt, plus enough
information for the reward function to grade whatever the model produces.
The model's own generations become the training signal, scored against
each other within the group of 8.
</div>

<div class="mt-6 panel-card good">
<div class="grid grid-cols-3 gap-4" style="text-align:center;">
<div><div class="metric good" style="font-size:1.3em;">23.5 min</div><div style="font-size:0.8em;color:var(--muted);">Colab T4, free tier</div></div>
<div><div class="metric good" style="font-size:1.3em;">$0</div><div style="font-size:0.8em;color:var(--muted);">no paid compute</div></div>
<div><div class="metric good" style="font-size:1.3em;">1,600</div><div style="font-size:0.8em;color:var(--muted);">generations (200 prompts &times; 8)</div></div>
</div>
</div>

<div class="mt-3" style="text-align:center; color:var(--muted); font-size:0.82em;">
Why GPU here too: 8 generations per prompt, every step — the laptop CPU could
do this, just not before the talk ends.
</div>

<!--
"Group relative" is the key word in the name: the model isn't compared against
an absolute bar, it's compared against its OWN other attempts on the same
prompt. That's what "advantage" means here — better than the group average,
or worse.
-->

---
title: Reward function walkthrough
---

## The reward function IS the field scorer

<div class="panel-card" style="font-size:0.72em; line-height:1.5;">

```python
def _score_one(gold, raw_output, source_text):
    parsed = parse_json_loose(raw_output)
    if parsed is None:
        return 0.0                    # invalid JSON: hard zero, no partial credit

    hits = 0
    if title_matches(gold):     hits += 1
    if date_matches(gold):      hits += 1
    if time_matches(gold):      hits += 1
    if location_matches(gold):  hits += 1
    if attendees_match(gold):   hits += 1
    field_score = hits / 5

    hallucinated = names_in_output_not_in(source_text)
    penalty = 0.1 * hallucinated
    return max(0.0, field_score - penalty)
```

</div>

<div class="mt-4" style="color:var(--muted);">
"Reward went up" and "field accuracy went up" are <i>the same measurement</i> —
deliberately, so the reward can't diverge from what we actually care about.
The hallucination penalty exists to close a specific loophole: without it,
always guessing <code>attendees: []</code> could farm reward on easy examples.
</div>

<div class="mt-3 stamp">Live: CodeTour walkthrough of this file, line by line — see .tours/ in the repo</div>

<!--
This is the "reward hacking" concept in one slide. If your reward function has
a blind spot, RL will find it — that's not a bug in RL, it's RL doing exactly
what it's designed to do: maximize the number you gave it, however it can.

If presenting from VS Code: switch over now and run CodeTour: Start Tour on
.tours/grpo-reward-function.tour instead of just showing this static slide —
it walks the exact same points (hard zero on invalid JSON, hallucination
penalty, group-relative advantage) directly in the real source file.
-->

---
title: GRPO results
---

## The full pipeline, four stages

| Stage | Seen | Held-out | Halluc. (seen/held) | Time &middot; infra |
|---|---|---|---|---|
| BASE | 40.0% | 38.7% | 36 / 37 | — |
| +SFT | 56.0% <span style="color:var(--good); font-size:0.8em;">(+16pp)</span> | 58.7% <span style="color:var(--good); font-size:0.8em;">(+20pp)</span> | 4 / 6 | 17 min &middot; CPU laptop |
| +SFT+DPO | 62.0% <span style="color:var(--good); font-size:0.8em;">(+6pp)</span> | 66.0% <span style="color:var(--good); font-size:0.8em;">(+7pp)</span> | 0 / 0 | 29 min &middot; Colab T4 |
| **+SFT+DPO+GRPO** | **64.0%** <span style="color:var(--muted); font-size:0.8em;">(+2pp)</span> | **66.0%** <span style="color:var(--muted); font-size:0.8em;">(+0pp)</span> | 0 / 0 | 23.5 min &middot; Colab T4 |

<div class="mt-6" style="color:var(--muted);">
A small, real gain on seen phrasing — flat on held-out. Consistent with the
training curve itself: reward drifted mildly upward over one epoch, not a
clean climb. Read as: GRPO didn't hurt, and helped a little, but one epoch on
200 prompts wasn't enough to move generalization further than DPO already had.
</div>

---
title: Training curves, all three stages
---

## What the training curves actually looked like

<div class="grid grid-cols-3 gap-3 mt-4">

<div class="curve-panel sft">
<h4>SFT &middot; loss</h4>
<div class="curve-bars">
<div class="bar" style="height:100%"></div><div class="bar" style="height:88%"></div><div class="bar" style="height:78%"></div><div class="bar" style="height:70%"></div><div class="bar" style="height:62%"></div><div class="bar" style="height:55%"></div><div class="bar" style="height:49%"></div><div class="bar" style="height:44%"></div><div class="bar" style="height:40%"></div><div class="bar" style="height:37%"></div><div class="bar" style="height:34%"></div><div class="bar" style="height:32%"></div><div class="bar" style="height:30%"></div><div class="bar" style="height:29%"></div><div class="bar" style="height:28%"></div><div class="bar" style="height:27%"></div>
</div>
<div class="curve-endlabels"><span>step 0</span><span>step 60</span></div>
<div class="curve-caption">smooth decline, 2 epochs &rarr; final loss 0.316. Clean fit — expected, it's imitating 150 fixed examples.</div>
</div>

<div class="curve-panel dpo">
<h4>DPO &middot; loss (after lr fix)</h4>
<div class="curve-bars">
<div class="bar" style="height:100%"></div><div class="bar" style="height:92%"></div><div class="bar" style="height:82%"></div><div class="bar" style="height:74%"></div><div class="bar" style="height:68%"></div><div class="bar" style="height:63%"></div><div class="bar" style="height:60%"></div><div class="bar" style="height:58%"></div><div class="bar" style="height:57%"></div><div class="bar" style="height:56%"></div><div class="bar" style="height:56%"></div><div class="bar" style="height:55%"></div><div class="bar" style="height:56%"></div><div class="bar" style="height:55%"></div><div class="bar" style="height:56%"></div><div class="bar" style="height:55%"></div>
</div>
<div class="curve-endlabels"><span>step 0</span><span>step ~90</span></div>
<div class="curve-caption">drops fast, then flattens ~0.55-0.60 for the back half of training — this IS a plateau, just a shallower one than GRPO's.</div>
</div>

<div class="curve-panel grpo">
<h4>GRPO &middot; reward</h4>
<div class="curve-bars">
<div class="bar" style="height:55%"></div><div class="bar" style="height:70%"></div><div class="bar" style="height:45%"></div><div class="bar" style="height:60%"></div><div class="bar" style="height:80%"></div><div class="bar" style="height:50%"></div><div class="bar" style="height:65%"></div><div class="bar" style="height:40%"></div><div class="bar" style="height:58%"></div><div class="bar" style="height:72%"></div><div class="bar" style="height:48%"></div><div class="bar" style="height:66%"></div><div class="bar" style="height:52%"></div><div class="bar" style="height:78%"></div><div class="bar" style="height:60%"></div><div class="bar" style="height:50%"></div>
</div>
<div class="curve-endlabels"><span>step 0</span><span>step 200</span></div>
<div class="curve-caption">no clean climb at all — noisy oscillation the entire run (reward 0.32-0.61). Higher bar = higher reward here, unlike the two loss panels.</div>
</div>

</div>

<div class="mt-6 stamp" style="display:block; width:fit-content; margin:1.2em auto 0;">Shapes reconstructed from each run's logged summary stats (start/end/range) — not a raw per-step export</div>

<!--
Point out the contrast in shapes deliberately: SFT looks like "normal" training
because it IS normal training (supervised, fixed target, clean loss surface).
DPO looks like normal training that plateaus early. GRPO doesn't even look like
a loss curve — it's noisy the whole time, because reward-based RL with only 200
prompts and 1 epoch just doesn't have enough steps to smooth out that noise.
This sets up the next-but-one slide: plateauing itself is completely normal,
what matters is WHY each one plateaus where it does.
-->

---
title: The honest problem
---

## Three methods. One field never moves.

<div class="mt-6 panel-card bad" style="text-align:center;">
<div style="font-family:'IBM Plex Mono',monospace; font-size:1em; color:var(--muted);">held-out <code>date</code> accuracy</div>
<div class="metric bad" style="font-size:2.2em; margin-top:0.3em;">4 / 30</div>
<div style="color:var(--muted); margin-top:0.3em;">...after SFT, DPO, AND GRPO</div>
</div>

<div class="mt-6" style="text-align:center; color:var(--muted);">
Relative-date arithmetic ("next Thursday", "in 3 days") is the one thing
three different training methods couldn't fix. This is the honest thing to
say out loud, and the most useful part of today for anyone doing this for real.
</div>

<!--
Pause here. This is the pivot point of the talk. Everything so far has been
"here's a method, here's the number going up" — this slide is "here's where
that stopped working," and the next section is the investigation into why.
-->

---
title: Follow-up experiment
---

## Chasing the date field

<div class="grid grid-cols-2 gap-5 mt-4">

<div class="panel-card">
<h3>Attempt 1: Chain-of-thought SFT</h3>
Teach the model to write out date reasoning before the JSON, via imitation
(200 fresh examples, "Reasoning: ... / JSON: ...").
<div class="mt-3 tag fail">date: 0/30 both sets</div>
</div>

<div class="panel-card">
<h3>Attempt 2: Date-weighted GRPO</h3>
On top of the CoT model, reward 60% on date correctness, 40% on everything
else — concentrate the signal specifically on the failing field.
<div class="mt-3 tag fail">date: 0/30 both sets</div>
</div>

</div>

<div class="mt-6" style="color:var(--muted); text-align:center;">
Neither fixed it. The next slide is why — and it's not a reward-design bug.
</div>

---
title: Root cause
---

## Why GRPO couldn't fix it

<div class="panel-card mt-4">
<div style="font-size:1.05em;">
GRPO can only reinforce behavior the model <b>already produces sometimes, by chance.</b>
</div>
</div>

<div class="mt-6 grid grid-cols-3 gap-3" style="text-align:center;">
<div class="pipeline-step">8 generations<br>per prompt</div>
<div class="pipeline-step">baseline date<br>accuracy &approx; 0-3%</div>
<div class="pipeline-step active">all 8 get the<br>same (zero) reward</div>
</div>

<div class="mt-6" style="color:var(--muted);">
No variance across the group means no contrast for the "group relative"
advantage to learn from. Weighting the reward toward date does nothing if
the policy essentially never samples a correct date to begin with — this is
the standard <b>RL exploration problem</b>, not a reward-shaping mistake.
Reward shaping amplifies existing signal; it can't manufacture signal from
a near-zero success rate.
</div>

<!--
This is the single most important concept slide in the whole talk. Everyone
who's heard "just add a reward for the thing you want" needs to hear the
counter-case: it only works if the model can already stumble into the right
answer occasionally. Set up Part 2 as the demonstration of GRPO working when
that condition IS met.
-->

---
title: Why plateaus happen
---

## Every curve plateaus. This is normal — the question is why.

<div class="panel-card mt-4" style="text-align:center;">
<div style="font-size:1.05em;">
A curve flattens the moment more steps stop adding <b>new information</b> —
not a failure state, just the training signal running out.
</div>
</div>

<div class="grid grid-cols-3 gap-3 mt-6">

<div class="panel-card">
<h3 style="color:var(--navy);">SFT plateaus because</h3>
<div style="font-size:0.9em; color:var(--muted); margin-top:0.4em;">
150 <b>fixed</b> examples. More epochs on the same set refines the fit to
those examples — doesn't add a single new fact about date arithmetic.
</div>
</div>

<div class="panel-card">
<h3 style="color:var(--gold);">DPO plateaus because</h3>
<div style="font-size:0.9em; color:var(--muted); margin-top:0.4em;">
6 <b>fixed</b> corruption types. Only teaches "avoid this specific mistake" —
<code>wrong_date</code> was 1 of 6, never enough concentrated signal alone.
</div>
</div>

<div class="panel-card">
<h3 style="color:var(--good);">GRPO plateaus because</h3>
<div style="font-size:0.9em; color:var(--muted); margin-top:0.4em;">
Zero <b>reward variance</b> on date. Nothing new can be learned from a
signal that's constant across every sample — see the last slide.
</div>
</div>

</div>

<div class="mt-8 stamp" style="display:block; width:fit-content; margin:2em auto 0;">Same underlying cause, three different flavors: no new information entering the training signal</div>

<div class="mt-6" style="text-align:center; color:var(--muted); font-size:0.88em;">
Not unique to our small task — the same shape (fast improvement, then flat)
is why pretraining needs ever <i>more and different</i> data, not just more
steps on the same corpus. Pushing past a plateau always means changing what
the model sees, not just how long it sees it.
</div>

---
layout: section
---

# Part 2
## What if we skip SFT entirely?

<div style="color:var(--muted); margin-top:1em;">Base model &middot; straight to RL &middot; a task with genuine exploration signal</div>

<!--
Transition: Part 1 showed GRPO failing on a task where the base model almost
never got it right by chance. Part 2 is the opposite setup on purpose — a task
where the base model gets SOME partial credit sometimes, so there's real
variance for GRPO to climb.
-->

---
title: The idea
---

## Countdown: a classic RL testbed

<div class="panel-card mt-4">
<div style="font-family:'IBM Plex Mono',monospace; font-size:0.95em;">
Given numbers <b>[58, 80, 36]</b> and target <b>14</b>, write an equation using
each number exactly once that equals the target.
</div>
<div class="mt-3" style="color:var(--good); font-family:'IBM Plex Mono',monospace;">
(58 − 80) + 36 = 14 &check;
</div>
</div>

<div class="mt-6 grid grid-cols-3 gap-4" style="text-align:center;">
<div class="panel-card"><div class="metric" style="font-size:1.2em;">Qwen2.5-0.5B</div><div style="font-size:0.8em;color:var(--muted);">base, no instruct tuning</div></div>
<div class="panel-card"><div class="metric" style="font-size:1.2em;">no SFT</div><div style="font-size:0.8em;color:var(--muted);">zero labeled examples, ever</div></div>
<div class="panel-card"><div class="metric" style="font-size:1.2em;">GRPO only</div><div style="font-size:0.8em;color:var(--muted);">150 steps, ~47 min, one Colab T4</div></div>
</div>

<div class="mt-4 stamp" style="display:block; width:fit-content; margin:1.5em auto 0;">This is the setup from the TinyZero / Jiayi-Pan Countdown reproductions</div>

---
title: Countdown reward function
---

## Reward: pure arithmetic, no reward model

<div class="panel-card" style="font-size:0.75em; line-height:1.55;">

```python
def score_completion(completion, target, nums):
    match = ANSWER_RE.search(completion)
    if not match:
        return 0.0                 # no answer tag at all

    used = numbers_used(match)
    if used != nums:
        return 0.1                 # right format, wrong numbers

    result = safe_eval(match)      # ast-based, not eval() -- no code injection
    if result == target:
        return 1.0                 # solved
    return 0.2                     # right numbers, wrong arithmetic
```

</div>

<div class="mt-4" style="color:var(--muted);">
Four reward levels, not just 0/1 — partial credit gives the model something
to climb even before it's fully correct. Ground truth, computed in Python,
no learned reward model anywhere in the loop.
</div>

---
title: Countdown results
---

## Base model vs. after 150 GRPO steps

<div class="grid grid-cols-2 gap-6 mt-4">

<div class="panel-card bad">
<h3>BASE (untrained)</h3>
<div class="mt-3">
<div>Solved: <span class="metric bad">0 / 100</span></div>
<div>Valid format: <span class="metric bad">5 / 100</span></div>
<div>Mean reward: <span class="metric bad">0.006</span></div>
</div>
</div>

<div class="panel-card good">
<h3>+GRPO, 150 steps, no SFT</h3>
<div class="mt-3">
<div>Solved: <span class="metric good">5 / 100</span></div>
<div>Valid format: <span class="metric good">67 / 100</span></div>
<div>Mean reward: <span class="metric good">0.121</span> <span style="color:var(--muted); font-size:0.8em;">(20&times;)</span></div>
</div>
</div>

</div>

<div class="mt-6" style="color:var(--muted); text-align:center;">
Format compliance moved the most (5%&rarr;67%), correctness moved less but real
(0%&rarr;5%) — the textbook order: structure is learned before correctness,
because the reward can't teach arithmetic until the model reliably emits
something parseable in the first place.
</div>

<!--
This is the payoff slide for the whole talk. Contrast explicitly with the
date-field failure: here the base model had SOME chance of getting partial
credit (it can emit numbers, it can attempt arithmetic), so GRPO had variance
to climb. There it had essentially none. Same algorithm, opposite outcome,
and now the room understands why.
-->

---
title: Live demo
---

## Live: run it yourself, right now

<div class="panel-card mt-4" style="font-family:'IBM Plex Mono',monospace; font-size:1em;">
<div style="color:var(--muted);">$</div>
<div>cd finetune-demo && uv sync</div>
<div>cd countdown-rl && uv run --project .. python showcase_demo.py</div>
</div>

<div class="mt-6" style="color:var(--muted);">
No GPU needed to run this — the trained adapter is a few MB, committed to the
repo. Loads the base model and the trained model side by side, runs both on
the same puzzles, prints the comparison live. Takes about a minute to load,
then seconds per example.
</div>

<div class="mt-4 stamp">Add --interactive to type your own numbers and target</div>

<!--
Actually run this on stage. Let the room watch the base model fail to close
an <answer> tag, then the trained model solve one live. This is the moment
of the whole talk — point it out explicitly.
-->

---
title: If you run it yourself
---

## If you run it yourself: two things you'll notice

<div class="grid grid-cols-2 gap-5 mt-4">

<div class="panel-card">
<h3>There's no system prompt at all</h3>
<div style="font-size:0.68em; margin-top:0.6em; line-height:1.5;">

```text
"Using the numbers {nums}, create an
equation that equals {target}. ...
Show your work in <think> </think>
tags, then return the final equation
in <answer> </answer> tags...
<think>"      <- prompt ends HERE, unclosed
```

</div>
<div class="mt-3" style="color:var(--muted); font-size:0.88em;">
One flat string, tokenized directly — no chat roles, no <code>apply_chat_template</code>.
It ends mid-tag on purpose, priming the model to continue as if already reasoning.
<b>The think/answer format isn't a model feature</b> — it's plain text the model
was taught to produce because the reward function only pays out when it finds
a well-formed <code>&lt;answer&gt;</code> block.
</div>
</div>

<div class="panel-card bad">
<h3>The [10, 3, 5] example can loop forever</h3>
<div style="color:var(--muted); font-size:0.88em; margin-top:0.6em;">
Greedy decoding (<code>do_sample=False</code>) always picks the single most
probable next token — no randomness to break a repeating pattern. With no
repetition penalty either (tried, made things worse — hallucinated
<code>x</code>/<code>y</code>/<code>z</code> variables), a model that gets stuck
predicting <code>&lt;think&gt;10 + 3&lt;/think&gt;</code> as "most probable"
will keep predicting exactly that, forever. <code>&lt;/answer&gt;</code> never
appears, so nothing stops it early.
</div>
<div class="mt-3 tag fail">Kept in the demo on purpose — an honest failure, not a bug</div>
</div>

</div>

<div class="mt-6 stamp" style="display:block; width:fit-content; margin:0 auto;">Same exact decoding setup that produced every number in RESULTS.md — no cherry-picking for the demo</div>

<!--
This slide exists because someone will run showcase_demo.py themselves and hit
the [10,3,5] infinite loop, or ask "wait, where's the system prompt?" — answer
both before they ask. The prompt-engineering point (plain string vs chat format)
is also a good aside: this model is a base model, may not even have a usable
chat template, so plain-string prompting is the standard approach here, matching
the TinyZero/Jiayi-Pan reproductions this demo is modeled on.
-->

---
title: What this actually cost
---

## The whole session, start to finish

<div class="grid grid-cols-4 gap-3 mt-4" style="text-align:center;">
<div class="panel-card">
<div class="metric" style="font-size:1.3em; color:var(--navy);">SFT</div>
<div style="font-size:0.85em; color:var(--muted); margin-top:0.3em;">17 min<br>laptop CPU</div>
</div>
<div class="panel-card">
<div class="metric" style="font-size:1.3em; color:var(--gold);">DPO</div>
<div style="font-size:0.85em; color:var(--muted); margin-top:0.3em;">29 min<br>Colab T4</div>
</div>
<div class="panel-card">
<div class="metric" style="font-size:1.3em; color:var(--good);">GRPO</div>
<div style="font-size:0.85em; color:var(--muted); margin-top:0.3em;">23.5 min<br>Colab T4</div>
</div>
<div class="panel-card">
<div class="metric" style="font-size:1.3em; color:var(--good);">Countdown</div>
<div style="font-size:0.85em; color:var(--muted); margin-top:0.3em;">47 min<br>Colab T4</div>
</div>
</div>

<div class="mt-8 panel-card good" style="text-align:center;">
<div class="grid grid-cols-2 gap-6">
<div><div class="metric good" style="font-size:1.8em;">~2 hours</div><div style="font-size:0.85em;color:var(--muted); margin-top:0.3em;">total training compute, all four runs combined</div></div>
<div><div class="metric good" style="font-size:1.8em;">$0</div><div style="font-size:0.85em;color:var(--muted); margin-top:0.3em;">own laptop + Colab's free tier, nothing paid</div></div>
</div>
</div>

<div class="mt-6" style="text-align:center; color:var(--muted); font-size:0.9em;">
<b>Why CPU for SFT but GPU for everything else:</b> SFT is one forward+backward
pass per example — small and fast enough for a laptop. DPO needs two forward
passes per step (policy + reference); GRPO needs eight generations per prompt,
every step. Same laptop, same budget — the <i>training signal's</i> shape is
what decided the hardware, not the task's difficulty.
</div>

---
title: Takeaways
---

## What to take home

<div class="mt-4 grid grid-cols-1 gap-3">

<div class="panel-card"><b>LoRA</b> is how cheaply you update weights — orthogonal to which training signal you use.</div>
<div class="panel-card"><b>SFT</b> teaches by imitation — only as good as the labels, and the failure mode is silent (it can memorize shape without understanding).</div>
<div class="panel-card"><b>DPO</b> teaches by preference between two options — excellent at stamping out specific, named failure modes.</div>
<div class="panel-card"><b>GRPO</b> teaches by trial and reward — but it can only amplify a skill the model already stumbles into sometimes. It cannot manufacture a skill from a near-zero base rate.</div>
<div class="panel-card good"><b>Verify before you trust a reward.</b> If the reward function has a blind spot, the model will find it — that's not misbehavior, that's the algorithm working correctly on the wrong target.</div>

</div>

---
layout: center
---

<div style="text-align:center;">
<h1 style="border:none;">Thank you</h1>
<div class="mt-4 stamp">Questions &middot; Repo linked in the meetup post</div>
</div>

<!--
Repo has everything: datasets, scripts, adapters, RESULTS.md and EXPERIMENTS.md
with the full negative-result writeups, and showcase_demo.py to try at home.
-->
