# Bedtime Story Agent

Generates bedtime stories for ages 5-10 from a free-form request, using a
planner / storyteller / judge / reviser pipeline on top of `gpt-3.5-turbo`.
Everything lives in `main.py` with plain `asyncio` for parallelism; no agent
framework. Requires Python 3.10+.

## Design

The request goes through six stages rather than a single completion:

1. Request analysis. The request is parsed into structured requirements
   (theme, protagonist, setting, tone, age range, risk notes) plus a
   category (adventure, friendship, animal, magic, family, calming).
   The category selects a writing strategy: a "calming" request slows the
   rhythm toward the end, a "magic" request keeps magic soft, and so on.
   If the request contains scary or violent elements, they are listed in
   `risk_notes` and the planner is instructed to transform them into
   gentle equivalents while keeping the requested subject.

2. Outline candidates. Five outlines are drafted in parallel, each
   leaning toward a different narrative angle. A single comparative call
   picks the best one and prints its reason. Comparative selection is
   used here instead of per-outline 1-5 scoring because scores from the
   judge cluster near the ceiling and tie too often to be useful.

3. Drafting. Ten drafts are written in parallel from the winning
   outline, sweeping temperature from 0.8 to 1.1 for diversity.

4. Judging. Each draft is scored by a three-judge panel (30 calls in
   parallel). Each judge reads with a different emphasis: safety and age
   appropriateness, bedtime tone and clarity, structure and creativity.
   Judges receive the original request together with the story, and one
   of the seven scored dimensions is `request_fidelity`, so a
   well-written story that drops the requested subject still loses.
   Dimension scores are averaged across the panel; the overall score is
   the mean of the seven dimension averages.

5. Polish. The winning draft always gets one revision pass driven by the
   panel's feedback, targeting its weakest dimensions. The panel
   re-scores the revision and the higher-scoring version is kept, so the
   revision step cannot make the story worse. If the result is still
   below 4.2 overall, or below 3 on any single dimension, one extra
   repair round runs.

6. User feedback. After the story is printed, the user can request
   changes. Each change is applied by a reviser and re-checked by the
   same judge panel; if the updated story falls below the threshold it
   is repaired once before being shown. User requests do not bypass the
   safety checks.

Model calls run up to 30 concurrent under an `asyncio.Semaphore`, so the
~50 calls of a full run typically finish in 15-25 seconds.

When stdout is a terminal, the final story prints sentence by sentence
with gradually increasing delays (read-aloud pacing). Piped output stays
plain so transcripts and tests are unaffected.

## Block diagram

```text
User ──────────── story request
    |
    v
[1] Request Analyzer ── requirements + category ─── 1 call
    |     (category selects writing strategy;
    |      risky elements flagged for gentle transformation)
    v
[2] Story Planner ───── 5 candidate outlines ────── 5 parallel calls
    |
    v
[3] Plan Selector ───── comparative pick + reason ── 1 call
    |  best outline
    v
[4] Storyteller ─────── 10 drafts, temp 0.8-1.1 ─── 10 parallel calls
    |
    v
[5] Judge Panel ─────── 3 lenses x 10 drafts ────── 30 parallel calls
    |  7 dimensions averaged, best draft selected
    v
[6] Polish Pass ─────── revise weakest dimensions ── 1 + 3 calls
    |  keep whichever version the panel scores higher
    v
[7] still below bar? ── yes ──> repair round ──> back to [5]
    |  no                       (max 1 more)
    v
Final Story + Judge Summary ──────> User
    ^                                |  change request / Enter to accept
    |                                v
    |                       [8] Feedback Reviser ── apply user changes
    |                                |
    +──── [5] Judge Panel <──────────+
          (re-check; auto-repair once if below threshold)
```

## Judge output

Each judge returns 1-5 scores on seven dimensions plus written feedback.
The panel aggregate looks like:

```json
{
  "scores": {
    "age_appropriateness": 5.0,
    "bedtime_tone": 5.0,
    "safety": 5.0,
    "story_structure": 4.67,
    "creativity": 4.33,
    "clarity": 5.0,
    "request_fidelity": 5.0
  },
  "overall_score": 4.86,
  "weakest_dimensions": ["creativity", "story_structure"],
  "feedback": ["...", "...", "..."],
  "revision_needed": false
}
```

Revision triggers when the overall score is below 4.2 or any single
dimension is below 3, so a story that collapses on one dimension (ignored
request, unsafe content) fails even if its average looks fine. One
exception: when the request itself contained unsafe elements, deviating
from it is deliberate, so `request_fidelity` is still reported for
transparency but excluded from the pass/fail decision — otherwise every
unsafe request would trigger repair rounds that cannot succeed. Judges
run at temperature 0.2 for score stability; writers run at 0.8-1.1.
Judges are instructed to score and critique only, never rewrite.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env   # put your OPENAI_API_KEY inside
python main.py
```

Example session:

```text
What kind of bedtime story do you want? A story about a girl named Alice and her best friend Bob, who happens to be a cat.
...
Any changes you'd like? (press Enter to keep the story) Could Bob find a lost kitten?
```

## Examples

Finished stories from four real runs are collected in
[`examples/stories.pdf`](examples/stories.pdf) — request in, story out,
with panel scores. When you want to see how a story was produced, the
complete unedited transcript of each run (request analysis, outline
selection with the judge's reason, the 10-draft leaderboard, the polish
pass, and the final judge summary) is next to it:

| Transcript | Request | What the trace demonstrates |
|---|---|---|
| [`shy_star.txt`](examples/shy_star.txt) | A shy little star afraid to shine | Standard flow: clean 5.0 run in ~21s |
| [`thunderstorm.txt`](examples/thunderstorm.txt) | Calming story for a child scared of thunderstorms | Fear-of-X request handled with a reassuring, de-escalating story |
| [`trex_safety.txt`](examples/trex_safety.txt) | A T-Rex battle with lots of fighting | Safety transformation: violent request reframed, T-Rex kept |
| [`alice_and_bob.txt`](examples/alice_and_bob.txt) | Alice and Bob the cat | User feedback loop: "Can the ending be a bit sleepier?" applied and re-judged |
