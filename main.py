"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

- Evals before features: log every run (request, plans, drafts, panel scores,
  revisions) as JSONL, keep a golden set of ~20 requests including adversarial
  ones like "a T-Rex battle with lots of fighting", and run the judge panel
  over it in CI so every prompt change shows its score diff before it ships.
- Operational hardening: per-call tracing with latency and cost, a degraded
  mode that drops to 3 drafts under rate limiting, and alerting on judge-score
  drift - usually the first sign that a provider-side model change altered
  behavior.
- Quality follow-ups: make revision surgical (regenerate only the weakest
  story beat instead of the whole story), and A/B a stronger judge model -
  the scoring noise I hit on request_fidelity is a judge-capability limit,
  not an architecture limit.
"""

import asyncio
import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL = "gpt-3.5-turbo"
# High bar on purpose: the polish pass always runs; this threshold only decides
# whether the story needs *further* repair rounds afterwards.
JUDGE_THRESHOLD = 4.2
MAX_REVISIONS = 2
CONCURRENCY = 30

# One candidate outline per angle (outline diversity before any prose is written).
PLAN_ANGLES = [
    "a gentle adventure with a safe return home",
    "a friendship story about helping someone",
    "a quiet, sensory story that gets sleepier as it goes",
    "a story with a small touch of soft magic",
    "a story about family warmth and feeling secure",
]

# 10 drafts spreads temperature 0.8-1.1 finely enough that the panel usually
# sees real spread; fewer drafts made the sweep too coarse in testing.
N_STORIES = 10

# Tailored generation strategy per request category.
CATEGORY_STYLES = {
    "adventure": "Pace it as a gentle quest: curiosity, one small challenge, and a safe return home.",
    "friendship": "Center feelings and kindness; let the characters talk warmly to each other.",
    "animal": "Give the animals cozy, familiar habits; keep nature sounds and textures vivid but soft.",
    "magic": "Keep the magic soft and wondrous, never powerful or dangerous.",
    "family": "Anchor the story in everyday rituals: shared meals, tucking in, a goodnight hug.",
    "calming": "Slow the rhythm as the story goes; end with the whole world settling down to sleep.",
}

# Perspective-diverse judge panel.
JUDGE_LENSES = [
    "Pay special attention to safety and age appropriateness.",
    "Pay special attention to bedtime tone and clarity.",
    "Pay special attention to story structure and creativity.",
]

SCORE_KEYS = [
    "age_appropriateness", "bedtime_tone", "safety",
    "story_structure", "creativity", "clarity", "request_fidelity",
]

sem = asyncio.Semaphore(CONCURRENCY)


async def call_model(system_prompt, user_prompt, json_mode=False, temperature=1.0):
    async with sem:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"} if json_mode else None,
        )
    return response.choices[0].message.content


async def call_json(system_prompt, user_prompt, required_keys, temperature=1.0):
    # json_object mode guarantees valid JSON syntax but not the schema; one
    # retry keeps a single flaky reply from killing a ~50-call run.
    for _ in range(2):
        data = json.loads(await call_model(
            system_prompt, user_prompt, json_mode=True, temperature=temperature
        ))
        if all(k in data for k in required_keys):
            return data
    raise RuntimeError(f"model kept omitting keys {required_keys}")


async def analyze_request(user_input):
    system = (
        "You analyze a bedtime story request from a parent or child. "
        "Extract the story requirements as a JSON object with keys: "
        "theme, protagonist, setting, tone, age_range, risk_notes, "
        "category (one of: adventure, friendship, animal, magic, family, calming). "
        "In risk_notes, list any requested elements that are scary, violent, or "
        "otherwise unsuitable for ages 5-10, or 'none'."
    )
    keys = ["theme", "protagonist", "setting", "tone", "age_range", "risk_notes", "category"]
    return await call_json(system, user_input, keys)


async def create_story_plan(request, analysis, angle, style):
    system = (
        "You are a children's story planner. Given the original request and the "
        "story requirements, produce a story outline as a JSON object with keys: "
        "opening, problem, journey, climax, ending, lesson. The outline must "
        "deliver exactly what the original request asks for - keep its specific "
        "subject, do not abstract it away, and keep it as the main plotline "
        "from opening to ending. Supporting characters may appear, but the "
        "requested characters stay the protagonists. If risk_notes lists anything scary, "
        "violent, or unsuitable, transform those elements into gentle, "
        "comforting equivalents while keeping the requested subject. "
        f"For this outline, lean toward {angle}. {style}"
    )
    user = json.dumps({"original_request": request, "requirements": analysis})
    keys = ["opening", "problem", "journey", "climax", "ending", "lesson"]
    return await call_json(system, user, keys)


async def select_best_plan(request, analysis, plans):
    # Comparative choice: LLM judges rank far more reliably than they score,
    # and pointwise 1-5 scores kept tying at the ceiling.
    system = (
        "You compare candidate bedtime story outlines for the same request. "
        "Pick the one that best delivers exactly what the original request asks "
        "for and is most promising as a calm, warm bedtime story for ages 5-10. "
        "Return a JSON object with keys: best_index (0-based integer), reason."
    )
    user = json.dumps({
        "original_request": request, "requirements": analysis, "outlines": plans,
    })
    choice = await call_json(system, user, ["best_index", "reason"], temperature=0.2)
    i = int(choice["best_index"])
    if not 0 <= i < len(plans):
        i = 0
    return i, choice["reason"]


async def generate_story(request, analysis, plan, temperature, style):
    system = (
        "You are a bedtime story writer for children ages 5-10. Write in English. "
        "The story must be warm, calm, and suitable for bedtime: no violence, no "
        "frightening content, 500-800 words, with a clear opening, middle, and "
        "ending. The story must deliver what the original request asks for. "
        "Use the outline only as a guide: write fresh prose and never copy "
        "outline sentences. Tell it in past tense with a little warm dialogue. "
        "Use simple words a five year old understands when read aloud. Keep "
        "the requested characters at the center of every scene. Never "
        "explain the moral or say what the story teaches - let it show through "
        f"the events. {style} "
        "Write flowing prose only: no title, no headings, no section labels."
    )
    user = json.dumps({
        "original_request": request, "requirements": analysis, "outline": plan,
    })
    return await call_model(system, user, temperature=temperature)


async def judge_story(request, story, lens, safety_note=None):
    # Anchored rubric + evidence-before-score: the checklist key comes first
    # so the judge enumerates what the request asked for before scoring,
    # and fidelity is judged only against the request's explicit content.
    system = (
        f"You evaluate children's bedtime stories for ages 5-10. {lens} "
        "First build a checklist: list each element the request explicitly "
        "asks for and mark it PRESENT or MISSING in the story. Judge only "
        "against what the request says - never invent expectations it does "
        "not mention. Then score each dimension 1-5. Anchors for "
        "request_fidelity: 5 = every requested element is present (character "
        "growth over the story, or a gentle reframe of unsafe elements, still "
        "counts as present); 3 = the requested subject is there but one "
        "requested detail is missing; 1 = the requested subject is dropped or "
        "replaced. Anchors for the other dimensions: 5 = nothing a parent "
        "would object to or a child would stumble on; 3 = one clear issue; "
        "1 = unsuitable for bedtime. Return a JSON object with keys: "
        "checklist (array of strings), age_appropriateness, bedtime_tone, "
        "safety, story_structure, creativity, clarity, request_fidelity, "
        "feedback (string). Do not rewrite the story."
    )
    payload = {"original_request": request, "story": story}
    if safety_note:
        payload["safety_note"] = safety_note
    user = json.dumps(payload)
    try:
        return await call_json(
            system, user, SCORE_KEYS + ["checklist", "feedback"], temperature=0.2
        )
    except RuntimeError:
        # a single broken judge must not sink the whole run
        return {**{k: 3 for k in SCORE_KEYS}, "feedback": "judge returned invalid scores"}


async def panel_judge(request, story, safety_transformed=False):
    safety_note = (
        "Unsafe elements in the request were intentionally transformed into "
        "gentle equivalents; judge request_fidelity against the transformed intent."
        if safety_transformed else None
    )
    judgments = await asyncio.gather(
        *[judge_story(request, story, lens, safety_note) for lens in JUDGE_LENSES]
    )
    scores = {
        k: round(sum(float(j[k]) for j in judgments) / len(judgments), 2)
        for k in SCORE_KEYS
    }
    # When the request itself was unsafe, deviating from it is deliberate:
    # report request_fidelity for transparency but keep it out of the
    # pass/fail math, or every unsafe request triggers unfixable repairs.
    gated = (
        {k: v for k, v in scores.items() if k != "request_fidelity"}
        if safety_transformed else scores
    )
    overall = round(sum(gated.values()) / len(gated), 2)
    weakest = [k for k in sorted(gated, key=gated.get) if gated[k] < 4.5][:2]
    return {
        "scores": scores,
        "overall_score": overall,
        "weakest_dimensions": weakest or ["overall warmth and flow"],
        "feedback": [j["feedback"] for j in judgments],
        # A single collapsed dimension (e.g. the story ignored the request)
        # must fail the story even when the overall average looks fine.
        "revision_needed": overall < JUDGE_THRESHOLD or min(gated.values()) < 3.0,
    }


async def revise_story(request, story, judgment):
    system = (
        "You revise a bedtime story using a judge panel's feedback. Fix the "
        "weak areas, especially: "
        + ", ".join(judgment["weakest_dimensions"]) + ". "
        "The revised story must deliver what the original request asks for. "
        "Do not add scary or violent content. Write in English, 500-800 words, "
        "flowing prose only with no title or headings. Return the complete "
        "story from beginning to end, never just the changed part."
    )
    user = json.dumps({
        "original_request": request,
        "story": story,
        "panel_feedback": judgment["feedback"],
    })
    return await call_model(system, user)


async def revise_with_user_feedback(story, feedback):
    system = (
        "You revise a bedtime story according to the user's requested changes. "
        "Apply the changes faithfully while keeping the story warm, calm, and "
        "suitable for bedtime for ages 5-10: no violence, no frightening "
        "content, simple language, 500-800 words, flowing prose only with no "
        "title or headings. Return the complete story from beginning to end "
        "with the changes applied, never just the changed part."
    )
    user = json.dumps({"story": story, "requested_changes": feedback})
    return await call_model(system, user)


async def draft_and_judge(request, analysis, plan, temperature, style, safety_transformed):
    story = await generate_story(request, analysis, plan, temperature, style)
    return story, await panel_judge(request, story, safety_transformed)


def print_story(story):
    # Bedtime pacing in an interactive terminal: read sentence by sentence,
    # slowing down toward the end. Piped output stays plain for logs/tests.
    if not sys.stdout.isatty():
        print(story)
        return
    sentences = re.split(r"(?<=[.!?]) +", story)
    for i, s in enumerate(sentences):
        print(s)
        time.sleep(0.25 + 0.5 * i / max(len(sentences) - 1, 1))


async def pipeline(user_input):
    start = time.perf_counter()

    analysis = await analyze_request(user_input)
    style = CATEGORY_STYLES.get(str(analysis["category"]).lower(), "")
    safety_transformed = str(analysis["risk_notes"]).strip().lower() != "none"
    print("\n=== ANALYSIS ===\n")
    print(json.dumps(analysis, indent=2))
    if style:
        print(f"\nTailored strategy [{analysis['category']}]: {style}")
    if safety_transformed:
        print("\nSafety: risky elements will be gently transformed; "
              "request_fidelity is reported but not gated for this run.")

    print(f"\nPlanning: {len(PLAN_ANGLES)} candidate outlines in parallel...")
    plans = await asyncio.gather(
        *[create_story_plan(user_input, analysis, a, style) for a in PLAN_ANGLES]
    )
    best_plan_i, reason = await select_best_plan(user_input, analysis, plans)
    print(f"  selected outline {best_plan_i} ({PLAN_ANGLES[best_plan_i]})")
    print(f"  reason: {reason}")
    plan = plans[best_plan_i]
    print("\n=== SELECTED PLAN ===\n")
    print(json.dumps(plan, indent=2))

    n_judges = N_STORIES * len(JUDGE_LENSES)
    print(f"\nWriting: {N_STORIES} parallel drafts, each scored by "
          f"{len(JUDGE_LENSES)} judges ({n_judges} judge calls)...")
    temperatures = [round(0.8 + 0.3 * i / (N_STORIES - 1), 2) for i in range(N_STORIES)]
    results = await asyncio.gather(
        *[draft_and_judge(user_input, analysis, plan, t, style, safety_transformed)
          for t in temperatures]
    )
    best_i = max(
        range(len(results)),
        key=lambda i: (not results[i][1]["revision_needed"], results[i][1]["overall_score"]),
    )
    for i, (_, j) in enumerate(results):
        marker = "  <-- selected" if i == best_i else ""
        print(f"  draft {i} (temp {temperatures[i]}): {j['overall_score']}{marker}")
    story, judgment = results[best_i]

    # Polish pass: the judge panel's feedback always gets one chance to improve
    # the winning draft; keep whichever version the panel scores higher.
    print(f"\nPolish pass targeting: {judgment['weakest_dimensions']}...")
    revised = await revise_story(user_input, story, judgment)
    revised_judgment = await panel_judge(user_input, revised, safety_transformed)
    print(f"  panel scores: original {judgment['overall_score']}, "
          f"revised {revised_judgment['overall_score']}")
    if revised_judgment["overall_score"] >= judgment["overall_score"]:
        story, judgment = revised, revised_judgment
        print("  kept the revision")
    else:
        print("  kept the original")

    revisions = 1
    while judgment["revision_needed"] and revisions < MAX_REVISIONS:
        print(f"\nRepair round {revisions}, "
              f"weakest: {judgment['weakest_dimensions']}...")
        story = await revise_story(user_input, story, judgment)
        judgment = await panel_judge(user_input, story, safety_transformed)
        revisions += 1
        print(f"  re-judged: {judgment['overall_score']}")

    elapsed = time.perf_counter() - start
    print("\n=== FINAL STORY ===\n")
    print_story(story)
    print("\n=== FINAL JUDGE SUMMARY ===\n")
    print(json.dumps(judgment, indent=2))
    print(f"\nPipeline done in {elapsed:.1f}s")

    # User feedback loop: every user-requested change is re-checked by the
    # judge panel, and auto-repaired once if it falls below the threshold.
    request_ctx = user_input
    while True:
        try:
            feedback = input("\nAny changes you'd like? (press Enter to keep the story) ").strip()
        except EOFError:
            break
        if not feedback:
            break
        request_ctx += f"; the user then asked: {feedback}"
        story = await revise_with_user_feedback(story, feedback)
        judgment = await panel_judge(request_ctx, story, safety_transformed)
        if judgment["revision_needed"]:
            print(f"  judge panel flagged the updated story "
                  f"({judgment['overall_score']}), auto-repairing "
                  f"{judgment['weakest_dimensions']}...")
            story = await revise_story(request_ctx, story, judgment)
            judgment = await panel_judge(request_ctx, story, safety_transformed)
        print("\n=== UPDATED STORY ===\n")
        print_story(story)
        print("\n=== JUDGE SUMMARY ===\n")
        print(json.dumps(judgment, indent=2))


def main():
    user_input = input("What kind of bedtime story do you want? ")
    asyncio.run(pipeline(user_input))


if __name__ == "__main__":
    main()
