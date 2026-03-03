"""
AI Chef Agent endpoint.

POST /chef/parse
    Body (JSON):
        - recipe_name:   str           (dish name)
        - instructions:  str           (raw cooking instructions from DB or user)
        - ingredients:   str | None    (raw ingredients list, optional)

Returns a structured cook-session object:
    - mise_en_place: prep tasks to complete before cooking
    - steps: sequential cooking steps, each with optional timer_seconds + tool
    - tools_required: list of kitchen tools
    - estimated_total_minutes: approximate total cook time

POST /chef/intent
    Body (JSON):
        - raw_text:        str  (voice transcript from phone)
        - recipe_name:     str
        - current_step:    int
        - total_steps:     int
        - current_action:  str
        - timer_running:   bool
        - timer_seconds_left: int | None

Returns a structured voice intent:
    - action:     NEXT | PREV | DONE | STRIKE | TIMER_START | TIMER_PAUSE | ...
    - step:       int | None
    - question:   str | None
    - confidence: float
    - filtered:   bool
"""

import json
import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from Backend.dependencies.router import get_router
from Backend.schemas.chef import (
    ChefIntentRequest,
    ChefIntentResponse,
    ChefParseRequest,
    ChefParseResponse,
    CookStep,
    MiseEnPlaceItem,
    VoiceAction,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chef"])

# ── Helpers ────────────────────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json_brace_match(text: str) -> str | None:
    """Extract JSON by matching braces properly (handles nested objects)."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json(text: str) -> str:
    """Strip markdown code fences if present, return raw JSON string."""
    # Strategy 1: code-fenced JSON
    match = _JSON_BLOCK_RE.search(text)
    if match:
        inner = match.group(1).strip()
        # Try brace matching inside the code fence
        brace_matched = _extract_json_brace_match(inner)
        if brace_matched:
            return brace_matched
        return inner

    # Strategy 2: brace-matched extraction
    brace_matched = _extract_json_brace_match(text)
    if brace_matched:
        return brace_matched

    # Strategy 3: simple first/last brace
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text.strip()


def _repair_json(text: str) -> str:
    """Attempt basic JSON repairs for common LLM issues."""
    # Remove control characters (keep newlines for now)
    repaired = re.sub(r"[\x00-\x09\x0b\x0c\x0e-\x1f]", " ", text)
    # Remove trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    # Replace single quotes with double quotes if no double quotes present
    if '"' not in repaired and "'" in repaired:
        repaired = repaired.replace("'", '"')
    # Fix null variants the LLM sometimes uses
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    return repaired


def _fix_unescaped_newlines_in_strings(text: str) -> str:
    """Replace literal newlines inside JSON string values with spaces."""
    result = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ch in ("\n", "\r"):
            result.append(" ")
            continue
        result.append(ch)
    return "".join(result)


def _aggressive_repair_json(text: str) -> str:
    """More aggressive repair: strip all non-JSON wrapper text."""
    repaired = _repair_json(text)
    # Remove any text before the first { and after the last }
    brace_matched = _extract_json_brace_match(repaired)
    if brace_matched:
        repaired = brace_matched
    # Fix unescaped newlines inside JSON string values
    repaired = _fix_unescaped_newlines_in_strings(repaired)
    return repaired


def _looks_like_json(text: str) -> bool:
    """Check if text appears to be JSON (has json-like structure)."""
    stripped = text.strip()
    return (
        stripped.startswith("{") or
        '"mise_en_place"' in stripped or
        '"steps"' in stripped or
        '"action"' in stripped
    )


def _regex_extract_steps(raw: str, recipe_name: str) -> ChefParseResponse | None:
    """Last-resort extraction: pull individual step actions from raw text via regex."""
    # Try to find all action strings
    action_pattern = re.compile(r'"action"\s*:\s*"((?:[^"\\]|\\.)*?)"', re.DOTALL)
    action_matches = action_pattern.findall(raw)
    if not action_matches or len(action_matches) < 2:
        return None

    # Try to find mise_en_place text entries
    mise_pattern = re.compile(
        r'"text"\s*:\s*"((?:[^"\\]|\\.)*?)"\s*,\s*"duration_minutes"\s*:\s*(\d+|null)',
        re.DOTALL,
    )
    mise_matches = mise_pattern.findall(raw)

    # Try to find timer_seconds for each step
    step_block_pattern = re.compile(
        r'"action"\s*:\s*"((?:[^"\\]|\\.)*?)"\s*,\s*"timer_seconds"\s*:\s*(\d+|null)',
        re.DOTALL,
    )
    step_blocks = step_block_pattern.findall(raw)

    # Try to find tool for each step
    full_step_pattern = re.compile(
        r'"action"\s*:\s*"((?:[^"\\]|\\.)*?)"'
        r'\s*,\s*"timer_seconds"\s*:\s*(\d+|null)'
        r'\s*,\s*"tool"\s*:\s*(?:"((?:[^"\\]|\\.)*?)"|null)'
        r'(?:\s*,\s*"tip"\s*:\s*(?:"((?:[^"\\]|\\.)*?)"|null))?',
        re.DOTALL,
    )
    full_matches = full_step_pattern.findall(raw)

    steps: list[CookStep] = []
    if full_matches:
        for i, m in enumerate(full_matches, 1):
            action_text, timer_str, tool_str, tip_str = m
            steps.append(CookStep(
                id=i,
                action=action_text.replace("\\n", " ").strip(),
                timer_seconds=int(timer_str) if timer_str != "null" else None,
                tool=tool_str if tool_str else None,
                tip=tip_str if tip_str else None,
            ))
    elif step_blocks:
        for i, (act, timer_str) in enumerate(step_blocks, 1):
            steps.append(CookStep(
                id=i,
                action=act.replace("\\n", " ").strip(),
                timer_seconds=int(timer_str) if timer_str != "null" else None,
            ))
    else:
        for i, act in enumerate(action_matches, 1):
            steps.append(CookStep(
                id=i,
                action=act.replace("\\n", " ").strip(),
            ))

    mise_items = []
    for i, (text_val, dur_str) in enumerate(mise_matches, 1):
        mise_items.append(MiseEnPlaceItem(
            id=i,
            text=text_val.replace("\\n", " ").strip(),
            duration_minutes=int(dur_str) if dur_str != "null" else None,
        ))

    # Extract tools_required
    tools_match = re.search(r'"tools_required"\s*:\s*\[([^\]]*)\]', raw)
    tools = []
    if tools_match:
        tools = [t.strip().strip('"') for t in tools_match.group(1).split(",") if t.strip().strip('"')]

    # Extract estimated_total_minutes
    minutes_match = re.search(r'"estimated_total_minutes"\s*:\s*(\d+)', raw)
    total_min = int(minutes_match.group(1)) if minutes_match else None

    logger.info("Regex extraction recovered %d steps, %d mise_en_place items", len(steps), len(mise_items))
    return ChefParseResponse(
        recipe_name=recipe_name,
        mise_en_place=mise_items,
        steps=steps,
        tools_required=tools,
        estimated_total_minutes=total_min,
    )


def _fallback_from_raw_text(raw: str, recipe_name: str) -> ChefParseResponse:
    """Create a structured response by splitting raw text into numbered steps."""
    # If the text looks like JSON, try regex extraction first
    if _looks_like_json(raw):
        regex_result = _regex_extract_steps(raw, recipe_name)
        if regex_result and len(regex_result.steps) >= 2:
            return regex_result
        # Regex extraction failed — return an error, NEVER dump raw JSON as step text
        return ChefParseResponse(
            recipe_name=recipe_name,
            steps=[],
            parse_error="Could not parse the AI-generated recipe. Please try again.",
        )

    # For genuine plain-text instructions, split on numbered patterns or sentences
    lines = re.split(r"(?:\d+[.)\-]\s*|\n{2,})", raw.strip())
    steps = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if len(line) > 10:  # skip very short fragments
            steps.append(CookStep(id=i, action=line[:500]))

    if not steps:
        # Last resort: split on single newlines
        for i, line in enumerate(raw.strip().split("\n"), 1):
            line = line.strip()
            if len(line) > 10:
                steps.append(CookStep(id=i, action=line[:500]))

    if not steps:
        steps = [CookStep(id=1, action=raw.strip()[:500])]

    return ChefParseResponse(
        recipe_name=recipe_name,
        steps=steps,
        parse_error="The AI could not structure this recipe perfectly. Steps were extracted from the raw instructions.",
    )


def _build_prompt(recipe_name: str, instructions: str, ingredients: Optional[str]) -> str:
    ingr_block = ""
    if ingredients and ingredients.strip():
        ingr_block = f"INGREDIENTS:\n{ingredients.strip()}\n\n"

    has_stored_instructions = bool(
        instructions and instructions.strip() and len(instructions.strip()) > 30
    )

    if has_stored_instructions:
        source_block = f"RAW INSTRUCTIONS TO PARSE:\n{instructions.strip()}"
        task_line = (
            f'Parse and structure the raw instructions below for the dish "{recipe_name}" '
            f"into the JSON format defined in OUTPUT_FORMAT."
        )
    else:
        source_block = ""
        task_line = (
            f'No stored instructions are available for "{recipe_name}". '
            f"Use your culinary expertise to generate accurate, authentic, and complete "
            f"cooking steps for this dish based on its name and any available ingredient information."
        )

    return f"""<SYSTEM>
You are an expert culinary AI that converts recipe information into a precise, structured JSON cooking session. You specialise in Indian cuisine and home-cook-friendly instructions. You output ONLY valid JSON — no explanations, no markdown fences, no text before or after the JSON object.
</SYSTEM>

<TASK>
{task_line}

Dish: {recipe_name}

{ingr_block}{source_block}
</TASK>

<OUTPUT_FORMAT>
Respond with exactly this JSON structure (fill with real data, not the examples):
{{
  "mise_en_place": [
    {{"id": 1, "text": "Soak 1 cup chana dal in cold water for 30 minutes", "duration_minutes": 30}},
    {{"id": 2, "text": "Finely dice 2 medium onions", "duration_minutes": null}}
  ],
  "steps": [
    {{"id": 1, "action": "Heat 2 tablespoons oil in a heavy-bottomed pan over medium-high heat.", "timer_seconds": null, "tool": "pan", "tip": null}},
    {{"id": 2, "action": "Add cumin seeds and stir for 30 seconds until they splutter.", "timer_seconds": 30, "tool": "pan", "tip": "Oil is ready when a single cumin seed sizzles immediately on contact."}},
    {{"id": 3, "action": "Add diced onions and fry, stirring occasionally, until deep golden brown.", "timer_seconds": 480, "tool": "pan", "tip": null}},
    {{"id": 4, "action": "Add ginger-garlic paste and cook for 2 minutes until the raw smell disappears.", "timer_seconds": 120, "tool": "pan", "tip": null}},
    {{"id": 5, "action": "Pressure cook on high for 10 minutes, then allow natural pressure release for 5 minutes.", "timer_seconds": 600, "tool": "pressure cooker", "tip": "Never open the lid while the pressure pin is still raised."}}
  ],
  "tools_required": ["pan", "pressure cooker", "knife", "cutting board"],
  "estimated_total_minutes": 45
}}
</OUTPUT_FORMAT>

<RULES>

PREPARATION rules:
- Include ONLY passive preparation tasks done BEFORE cooking: chopping, soaking, measuring, marinating, grinding, peeling, draining, washing.
- NEVER include active cooking actions (frying, sauteing, boiling, pressure cooking) in mise_en_place.
- Set duration_minutes to an integer only when the task requires waiting time longer than 5 minutes (e.g., soaking = 30, marinating = 60). Use null for quick tasks under 5 minutes.

STEPS rules:
- Each step must be a single, atomic cooking action beginning with an imperative verb (Heat, Add, Stir, Simmer, Cover, Drain, Garnish, etc.).
- Maximum 2 sentences per action. Write at the level of a confident home cook — precise but not verbose.
- Steps must be in correct chronological cooking order with no gaps in the sequence.
- Number all IDs starting from 1 and incrementing by 1.

TIMER RULES — read carefully:
SET timer_seconds when the step specifies a concrete, measurable waiting duration:
  - "fry for 2 minutes"        → timer_seconds: 120
  - "simmer 20 minutes"        → timer_seconds: 1200
  - "boil for 15 minutes"      → timer_seconds: 900
  - "roast for 45 seconds"     → timer_seconds: 45
  - "bake at 180C for 30 min"  → timer_seconds: 1800
  - "cook on low for 8 min"    → timer_seconds: 480
  - "rest for 5 minutes"       → timer_seconds: 300
  - "cook for about 10 minutes" → timer_seconds: 600
  - "5 to 7 minutes"           → timer_seconds: 360  (use midpoint)

SET timer_seconds to null for subjective or instant actions:
  - "fry until golden brown"   → null  (visual cue, not a fixed time)
  - "cook until soft"          → null  (subjective doneness)
  - "add the spices"           → null  (instant action)
  - "stir to combine"          → null  (instant action)
  - "bring to a boil"          → null  (variable, no fixed duration)
  - "season to taste"          → null  (instant, no waiting)
  - "garnish and serve"        → null  (instant action)

Minimum meaningful timer: 20 seconds. Steps shorter than 20 seconds must use null.

TOOL rules:
- Set tool to the primary kitchen utensil for that step: "pan", "wok", "kadai", "tawa", "pressure cooker", "oven", "blender", "grinder", "steamer", "griddle", "mixing bowl", "mortar and pestle", "knife".
- Use null if no specific tool is required.

TIP rules:
- Include a tip ONLY when there is a genuinely useful, non-obvious technique hint: visual doneness cues, common mistake warnings, temperature guidance, substitution notes.
- Use null for straightforward steps that need no extra guidance.

TOOLS REQUIRED:
- Provide a deduplicated list of every unique tool mentioned across all steps (lowercase).

ESTIMATED TOTAL MINUTES:
- Integer. Active cooking time only — do not include mise en place prep time.

LANGUAGE AND FORMAT:
- All text must be in English only. Translate any non-English source text.
- No emojis anywhere.
- Output the JSON object and absolutely nothing else. No preamble, no explanation, no trailing text.
- The JSON must be valid: properly quoted string keys, no trailing commas, no JavaScript comments.

</RULES>"""


def _parse_response(raw: str, recipe_name: str) -> ChefParseResponse:
    """Parse LLM output into ChefParseResponse, with graceful fallback."""
    json_str = _extract_json(raw)

    # Try multiple JSON repair strategies (increasingly aggressive)
    data = None
    candidates = [
        json_str,
        _repair_json(json_str),
        _aggressive_repair_json(json_str),
        _fix_unescaped_newlines_in_strings(_repair_json(json_str)),
        _repair_json(_extract_json(raw)),  # retry extraction after repair
        _aggressive_repair_json(raw),       # try repair on raw text directly
        _fix_unescaped_newlines_in_strings(_aggressive_repair_json(raw)),
    ]
    for attempt_str in candidates:
        try:
            data = json.loads(attempt_str)
            if isinstance(data, dict) and ("steps" in data or "mise_en_place" in data):
                break  # found valid data
            data = None  # parsed but not the right structure
        except (json.JSONDecodeError, TypeError):
            continue

    if data is None:
        logger.warning("Chef parse JSON decode failed, trying regex extraction | raw=%r", raw[:500])
        # Try regex-based extraction before falling back to raw text splitting
        regex_result = _regex_extract_steps(raw, recipe_name)
        if regex_result and len(regex_result.steps) >= 2:
            return regex_result
        return _fallback_from_raw_text(raw, recipe_name)

    mise_items = []
    for item in data.get("mise_en_place", []):
        if isinstance(item, dict) and item.get("text"):
            mise_items.append(
                MiseEnPlaceItem(
                    id=item.get("id", len(mise_items) + 1),
                    text=item["text"],
                    duration_minutes=item.get("duration_minutes"),
                )
            )

    steps = []
    for step in data.get("steps", []):
        if isinstance(step, dict) and step.get("action"):
            steps.append(
                CookStep(
                    id=step.get("id", len(steps) + 1),
                    action=step["action"],
                    timer_seconds=step.get("timer_seconds"),
                    tool=step.get("tool"),
                    tip=step.get("tip"),
                )
            )

    return ChefParseResponse(
        recipe_name=recipe_name,
        mise_en_place=mise_items,
        steps=steps,
        tools_required=[str(t) for t in data.get("tools_required", []) if t],
        estimated_total_minutes=data.get("estimated_total_minutes"),
    )


# ── Endpoint ────────────────────────────────────────────────────────

_MAX_RETRIES = 2


@router.post("/chef/parse", response_model=ChefParseResponse)
async def chef_parse(body: ChefParseRequest, nutri_router=Depends(get_router)):
    """
    Parse raw recipe instructions into a structured cook-session.

    If `instructions` is not provided or is empty, the LLM will generate
    cooking steps from scratch using the dish name and any available ingredients.

    The LLM extracts (or generates):
    - Mise en place (prep tasks)
    - Cooking steps with optional timers and tool annotations
    - Tools required
    - Estimated total cooking time

    Retries up to _MAX_RETRIES times if parsing fails completely.
    """
    prompt = _build_prompt(body.recipe_name, body.instructions or "", body.ingredients)
    llm_engine = nutri_router.engine

    last_result: ChefParseResponse | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            raw = await llm_engine.llm.generate_async(prompt)
        except Exception as exc:
            logger.exception("Chef LLM generation failed (attempt %d): %s", attempt + 1, exc)
            if attempt == _MAX_RETRIES - 1:
                raise HTTPException(status_code=500, detail="Chef service unavailable. Please try again.")
            continue

        result = _parse_response(raw, body.recipe_name)

        # If we got actual steps, return immediately
        if result.steps and len(result.steps) >= 1 and not result.parse_error:
            return result

        # If we got steps from regex fallback (no parse_error), accept it
        if result.steps and len(result.steps) >= 2:
            return result

        last_result = result
        logger.warning(
            "Chef parse attempt %d/%d failed for '%s' — retrying",
            attempt + 1, _MAX_RETRIES, body.recipe_name,
        )

    # All retries exhausted — return best result or error
    if last_result and last_result.steps:
        return last_result

    return ChefParseResponse(
        recipe_name=body.recipe_name,
        steps=[],
        parse_error="Could not parse the recipe after multiple attempts. Please try again.",
    )


# ══════════════════════════════════════════════════════════════════════
#  Voice Intent Parsing  (P2P Kitchen Remote)
# ══════════════════════════════════════════════════════════════════════

# ── Keyword gate: only cooking-relevant speech passes to the LLM ──

_COOKING_KEYWORDS = re.compile(
    r"\b("
    r"next|previous|prev|back|go back|skip|done|finish|finished|complete|completed"
    r"|start|stop|pause|resume|timer|reset|restart"
    r"|repeat|again|say that|what|how|why|help|tip|ingredient"
    r"|ready|move on|mark|strike|check|uncheck"
    r"|first|last|step|number|one|two|three|four|five|six|seven|eight|nine|ten"
    r")\b",
    re.IGNORECASE,
)


def _is_cooking_relevant(text: str) -> bool:
    """Return True only if the text contains cooking-intent keywords.

    This is the pre-filter that prevents random kitchen chatter
    ('pass me the salt', 'the dog is barking') from ever reaching the LLM.
    """
    return bool(_COOKING_KEYWORDS.search(text))


# ── Fast heuristic shortcuts (no LLM needed) ──

_HEURISTIC_PATTERNS: list[tuple[re.Pattern, VoiceAction]] = [
    (re.compile(r"\b(next\s*(step)?|move\s*on|skip|go\s*ahead)\b", re.I), VoiceAction.NEXT),
    (re.compile(r"\b(prev(ious)?\s*(step)?|go\s*back|back)\b", re.I), VoiceAction.PREV),
    (re.compile(r"\b(done|finish(ed)?|complete(d)?|all\s*done)\b", re.I), VoiceAction.DONE),
    (re.compile(r"\b(start\s*timer|begin\s*timer|timer\s*start|start\s*the\s*timer)\b", re.I), VoiceAction.TIMER_START),
    (re.compile(r"\b(stop\s*timer|pause\s*timer|timer\s*(stop|pause)|pause\s*the\s*timer)\b", re.I), VoiceAction.TIMER_PAUSE),
    (re.compile(r"\b(reset\s*timer|timer\s*reset|restart\s*timer)\b", re.I), VoiceAction.TIMER_RESET),
    (re.compile(r"\b(repeat|say\s*(that|it)\s*again|read\s*(it|step)\s*(again)?)\b", re.I), VoiceAction.REPEAT),
]


def _try_heuristic(text: str) -> VoiceAction | None:
    """Attempt to resolve intent via fast regex without calling the LLM."""
    for pattern, action in _HEURISTIC_PATTERNS:
        if pattern.search(text):
            return action
    return None


# ── LLM intent prompt ──

def _build_intent_prompt(body: ChefIntentRequest) -> str:
    timer_ctx = ""
    if body.timer_running and body.timer_seconds_left is not None:
        timer_ctx = f"Timer is RUNNING with {body.timer_seconds_left}s left."
    elif body.timer_seconds_left is not None:
        timer_ctx = f"Timer is PAUSED at {body.timer_seconds_left}s."
    else:
        timer_ctx = "No timer for this step."

    return f"""<SYSTEM>
You are a kitchen voice-command interpreter. Parse the user's spoken command into exactly one JSON action.
Respond with ONLY a JSON object — no explanations.
</SYSTEM>

<CONTEXT>
Dish: {body.recipe_name}
Step {body.current_step} of {body.total_steps}: "{body.current_action}"
{timer_ctx}
</CONTEXT>

<USER_SPEECH>
"{body.raw_text}"
</USER_SPEECH>

<ACTIONS>
- NEXT: go to the next cooking step
- PREV: go to the previous step
- DONE: mark current step as finished and advance (same as NEXT but marks completion)
- STRIKE: mark a specific step as done by number  (requires "step" field, 1-based)
- TIMER_START: start/resume the timer for the current step
- TIMER_PAUSE: pause the timer
- TIMER_RESET: reset the timer to its original duration
- REPEAT: read the current step instructions again
- ASK: the user is asking a cooking question (set "question" field with the question text)
- NOOP: the speech is unrelated to cooking navigation
</ACTIONS>

<OUTPUT_FORMAT>
{{"action": "NEXT", "step": null, "question": null}}
</OUTPUT_FORMAT>

<RULES>
- Pick the single most likely action.
- For ASK, extract the core question into the "question" field.
- For STRIKE, set "step" to the step number mentioned (1-based integer).
- If the speech is clearly not a cooking command, return NOOP.
- Output ONLY the JSON object. No markdown, no text.
</RULES>"""


# ── Endpoint ──

@router.post("/chef/intent", response_model=ChefIntentResponse)
async def chef_intent(body: ChefIntentRequest, nutri_router=Depends(get_router)):
    """Parse raw voice text into a structured cooking intent.

    Two-stage filter:
      1. Keyword gate — rejects non-cooking speech without touching the LLM.
      2. Heuristic patterns — resolves simple commands (next, done, timer) instantly.
      3. LLM fallback — for ambiguous or complex commands only.
    """
    text = body.raw_text.strip()

    # Stage 1: keyword gate — reject noise
    if not _is_cooking_relevant(text):
        logger.debug("Voice filtered (no cooking keywords): %r", text)
        return ChefIntentResponse(
            action=VoiceAction.NOOP,
            confidence=1.0,
            filtered=True,
        )

    # Stage 2: fast heuristic
    heuristic = _try_heuristic(text)
    if heuristic is not None:
        logger.info("Voice heuristic match: %r → %s", text, heuristic.value)
        return ChefIntentResponse(
            action=heuristic,
            confidence=0.95,
            filtered=False,
        )

    # Stage 3: LLM intent parsing
    prompt = _build_intent_prompt(body)
    llm_engine = nutri_router.engine

    try:
        raw = await llm_engine.llm.generate_async(prompt)
    except Exception as exc:
        logger.exception("Chef intent LLM failed: %s", exc)
        raise HTTPException(status_code=500, detail="Voice processing unavailable.")

    # Parse LLM JSON response
    json_str = _extract_json(raw)
    try:
        data = json.loads(_repair_json(json_str))
    except (json.JSONDecodeError, TypeError):
        logger.warning("Chef intent JSON parse failed: %r", raw[:300])
        return ChefIntentResponse(
            action=VoiceAction.NOOP,
            confidence=0.3,
            filtered=False,
        )

    action_str = str(data.get("action", "NOOP")).upper()
    try:
        action = VoiceAction(action_str)
    except ValueError:
        action = VoiceAction.NOOP

    return ChefIntentResponse(
        action=action,
        step=data.get("step"),
        question=data.get("question"),
        confidence=0.85,
        filtered=False,
    )
