"""Lumina's FastAPI application and real-time reading adaptation endpoint.

Adaptation is fully rule-based (no external LLM call, no API key required):
it splits overly long sentences, swaps in simpler vocabulary for detected
jargon, and breaks the result into short readable chunks. Claude-Mem is used
to track recurring struggle patterns across a reading session so repeat
patterns get a more aggressive rewrite.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import BackgroundTasks, FastAPI, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)
app = FastAPI(title="Lumina")


class AdaptRequest(BaseModel):
    """The dwell event sent by the reading interface."""

    paragraph_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=12_000)
    dwell_seconds: float = Field(ge=0, le=3_600)
    # The browser generates this once and keeps it in sessionStorage.
    reader_session_id: str = Field(default="demo-reader", min_length=1, max_length=100)


class FrictionProfile(BaseModel):
    struggle_patterns: list[str]


class AdaptResponse(BaseModel):
    adapted_text: str
    user_friction_profile: FrictionProfile
    personalized: bool = False
    personalization_trigger: str | None = None
    memory_pattern_count: int = 0
    intensity: str = "light"


class MemoryProfile(BaseModel):
    """The reader's pattern counts, retrieved from Claude-Mem's worker API."""

    pattern_count: int = 0
    recurring_pattern: str | None = None
    available: bool = False


CLAUDE_MEM_URL = os.getenv("CLAUDE_MEM_URL", "http://127.0.0.1:37701").rstrip("/")
CLAUDE_MEM_PROJECT_PREFIX = "lumina-reader"
RECURRENCE_THRESHOLD = 3
PATTERN_MARKER = re.compile(r"lumina_pattern:([a-z0-9_]+)")


def claude_mem_project(reader_session_id: str) -> str:
    """Scope a browser reading session to one real Claude-Mem project."""

    safe_session_id = re.sub(r"[^a-zA-Z0-9_-]", "-", reader_session_id)[:64]
    return f"{CLAUDE_MEM_PROJECT_PREFIX}-{safe_session_id or 'demo'}"


def claude_mem_request(
    method: str, path: str, payload: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Use Claude-Mem's local worker API without letting memory failures block reading."""

    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{CLAUDE_MEM_URL}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=1) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as error:
        logger.info("Claude-Mem unavailable: %s", error)
        return None


def memory_profile(reader_session_id: str) -> MemoryProfile:
    """Count processed Claude-Mem observations before a fresh adaptation is generated."""

    project = claude_mem_project(reader_session_id)
    result = claude_mem_request(
        "GET", f"/api/observations?{urlencode({'project': project, 'limit': 200})}"
    )
    if result is None:
        return MemoryProfile()

    observations = result.get("items", result.get("observations", []))
    counts: dict[str, int] = {}
    for observation in observations:
        serialized = json.dumps(observation).lower()
        for pattern in set(PATTERN_MARKER.findall(serialized)):
            counts[pattern] = counts.get(pattern, 0) + 1

    recurring = next(
        (pattern for pattern, count in sorted(counts.items()) if count >= RECURRENCE_THRESHOLD),
        None,
    )
    return MemoryProfile(
        pattern_count=sum(counts.values()),
        recurring_pattern=recurring,
        available=True,
    )


def record_claude_mem_observation(
    request: AdaptRequest,
    patterns: list[str],
    personalized: bool,
    trigger: str | None,
) -> None:
    """Fire-and-forget a structured reading event into Claude-Mem's real observer."""

    project = claude_mem_project(request.reader_session_id)
    content_session_id = f"lumina-{project}"
    event = {
        "event_type": "lumina_adaptation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paragraph_id": request.paragraph_id,
        "dwell_seconds": request.dwell_seconds,
        "struggle_patterns": patterns,
        "pattern_markers": [f"lumina_pattern:{pattern}" for pattern in patterns],
        "personalized": personalized,
        "personalization_trigger": trigger,
    }
    claude_mem_request(
        "POST",
        "/api/sessions/init",
        {
            "contentSessionId": content_session_id,
            "project": project,
            "prompt": "Lumina adaptive reading session",
            "platformSource": "codex-cli",
        },
    )
    claude_mem_request(
        "POST",
        "/api/sessions/observations",
        {
            "contentSessionId": content_session_id,
            "tool_name": "lumina_adapt",
            "tool_input": event,
            "tool_response": {"status": "adapted", "pattern_markers": event["pattern_markers"]},
            "cwd": str(Path.cwd()),
            "platformSource": "codex-cli",
        },
    )


# ---------------------------------------------------------------------------
# Rule-based text simplification (no LLM / API key required)
# ---------------------------------------------------------------------------

# Longer, more specific phrases first so they match before their component words do.
SIMPLIFICATION_MAP: list[tuple[str, str]] = [
    (r"\bsynthetic computational architectures\b", "computer systems"),
    (r"\bbiological neural networks\b", "the brain's networks"),
    (r"\bneural engineering\b", "brain-computer engineering"),
    (r"\bcognitive fatigue\b", "mental tiredness"),
    (r"\bdense technical documentation\b", "hard technical text"),
    (r"\bin order to\b", "to"),
    (r"\bprior to\b", "before"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bwith respect to\b", "about"),
    (r"\butiliz(e|es|ed|ing)\b", "use"),
    (r"\bfacilitat(e|es|ed|ing)\b", "help"),
    (r"\bnecessitat(e|es|ed|ing)\b", "need"),
    (r"\bsubsequently\b", "then"),
    (r"\bapproximately\b", "about"),
    (r"\bdemonstrat(e|es|ed|ing)\b", "show"),
    (r"\bregarding\b", "about"),
    (r"\bsignificantly\b", "greatly"),
    (r"\bsignificant\b", "big"),
    (r"\bnumerous\b", "many"),
    (r"\bindividuals\b", "people"),
    (r"\bmethodology\b", "method"),
    (r"\badditional\b", "more"),
    (r"\bsufficient\b", "enough"),
    (r"\bcomprehensive\b", "complete"),
    (r"\bimplement(s|ed|ing)?\b", "do"),
    (r"\bcomponents?\b", "parts"),
    (r"\bfurthermore\b", "also"),
    (r"\btherefore\b", "so"),
    (r"\bhowever\b", "but"),
    (r"\bcommenc(e|es|ed|ing)\b", "start"),
    (r"\bterminat(e|es|ed|ing)\b", "end"),
    (r"\bobtain(s|ed|ing)?\b", "get"),
    (r"\brequir(e|es|ed|ing)\b", "need"),
    (r"\bassist(s|ed|ing)?\b", "help"),
    (r"\bindicat(e|es|ed|ing)\b", "show"),
    (r"\bconstruct(s|ed|ing)?\b", "build"),
    (r"\boccur(s|red|ring)?\b", "happen"),
    (r"\battempt(s|ed|ing)?\b", "try"),
    (r"\binduces\b", "causes"),
    (r"\binduce\b", "cause"),
    (r"\binduced\b", "caused"),
    (r"\binducing\b", "causing"),
    (r"\bprocessing\b", "working through"),
    (r"\butilization\b", "use"),
]


def simplify_vocabulary(text: str) -> str:
    """Swap detected jargon/complex words for simpler equivalents, preserving case."""

    result = text
    for pattern, replacement in SIMPLIFICATION_MAP:
        def repl(match: re.Match, replacement: str = replacement) -> str:
            original = match.group(0)
            if original[:1].isupper():
                return replacement[:1].upper() + replacement[1:]
            return replacement

        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
    return result


def intensity_for_dwell(dwell_seconds: float, forced_max: bool = False) -> str:
    """Map how long a reader has been stuck on a paragraph to a simplification tier."""

    if forced_max or dwell_seconds >= 6:
        return "aggressive"
    if dwell_seconds >= 4:
        return "moderate"
    return "light"


_MAX_WORDS_BY_INTENSITY = {"light": 18, "moderate": 10, "aggressive": 6}


def hard_wrap(piece: str, max_words: int) -> list[str]:
    """Force-wrap a chunk into fixed-size word groups when no natural break exists.

    Used only at the highest intensity so a stubborn short-but-dense sentence
    still visibly fragments further, rather than looking identical to a lower tier.
    """

    core = piece.rstrip(".!?")
    words = core.split()
    if len(words) <= max_words:
        return [piece]
    out = []
    for i in range(0, len(words), max_words):
        group = " ".join(words[i : i + max_words])
        if not group.endswith((".", "!", "?")):
            group += "."
        out.append(group)
    return out


def split_into_chunks(text: str, intensity: str = "light") -> list[str]:
    """Break long sentences into short, readable chunks. Higher intensity -> shorter chunks."""

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    max_words = _MAX_WORDS_BY_INTENSITY.get(intensity, 18)
    split_markers = [", which", ", and", ", but", ", because", ", since", " and ", "; "]

    chunks: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words:
            chunks.append(sentence)
            continue

        best_idx, best_marker = -1, ""
        midpoint = len(sentence) // 2
        for marker in split_markers:
            idx = sentence.find(marker)
            if idx != -1 and (best_idx == -1 or abs(idx - midpoint) < abs(best_idx - midpoint)):
                best_idx, best_marker = idx, marker

        if best_idx != -1:
            first = sentence[:best_idx].strip()
            second = sentence[best_idx + len(best_marker):].strip()
            if first and not first.endswith((".", "!", "?")):
                first += "."
            if second:
                second = second[0].upper() + second[1:]
                if not second.endswith((".", "!", "?")):
                    second += "."
            chunks.extend([c for c in (first, second) if c])
        else:
            chunks.append(sentence)

    if intensity == "aggressive":
        expanded: list[str] = []
        for chunk in chunks:
            expanded.extend(hard_wrap(chunk, max_words))
        chunks = expanded

    return chunks


def analyze_struggle_patterns(text: str) -> list[str]:
    """Derive friction tags from real features of the input text."""

    patterns: list[str] = []
    sentences = re.split(r"[.!?]+", text)
    if any(len(s.split()) >= 20 for s in sentences):
        patterns.append("long_sentence")
    if len(re.findall(r"\b\w{12,}\b", text)) >= 2:
        patterns.append("complex_vocabulary")
    if text.count(",") >= 3 or text.count(";") >= 1:
        patterns.append("dense_clause")
    if re.search(r"\b(is|are|was|were|been|being)\s+\w+ed\b", text, re.IGNORECASE):
        patterns.append("passive_voice")
    if not patterns:
        patterns.append("general_complexity")
    return patterns


def adapt_rule_based(
    text: str, dwell_seconds: float = 2.0, recurring_pattern: str | None = None
) -> tuple[AdaptResponse, str]:
    """Rewrite text using rules only -- no network call, no API key needed.

    Simplification gets more aggressive the longer the reader has been stuck
    (dwell_seconds), and maxes out immediately if Claude-Mem flagged a
    recurring struggle pattern for this reader.
    """

    patterns = analyze_struggle_patterns(text)
    simplified = simplify_vocabulary(text)
    intensity = intensity_for_dwell(dwell_seconds, forced_max=recurring_pattern is not None)
    chunks = split_into_chunks(simplified, intensity=intensity)
    adapted_text = "\n".join(chunks)
    response = AdaptResponse(
        adapted_text=adapted_text,
        user_friction_profile=FrictionProfile(struggle_patterns=patterns),
    )
    return response, intensity


@app.post("/api/adapt", response_model=AdaptResponse)
async def adapt(request: AdaptRequest, background_tasks: BackgroundTasks) -> AdaptResponse:
    """Adapt a paragraph after a reader's dwell signal without breaking the demo."""

    profile = await run_in_threadpool(memory_profile, request.reader_session_id)
    try:
        result, intensity = await run_in_threadpool(
            adapt_rule_based, request.text, request.dwell_seconds, profile.recurring_pattern
        )
        result.intensity = intensity
    except Exception:
        logger.exception("Lumina adaptation failed for paragraph %s", request.paragraph_id)
        result = AdaptResponse(
            adapted_text=request.text,
            user_friction_profile=FrictionProfile(struggle_patterns=["general_complexity"]),
        )

    result.personalized = profile.recurring_pattern is not None
    result.personalization_trigger = profile.recurring_pattern
    result.memory_pattern_count = profile.pattern_count
    background_tasks.add_task(
        record_claude_mem_observation,
        request,
        result.user_friction_profile.struggle_patterns,
        result.personalized,
        result.personalization_trigger,
    )
    return result


@app.get("/api/memory", response_model=MemoryProfile)
async def get_memory(
    reader_session_id: str = Query(default="demo-reader", min_length=1, max_length=100),
) -> MemoryProfile:
    """Give the UI a live Claude-Mem-derived count for its memory badge."""

    return await run_in_threadpool(memory_profile, reader_session_id)


app.mount("/", StaticFiles(directory=Path(__file__).parent / "public", html=True), name="public")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
