"""Lumina's FastAPI application and real-time reading adaptation endpoint."""

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


class MemoryProfile(BaseModel):
    """The reader's pattern counts, retrieved from Claude-Mem's worker API."""

    pattern_count: int = 0
    recurring_pattern: str | None = None
    available: bool = False


ADAPTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "adapted_text": {"type": "string"},
        "struggle_patterns": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["adapted_text", "struggle_patterns"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are Lumina, an accessibility-focused reading assistant.
Rewrite the supplied passage to reduce reading friction. Preserve its meaning and
important details. Use shorter sentences, common vocabulary, active phrasing, and
small readable chunks where useful. Do not add facts or commentary.

Also identify why this specific original passage could be difficult to read. Return
short, lowercase snake_case tags based only on real features in the input, such as
long_sentence, technical_jargon, passive_voice, dense_clause, or abstract_language.
Return an empty list if no clear friction pattern is present."""

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

    # Claude-Mem v13 calls this list `items`; older workers call it `observations`.
    observations = result.get("items", result.get("observations", []))
    counts: dict[str, int] = {}
    for observation in observations:
        serialized = json.dumps(observation).lower()
        # The marker is embedded in every Lumina event, so the worker's processed
        # observation remains machine-readable while still benefiting from its AI summary.
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
    # Initialize is idempotent. Claude-Mem immediately queues the observation and
    # compresses/stores it asynchronously, so Lumina never waits for that work.
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


def adapt_with_llm(text: str, recurring_pattern: str | None = None) -> AdaptResponse:
    """Call OpenAI with a strict JSON schema so the frontend always gets usable data."""

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=api_key)
    personalization_instruction = ""
    if recurring_pattern:
        personalization_instruction = (
            f"\nThis reader has repeatedly struggled with {recurring_pattern}. "
            "Adapt more proactively for that pattern: reduce it as much as possible, "
            "use one clear idea per sentence, and add short line breaks where helpful."
        )

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        input=[
            {"role": "system", "content": SYSTEM_PROMPT + personalization_instruction},
            {"role": "user", "content": text},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "lumina_adaptation",
                "strict": True,
                "schema": ADAPTATION_SCHEMA,
            }
        },
    )
    result = json.loads(response.output_text)
    return AdaptResponse(
        adapted_text=result["adapted_text"].strip(),
        user_friction_profile=FrictionProfile(
            struggle_patterns=[tag.strip().lower() for tag in result["struggle_patterns"]]
        ),
    )


def fallback_profile(text: str) -> list[str]:
    """Give a transparent, text-derived profile when the LLM is unavailable."""

    patterns: list[str] = []
    sentences = re.split(r"[.!?]+", text)
    if any(len(sentence.split()) >= 25 for sentence in sentences):
        patterns.append("long_sentence")
    if len(re.findall(r"\b\w{12,}\b", text)) >= 2:
        patterns.append("complex_vocabulary")
    if text.count(",") >= 3 or text.count(";") >= 1:
        patterns.append("dense_clause")
    return patterns


@app.post("/api/adapt", response_model=AdaptResponse)
async def adapt(request: AdaptRequest, background_tasks: BackgroundTasks) -> AdaptResponse:
    """Adapt a paragraph after a reader's dwell signal without breaking the demo."""

    profile = await run_in_threadpool(memory_profile, request.reader_session_id)
    try:
        result = await run_in_threadpool(
            adapt_with_llm, request.text, profile.recurring_pattern
        )
    except Exception:
        logger.exception("Lumina adaptation failed for paragraph %s", request.paragraph_id)
        result = AdaptResponse(
            adapted_text=request.text,
            user_friction_profile=FrictionProfile(
                struggle_patterns=fallback_profile(request.text)
            ),
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


app.mount("/", StaticFiles(directory=Path(__file__).parent, html=True), name="public")

if __name__ == "__main__":
    import
