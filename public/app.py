"""Lumina's FastAPI application and real-time reading adaptation endpoint."""

import json
import logging
import os
import re
from pathlib import Path

from fastapi import FastAPI
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


class FrictionProfile(BaseModel):
    struggle_patterns: list[str]


class AdaptResponse(BaseModel):
    adapted_text: str
    user_friction_profile: FrictionProfile


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


def adapt_with_llm(text: str) -> AdaptResponse:
    """Call OpenAI with a strict JSON schema so the frontend always gets usable data."""


    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(

        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
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
async def adapt(request: AdaptRequest) -> AdaptResponse:
    """Adapt a paragraph after a reader's dwell signal without breaking the demo."""

    try:
        return await run_in_threadpool(adapt_with_llm, request.text)
    except Exception:

        logger.exception("Lumina adaptation failed for paragraph %s", request.paragraph_id)
        return AdaptResponse(
            adapted_text=request.text,
            user_friction_profile=FrictionProfile(
                struggle_patterns=fallback_profile(request.text)
            ),
        )



app.mount("/", StaticFiles(directory=Path(__file__).parent, html=True), name="public")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
