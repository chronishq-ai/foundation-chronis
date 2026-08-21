"""
Core Pod B module: analyze_event().

Pipeline: build prompt -> call LLM -> strip markdown fences (defensive,
in case the provider ignores the "no markdown" instruction) -> parse JSON
-> validate against EventSignals -> log -> return dict.

On invalid JSON or a failed Pydantic validation, the prompt is re-sent with
the specific error appended, up to `settings.max_retries` attempts.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import ValidationError

from config import settings
from llm_client import LLMProvider, get_provider
from logger import EventLogger
from prompt import build_prompt
from schemas import EventSignals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pod_b.analyzer")

_event_logger = EventLogger(settings.log_path)
_provider: Optional[LLMProvider] = None


def _get_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        _provider = get_provider(settings)
    return _provider


def set_provider(provider: LLMProvider) -> None:
    """Test/dependency-injection hook - bypasses the config-driven factory."""
    global _provider
    _provider = provider


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_and_validate(raw: str) -> EventSignals:
    cleaned = _strip_markdown_fences(raw)
    data = json.loads(cleaned)  # raises json.JSONDecodeError on malformed JSON
    return EventSignals(**data)  # raises pydantic.ValidationError on bad schema


def analyze_event(event: str, max_retries: Optional[int] = None) -> dict:
    """
    Convert a plain-English event sentence into structured signal JSON.

    Returns: {"signals": {"<variable>": {"value": float, "confidence": float}, ...}}
    Raises: RuntimeError if no valid output is produced within the retry budget.
    """
    provider = _get_provider()
    retries = settings.max_retries if max_retries is None else max_retries

    prompt = build_prompt(event)
    last_error: Optional[str] = None
    raw_response = ""

    for attempt in range(1, retries + 1):
        try:
            raw_response = provider.generate(prompt)
            result = _parse_and_validate(raw_response)
            payload = result.model_dump()
            _event_logger.log(event, raw_response, payload)
            return payload
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            logger.warning(
                "analyze_event: invalid output on attempt %d/%d for event=%r: %s",
                attempt, retries, event, last_error,
            )
            prompt = (
                build_prompt(event)
                + f"\n\nYour previous response was invalid ({last_error}). "
                  "Return ONLY valid JSON matching the schema, with no extra text."
            )
        except Exception as exc:  # provider/network-level failure
            last_error = str(exc)
            logger.error(
                "analyze_event: provider error on attempt %d/%d for event=%r: %s",
                attempt, retries, event, last_error,
            )

    _event_logger.log(event, raw_response, None, error=last_error)
    raise RuntimeError(
        f"Failed to get valid signals for event={event!r} after {retries} attempts: {last_error}"
    )
