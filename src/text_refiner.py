from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from src import log

SYSTEM_PROMPT = """You clean up speech-to-text transcripts.
Fix punctuation, capitalization, spacing, and obvious transcription mistakes.
Remove filler words and stutters only when the meaning stays the same.
Keep commands, code terms, filenames, product names, and technical words exactly when possible.
Do not add new information.
Return only the corrected text."""

USER_AGENT = "vibeVoice/1.0"


@dataclass(frozen=True)
class RefinerSettings:
    enabled: bool
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float
    max_tokens: int
    temperature: float
    top_p: float
    seed: int


@dataclass(frozen=True)
class TranscriptRefinementResult:
    text: str
    status: str
    detail: str
    elapsed_seconds: float


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def load_refiner_settings() -> RefinerSettings:
    return RefinerSettings(
        enabled=_env_flag("TEXT_REFINER_ENABLED", False),
        api_key=os.environ.get("GROQ_API_KEY", "").strip(),
        base_url=os.environ.get("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
        model=os.environ.get("GROQ_REFINER_MODEL", "llama-3.1-8b-instant").strip(),
        timeout_seconds=float(os.environ.get("TEXT_REFINER_TIMEOUT_SECONDS", "4.0")),
        max_tokens=int(os.environ.get("TEXT_REFINER_MAX_TOKENS", "128")),
        temperature=float(os.environ.get("TEXT_REFINER_TEMPERATURE", "0.0")),
        top_p=float(os.environ.get("TEXT_REFINER_TOP_P", "1.0")),
        seed=int(os.environ.get("TEXT_REFINER_SEED", "7")),
    )


def build_refiner_messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Transcript:\n{text.strip()}"},
    ]


class TranscriptRefiner:
    def __init__(self, settings: RefinerSettings):
        self.settings = settings

    def _log_refinement_summary(
        self,
        *,
        session_id: str | None,
        status: str,
        input_text: str,
        output_text: str,
        elapsed_seconds: float,
        detail: str = "",
    ) -> None:
        short_session_id = session_id or "-"
        summary = (
            f"[Refiner] session={short_session_id} status={status} "
            f"elapsed={elapsed_seconds:.2f}s "
            f"input_chars={len(input_text)} "
            f"output_chars={len(output_text)}"
        )
        if detail:
            summary = f"{summary} detail={detail}"
        log.info(summary)

    def refine_with_result(self, text: str, session_id: str | None = None) -> TranscriptRefinementResult:
        original_text = text.strip()
        if not original_text:
            return TranscriptRefinementResult(text="", status="skipped_empty", detail="", elapsed_seconds=0.0)
        if not self.settings.enabled or not self.settings.api_key:
            return TranscriptRefinementResult(text=original_text, status="disabled", detail="", elapsed_seconds=0.0)

        payload = {
            "model": self.settings.model,
            "messages": build_refiner_messages(original_text),
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens,
            "seed": self.settings.seed,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.settings.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.api_key}",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )

        started_at = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            refined_text = str(body["choices"][0]["message"]["content"]).strip()
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            socket.timeout,
            TimeoutError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            elapsed_seconds = time.perf_counter() - started_at
            detail = str(exc)
            self._log_refinement_summary(
                session_id=session_id,
                status="fallback_error",
                input_text=original_text,
                output_text=original_text,
                elapsed_seconds=elapsed_seconds,
                detail=detail,
            )
            return TranscriptRefinementResult(
                text=original_text,
                status="fallback_error",
                detail=detail,
                elapsed_seconds=elapsed_seconds,
            )

        elapsed_seconds = time.perf_counter() - started_at
        if not refined_text:
            self._log_refinement_summary(
                session_id=session_id,
                status="fallback_blank",
                input_text=original_text,
                output_text=original_text,
                elapsed_seconds=elapsed_seconds,
                detail="blank response",
            )
            return TranscriptRefinementResult(
                text=original_text,
                status="fallback_blank",
                detail="blank response",
                elapsed_seconds=elapsed_seconds,
            )

        if refined_text == original_text:
            self._log_refinement_summary(
                session_id=session_id,
                status="unchanged",
                input_text=original_text,
                output_text=refined_text,
                elapsed_seconds=elapsed_seconds,
            )
            return TranscriptRefinementResult(
                text=original_text,
                status="unchanged",
                detail="",
                elapsed_seconds=elapsed_seconds,
            )

        self._log_refinement_summary(
            session_id=session_id,
            status="refined",
            input_text=original_text,
            output_text=refined_text,
            elapsed_seconds=elapsed_seconds,
        )
        return TranscriptRefinementResult(
            text=refined_text,
            status="refined",
            detail="",
            elapsed_seconds=elapsed_seconds,
        )

    def refine(self, text: str) -> str:
        return self.refine_with_result(text).text
