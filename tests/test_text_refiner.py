import json
import urllib.error

from text_refiner import (
    RefinerSettings,
    TranscriptRefiner,
    TranscriptRefinementResult,
    build_refiner_messages,
    load_refiner_settings,
)


def test_load_refiner_settings_reads_env_defaults(monkeypatch):
    monkeypatch.delenv("TEXT_REFINER_ENABLED", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_BASE_URL", raising=False)
    monkeypatch.delenv("GROQ_REFINER_MODEL", raising=False)
    monkeypatch.delenv("TEXT_REFINER_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("TEXT_REFINER_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TEXT_REFINER_TEMPERATURE", raising=False)
    monkeypatch.delenv("TEXT_REFINER_TOP_P", raising=False)
    monkeypatch.delenv("TEXT_REFINER_SEED", raising=False)

    settings = load_refiner_settings()

    assert settings == RefinerSettings(
        enabled=False,
        api_key="",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        timeout_seconds=4.0,
        max_tokens=128,
        temperature=0.0,
        top_p=1.0,
        seed=7,
    )


def test_build_refiner_messages_uses_expected_prompt():
    messages = build_refiner_messages("hello world")

    assert messages[0]["role"] == "system"
    assert "Return only the corrected text." in messages[0]["content"]
    assert messages[1] == {
        "role": "user",
        "content": "Transcript:\nhello world",
    }


def test_refine_returns_original_text_when_disabled():
    refiner = TranscriptRefiner(
        RefinerSettings(
            enabled=False,
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
            timeout_seconds=4.0,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            seed=7,
        )
    )

    assert refiner.refine("hello world") == "hello world"


def test_refine_posts_expected_payload_and_returns_cleaned_text(monkeypatch):
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "Hello, world."
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["headers"] = dict(request.headers)
        seen["timeout"] = timeout
        seen["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    refiner = TranscriptRefiner(
        RefinerSettings(
            enabled=True,
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
            timeout_seconds=4.0,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            seed=7,
        )
    )

    result = refiner.refine("hello world")

    assert result == "Hello, world."
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["headers"]["User-agent"] == "vibeVoice/1.0"
    assert seen["timeout"] == 4.0
    assert seen["payload"]["model"] == "llama-3.1-8b-instant"
    assert seen["payload"]["temperature"] == 0.0
    assert seen["payload"]["top_p"] == 1.0
    assert seen["payload"]["max_tokens"] == 128
    assert seen["payload"]["seed"] == 7
    assert seen["payload"]["messages"][1]["content"] == "Transcript:\nhello world"


def test_refine_with_result_reports_successful_refinement(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "Hello, world."
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(_request, timeout):
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    refiner = TranscriptRefiner(
        RefinerSettings(
            enabled=True,
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
            timeout_seconds=4.0,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            seed=7,
        )
    )

    result = refiner.refine_with_result("hello world")

    assert result.text == "Hello, world."
    assert result.status == "refined"
    assert result.detail == ""
    assert result.elapsed_seconds >= 0.0


def test_refine_returns_original_text_on_http_error(monkeypatch):
    def fake_urlopen(_request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    refiner = TranscriptRefiner(
        RefinerSettings(
            enabled=True,
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
            timeout_seconds=4.0,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            seed=7,
        )
    )

    assert refiner.refine("hello world") == "hello world"


def test_refine_with_result_reports_timeout_fallback(monkeypatch):
    def fake_urlopen(_request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    refiner = TranscriptRefiner(
        RefinerSettings(
            enabled=True,
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
            timeout_seconds=4.0,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            seed=7,
        )
    )

    result = refiner.refine_with_result("hello world")

    assert result.text == "hello world"
    assert result.status == "fallback_error"
    assert "timed out" in result.detail
    assert result.elapsed_seconds >= 0.0


def test_refine_returns_empty_string_for_empty_input():
    refiner = TranscriptRefiner(
        RefinerSettings(
            enabled=True,
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
            timeout_seconds=4.0,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            seed=7,
        )
    )

    assert refiner.refine("   ") == ""


def test_refine_with_result_returns_original_text_on_http_429(monkeypatch):
    def fake_urlopen(_request, timeout):
        raise urllib.error.HTTPError(
            url="https://api.groq.com/openai/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    refiner = TranscriptRefiner(
        RefinerSettings(
            enabled=True,
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
            timeout_seconds=4.0,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            seed=7,
        )
    )

    result = refiner.refine_with_result("hello world")

    assert result.text == "hello world"
    assert result.status == "fallback_error"
    assert "429" in result.detail
