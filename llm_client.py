from __future__ import annotations

from abc import ABC, abstractmethod

from config import Settings


class LLMProvider(ABC):
    """Minimal interface every provider must implement."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send `prompt` to the model and return the raw text response."""
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        from google import genai  # local import: keeps this an optional dep

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        return response.text


class AnthropicProvider(LLMProvider):
    """Drop-in replacement for GeminiProvider - identical `generate()` shape."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        import anthropic  # local import: keeps this an optional dep

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class GroqProvider(LLMProvider):
    """Drop-in replacement - identical `generate()` shape. OpenAI-compatible API."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        from groq import Groq  # local import: keeps this an optional dep

        self._client = Groq(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return response.choices[0].message.content


def get_provider(settings: Settings) -> LLMProvider:
    """Factory: builds whichever provider is configured as active."""
    if settings.active_provider == "gemini":
        return GeminiProvider(settings.gemini_api_key, settings.gemini_model)
    if settings.active_provider == "anthropic":
        return AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model)
    if settings.active_provider == "groq":
        return GroqProvider(settings.groq_api_key, settings.groq_model)
    raise ValueError(
        f"Unknown LLM_PROVIDER={settings.active_provider!r}; "
        "expected 'gemini', 'anthropic', or 'groq'"
    )