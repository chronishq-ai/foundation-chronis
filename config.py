from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # The CLI remains usable when optional dotenv is not installed.
    def load_dotenv() -> bool:
        env_path = os.getenv("CHRONIS_ENV_FILE", ".env")
        try:
            for line in open(env_path, encoding="utf-8"):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'") )
        except FileNotFoundError:
            return False
        return True

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    anthropic_api_key: Optional[str]
    anthropic_model: str
    groq_api_key: Optional[str]
    groq_model: str
    active_provider: str  # "gemini" | "anthropic" | "groq"
    max_retries: int
    retry_initial_seconds: float
    log_path: str


def load_settings() -> Settings:
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        active_provider=os.getenv("LLM_PROVIDER", "gemini").lower(),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        retry_initial_seconds=float(os.getenv("RETRY_INITIAL_SECONDS", "1")),
        log_path=os.getenv("LOG_PATH", "logs/outputs.json"),
    )


settings = load_settings()
