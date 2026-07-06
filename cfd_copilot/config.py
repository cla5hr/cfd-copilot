"""Runtime configuration, read from environment with sensible local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    """Configuration for the copilot.

    Everything defaults to a local-first setup (Ollama on localhost) so the tool
    runs offline on a laptop/WSL with no cloud API keys.
    """

    # --- LLM (Ollama) ---
    ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model: str = _env("CFD_CHAT_MODEL", "qwen2.5-coder:7b")
    embed_model: str = _env("CFD_EMBED_MODEL", "nomic-embed-text")
    llm_temperature: float = float(_env("CFD_LLM_TEMPERATURE", "0.0"))

    # --- Paths ---
    work_dir: Path = Path(_env("CFD_WORK_DIR", "runs")).expanduser()
    rag_index_dir: Path = Path(_env("CFD_RAG_INDEX", ".rag_index")).expanduser()
    # Where OpenFOAM tutorials live (used to build the RAG index). Auto-detected
    # from $FOAM_TUTORIALS if not set.
    foam_tutorials: str = _env("FOAM_TUTORIALS", os.environ.get("FOAM_TUTORIALS", ""))

    # --- Agent behaviour ---
    max_repair_attempts: int = int(_env("CFD_MAX_REPAIRS", "3"))
    # If true, only mesh+validate; do not run the solver (faster iteration).
    validate_only_default: bool = _env("CFD_VALIDATE_ONLY", "0") == "1"

    def with_overrides(self, **kwargs) -> "Settings":
        data = {**self.__dict__, **{k: v for k, v in kwargs.items() if v is not None}}
        return Settings(**data)


settings = Settings()
