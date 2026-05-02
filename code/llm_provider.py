"""
LLM Provider Interface — abstraction layer for multiple LLM backends.

Supports:
    - Groq (LLaMA models via Groq API)
    - Gemini (Google Gemini models)
    - NVIDIA (NVIDIA NIM API — OpenAI-compatible)

Usage:
    from llm_provider import get_llm_provider
    llm = get_llm_provider()  # Reads LLM_PROVIDER env var
    response = llm.invoke(prompt)
"""

import os
import sys
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """Send a prompt and return the response text."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name for logging."""
        ...


class GroqProvider(LLMProvider):
    """Groq API provider using LangChain's ChatGroq."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.0):
        from langchain_groq import ChatGroq

        self._model_name = model
        self._llm = ChatGroq(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_retries=3,
        )

    @property
    def name(self) -> str:
        return f"Groq ({self._model_name})"

    def invoke(self, prompt: str) -> str:
        result = self._llm.invoke(prompt)
        content = result.content if result and result.content else ""
        if not content.strip():
            raise ValueError("Empty LLM response from Groq")
        return content


class GeminiProvider(LLMProvider):
    """Google Gemini API provider using LangChain's ChatGoogleGenerativeAI."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.0):
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._model_name = model
        self._llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
            max_retries=3,
        )

    @property
    def name(self) -> str:
        return f"Gemini ({self._model_name})"

    def invoke(self, prompt: str) -> str:
        result = self._llm.invoke(prompt)
        content = result.content if result and result.content else ""
        if not content.strip():
            raise ValueError("Empty LLM response from Gemini")
        return content


class NvidiaProvider(LLMProvider):
    """NVIDIA NIM API provider (OpenAI-compatible endpoint)."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.0):
        from langchain_openai import ChatOpenAI

        self._model_name = model
        self._llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            max_retries=3,
            max_tokens=32768,
        )

    @property
    def name(self) -> str:
        return f"NVIDIA ({self._model_name})"

    def invoke(self, prompt: str) -> str:
        result = self._llm.invoke(prompt)
        content = result.content if result and result.content else ""
        if not content.strip():
            raise ValueError("Empty LLM response from NVIDIA")
        return content


# ── Provider Registry ────────────────────────────────────────────────────────

PROVIDERS = {
    "groq": GroqProvider,
    "gemini": GeminiProvider,
    "nvidia": NvidiaProvider,
}


def get_llm_provider(
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
) -> LLMProvider:
    """
    Factory function to create the appropriate LLM provider.

    Parameters
    ----------
    provider : str, optional
        "groq" or "gemini". Defaults to LLM_PROVIDER env var, then "groq".
    api_key : str, optional
        API key. If not provided, reads from env (GROQ_API_KEY or GEMINI_API_KEY).
    model : str, optional
        Model name. If not provided, reads from env or uses defaults.
    temperature : float
        Sampling temperature (0.0 for deterministic).

    Returns
    -------
    LLMProvider
        An initialized provider instance.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower().strip()

    if provider not in PROVIDERS:
        print(
            f"ERROR: Unknown LLM_PROVIDER '{provider}'. "
            f"Supported: {', '.join(PROVIDERS.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    if provider == "groq":
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            print("ERROR: GROQ_API_KEY not set.", file=sys.stderr)
            sys.exit(1)
        model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        return GroqProvider(api_key=api_key, model=model, temperature=temperature)

    elif provider == "gemini":
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY not set.", file=sys.stderr)
            sys.exit(1)
        model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return GeminiProvider(api_key=api_key, model=model, temperature=temperature)

    elif provider == "nvidia":
        api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not api_key:
            print("ERROR: NVIDIA_API_KEY not set.", file=sys.stderr)
            sys.exit(1)
        model = model or os.getenv("NVIDIA_MODEL", "mistralai/mistral-medium-3.5-128b")
        return NvidiaProvider(api_key=api_key, model=model, temperature=temperature)
