"""
llm_loader.py — Loads and returns the correct AI model (LLM) based on settings.

What this file does:
    Reads the LLM_PROVIDER environment variable from .env and returns
    the appropriate LangChain chat model — either OpenAI or Google Gemini.

What is an LLM?
    LLM stands for Large Language Model. It is the AI model (like GPT-4 or
    Gemini) that reads the transcript and generates evaluations, scores,
    and recommendations.

Why ChatOpenAI?
    ChatOpenAI is LangChain's wrapper around OpenAI's chat models (e.g., GPT-4o-mini).
    It gives you a consistent interface to call the model with prompts and
    receive structured responses.

    When to use: Set LLM_PROVIDER=openai in your .env file.
    Best for: High accuracy, rich reasoning, widely tested in production.

Why ChatGoogleGenerativeAI?
    ChatGoogleGenerativeAI is LangChain's wrapper around Google's Gemini models.
    It works the same way as ChatOpenAI — just a different model underneath.

    When to use: Set LLM_PROVIDER=gemini in your .env file.
    Best for: Cost-effective alternative, works well for most evaluation tasks.

Why temperature=0.3?
    Temperature controls how "creative" or "random" the AI's answers are.
    - 0.0 = very predictable, always picks the most likely answer
    - 1.0 = more creative and varied
    - 0.3 = slightly flexible but mostly consistent — ideal for evaluation tasks
      where you want reliable, repeatable scores.

Best Practices:
    - Never hardcode the provider name — always read from environment variables.
    - Keep temperature low (0.1–0.3) for classification and scoring tasks.
    - Raise a clear error if an unsupported provider is set, so it's easy to debug.

When to use:
    Call load_llm(config) once at startup. Then share the same LLM instance
    across all chains — avoids creating multiple connections unnecessarily.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


def load_llm(config: dict):
    """
    Reads LLM_PROVIDER from the environment and returns the correct LangChain model.

    Args:
        config: The loaded config dictionary (from config.json).

    Returns:
        A LangChain-compatible chat model instance (OpenAI or Gemini).

    Raises:
        ValueError: If LLM_PROVIDER is not set or is not a supported value.
    """
    provider = os.getenv("LLM_PROVIDER")

    if provider is None:
        raise ValueError("LLM_PROVIDER is not set in your .env file. Set it to 'openai' or 'gemini'.")

    provider = provider.lower()

    if provider == "openai":
        # Use OpenAI's GPT model with the model name and temperature from config.json
        return ChatOpenAI(
            model=config["llm"]["openai_model"],
            temperature=config["llm"]["temperature"]
        )
    elif provider == "gemini":
        # Use Google's Gemini model with the model name and temperature from config.json
        return ChatGoogleGenerativeAI(
            model=config["llm"]["gemini_model"],
            temperature=config["llm"]["temperature"]
        )
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{provider}'. "
            "Please set it to 'openai' or 'gemini' in your .env file."
        )