"""
config_loader.py — Loads settings from the .env file and config.json.

What this file does:
    Reads two sources of configuration:
      1. .env file     → Contains secret values like API keys (LLM_PROVIDER, API keys)
      2. config.json   → Contains non-secret settings like model names, labels, score range

Why separate .env and config.json?
    - .env holds sensitive information (API keys) that should NEVER be committed to Git.
    - config.json holds safe settings that can be version-controlled and shared.
    - Keeping them separate is a security best practice.

Why use python-dotenv (load_dotenv)?
    load_dotenv() reads the .env file and loads each key=value pair into
    the system's environment variables. After calling it, you can access
    them with os.getenv("KEY_NAME").

Best Practices:
    - Always call load_dotenv() before reading any os.getenv() values.
    - Never hardcode API keys in your code — always use environment variables.
    - Raise a clear error if the config file is missing so it's easy to debug.

When to use:
    Call load_config() at the very start of your program (before anything else)
    to make all settings available throughout the application.
"""

import os
import json
from dotenv import load_dotenv


def load_config(config_path: str = "config/config.json") -> dict:
    """
    Loads environment variables from .env and reads settings from config.json.

    Args:
        config_path: Path to the JSON config file. Defaults to config/config.json.

    Returns:
        A dictionary containing all config settings.

    Raises:
        FileNotFoundError: If config.json does not exist at the given path.
    """
    # Load .env file into environment variables
    load_dotenv()

    # Make sure the config file actually exists before trying to open it
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # Read and parse the JSON config file
    with open(config_path, "r") as f:
        config = json.load(f)

    return config