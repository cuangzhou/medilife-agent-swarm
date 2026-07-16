"""Optional local configuration for MediLife.

Prefer environment variables in deployed environments. Copy this file to
``config.py`` only for local development; ``config.py`` is ignored by Git.
"""

LLM_CONFIG = {
    "api_key": "",
    "model_name": "gpt-4.1-mini",
    "base_url": "https://api.openai.com/v1",
    "temperature": 0.7,
    "max_tokens": 8192,
}

MEM0_CONFIG = {"api_key": ""}

