"""Shared test setup.

`src.utils.models` builds a `ChatOpenAI` at import time, so merely importing
`src.utils.nodes` (which is where `finalize_report` lives) raises `OpenAIError`
on a clean checkout with no `.env`. The routing functions under test never touch
the network, so a placeholder key is enough to make the module importable in CI
and on a fresh clone.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")
os.environ.setdefault("TAVILY_API_KEY", "tvly-test-not-a-real-key")
