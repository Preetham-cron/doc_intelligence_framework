"""
config.py  —  loads secrets from .envs/.local (never hard-code keys).

Expected file:  <project_root>/.envs/.local
Expected line:  GEMINI_API_KEY=your_key_here   (also accepts GOOGLE_API_KEY)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# project root = one level up from src/
ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".envs" / ".local"

def load_env():
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)
    # accept either name
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            f"No GEMINI_API_KEY found. Put it in {ENV_FILE} as:\n"
            f"    GEMINI_API_KEY=your_key_here"
        )
    os.environ["GEMINI_API_KEY"] = key   # normalise
    return key
