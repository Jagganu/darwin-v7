import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
PAPERS_DIR = DATA_DIR / "papers"
DB_PATH    = DATA_DIR / "darwin.db"

DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
PAPERS_DIR.mkdir(exist_ok=True)

# ── LLM Provider Config ────────────────────────────────────────────────────
# Choose your primary and fallback providers.
# Options: "ollama" | "anthropic" | "nvidia" | "openai_compatible"

PRIMARY_PROVIDER  = os.getenv("DARWIN_PRIMARY",  "ollama")
PRIMARY_MODEL     = os.getenv("DARWIN_MODEL",     "qwen2.5:14b")

FALLBACK_PROVIDER = os.getenv("DARWIN_FALLBACK",  "anthropic")
FALLBACK_MODEL    = os.getenv("DARWIN_FB_MODEL",  "claude-sonnet-4-6")

# ── API Keys ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY",  "")
NVIDIA_API_KEY     = os.getenv("NVIDIA_API_KEY",     "")   # NVIDIA NIM
OPENAI_COMPAT_URL  = os.getenv("OPENAI_COMPAT_URL",  "")   # OpenRouter / Together / etc.
OPENAI_COMPAT_KEY  = os.getenv("OPENAI_COMPAT_KEY",  "")

# ── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL    = PRIMARY_MODEL  # kept for backward compat

# ── NVIDIA NIM popular models ───────────────────────────────────────────────
# "meta/llama-3.1-70b-instruct"
# "mistralai/mixtral-8x7b-instruct-v0.1"
# "01-ai/yi-large"
# Set PRIMARY_PROVIDER=nvidia and DARWIN_MODEL=<model> to use

# ── ArXiv ───────────────────────────────────────────────────────────────────
ARXIV_MAX_RESULTS = 20

# ── Embedding model (always local, no API needed) ───────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
