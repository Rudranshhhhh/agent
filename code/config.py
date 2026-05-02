"""
Configuration constants for the support triage agent.
All paths are resolved relative to this file's location.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
CORPUS_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INPUT_CSV = PROJECT_ROOT / "support_tickets" / "support_tickets.csv"
SAMPLE_CSV = PROJECT_ROOT / "support_tickets" / "sample_support_tickets.csv"
OUTPUT_CSV = PROJECT_ROOT / "support_tickets" / "output.csv"

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq", "gemini", or "nvidia"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "mistralai/mistral-medium-3.5-128b")
LLM_TEMPERATURE = 0.0  # Deterministic where possible (§6.6)

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── RAG ──────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1500       # Characters per chunk
CHUNK_OVERLAP = 300     # Overlap between chunks for context continuity
TOP_K = 6               # Number of retrieval results per query

# ── Company mapping ──────────────────────────────────────────────────────────
COMPANY_DIR_MAP = {
    "hackerrank": "HackerRank",
    "claude": "Claude",
    "visa": "Visa",
}
