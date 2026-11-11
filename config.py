"""Central configuration for EduRAG."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index_store"
DATA_DIR.mkdir(exist_ok=True)

# Embeddings (runs locally, free)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking (characters)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = 4
MIN_SCORE = 0.25  # below this similarity we say "not in the material" instead of guessing

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
