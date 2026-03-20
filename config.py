"""
config.py — Central configuration for RepoAsk.
Edit these values to switch between vector stores or LLMs.
"""

import os
from pathlib import Path

# ─── Vector Store ──────────────────────────────────────────────────────────────
# Options: "chroma" (default, no account needed) | "pinecone" (requires API key)
VECTOR_STORE: str = os.getenv("VECTOR_STORE", "chroma")

# Pinecone settings (only used when VECTOR_STORE="pinecone")
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "repoask")

# ─── Ollama / LLM ──────────────────────────────────────────────────────────────
# Options: "llama3" | "codellama" | "mistral"
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ─── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"

# ─── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVER_K: int = 6          # number of chunks to retrieve per query
MAX_FILES_WARN: int = 500     # warn user when repo exceeds this many files

# ─── Cache ─────────────────────────────────────────────────────────────────────
CACHE_DIR: Path = Path.home() / ".repoask" / "cache"
CHROMA_STORE_BASE: Path = Path.home() / ".repoask" / "chroma_stores"

# ─── Supported File Extensions ─────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".cpp", ".c", ".java", ".go", ".rs",
    ".md", ".rst", ".html", ".css",
}

# Language tag per extension (used for LangChain splitter)
EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py":   "python",
    ".js":   "js",
    ".ts":   "ts",
    ".jsx":  "js",
    ".tsx":  "ts",
    ".cpp":  "cpp",
    ".c":    "c",
    ".java": "java",
    ".go":   "go",
    ".rs":   "rust",
    ".md":   "markdown",
    ".rst":  "rst",
    ".html": "html",
    ".css":  "css",
}

# ─── Ignored Directories ───────────────────────────────────────────────────────
IGNORED_DIRS: set[str] = {
    "node_modules", ".git", "__pycache__", ".env",
    ".venv", "venv", "dist", "build", ".next",
    ".nuxt", "coverage", ".pytest_cache", ".mypy_cache",
}
