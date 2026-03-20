"""
cache.py — Persist index state so the same repo is not re-indexed every run.

Cache key: SHA-256(repo_url + latest_commit_hash)
Storage:   ~/.repoask/cache/index.json
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import git

from config import CACHE_DIR, CHROMA_STORE_BASE

logger = logging.getLogger(__name__)


def _ensure_cache_dir() -> Path:
    """Create ~/.repoask/cache/ if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_STORE_BASE.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _cache_file() -> Path:
    return _ensure_cache_dir() / "index.json"


def _load_cache() -> dict:
    cf = _cache_file()
    if cf.exists():
        try:
            return json.loads(cf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Cache file corrupt – starting fresh.")
    return {}


def _save_all(data: dict) -> None:
    _cache_file().write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8"
    )


# ─── Public API ────────────────────────────────────────────────────────────────

def get_latest_commit_hash(repo_path: str) -> str:
    """Return the short HEAD commit hash for the cloned repo."""
    try:
        repo = git.Repo(repo_path)
        return repo.head.commit.hexsha[:12]
    except Exception as exc:
        logger.warning("Could not read commit hash: %s", exc)
        return "unknown"


def make_cache_key(repo_url: str, commit_hash: str) -> str:
    """Stable, filesystem-safe cache key from URL + commit."""
    raw = f"{repo_url.strip()}|{commit_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def is_cached(repo_url: str, commit_hash: str) -> bool:
    """Return True if this exact repo+commit is already indexed."""
    key = make_cache_key(repo_url, commit_hash)
    cache = _load_cache()
    return key in cache


def get_store_path(repo_url: str, commit_hash: str) -> Path:
    """Return the Chroma persist directory for this repo+commit."""
    key = make_cache_key(repo_url, commit_hash)
    return CHROMA_STORE_BASE / key


def save_cache(
    repo_url: str,
    commit_hash: str,
    *,
    num_chunks: int = 0,
    num_files: int = 0,
) -> None:
    """Persist cache metadata for a successfully indexed repo."""
    key = make_cache_key(repo_url, commit_hash)
    store_path = get_store_path(repo_url, commit_hash)
    cache = _load_cache()
    cache[key] = {
        "repo_url": repo_url,
        "commit_hash": commit_hash,
        "store_path": str(store_path),
        "num_chunks": num_chunks,
        "num_files": num_files,
        "indexed_at": datetime.utcnow().isoformat(),
    }
    _save_all(cache)
    logger.info("Cache saved for %s @ %s", repo_url, commit_hash)


def load_cache_entry(repo_url: str, commit_hash: str) -> Optional[dict]:
    """Return the cache record for a repo+commit, or None."""
    key = make_cache_key(repo_url, commit_hash)
    return _load_cache().get(key)


def list_cached_repos() -> list[dict]:
    """Return all cached repo entries (for a potential UI display)."""
    return list(_load_cache().values())
