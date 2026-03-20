"""
ingest.py — Clone a GitHub repo, parse files, chunk code, build vector index.
"""

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Generator, Optional

import git
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config
from cache import get_store_path, save_cache

logger = logging.getLogger(__name__)

# ─── Extension → LangChain Language enum ───────────────────────────────────────
_LANG_ENUM_MAP: dict[str, Optional[Language]] = {
    ".py":   Language.PYTHON,
    ".js":   Language.JS,
    ".ts":   Language.TS,
    ".jsx":  Language.JS,
    ".tsx":  Language.TS,
    ".cpp":  Language.CPP,
    ".c":    Language.C,
    ".java": Language.JAVA,
    ".go":   Language.GO,
    ".rs":   Language.RUST,
    ".md":   Language.MARKDOWN,
    ".rst":  None,
    ".html": Language.HTML,
    ".css":  None,
}

# Best-effort regex patterns to extract a function/class name from the start of a chunk
_FUNC_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*(?:async\s+)?def\s+(\w+)"),           # Python def / async def
    re.compile(r"^\s*class\s+(\w+)"),                        # Python class / Java class
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)"),  # JS/TS function
    re.compile(r"^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\("),            # JS/TS arrow
    re.compile(r"^\s*func\s+(\w+)"),                         # Go
    re.compile(r"^\s*fn\s+(\w+)"),                           # Rust
    re.compile(r"^\s*public\s+\w+\s+(\w+)\s*\("),           # Java method
]


def _extract_function_name(text: str) -> str:
    """Return the first function/class name found in text, or empty string."""
    for line in text.splitlines()[:10]:
        for pat in _FUNC_PATTERNS:
            m = pat.match(line)
            if m:
                return m.group(1)
    return ""


def clone_repo(repo_url: str, token: Optional[str] = None) -> str:
    """
    Clone *repo_url* into a fresh temp directory and return its path.
    If *token* (or env var GITHUB_TOKEN) is set, embed it for private repos.
    """
    token = token or os.getenv("GITHUB_TOKEN", "")
    clone_url = repo_url.strip()

    if token and clone_url.startswith("https://github.com/"):
        clone_url = clone_url.replace(
            "https://github.com/", f"https://{token}@github.com/"
        )

    dest = tempfile.mkdtemp(prefix="repoask_")
    logger.info("Cloning %s → %s", repo_url, dest)
    git.Repo.clone_from(clone_url, dest, depth=1)
    return dest


def get_files(repo_path: str) -> tuple[list[Path], bool]:
    """
    Walk repo_path and return (list_of_file_paths, too_large).
    too_large is True when more than config.MAX_FILES_WARN files are found.
    """
    root = Path(repo_path)
    files: list[Path] = []

    for path in root.rglob("*"):
        # Skip ignored directories
        if any(part in config.IGNORED_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            continue
        files.append(path)

    too_large = len(files) > config.MAX_FILES_WARN
    return files, too_large


def _get_splitter(ext: str) -> RecursiveCharacterTextSplitter:
    """Return a language-aware splitter for the given file extension."""
    lang = _LANG_ENUM_MAP.get(ext.lower())
    if lang is not None:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=1200, chunk_overlap=200
        )
    # Generic splitter for .rst, .css, and other text formats
    return RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)


def chunk_files(
    files: list[Path],
    repo_path: str,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> list[Document]:
    """
    Parse and chunk *files* into LangChain Documents with rich metadata.
    Calls progress_cb(current, total, filename) for each file processed.
    """
    root = Path(repo_path)
    all_docs: list[Document] = []
    total = len(files)

    for idx, fpath in enumerate(files):
        if progress_cb:
            progress_cb(idx + 1, total, str(fpath.name))

        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            logger.warning("Skipping %s: %s", fpath, exc)
            continue

        if not text.strip():
            continue

        ext = fpath.suffix.lower()
        rel_path = str(fpath.relative_to(root)).replace("\\", "/")
        language = config.EXTENSION_LANGUAGE_MAP.get(ext, "text")
        splitter = _get_splitter(ext)

        lines = text.splitlines()
        try:
            chunks: list[str] = splitter.split_text(text)
        except Exception as exc:
            logger.warning("Splitter failed for %s: %s", fpath, exc)
            continue

        char_pos = 0
        for chunk_text in chunks:
            # Approximate line numbers by scanning from current char position
            start_char = text.find(chunk_text, char_pos)
            if start_char == -1:
                start_char = char_pos
            end_char = start_char + len(chunk_text)

            start_line = text[:start_char].count("\n") + 1
            end_line = text[:end_char].count("\n") + 1
            char_pos = end_char

            func_name = _extract_function_name(chunk_text)

            doc = Document(
                page_content=chunk_text,
                metadata={
                    "file_path": rel_path,
                    "language": language,
                    "start_line": start_line,
                    "end_line": end_line,
                    "function_name": func_name,
                    "source": rel_path,
                },
            )
            all_docs.append(doc)

    logger.info("Total chunks produced: %d", len(all_docs))
    return all_docs


def build_index(
    chunks: list[Document],
    repo_url: str,
    commit_hash: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Embed *chunks* and persist to ChromaDB (or Pinecone).
    Returns the store_path string so the caller can load it later.
    """
    if progress_cb:
        progress_cb("Generating embeddings…")

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    store_path = get_store_path(repo_url, commit_hash)

    if config.VECTOR_STORE == "pinecone":
        _build_pinecone(chunks, embeddings, progress_cb)
        store_path_str = f"pinecone:{config.PINECONE_INDEX_NAME}"
    else:
        _build_chroma(chunks, embeddings, store_path, progress_cb)
        store_path_str = str(store_path)

    if progress_cb:
        progress_cb("Saving cache metadata…")

    save_cache(
        repo_url,
        commit_hash,
        num_chunks=len(chunks),
        num_files=len({d.metadata["file_path"] for d in chunks}),
    )

    return store_path_str


def _build_chroma(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    store_path: Path,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    if progress_cb:
        progress_cb("Building ChromaDB vector index…")
    store_path.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(store_path),
    )
    logger.info("ChromaDB index saved at %s", store_path)


def _build_pinecone(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    if not config.PINECONE_API_KEY:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Switch VECTOR_STORE to 'chroma' or set the key."
        )
    if progress_cb:
        progress_cb("Building Pinecone vector index…")

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existing = [i.name for i in pc.list_indexes()]
    if config.PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=768,  # BGE-base output dim
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=config.PINECONE_INDEX_NAME,
    )
    logger.info("Pinecone index '%s' populated.", config.PINECONE_INDEX_NAME)


def cleanup_repo(repo_path: str) -> None:
    """Remove the temporary clone directory."""
    try:
        shutil.rmtree(repo_path, ignore_errors=True)
        logger.info("Cleaned up %s", repo_path)
    except Exception as exc:
        logger.warning("Cleanup failed: %s", exc)
