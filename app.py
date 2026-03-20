"""
app.py — Streamlit entry point for RepoAsk.
Run with: streamlit run app.py
"""

import logging
import os
from pathlib import Path

import streamlit as st

import config
from cache import get_latest_commit_hash, is_cached, get_store_path, load_cache_entry
from ingest import cleanup_repo, clone_repo, get_files, chunk_files, build_index
from query import ask, build_chain, check_ollama


# ─── Helper: Source Citation Renderer ──────────────────────────────────────────
def _render_sources(sources: list[dict]) -> None:
    """Render source metadata as a formatted citation block."""
    if not sources:
        return
    lines = []
    seen = set()
    for s in sources:
        fp = s.get("file_path") or s.get("source", "")
        fn = s.get("function_name", "")
        sl = s.get("start_line", "")
        el = s.get("end_line", "")
        key = f"{fp}|{fn}|{sl}"
        if key in seen or not fp:
            continue
        seen.add(key)
        parts = [f"→ {fp}"]
        if fn:
            parts.append(f" — {fn}()")
        if sl and el:
            parts.append(f" [lines {sl}–{el}]")
        lines.append("".join(parts))
    if lines:
        block = "\n".join(lines)
        st.markdown(
            f'<div class="source-box"><strong>Sources</strong><br>'
            f'{block.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RepoAsk",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Dark background */
    .stApp { background-color: #0d0d0d; color: #e0e0e0; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #1f1f1f; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: #a0a0a0 !important; }

    /* Inputs */
    .stTextInput > div > div > input {
        background: #1a1a1a !important; border: 1px solid #2a2a2a !important;
        color: #e0e0e0 !important; border-radius: 8px;
    }
    .stTextInput > div > div > input:focus { border-color: #00c896 !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00c896, #007b5e);
        color: #000 !important; font-weight: 700; border-radius: 10px;
        border: none; padding: 0.6rem 1.4rem;
        transition: opacity 0.2s ease;
    }
    .stButton > button:hover { opacity: 0.88; }
    .stButton > button:disabled { background: #2a2a2a; color: #555 !important; }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #151515; border: 1px solid #1f1f1f;
        border-radius: 12px; padding: 0.8rem 1rem; margin-bottom: 0.5rem;
    }

    /* Source box */
    .source-box {
        background: #0f1f1a; border: 1px solid #00c89640;
        border-radius: 8px; padding: 0.6rem 1rem;
        font-size: 0.82rem; color: #7ecfb8;
        margin-top: 0.8rem; font-family: monospace;
    }

    /* Status banner */
    .status-ok   { color: #00c896; background: #0d1f19; border: 1px solid #00c89630;
                   border-radius: 8px; padding: 0.5rem 1rem; }
    .status-warn { color: #ffc947; background: #1f1a0d; border: 1px solid #ffc94740;
                   border-radius: 8px; padding: 0.5rem 1rem; }
    .status-err  { color: #ff6b6b; background: #1f0d0d; border: 1px solid #ff6b6b40;
                   border-radius: 8px; padding: 0.5rem 1rem; }

    /* Divider */
    hr { border-color: #1f1f1f; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Session State Init ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "indexed_repo" not in st.session_state:
    st.session_state.indexed_repo = None
if "store_path" not in st.session_state:
    st.session_state.store_path = None


# ─── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([0.06, 0.94])
with col_logo:
    st.markdown("### ⚡")
with col_title:
    st.markdown("## **RepoAsk**")
    st.caption("Ask natural language questions about any public GitHub repository — powered by local LLMs.")

st.markdown("---")

# ─── Ollama Check ───────────────────────────────────────────────────────────────
ollama_ok, ollama_err = check_ollama()

if not ollama_ok:
    st.error("**Ollama is not running.**")
    st.markdown(
        f"""
**Error:** `{ollama_err}`

Please follow these steps to set up Ollama:

1. **Download** Ollama from [https://ollama.ai](https://ollama.ai)
2. **Pull** the default model:
   ```bash
   ollama pull {config.OLLAMA_MODEL}
   ```
3. **Start** the server:
   ```bash
   ollama serve
   ```
4. **Refresh** this page.

> **Alternative models** you can use by editing `config.py`:
> - `codellama` — better for code-heavy repos
> - `mistral` — lighter and faster
        """
    )
    st.stop()

st.markdown(
    f'<p class="status-ok">✅ Ollama is running — model: <code>{config.OLLAMA_MODEL}</code> &nbsp;|&nbsp; '
    f'Vector store: <code>{config.VECTOR_STORE}</code></p>',
    unsafe_allow_html=True,
)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔗 Repository")

    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repo",
        help="Enter the URL of any public GitHub repository.",
    )

    force_reindex = st.checkbox(
        "Force Re-index",
        value=False,
        help="Re-index even if this commit is already cached.",
    )

    if st.button("🗂️ Index Repository", use_container_width=True, disabled=not repo_url.strip()):
        if not repo_url.strip():
            st.warning("Please enter a repository URL.")
        elif config.VECTOR_STORE == "pinecone" and not config.PINECONE_API_KEY:
            st.error(
                "**Pinecone API key not set.**\n\n"
                "Set `PINECONE_API_KEY` in your `.env` file or switch to "
                "`VECTOR_STORE='chroma'` in `config.py`."
            )
        else:
            repo_url_clean = repo_url.strip()
            st.session_state.messages = []
            st.session_state.chain = None

            # ── Step 1: Clone ──────────────────────────────────────────
            with st.status("⏳ Cloning repository…", expanded=True) as status:
                try:
                    st.write("Cloning repository…")
                    repo_path = clone_repo(repo_url_clean)
                except Exception as exc:
                    status.update(label="❌ Clone failed", state="error")
                    if "Authentication" in str(exc) or "not found" in str(exc).lower():
                        st.error(
                            "Repository not found or private.\n\n"
                            "For private repos, set `GITHUB_TOKEN` as an environment variable."
                        )
                    else:
                        st.error(f"Clone failed: {exc}")
                    st.stop()

                # ── Step 2: Commit hash + cache check ─────────────────
                st.write("Checking index cache…")
                commit_hash = get_latest_commit_hash(repo_path)
                cached = is_cached(repo_url_clean, commit_hash)

                if cached and not force_reindex:
                    entry = load_cache_entry(repo_url_clean, commit_hash)
                    store_path = entry["store_path"] if entry else str(get_store_path(repo_url_clean, commit_hash))
                    status.update(
                        label=f"✅ Already indexed ({entry['num_chunks']} chunks, "
                              f"{entry['num_files']} files) — loaded from cache.",
                        state="complete",
                        expanded=False,
                    )
                    cleanup_repo(repo_path)
                    st.session_state.store_path = store_path
                    st.session_state.indexed_repo = repo_url_clean
                    st.session_state.chain = build_chain(store_path)  # returns (chain, retriever)
                    st.rerun()

                # ── Step 3: File scan ─────────────────────────────────
                st.write("Parsing files…")
                files, too_large = get_files(repo_path)

                if too_large:
                    status.update(label="⚠️ Large repository detected", state="running")

                if too_large:
                    st.warning(
                        f"This repository contains **{len(files)} files** "
                        f"(>{config.MAX_FILES_WARN}). Indexing may take several minutes."
                    )
                    proceed = st.button("Proceed with indexing anyway", key="proceed_large")
                    if not proceed:
                        cleanup_repo(repo_path)
                        st.stop()

                # ── Step 4: Chunking ──────────────────────────────────
                st.write(f"Chunking {len(files)} files…")
                progress_bar = st.progress(0.0)
                progress_text = st.empty()

                def chunk_progress(current: int, total: int, fname: str) -> None:
                    frac = current / max(total, 1)
                    progress_bar.progress(frac)
                    progress_text.caption(f"[{current}/{total}] {fname}")

                chunks = chunk_files(files, repo_path, progress_cb=chunk_progress)
                progress_bar.empty()
                progress_text.empty()

                if not chunks:
                    status.update(label="❌ No content could be extracted.", state="error")
                    cleanup_repo(repo_path)
                    st.stop()

                # ── Step 5: Embed + store ──────────────────────────────
                embed_status = st.empty()

                def embed_progress(msg: str) -> None:
                    embed_status.caption(f"🔧 {msg}")

                st.write(f"Generating embeddings for {len(chunks)} chunks…")
                try:
                    store_path_str = build_index(
                        chunks,
                        repo_url_clean,
                        commit_hash,
                        progress_cb=embed_progress,
                    )
                except RuntimeError as exc:
                    status.update(label="❌ Index build failed", state="error")
                    st.error(str(exc))
                    cleanup_repo(repo_path)
                    st.stop()
                except Exception as exc:
                    status.update(label="❌ Index build failed", state="error")
                    st.error(f"Unexpected error: {exc}")
                    cleanup_repo(repo_path)
                    st.stop()

                embed_status.empty()
                cleanup_repo(repo_path)

                status.update(
                    label=f"✅ Indexed {len(chunks)} chunks from {len(files)} files.",
                    state="complete",
                    expanded=False,
                )

            st.session_state.store_path = store_path_str
            st.session_state.indexed_repo = repo_url_clean
            st.session_state.chain = build_chain(store_path_str)  # returns (chain, retriever)
            st.rerun()

    # ── Sidebar info ────────────────────────────────────────────────────────
    st.markdown("---")
    if st.session_state.indexed_repo:
        st.markdown(
            f'<p class="status-ok">✅ Indexed:<br><small><code>{st.session_state.indexed_repo}</code></small></p>',
            unsafe_allow_html=True,
        )

    st.markdown("### ⚙️ Configuration")
    st.caption(f"**Model:** `{config.OLLAMA_MODEL}`")
    st.caption(f"**Embeddings:** `{config.EMBEDDING_MODEL}`")
    st.caption(f"**Vector store:** `{config.VECTOR_STORE}`")
    st.caption(f"**Top-K:** `{config.RETRIEVER_K}` chunks")

    st.markdown("---")
    st.markdown(
        "**Example questions:**\n"
        "- How does authentication work?\n"
        "- Where is the database connection created?\n"
        "- Explain how the API routes are structured.\n"
        "- What does the main entry point do?"
    )


# ─── Main Chat Area ─────────────────────────────────────────────────────────────
if not st.session_state.chain:
    st.markdown(
        """
        <div style="text-align:center; padding: 5rem 2rem; color: #444;">
            <div style="font-size:3rem; margin-bottom:1rem;">🗂️</div>
            <h3 style="color:#666;">No repository indexed yet</h3>
            <p>Enter a GitHub repository URL in the sidebar and click <strong>Index Repository</strong> to begin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # ── Render conversation history ─────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "⚡"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    # ── Chat input ──────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about the codebase…"):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🙋"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Searching codebase…"):
                answer, sources = ask(st.session_state.chain, prompt)

            st.markdown(answer)
            if sources:
                _render_sources(sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

