# RepoAsk

Ask natural language questions about any public GitHub repository and get answers with source citations — powered entirely by **free, local, open-source tools**. No paid API keys required.

---

## Features

- 🔎 **Language-aware chunking** — `RecursiveCharacterTextSplitter` with per-language separators
- 🧠 **Local embeddings** — `BAAI/bge-base-en-v1.5` via `sentence-transformers` (auto-downloads on first run)
- 💾 **Index caching** — skips re-indexing if the same repo + commit was already processed
- 🗄️ **Vector storage** — ChromaDB (default, zero config) or Pinecone (free tier)
- 📑 **Source citations** — every answer includes file paths, function names, and line ranges
- 🦙 **Local LLM** — fully via [Ollama](https://ollama.ai); supports `llama3`, `codellama`, `mistral`

---

## Prerequisites

### 1. Python 3.11+

```bash
python --version
```

### 2. Create a virtual environment (recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate     # Linux / macOS
```

> If PowerShell blocks the activation script, run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ On first run, `sentence-transformers` will download **~440 MB** for `BAAI/bge-base-en-v1.5` into `~/.cache/huggingface/`. Subsequent runs use the cache.

### 4. Install and start Ollama

1. Download from [https://ollama.ai](https://ollama.ai)
2. Pull the default model (once):
   ```bash
   ollama pull llama3
   ```
3. Start the server (keep this terminal open):
   ```bash
   ollama serve
   ```

The app will detect if Ollama isn't running and show setup instructions in the UI.

**Alternative models** (set `OLLAMA_MODEL` in `config.py`):

| Model | Pull command | Notes |
|---|---|---|
| `llama3` | `ollama pull llama3` | Default, well-rounded |
| `codellama` | `ollama pull codellama` | Better for code-heavy repos |
| `mistral` | `ollama pull mistral` | Lighter and faster |

---

## Running the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Paste a GitHub repository URL** in the sidebar (e.g. `https://github.com/pallets/flask`)
2. Click **Index Repository** — the app clones, parses, chunks, embeds and stores the code
3. Ask questions in the chat:
   - *How does authentication work?*
   - *Where is the database connection created?*
   - *Explain how the API routes are structured.*

Every answer follows this format:

```
Answer:
[explanation referencing actual functions, classes, variables, or modules]

Sources:
→ path/to/file.py — function_name() [lines 12–34]
→ path/to/other.js — anotherFunction() [lines 56–78]
```

### First-run expectations

| Step | What happens |
|---|---|
| `pip install` | Downloads ~1–2 GB of packages |
| First app launch | BGE model downloads (~440 MB), one time only |
| First index of a repo | Embeddings computed locally, ~1–5 min depending on repo size |
| Same repo on next run | Instant — loaded from cache |

---

## Configuration

Edit `config.py`:

| Setting | Default | Description |
|---|---|---|
| `VECTOR_STORE` | `"chroma"` | `"chroma"` or `"pinecone"` |
| `OLLAMA_MODEL` | `"llama3"` | LLM model name |
| `OLLAMA_BASE_URL` | `"http://localhost:11434"` | Ollama server URL |
| `RETRIEVER_K` | `6` | Chunks retrieved per query |
| `MAX_FILES_WARN` | `500` | Warn before indexing large repos |

### Pinecone (optional)

1. Create a free account at [pinecone.io](https://pinecone.io)
2. Copy `.env.example` → `.env` and fill in your key:
   ```
   PINECONE_API_KEY=your_key_here
   PINECONE_INDEX_NAME=repoask
   ```
3. Set `VECTOR_STORE = "pinecone"` in `config.py`

### Private repositories

```powershell
$env:GITHUB_TOKEN = "ghp_your_token_here"   # Windows
# export GITHUB_TOKEN=ghp_your_token_here   # Linux / macOS
```

---

## Project Structure

```
repoask/
  app.py           # Streamlit application entry point
  ingest.py        # Clone repo, parse files, build index
  query.py         # Retrieval chain and question answering
  cache.py         # Check if repo already indexed
  config.py        # Model settings, vector store configuration
  requirements.txt
  README.md
  .env.example
```

### Index cache location

```
~/.repoask/
  cache/index.json          # Cache metadata (repo → commit → store path)
  chroma_stores/<key>/      # Persisted ChromaDB index per repo+commit
```

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| LLM | Ollama (`langchain-ollama`) |
| Embeddings | `sentence-transformers` / `langchain-huggingface` |
| Vector store | ChromaDB (`langchain-chroma`) or Pinecone (`langchain-pinecone`) |
| RAG chain | LangChain `RetrievalQA` |
| Git | GitPython |
