# RepoAsk

Ask natural language questions about any public GitHub repository and get answers with source citations powered entirely by free, local, open-source tools. 

## Features

-  **Language-aware chunking** — `RecursiveCharacterTextSplitter` with per-language separators
-  **Local embeddings** — `BAAI/bge-base-en-v1.5` via `sentence-transformers` 
-  **Index caching** — skips re-indexing if the same repo + commit was already processed
-  **Vector storage** — ChromaDB 
-  **Source citations** — every answer includes file paths, function names, and line ranges
-  **Local LLM** — fully via [Ollama](https://ollama.ai); supports `llama3`, `codellama`, `mistral`

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

```


### 3. Install dependencies

```bash
pip install -r requirements.txt
```


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


## Running the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Paste a GitHub repository URL** in the sidebar 
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



## Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| LLM | Ollama (`langchain-ollama`) |
| Embeddings | `sentence-transformers` / `langchain-huggingface` |
| Vector store | ChromaDB (`langchain-chroma`) or Pinecone (`langchain-pinecone`) |
| RAG chain | LangChain `RetrievalQA` |
| Git | GitPython |
