"""
query.py — Retrieval-Augmented QA using Ollama + ChromaDB/Pinecone (LangChain v1.x LCEL).
"""

import logging

import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

import config

logger = logging.getLogger(__name__)

# ─── Prompt ────────────────────────────────────────────────────────────────────
_PROMPT_TEMPLATE = """You are a technical expert assistant for the "RepoAsk" tool.
Answer questions about the codebase using ONLY the provided context.

Rules (follow strictly):
1. Only answer using the retrieved code context below.
2. Always cite every source file you reference.
3. Never invent functions, classes, or file paths not present in the context.
4. If the answer is not present in the context, respond with EXACTLY:
   I could not find this in the indexed codebase.

Context:
{context}

Question: {question}

Format your answer EXACTLY as follows:

Answer:
[Your detailed explanation referencing specific functions, classes, variables, or modules]

Sources:
→ <file_path> — <function_name> [lines <start>–<end>]
(list every file you used; omit function/line info if unavailable)
"""

_PROMPT = PromptTemplate(
    template=_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


# ─── Ollama Health Check ────────────────────────────────────────────────────────

def check_ollama() -> tuple[bool, str]:
    """Return (True, '') if Ollama is reachable, else (False, error_message)."""
    try:
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            return True, ""
        return False, f"Ollama returned HTTP {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused — is Ollama running?"
    except requests.exceptions.Timeout:
        return False, "Connection timed out."
    except Exception as exc:
        return False, str(exc)


# ─── Embeddings ─────────────────────────────────────────────────────────────────

def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ─── Vector Store Loader ────────────────────────────────────────────────────────

def load_vectorstore(store_path: str):
    """Load and return the persisted vector store."""
    if store_path.startswith("pinecone:"):
        from langchain_pinecone import PineconeVectorStore
        index_name = store_path.split(":", 1)[1]
        return PineconeVectorStore(index_name=index_name, embedding=_get_embeddings())
    return Chroma(
        persist_directory=store_path,
        embedding_function=_get_embeddings(),
    )


# ─── Chain Builder (LCEL) ──────────────────────────────────────────────────────

def _format_docs(docs) -> str:
    """Concatenate retrieved documents into a single context string."""
    parts = []
    for doc in docs:
        meta = doc.metadata
        header = f"[{meta.get('file_path', 'unknown')}]"
        if meta.get("function_name"):
            header += f" — {meta['function_name']}()"
        if meta.get("start_line") and meta.get("end_line"):
            header += f" lines {meta['start_line']}–{meta['end_line']}"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain(store_path: str) -> RunnableSerializable:
    """
    Build and return an LCEL retrieval chain backed by the given vector store.
    Returns a dict-in / str-out runnable: invoke({"question": "..."})
    """
    vectorstore = load_vectorstore(store_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

    llm = ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
    )

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | _PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ─── Ask ───────────────────────────────────────────────────────────────────────

def ask(chain_and_retriever: tuple, question: str) -> tuple[str, list[dict]]:
    """
    Run *question* through the chain.
    Returns (answer_str, list_of_source_metadata_dicts).
    """
    chain, retriever = chain_and_retriever
    try:
        # Get source docs separately so we can show citations
        source_docs = retriever.invoke(question)
        answer: str = chain.invoke(question)
    except Exception as exc:
        logger.error("Chain error: %s", exc)
        return f"Error generating answer: {exc}", []

    sources = [doc.metadata for doc in source_docs]
    return answer, sources
