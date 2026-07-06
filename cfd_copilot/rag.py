"""Retrieval over real OpenFOAM tutorial dictionaries.

Why this matters: OpenFOAM ships hundreds of *working* cases. Indexing them gives
the copilot a grounded knowledge base of valid dictionary syntax and parameter
choices, which (a) powers a "ask the docs" feature and (b) can be fed to the LLM
as context to improve spec extraction.

Embeddings use Ollama (``nomic-embed-text``) so everything stays local. This
module requires the optional ``agent`` extras (langchain + faiss) and a running
Ollama server.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from cfd_copilot.config import Settings, settings as default_settings

# Dictionary filenames worth indexing (small, information-dense, valid by design).
_INTERESTING = {
    "U", "p", "T", "k", "omega", "epsilon", "nut", "alphat", "p_rgh",
    "controlDict", "fvSchemes", "fvSolution", "blockMeshDict",
    "transportProperties", "thermophysicalProperties", "turbulenceProperties",
    "momentumTransport", "physicalProperties",
}


def _tutorials_dir(settings: Settings) -> Optional[Path]:
    cand = settings.foam_tutorials or os.environ.get("FOAM_TUTORIALS", "")
    if cand and Path(cand).is_dir():
        return Path(cand)
    return None


def gather_documents(tutorials_dir: Path, max_files: int = 4000) -> List:
    """Collect tutorial dictionary files as LangChain Documents."""
    from langchain_core.documents import Document

    docs: List = []
    for path in sorted(Path(tutorials_dir).rglob("*")):
        if not path.is_file() or path.name not in _INTERESTING:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue
        rel = path.relative_to(tutorials_dir)
        # The parent chain encodes <category>/<solver>/<case>/...
        parts = rel.parts
        solver = parts[1] if len(parts) > 2 else "unknown"
        case = parts[-2] if len(parts) >= 2 else "unknown"
        header = f"# OpenFOAM tutorial: {rel}\n# solver: {solver}  case: {case}\n\n"
        docs.append(
            Document(
                page_content=header + text[:4000],
                metadata={"source": str(rel), "solver": solver, "case": case, "file": path.name},
            )
        )
        if len(docs) >= max_files:
            break
    return docs


def build_index(settings: Settings = default_settings, tutorials_dir: Optional[Path] = None) -> int:
    """Build and persist a FAISS index over the OpenFOAM tutorials.

    Returns the number of documents indexed.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_ollama import OllamaEmbeddings

    tut = Path(tutorials_dir) if tutorials_dir else _tutorials_dir(settings)
    if tut is None:
        raise FileNotFoundError(
            "Could not find OpenFOAM tutorials. Set FOAM_TUTORIALS to the tutorials path."
        )
    docs = gather_documents(tut)
    if not docs:
        raise RuntimeError(f"No indexable dictionary files found under {tut}")

    embeddings = OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)
    store = FAISS.from_documents(docs, embeddings)
    settings.rag_index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(settings.rag_index_dir))
    return len(docs)


def load_index(settings: Settings = default_settings):
    from langchain_community.vectorstores import FAISS
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)
    return FAISS.load_local(
        str(settings.rag_index_dir), embeddings, allow_dangerous_deserialization=True
    )


def retrieve(question: str, k: int = 4, settings: Settings = default_settings) -> List:
    store = load_index(settings)
    return store.similarity_search(question, k=k)


def answer(question: str, k: int = 4, settings: Settings = default_settings) -> str:
    """Answer an OpenFOAM question grounded in retrieved tutorial snippets."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama

    docs = retrieve(question, k=k, settings=settings)
    context = "\n\n---\n\n".join(
        f"[source: {d.metadata.get('source')}]\n{d.page_content}" for d in docs
    )
    llm = ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
    )
    sys = (
        "You are an OpenFOAM expert. Answer using ONLY the provided tutorial "
        "snippets. Cite the source path(s) you used. If the answer is not in the "
        "context, say so."
    )
    msg = f"Question: {question}\n\nContext:\n{context}"
    resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=msg)])
    sources = ", ".join(sorted({d.metadata.get("source", "?") for d in docs}))
    return f"{resp.content}\n\nSources: {sources}"
