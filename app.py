"""Streamlit user interface for the ShryAI document assistant."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document

from src.evaluation import evaluate_response
from src.generator import FALLBACK_ANSWER, generate_answer
from src.ingestion import (
    create_embeddings,
    get_index_stats,
    get_vector_store,
    rebuild_vector_store,
)
from src.retriever import HybridRerankingRetriever

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading local embedding model…")
def cached_embeddings():
    return create_embeddings()


@st.cache_resource(show_spinner="Opening persistent document index…")
def cached_vector_store():
    return get_vector_store(VECTOR_STORE_DIR, embeddings=cached_embeddings())


@st.cache_resource(show_spinner="Loading local cross-encoder…")
def cached_retriever(indexed_count: int, top_k: int):
    del indexed_count
    return HybridRerankingRetriever(
        cached_vector_store(),
        initial_k=10,
        top_k=top_k,
    )


def _safe_pdf_name(filename: str) -> str:
    basename = Path(filename).name
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    if not sanitized or Path(sanitized).suffix.lower() != ".pdf":
        raise ValueError(f"Invalid PDF filename: {filename}")
    return sanitized


def _store_uploads(uploads) -> list[Path]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for upload in uploads:
        filename = _safe_pdf_name(upload.name)
        content = upload.getvalue()
        if not content.lstrip().startswith(b"%PDF-"):
            raise ValueError(f"{filename} does not appear to be a valid PDF file.")
        destination = (UPLOAD_DIR / filename).resolve()
        if destination.parent != UPLOAD_DIR.resolve():
            raise ValueError("The uploaded filename resolves outside the upload directory.")
        destination.write_bytes(content)
        paths.append(destination)
    return paths


def _render_sources(documents: list[Document]) -> None:
    with st.expander("Source References", expanded=False):
        if not documents:
            st.caption("No source passages were retrieved.")
            return
        for index, document in enumerate(documents, start=1):
            source = Path(str(document.metadata.get("source", "Unknown source"))).name
            page = document.metadata.get("page", "Unknown")
            st.markdown(f"**{index}. {source} — page {page}**")
            st.caption(document.page_content)


def _render_sidebar() -> tuple[float, int]:
    with st.sidebar:
        st.title("ShryAI")
        st.caption("Enterprise knowledge retrieval & compliance")
        st.subheader("Document ingestion")
        uploads = st.file_uploader(
            "Upload enterprise PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Uploading and indexing replaces the current index with all saved PDFs.",
        )
        if st.button("Build / refresh index", type="primary", use_container_width=True):
            if not uploads:
                st.error("Upload at least one PDF before building the index.")
            else:
                try:
                    with st.status("Saving files and building the local index…", expanded=True) as status:
                        upload_paths = _store_uploads(uploads)
                        pdf_paths = sorted(UPLOAD_DIR.glob("*.pdf"))
                        progress = st.progress(0.15, text="Reading and splitting PDF pages")
                        _, chunk_count = rebuild_vector_store(
                            pdf_paths,
                            VECTOR_STORE_DIR,
                            embeddings=cached_embeddings(),
                        )
                        progress.progress(1.0, text="Index ready")
                        status.update(label="Document index is ready", state="complete")
                    st.cache_resource.clear()
                    st.session_state["last_ingestion"] = {
                        "files": len(pdf_paths),
                        "chunks": chunk_count,
                        "uploaded": len(upload_paths),
                    }
                    st.success(f"Indexed {chunk_count} chunks from {len(pdf_paths)} PDFs.")
                    st.rerun()
                except Exception as exc:
                    logger.exception("Document ingestion failed")
                    st.error(f"Indexing failed: {exc}")

        temperature = st.slider("Generation temperature", 0.0, 1.0, 0.0, 0.1)
        top_k = st.slider("Retrieved passages", min_value=1, max_value=8, value=4)
        st.subheader("Index status")
        try:
            chunk_count, source_count = get_index_stats(
                VECTOR_STORE_DIR,
                embeddings=cached_embeddings(),
            )
            st.metric("Indexed chunks", chunk_count)
            st.metric("Source documents", source_count)
            st.caption("Persistent local ChromaDB")
        except Exception as exc:
            logger.exception("Could not read index statistics")
            st.error(f"Could not read index status: {exc}")
        return temperature, top_k


def _render_chat(temperature: float, top_k: int) -> None:
    st.header("Ask your documents")
    st.caption(
        "Answers are grounded in indexed PDF passages. Every supported claim must cite its source."
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_sources(message.get("sources", []))

    question = st.chat_input("Ask a question about your indexed documents")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            indexed_count, _ = get_index_stats(
                VECTOR_STORE_DIR,
                embeddings=cached_embeddings(),
            )
            if indexed_count == 0:
                answer = FALLBACK_ANSWER
                sources: list[Document] = []
            else:
                retriever = cached_retriever(indexed_count, top_k)
                sources = retriever.retrieve(question, top_k=top_k)
                answer = generate_answer(
                    question,
                    sources,
                    temperature=temperature,
                    chat_history=[
                        (message["role"], message["content"])
                        for message in st.session_state.messages[:-1]
                    ],
                )
            with st.chat_message("assistant"):
                st.markdown(answer)
                _render_sources(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            st.session_state["last_evaluation"] = {
                "question": question,
                "answer": answer,
                "documents": sources,
            }
        except Exception as exc:
            logger.exception("Question answering failed")
            with st.chat_message("assistant"):
                st.error(f"Could not answer this question: {exc}")


def _render_evaluation() -> None:
    st.header("Evaluation")
    st.write(
        "Run a lightweight lexical demonstration against the most recent chat response. "
        "These scores are proxies, not canonical RAGAS Faithfulness or Answer Relevance metrics."
    )
    last = st.session_state.get("last_evaluation")
    if not last:
        st.info("Ask a question in the Chat tab first, then evaluate its answer.")
        return

    st.markdown("**Sample query**")
    st.info(last["question"])
    st.markdown("**Sample response**")
    st.write(last["answer"])
    if st.button("Evaluate sample response", type="primary"):
        try:
            metrics = evaluate_response(
                last["question"],
                last["answer"],
                last["documents"],
                log_path=DATA_DIR / "evaluation_metrics.jsonl",
            )
            first, second, third = st.columns(3)
            first.metric("Faithfulness proxy", f'{metrics["faithfulness_proxy"]:.0%}')
            second.metric("Answer relevance proxy", f'{metrics["answer_relevance_proxy"]:.0%}')
            third.metric("Retrieved chunks", metrics["retrieved_chunks"])
            st.caption(
                "The JSONL log stores metric values and counts only; it does not store document "
                "text, questions, or generated answers."
            )
        except Exception as exc:
            logger.exception("Evaluation failed")
            st.error(f"Evaluation failed: {exc}")


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    st.set_page_config(page_title="ShryAI | Knowledge Assistant", page_icon="📚", layout="wide")
    st.title("ShryAI")
    st.caption("Grounded enterprise knowledge retrieval and compliance assistant")
    temperature, top_k = _render_sidebar()
    chat_tab, evaluation_tab = st.tabs(["Chat", "Evaluation"])
    with chat_tab:
        _render_chat(temperature, top_k)
    with evaluation_tab:
        _render_evaluation()


if __name__ == "__main__":
    main()
