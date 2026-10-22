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


@st.cache_data(show_spinner=False)
def cached_index_stats() -> tuple[int, int]:
    return get_index_stats(VECTOR_STORE_DIR)


@st.cache_resource(show_spinner="Preparing document search…")
def cached_retriever():
    return HybridRerankingRetriever(
        cached_vector_store(),
        initial_k=10,
        top_k=4,
    )


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --shry-bg: #0a0f1a;
            --shry-panel: #111a2a;
            --shry-border: rgba(155, 177, 216, 0.16);
            --shry-muted: #9eacc3;
            --shry-text: #edf3ff;
            --shry-accent: #8aa8ff;
            --shry-cyan: #6fe2d0;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 72% 0%, rgba(55, 93, 172, 0.20), transparent 38%),
                var(--shry-bg);
            color: var(--shry-text);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #10192a 0%, #0c1422 100%);
            border-right: 1px solid var(--shry-border);
        }
        [data-testid="stSidebar"] > div { padding-top: 1.3rem; }
        .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }
        h1, h2, h3 { color: var(--shry-text); letter-spacing: -0.035em; }
        p, label, [data-testid="stMarkdownContainer"] { color: #d4deef; }
        .shry-brand {
            color: var(--shry-text);
            font-size: 1.55rem;
            font-weight: 750;
            letter-spacing: -0.05em;
            margin: 0 0 .2rem;
        }
        .shry-kicker {
            color: var(--shry-cyan);
            font-size: .72rem;
            font-weight: 750;
            letter-spacing: .16em;
            text-transform: uppercase;
        }
        .shry-hero {
            background: linear-gradient(120deg, rgba(23, 38, 63, .96), rgba(17, 27, 45, .92));
            border: 1px solid var(--shry-border);
            border-radius: 22px;
            margin: .6rem 0 1.4rem;
            padding: 1.7rem 1.9rem;
            box-shadow: 0 18px 60px rgba(0, 0, 0, .20);
        }
        .shry-hero h1 { font-size: clamp(2rem, 4vw, 3rem); margin: .3rem 0 .45rem; }
        .shry-hero p { color: var(--shry-muted); font-size: 1rem; margin: 0; max-width: 760px; }
        .shry-panel {
            background: rgba(17, 26, 42, .76);
            border: 1px solid var(--shry-border);
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            min-height: 112px;
        }
        .shry-panel strong { color: var(--shry-text); font-size: 1rem; }
        .shry-panel p { color: var(--shry-muted); font-size: .88rem; margin: .4rem 0 0; }
        [data-testid="stMetric"] {
            background: rgba(17, 26, 42, .72);
            border: 1px solid var(--shry-border);
            padding: .85rem 1rem;
            border-radius: 14px;
        }
        [data-testid="stMetricLabel"] { color: var(--shry-muted); }
        [data-testid="stMetricValue"] { color: var(--shry-text); }
        [data-testid="stChatMessage"] {
            background: rgba(17, 26, 42, .78);
            border: 1px solid var(--shry-border);
            border-radius: 16px;
            padding: .6rem .85rem;
            margin-bottom: .75rem;
        }
        [data-testid="stChatInput"] {
            background: #111a2a !important;
            background-image: none !important;
            border: 1px solid var(--shry-border) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInput"] textarea {
            background: #111a2a !important;
            background-image: none !important;
            color: #f7f9ff !important;
            -webkit-text-fill-color: #f7f9ff !important;
            caret-color: #6fe2d0 !important;
            opacity: 1 !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #aebbd0 !important;
            -webkit-text-fill-color: #aebbd0 !important;
            opacity: 1;
        }
        [data-testid="stExpander"] {
            border-color: var(--shry-border);
            border-radius: 12px;
        }
        .stButton > button, [data-testid="stFileUploader"] section {
            border-radius: 12px;
        }
        button[kind="primary"] {
            background: linear-gradient(100deg, #6689f4, #5fcbbb);
            border: 0;
            color: #08111c;
            font-weight: 700;
        }
        button[kind="primary"]:hover { border: 0; filter: brightness(1.08); }
        [data-testid="stTabs"] [role="tab"] { color: var(--shry-muted); }
        [data-testid="stTabs"] [aria-selected="true"] { color: var(--shry-cyan); }
        </style>
        """,
        unsafe_allow_html=True,
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


def _render_sidebar() -> tuple[float, int, int | None]:
    indexed_count: int | None = None
    with st.sidebar:
        st.markdown('<div class="shry-brand">ShryAI</div>', unsafe_allow_html=True)
        st.caption("Your private document workspace")
        st.divider()
        st.subheader("Document library")
        uploads = st.file_uploader(
            "Add PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help=(
                "Searchable text, scanned pages, and text inside PDF images are indexed. "
                "OCR requires Tesseract to be installed."
            ),
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
                    cached_retriever.clear()
                    cached_vector_store.clear()
                    cached_index_stats.clear()
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
            chunk_count, source_count = cached_index_stats()
            indexed_count = chunk_count
            st.metric("Indexed chunks", chunk_count)
            st.metric("Source documents", source_count)
            st.caption("Persistent local ChromaDB")
        except Exception as exc:
            logger.exception("Could not read index statistics")
            st.error(f"Could not read index status: {exc}")
        st.divider()
        st.caption("SEARCH SETTINGS")
        st.caption("Passages are retrieved locally before the answer is generated.")
        return temperature, top_k, indexed_count


def _render_chat(temperature: float, top_k: int, indexed_count: int | None) -> None:
    st.markdown(
        """
        <section class="shry-hero">
            <div class="shry-kicker">DOCUMENT INTELLIGENCE</div>
            <h1>Answers, grounded in your files.</h1>
            <p>Ask a question across your indexed PDFs. ShryAI finds the relevant passages,
            then shows you exactly where its answer came from.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="shry-panel">
                <strong>Start with a question</strong>
                <p>Try asking for a policy requirement, a definition, a deadline, or a
                comparison between two sections. Answers are only as reliable as the
                evidence retrieved from your documents.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

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
            with st.spinner("Searching your documents and preparing an answer…"):
                if indexed_count is None:
                    raise RuntimeError("The document index status could not be read.")
                if indexed_count == 0:
                    answer = FALLBACK_ANSWER
                    sources: list[Document] = []
                else:
                    retriever = cached_retriever()
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
    _apply_theme()
    temperature, top_k, indexed_count = _render_sidebar()
    chat_tab, evaluation_tab = st.tabs(["✦  Ask ShryAI", "◷  Quality check"])
    with chat_tab:
        _render_chat(temperature, top_k, indexed_count)
    with evaluation_tab:
        _render_evaluation()


if __name__ == "__main__":
    main()
