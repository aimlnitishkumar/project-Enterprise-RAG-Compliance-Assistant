"""PDF ingestion, chunking, embedding, and persistent Chroma indexing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "shryai_documents"


def create_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Create a normalized, local Sentence-Transformers embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_and_split_pdfs(
    pdf_paths: Iterable[str | Path],
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Load PDFs and split pages while retaining a one-based page number."""
    paths = [Path(path).expanduser().resolve() for path in pdf_paths]
    if not paths:
        raise ValueError("Select at least one PDF file to index.")
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks: list[Document] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"PDF file does not exist: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files can be indexed: {path.name}")

        try:
            pages = PyPDFLoader(str(path)).load()
        except Exception:
            logger.exception("Failed to load PDF %s", path)
            raise

        pages_with_text = [
            page for page in pages if page.page_content and page.page_content.strip()
        ]
        if not pages_with_text:
            raise ValueError(
                f"{path.name} has no extractable text. It may be a scanned or "
                "image-only PDF. Run OCR on the PDF or upload a searchable/text-based "
                "PDF, then try indexing again."
            )
        empty_page_numbers = [
            int(page.metadata.get("page", 0)) + 1
            for page in pages
            if not page.page_content or not page.page_content.strip()
        ]
        if empty_page_numbers:
            logger.warning(
                "%s has no extractable text on page(s) %s",
                path.name,
                ", ".join(map(str, empty_page_numbers)),
            )
        for page in pages:
            page.metadata["source"] = path.name
            page.metadata["page"] = int(page.metadata.get("page", 0)) + 1
        chunks.extend(splitter.split_documents(pages))

    if not chunks:
        raise ValueError("The selected PDFs did not contain any text to index.")
    logger.info("Loaded %d PDF chunks from %d files", len(chunks), len(paths))
    return chunks


def get_vector_store(
    persist_directory: str | Path,
    *,
    embeddings: HuggingFaceEmbeddings | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """Open the persistent Chroma collection, creating it if necessary."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings or create_embeddings(),
        persist_directory=str(Path(persist_directory).expanduser().resolve()),
    )


def rebuild_vector_store(
    pdf_paths: Iterable[str | Path],
    persist_directory: str | Path,
    *,
    embeddings: HuggingFaceEmbeddings | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> tuple[Chroma, int]:
    """Replace the collection contents with chunks from the selected PDFs."""
    documents = load_and_split_pdfs(pdf_paths)
    vector_store = get_vector_store(
        persist_directory,
        embeddings=embeddings,
        collection_name=collection_name,
    )

    # Chroma persists writes automatically; remove the prior index before adding
    # the new corpus so deleted or replaced PDFs cannot remain searchable.
    vector_store.delete_collection()
    vector_store = get_vector_store(
        persist_directory,
        embeddings=embeddings,
        collection_name=collection_name,
    )
    try:
        vector_store.add_documents(documents)
    except Exception:
        logger.exception("Failed to embed or persist the PDF chunks")
        raise
    logger.info("Rebuilt Chroma collection with %d chunks", len(documents))
    return vector_store, len(documents)


def get_index_stats(
    persist_directory: str | Path,
    *,
    embeddings: HuggingFaceEmbeddings | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> tuple[int, int]:
    """Return the indexed chunk count and distinct source-file count."""
    store = get_vector_store(
        persist_directory,
        embeddings=embeddings,
        collection_name=collection_name,
    )
    records = store.get(include=["metadatas"])
    metadatas = records.get("metadatas") or []
    sources = {metadata.get("source") for metadata in metadatas if metadata.get("source")}
    return len(records.get("ids") or []), len(sources)
