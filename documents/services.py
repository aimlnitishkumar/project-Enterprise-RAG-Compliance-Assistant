from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from django.conf import settings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

from documents.models import UserDocument
from src.generator import FALLBACK_ANSWER, generate_answer
from src.ingestion import create_embeddings, rebuild_vector_store
from src.retriever import DEFAULT_RERANKER_MODEL, HybridRerankingRetriever

logger = logging.getLogger(__name__)
VECTOR_STORE_DIR = settings.BASE_DIR / "vector_store"


def user_collection_name(user_id: int) -> str:
    return f"shryai_user_{user_id}"


@lru_cache(maxsize=1)
def _embeddings():
    return create_embeddings()


@lru_cache(maxsize=1)
def _cross_encoder() -> CrossEncoder:
    return CrossEncoder(os.getenv("SHRYAI_RERANKER_MODEL", DEFAULT_RERANKER_MODEL))


@lru_cache(maxsize=32)
def _retriever_for_user(user_id: int) -> HybridRerankingRetriever:
    from src.ingestion import get_vector_store

    store = get_vector_store(
        VECTOR_STORE_DIR,
        embeddings=_embeddings(),
        collection_name=user_collection_name(user_id),
    )
    return HybridRerankingRetriever(
        store,
        initial_k=10,
        top_k=4,
        cross_encoder=_cross_encoder(),
    )


def invalidate_retrievers() -> None:
    _retriever_for_user.cache_clear()


def _document_paths(documents: list[UserDocument]) -> tuple[list[Path], dict[str, str]]:
    paths: list[Path] = []
    source_names: dict[str, str] = {}
    for document in documents:
        path = Path(document.file.path).resolve()
        paths.append(path)
        source_names[str(path)] = document.original_name
    return paths, source_names


def _delete_collection(user_id: int) -> None:
    collection = Chroma(
        collection_name=user_collection_name(user_id),
        embedding_function=None,
        persist_directory=str(VECTOR_STORE_DIR.resolve()),
    )
    collection.delete_collection()


def rebuild_user_index(user_id: int, documents: list[UserDocument] | None = None) -> int:
    owned_documents = (
        documents
        if documents is not None
        else list(UserDocument.objects.filter(owner_id=user_id).order_by("pk"))
    )
    invalidate_retrievers()
    if not owned_documents:
        _delete_collection(user_id)
        return 0

    paths, source_names = _document_paths(owned_documents)
    _, chunk_count = rebuild_vector_store(
        paths,
        VECTOR_STORE_DIR,
        embeddings=_embeddings(),
        collection_name=user_collection_name(user_id),
        source_names=source_names,
    )
    invalidate_retrievers()
    return chunk_count


def delete_user_index(user_id: int) -> None:
    invalidate_retrievers()
    _delete_collection(user_id)


def answer_question(user_id: int, question: str, *, top_k: int = 4) -> dict:
    if not UserDocument.objects.filter(owner_id=user_id).exists():
        return {"answer": FALLBACK_ANSWER, "sources": []}
    retriever = _retriever_for_user(user_id)
    documents = retriever.retrieve(question, top_k=top_k)
    answer = generate_answer(question, documents)
    sources = [
        {
            "source": str(document.metadata.get("source", "unknown source")),
            "page": document.metadata.get("page", "unknown"),
            "text": document.page_content,
        }
        for document in documents
    ]
    return {"answer": answer, "sources": sources}
