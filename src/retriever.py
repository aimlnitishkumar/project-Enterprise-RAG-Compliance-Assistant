"""Hybrid semantic/lexical retrieval followed by local cross-encoder ranking."""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from heapq import nlargest
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.ingestion import get_vector_store

logger = logging.getLogger(__name__)
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class HybridRerankingRetriever:
    """Retrieve a hybrid candidate set and return its most relevant passages."""

    def __init__(
        self,
        vector_store: Chroma,
        *,
        initial_k: int = 10,
        top_k: int = 4,
        reranker_model: str | None = None,
        cross_encoder: CrossEncoder | None = None,
    ) -> None:
        if initial_k < 1 or top_k < 1:
            raise ValueError("Retrieval counts must be positive integers.")
        self.vector_store = vector_store
        self.initial_k = initial_k
        self.top_k = top_k
        self.reranker_model = reranker_model or os.getenv(
            "SHRYAI_RERANKER_MODEL", DEFAULT_RERANKER_MODEL
        )
        self.cross_encoder = cross_encoder or CrossEncoder(self.reranker_model)

        corpus = self.vector_store.get(include=["documents", "metadatas"])
        self._documents = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(corpus.get("documents") or [], corpus.get("metadatas") or [])
            if text
        ]
        tokenized_corpus = [_tokens(document.page_content) for document in self._documents]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    @classmethod
    def from_persist_directory(
        cls,
        persist_directory: str | Path,
        *,
        initial_k: int = 10,
        top_k: int = 4,
        cross_encoder: CrossEncoder | None = None,
    ) -> "HybridRerankingRetriever":
        """Open an existing persistent index and configure a retriever."""
        store = get_vector_store(persist_directory)
        return cls(
            store,
            initial_k=initial_k,
            top_k=top_k,
            cross_encoder=cross_encoder,
        )

    @staticmethod
    def _document_key(document: Document) -> tuple[str, int, str]:
        return (
            str(document.metadata.get("source", "")),
            int(document.metadata.get("page", 0)),
            document.page_content,
        )

    def _hybrid_candidates(self, query: str) -> list[Document]:
        semantic_documents = self.vector_store.similarity_search(query, k=self.initial_k)
        candidates: dict[tuple[str, int, str], Document] = {}
        reciprocal_ranks: defaultdict[tuple[str, int, str], float] = defaultdict(float)

        for rank, document in enumerate(semantic_documents, start=1):
            key = self._document_key(document)
            candidates[key] = document
            reciprocal_ranks[key] += 1 / (60 + rank)

        if self._bm25 is not None:
            scores = self._bm25.get_scores(_tokens(query))
            lexical_top_indices = nlargest(
                self.initial_k,
                range(len(scores)),
                key=lambda index: float(scores[index]),
            )
            for rank, index in enumerate(lexical_top_indices, start=1):
                document = self._documents[index]
                key = self._document_key(document)
                candidates[key] = document
                reciprocal_ranks[key] += 1 / (60 + rank)

        ordered_keys = sorted(
            reciprocal_ranks,
            key=lambda key: reciprocal_ranks[key],
            reverse=True,
        )
        return [candidates[key] for key in ordered_keys[: self.initial_k]]

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[Document]:
        """Return up to ``top_k`` passages after cross-encoder reranking."""
        if not query.strip():
            raise ValueError("Enter a question before retrieving documents.")
        candidates = self._hybrid_candidates(query)
        if not candidates:
            return []

        pairs = [(query, document.page_content) for document in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        result_count = self.top_k if top_k is None else top_k
        if result_count < 1:
            raise ValueError("top_k must be a positive integer.")
        results = [document for document, _ in ranked[:result_count]]
        logger.info("Reranked %d candidates to %d passages", len(candidates), len(results))
        return results
