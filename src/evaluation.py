"""Transparent, lightweight evaluation metrics for the Streamlit demonstration."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with",
}


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in _STOP_WORDS and len(token) > 1
    }


def faithfulness_proxy(answer: str, documents: list[Document]) -> float:
    """Estimate lexical evidence coverage; this is not a substitute for RAGAS."""
    if not answer.strip() or not documents:
        return 0.0
    context_tokens = set().union(
        *(_content_tokens(document.page_content) for document in documents)
    )
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer)
        if _content_tokens(sentence)
    ]
    if not sentences:
        return 0.0
    sentence_scores = [
        len(_content_tokens(sentence) & context_tokens) / len(_content_tokens(sentence))
        for sentence in sentences
    ]
    return round(sum(sentence_scores) / len(sentence_scores), 4)


def answer_relevance_proxy(question: str, answer: str) -> float:
    """Estimate query-term coverage in the response using lexical overlap."""
    query_tokens = _content_tokens(question)
    if not query_tokens or not answer.strip():
        return 0.0
    return round(len(query_tokens & _content_tokens(answer)) / len(query_tokens), 4)


def evaluate_response(
    question: str,
    answer: str,
    documents: list[Document],
    *,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute demonstration metrics and optionally append a privacy-conscious log row."""
    metrics: dict[str, Any] = {
        "faithfulness_proxy": faithfulness_proxy(answer, documents),
        "answer_relevance_proxy": answer_relevance_proxy(question, answer),
        "retrieved_chunks": len(documents),
    }
    if log_path is not None:
        path = Path(log_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        try:
            with path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=True) + "\n")
        except OSError:
            logger.exception("Could not append evaluation metrics to %s", path)
            raise
    return metrics
