"""Grounded answer generation and citation validation."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
FALLBACK_ANSWER = (
    "I cannot find the answer to your question within the provided enterprise documents."
)
SYSTEM_PROMPT = (
    "You are DocuMind, an enterprise compliance and document assistant. Answer the user query "
    "strictly using the provided context blocks. Do not use outside knowledge or hallucinate. "
    "If the answer cannot be found in the context, output: "
    "'I cannot find the answer to your question within the provided enterprise documents.' "
    "Always append inline citations pointing to the source file name and page number for every "
    "claim made. Cite each supported factual claim as [source filename, p. N]. If the context "
    "does not support a claim, do not make that claim."
)
CITATION_PATTERN = re.compile(r"\[([^\[\]]+?),\s*p\.?\s*(\d+)\]", re.IGNORECASE)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "The conversation history below is only for resolving follow-up wording. It is not "
            "evidence. Use only the context blocks as evidence.\n\n"
            "Conversation history:\n{chat_history}\n\n"
            "Context blocks:\n{context}\n\n"
            "Question: {question}",
        ),
    ]
)


def create_chat_model(
    *,
    provider: str | None = None,
    model_name: str | None = None,
    temperature: float = 0.0,
):
    """Create a configured Groq or OpenAI chat model."""
    load_dotenv()
    selected_provider = (provider or os.getenv("SHRYAI_LLM_PROVIDER", "groq")).lower()
    if not 0 <= temperature <= 2:
        raise ValueError("Temperature must be between 0 and 2.")

    if selected_provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Set GROQ_API_KEY in your .env file to use Groq.")
        return ChatGroq(
            model=model_name or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            temperature=temperature,
            api_key=api_key,
        )
    if selected_provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY in your .env file to use OpenAI.")
        return ChatOpenAI(
            model=model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            api_key=api_key,
        )
    raise ValueError("SHRYAI_LLM_PROVIDER must be either 'groq' or 'openai'.")


def _format_context(documents: Sequence[Document]) -> str:
    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = Path(str(document.metadata.get("source", "unknown source"))).name
        page = document.metadata.get("page", "unknown")
        blocks.append(
            f"[Context {index} | Source: {source} | Page: {page}]\n{document.page_content}"
        )
    return "\n\n".join(blocks)


def _citations_are_valid(answer: str, documents: Sequence[Document]) -> bool:
    citations = CITATION_PATTERN.findall(answer)
    if not citations:
        return False
    valid_sources = {
        (Path(str(document.metadata.get("source", ""))).name.casefold(), str(document.metadata.get("page", "")))
        for document in documents
    }
    return all((Path(source.strip()).name.casefold(), page) in valid_sources for source, page in citations)


def generate_answer(
    question: str,
    documents: Sequence[Document],
    *,
    temperature: float = 0.0,
    chat_history: Sequence[tuple[str, str]] | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    llm=None,
) -> str:
    """Generate a grounded response, falling back when evidence/citations fail."""
    if not question.strip():
        raise ValueError("Enter a question before generating an answer.")
    if not documents:
        return FALLBACK_ANSWER

    model = llm or create_chat_model(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
    )
    history_text = "\n".join(
        f"{role}: {content}" for role, content in (chat_history or [])[-6:]
    ) or "(No prior conversation.)"
    response = model.invoke(
        _PROMPT.format_messages(
            chat_history=history_text,
            context=_format_context(documents),
            question=question,
        )
    )
    answer = str(response.content).strip()
    if answer == FALLBACK_ANSWER:
        return FALLBACK_ANSWER
    if not _citations_are_valid(answer, documents):
        logger.warning("Generated answer lacked valid citations; returning the evidence fallback.")
        return FALLBACK_ANSWER
    return answer
