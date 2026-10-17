import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from src.retriever import HybridRerankingRetriever


class _FakeBM25:
    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, _query_tokens):
        return self.scores


class _FakeVectorStore:
    def __init__(self, documents, semantic_results=None):
        self.documents = documents
        self.semantic_results = semantic_results or []

    def get(self, include):
        return {
            "documents": [document.page_content for document in self.documents],
            "metadatas": [document.metadata for document in self.documents],
        }

    def similarity_search(self, _query, k):
        return self.semantic_results[:k]


class _FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        return self.scores[: len(pairs)]


class HybridRetrieverTests(unittest.TestCase):
    def test_bm25_selects_top_candidates_without_full_sort(self):
        documents = [
            Document(page_content="first", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="best", metadata={"source": "b.pdf", "page": 2}),
            Document(page_content="second", metadata={"source": "c.pdf", "page": 3}),
        ]
        store = _FakeVectorStore(documents)

        with patch("src.retriever.BM25Okapi", return_value=_FakeBM25([0.1, 0.9, 0.2])):
            retriever = HybridRerankingRetriever(
                store,
                initial_k=2,
                cross_encoder=_FakeCrossEncoder([]),
            )

        self.assertEqual(
            [document.page_content for document in retriever._hybrid_candidates("query")],
            ["best", "second"],
        )

    def test_cross_encoder_ranks_hybrid_candidates(self):
        semantic_documents = [
            Document(page_content="semantic result", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="lexical result", metadata={"source": "b.pdf", "page": 2}),
        ]
        store = _FakeVectorStore(semantic_documents, semantic_results=semantic_documents)
        cross_encoder = _FakeCrossEncoder([0.1, 0.9])

        with patch("src.retriever.BM25Okapi", return_value=_FakeBM25([0.0, 0.0])):
            retriever = HybridRerankingRetriever(
                store,
                initial_k=2,
                cross_encoder=cross_encoder,
            )

        results = retriever.retrieve("query", top_k=1)
        self.assertEqual([document.page_content for document in results], ["lexical result"])


if __name__ == "__main__":
    unittest.main()
