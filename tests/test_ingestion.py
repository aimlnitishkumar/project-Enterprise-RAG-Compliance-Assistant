import unittest
from unittest.mock import patch

from src.ingestion import get_index_stats


class IndexStatsTests(unittest.TestCase):
    @patch("src.ingestion.Chroma")
    def test_reads_counts_without_initializing_embeddings(self, chroma_factory):
        chroma_factory.return_value.get.return_value = {
            "ids": ["chunk-1", "chunk-2", "chunk-3"],
            "metadatas": [
                {"source": "policy.pdf"},
                {"source": "policy.pdf"},
                {"source": "guide.pdf"},
            ],
        }

        self.assertEqual(get_index_stats("vector_store"), (3, 2))
        self.assertIsNone(chroma_factory.call_args.kwargs["embedding_function"])


if __name__ == "__main__":
    unittest.main()
