import unittest

from langchain_core.documents import Document

from src.generator import SYSTEM_PROMPT, _citations_are_valid


class CitationValidationTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(
                page_content="A policy excerpt.",
                metadata={"source": "policies/security.pdf", "page": 4},
            )
        ]

    def test_accepts_citation_to_retrieved_source_and_page(self):
        self.assertTrue(
            _citations_are_valid(
                "The policy requires review [security.pdf, p. 4].",
                self.documents,
            )
        )

    def test_rejects_missing_or_unretrieved_citation(self):
        self.assertFalse(_citations_are_valid("No citation here.", self.documents))
        self.assertFalse(
            _citations_are_valid("The policy says so [security.pdf, p. 5].", self.documents)
        )

    def test_health_answers_must_not_diagnose_or_replace_a_doctor(self):
        self.assertIn("do not diagnose", SYSTEM_PROMPT)
        self.assertIn("their doctor", SYSTEM_PROMPT)
        self.assertIn("emergency services", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
