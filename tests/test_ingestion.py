from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fitz
import pytesseract
from PIL import Image

from src.ingestion import _ocr_image, get_index_stats, load_and_split_pdfs


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


class PdfOcrIngestionTests(unittest.TestCase):
    @staticmethod
    def _image_bytes() -> bytes:
        image = Image.new("RGB", (160, 80), "white")
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    def test_combines_selectable_text_and_text_from_embedded_images(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "mixed.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            page.insert_text((40, 40), "Selectable page text")
            page.insert_image(
                fitz.Rect(40, 60, 200, 140),
                stream=self._image_bytes(),
            )
            pdf.save(pdf_path)
            pdf.close()

            with patch("src.ingestion.pytesseract.image_to_string", return_value="Image label"):
                chunks = load_and_split_pdfs([pdf_path])

        self.assertEqual(len(chunks), 1)
        self.assertIn("Selectable page text", chunks[0].page_content)
        self.assertIn("Image label", chunks[0].page_content)
        self.assertEqual(chunks[0].metadata["source"], "mixed.pdf")
        self.assertEqual(chunks[0].metadata["page"], 1)

    def test_ocr_reads_scanned_page_and_preserves_page_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "scan.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            page.insert_image(
                fitz.Rect(40, 60, 200, 140),
                stream=self._image_bytes(),
            )
            pdf.save(pdf_path)
            pdf.close()

            with patch("src.ingestion.pytesseract.image_to_string", return_value="Scanned page text"):
                chunks = load_and_split_pdfs([pdf_path])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, "Scanned page text")
        self.assertEqual(chunks[0].metadata["source"], "scan.pdf")
        self.assertEqual(chunks[0].metadata["page"], 1)

    def test_missing_tesseract_reports_installation_requirement(self):
        install_environment = {
            "ProgramFiles": r"C:\Missing",
            "ProgramFiles(x86)": r"C:\Missing",
            "LOCALAPPDATA": r"C:\Missing",
        }
        with (
            patch.dict("os.environ", {"SHRYAI_TESSERACT_CMD": ""}),
            patch("src.ingestion.shutil.which", return_value=None),
            patch("src.ingestion.os.getenv", side_effect=install_environment.get),
            patch("src.ingestion.Path.is_file", return_value=False),
            Image.new("RGB", (10, 10), "white") as image,
        ):
            with self.assertRaisesRegex(RuntimeError, "SHRYAI_TESSERACT_CMD"):
                _ocr_image(image)

    def test_finds_tesseract_in_standard_windows_install_directory(self):
        install_environment = {
            "ProgramFiles": r"C:\Program Files",
            "ProgramFiles(x86)": r"C:\Program Files (x86)",
            "LOCALAPPDATA": r"C:\Users\test\AppData\Local",
        }
        with (
            patch.dict("os.environ", {"SHRYAI_TESSERACT_CMD": ""}),
            patch("src.ingestion.shutil.which", return_value=None),
            patch("src.ingestion.os.getenv", side_effect=install_environment.get),
            patch("src.ingestion.Path.is_file", return_value=True),
            patch("src.ingestion.pytesseract.image_to_string", return_value="recognized"),
            Image.new("RGB", (10, 10), "white") as image,
        ):
            self.assertEqual(_ocr_image(image), "recognized")

        self.assertEqual(
            pytesseract.pytesseract.tesseract_cmd,
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        )

    def test_uses_configured_tesseract_executable_path(self):
        configured_path = r"C:\Custom OCR\tesseract.exe"
        with (
            patch.dict("os.environ", {"SHRYAI_TESSERACT_CMD": configured_path}),
            patch("src.ingestion.Path.is_file", return_value=True),
            patch("src.ingestion.shutil.which", return_value=None),
            patch("src.ingestion.pytesseract.image_to_string", return_value="recognized"),
            Image.new("RGB", (10, 10), "white") as image,
        ):
            self.assertEqual(_ocr_image(image), "recognized")

        self.assertEqual(pytesseract.pytesseract.tesseract_cmd, configured_path)

    def test_rejects_invalid_configured_tesseract_path(self):
        with (
            patch.dict(
                "os.environ",
                {"SHRYAI_TESSERACT_CMD": r"C:\missing\tesseract.exe"},
            ),
            patch("src.ingestion.Path.is_file", return_value=False),
            Image.new("RGB", (10, 10), "white") as image,
        ):
            with self.assertRaisesRegex(RuntimeError, "SHRYAI_TESSERACT_CMD"):
                _ocr_image(image)


if __name__ == "__main__":
    unittest.main()
