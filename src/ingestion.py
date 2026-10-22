"""PDF ingestion, chunking, embedding, and persistent Chroma indexing."""

from __future__ import annotations

import logging
import os
import shutil
from io import BytesIO
from pathlib import Path
from typing import Iterable, Mapping

import fitz
import pytesseract
from PIL import Image
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "shryai_documents"
OCR_RENDER_DPI = 200
TESSERACT_EXECUTABLE = "tesseract.exe"


def create_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Create a normalized, local Sentence-Transformers embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _configure_tesseract() -> None:
    configured_path = os.getenv("SHRYAI_TESSERACT_CMD", "").strip().strip('"')
    if configured_path:
        executable = Path(os.path.expandvars(configured_path)).expanduser()
        if not executable.is_file():
            raise RuntimeError(
                f"SHRYAI_TESSERACT_CMD does not point to a file: {executable}"
            )
    else:
        detected_path = shutil.which("tesseract")
        executable = Path(detected_path) if detected_path else None
        if executable is None:
            install_directories = (
                os.getenv("ProgramFiles", r"C:\Program Files"),
                os.getenv("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                str(Path(os.getenv("LOCALAPPDATA", "")) / "Programs"),
            )
            executable = next(
                (
                    Path(directory) / "Tesseract-OCR" / TESSERACT_EXECUTABLE
                    for directory in install_directories
                    if (Path(directory) / "Tesseract-OCR" / TESSERACT_EXECUTABLE).is_file()
                ),
                None,
            )
            if executable is None:
                raise RuntimeError(
                    "Tesseract OCR was not found on PATH or in its standard Windows "
                    "install location. Restart the terminal after installing it, or "
                    "set SHRYAI_TESSERACT_CMD to the full path of tesseract.exe."
                )

    pytesseract.pytesseract.tesseract_cmd = str(executable)


def _ocr_image(image: Image.Image) -> str:
    _configure_tesseract()
    try:
        return pytesseract.image_to_string(image).strip()
    except pytesseract.pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR could not be launched. Set SHRYAI_TESSERACT_CMD to the "
            "full path of tesseract.exe, or add its installation directory to PATH."
        ) from exc


def _extract_image_text(pdf: fitz.Document, page: fitz.Page) -> list[str]:
    image_text: list[str] = []
    seen_xrefs: set[int] = set()
    for image_info in page.get_images(full=True):
        xref = image_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        image_bytes = pdf.extract_image(xref)["image"]
        with Image.open(BytesIO(image_bytes)) as image:
            with image.convert("RGB") as rgb_image:
                text = _ocr_image(rgb_image)
        if text:
            image_text.append(text)
    return image_text


def _extract_pdf_pages(path: Path) -> list[Document]:
    pages = PyPDFLoader(str(path)).load()
    with fitz.open(path) as pdf:
        for page_number, pdf_page in enumerate(pdf):
            page_text = pages[page_number].page_content or ""
            if page_text.strip():
                image_text = _extract_image_text(pdf, pdf_page)
                if image_text:
                    pages[page_number].page_content = (
                        f"{page_text.rstrip()}\n\n"
                        "[Text recognized in images]\n"
                        + "\n\n".join(image_text)
                    )
                continue

            pixmap = pdf_page.get_pixmap(
                dpi=OCR_RENDER_DPI,
                colorspace=fitz.csRGB,
                alpha=False,
            )
            with Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            ) as page_image:
                pages[page_number].page_content = _ocr_image(page_image)
    return pages


def load_and_split_pdfs(
    pdf_paths: Iterable[str | Path],
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source_names: Mapping[str, str] | None = None,
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
            pages = _extract_pdf_pages(path)
        except Exception:
            logger.exception("Failed to extract text from an uploaded PDF")
            raise

        pages_with_text = [
            page for page in pages if page.page_content and page.page_content.strip()
        ]
        if not pages_with_text:
            raise ValueError(
                f"{path.name} has no readable text. OCR could not detect text in its "
                "pages or images. Check that the PDF contains legible text and that "
                "the required Tesseract language data is installed."
            )
        empty_page_numbers = [
            int(page.metadata.get("page", 0)) + 1
            for page in pages
            if not page.page_content or not page.page_content.strip()
        ]
        if empty_page_numbers:
            logger.warning(
                "A PDF has no extractable text on page(s) %s",
                ", ".join(map(str, empty_page_numbers)),
            )
        for page in pages:
            page.metadata["source"] = (source_names or {}).get(str(path), path.name)
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
    source_names: Mapping[str, str] | None = None,
) -> tuple[Chroma, int]:
    """Replace the collection contents with chunks from the selected PDFs."""
    documents = load_and_split_pdfs(pdf_paths, source_names=source_names)
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
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> tuple[int, int]:
    """Return the indexed chunk count and distinct source-file count."""
    store = Chroma(
        collection_name=collection_name,
        embedding_function=None,
        persist_directory=str(Path(persist_directory).expanduser().resolve()),
    )
    records = store.get(include=["metadatas"])
    metadatas = records.get("metadatas") or []
    sources = {metadata.get("source") for metadata in metadatas if metadata.get("source")}
    return len(records.get("ids") or []), len(sources)
