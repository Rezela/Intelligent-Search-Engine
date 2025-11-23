"""Utilities for loading raw text from various document formats."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

try:  # pragma: no cover - optional dependency
    from pypdf import PdfReader  # type: ignore
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore


class UnsupportedDocumentError(RuntimeError):
    pass


def _extract_pdf_text(file_path: Path) -> str:
    if PdfReader is None:
        raise UnsupportedDocumentError(
            "pypdf is required to read PDF files. Please install it via 'pip install pypdf'."
        )

    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:  # pragma: no cover - passthrough errors
        raise UnsupportedDocumentError(f"Unable to open PDF file {file_path}: {exc}") from exc

    texts = []
    for page_index, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover
            raise UnsupportedDocumentError(
                f"Failed to extract text from page {page_index + 1} of {file_path}: {exc}"
            ) from exc
        texts.append(page_text.strip())

    return "\n\n".join(text for text in texts if text)


def load_document_text(path: Union[str, Path]) -> str:
    """Load textual content from supported document formats.

    Currently supports plain text/markdown files and PDF documents. The caller is
    responsible for handling downstream chunking/embedding.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text(file_path)

    if suffix in {".txt", ".md", ".markdown", ".json", ""}:
        return file_path.read_text(encoding="utf-8")

    raise UnsupportedDocumentError(f"Unsupported document format: {file_path.suffix}")


def load_document_from_bytes(filename: str, data: bytes) -> str:
    """Decode document content from uploaded bytes."""

    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        if PdfReader is None:
            raise UnsupportedDocumentError(
                "pypdf is required to read PDF files. Please install it via 'pip install pypdf'."
            )
        try:
            reader = PdfReader(io.BytesIO(data))
        except Exception as exc:  # pragma: no cover
            raise UnsupportedDocumentError(f"Unable to open uploaded PDF {filename}: {exc}") from exc

        texts = []
        for page_index, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:  # pragma: no cover
                raise UnsupportedDocumentError(
                    f"Failed to extract text from uploaded PDF page {page_index + 1}: {exc}"
                ) from exc
            texts.append(page_text.strip())
        content = "\n\n".join(text for text in texts if text)
        if not content:
            raise UnsupportedDocumentError("Uploaded PDF did not contain extractable text.")
        return content

    if suffix in {".txt", ".md", ".markdown", ".json", ""}:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnsupportedDocumentError(f"Unable to decode uploaded text file {filename}: {exc}") from exc

    raise UnsupportedDocumentError(f"Unsupported uploaded document format: {suffix}")
