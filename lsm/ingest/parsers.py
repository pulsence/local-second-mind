from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import logging

from lsm.ingest.utils import normalize_whitespace

import fitz  # PyMuPDF
from docx import Document
from bs4 import BeautifulSoup

# OCR support (optional)
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)


# -----------------------------
# Metadata Extraction
# -----------------------------
def extract_pdf_metadata(doc: fitz.Document) -> Dict[str, Any]:
    """
    Extract metadata from a PDF document.

    Args:
        doc: Opened PyMuPDF document

    Returns:
        Dictionary containing available metadata fields
    """
    metadata = {}

    try:
        pdf_meta = doc.metadata

        if pdf_meta:
            # Common metadata fields
            if pdf_meta.get("author"):
                metadata["author"] = pdf_meta["author"]
            if pdf_meta.get("title"):
                metadata["title"] = pdf_meta["title"]
            if pdf_meta.get("subject"):
                metadata["subject"] = pdf_meta["subject"]
            if pdf_meta.get("creator"):
                metadata["creator"] = pdf_meta["creator"]
            if pdf_meta.get("producer"):
                metadata["producer"] = pdf_meta["producer"]
            if pdf_meta.get("creationDate"):
                metadata["creation_date"] = pdf_meta["creationDate"]
            if pdf_meta.get("modDate"):
                metadata["mod_date"] = pdf_meta["modDate"]

    except Exception as e:
        logger.debug(f"Failed to extract PDF metadata: {e}")

    return metadata


def extract_docx_metadata(doc: Document) -> Dict[str, Any]:
    """
    Extract metadata from a DOCX document.

    Args:
        doc: Opened python-docx Document

    Returns:
        Dictionary containing available metadata fields
    """
    metadata = {}

    try:
        core_props = doc.core_properties

        if core_props.author:
            metadata["author"] = core_props.author
        if core_props.title:
            metadata["title"] = core_props.title
        if core_props.subject:
            metadata["subject"] = core_props.subject
        if core_props.keywords:
            metadata["keywords"] = core_props.keywords
        if core_props.created:
            metadata["creation_date"] = core_props.created.isoformat() if core_props.created else None
        if core_props.modified:
            metadata["mod_date"] = core_props.modified.isoformat() if core_props.modified else None

    except Exception as e:
        logger.debug(f"Failed to extract DOCX metadata: {e}")

    return metadata


def extract_markdown_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """
    Extract YAML frontmatter from Markdown text.

    Args:
        text: Full markdown text

    Returns:
        Tuple of (metadata dict, text without frontmatter)
    """
    metadata = {}

    # Check for YAML frontmatter (--- at start and end)
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                import yaml
                frontmatter = parts[1].strip()
                metadata = yaml.safe_load(frontmatter) or {}
                text = parts[2].strip()
            except Exception as e:
                logger.debug(f"Failed to parse YAML frontmatter: {e}")

    return metadata, text


# -----------------------------
# OCR Detection & Extraction
# -----------------------------
def is_page_image_based(page: fitz.Page, min_text_threshold: int = 50) -> bool:
    """
    Detect if a PDF page is likely image-based (needs OCR).

    Args:
        page: PyMuPDF page object
        min_text_threshold: Minimum number of characters for text-based page

    Returns:
        True if page appears to be image-based
    """
    text = page.get_text()
    return len(text.strip()) < min_text_threshold


def ocr_page(page: fitz.Page) -> str:
    """
    Perform OCR on a PDF page.

    Args:
        page: PyMuPDF page object

    Returns:
        Extracted text from OCR
    """
    if not OCR_AVAILABLE:
        logger.warning("OCR requested but pytesseract not available. Install with: pip install pytesseract pillow")
        return ""

    try:
        # Render page to image
        pix = page.get_pixmap(dpi=300)  # Higher DPI for better OCR

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Perform OCR
        text = pytesseract.image_to_string(img)

        return text
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""


# -----------------------------
# Parse Functions (by extension)
# -----------------------------
def parse_txt(path: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Parse plain text file.

    Returns:
        Tuple of (text content, metadata dict)
    """
    # Try utf-8, then fallback
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1", errors="ignore")

    metadata = {}
    return text, metadata


def parse_md(path: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Parse Markdown file with optional YAML frontmatter.

    Returns:
        Tuple of (text content, metadata dict)
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1", errors="ignore")

    # Extract frontmatter if present
    metadata, text = extract_markdown_frontmatter(text)

    return text, metadata


def parse_pdf(path: Path, enable_ocr: bool = False) -> Tuple[str, Dict[str, Any]]:
    """
    Parse PDF file with optional OCR for image-based PDFs.

    Args:
        path: Path to PDF file
        enable_ocr: If True, use OCR for pages with little text

    Returns:
        Tuple of (text content, metadata dict)
    """
    parts: List[str] = []
    metadata = {}

    try:
        with fitz.open(str(path)) as doc:
            # Extract metadata
            metadata = extract_pdf_metadata(doc)

            # Extract text from each page
            for page_num, page in enumerate(doc):
                # Try regular text extraction first
                text = page.get_text()

                # If page appears image-based and OCR is enabled, use OCR
                if enable_ocr and is_page_image_based(page):
                    logger.debug(f"Page {page_num + 1} appears image-based, using OCR")
                    text = ocr_page(page)

                if text and text.strip():
                    parts.append(text)

    except Exception as e:
        logger.error(f"Failed to parse PDF {path}: {e}")
        return "", metadata

    return "\n\n".join(parts), metadata


def parse_docx(path: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Parse DOCX file and extract metadata.

    Returns:
        Tuple of (text content, metadata dict)
    """
    try:
        doc = Document(str(path))

        # Extract metadata
        metadata = extract_docx_metadata(doc)

        # Extract text
        parts: List[str] = []
        for para in doc.paragraphs:
            if para.text:
                parts.append(para.text)

        return "\n".join(parts), metadata
    except Exception as e:
        logger.error(f"Failed to parse DOCX {path}: {e}")
        return "", {}


def parse_html(path: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Parse HTML file.

    Returns:
        Tuple of (text content, metadata dict)
    """
    raw, _ = parse_txt(path)
    soup = BeautifulSoup(raw, "lxml")

    # Extract metadata from HTML meta tags
    metadata = {}

    try:
        # Try to extract title
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()

        # Try to extract author from meta tags
        author_tag = soup.find("meta", attrs={"name": "author"})
        if author_tag and author_tag.get("content"):
            metadata["author"] = author_tag["content"]

        # Try to extract description
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag and desc_tag.get("content"):
            metadata["description"] = desc_tag["content"]

    except Exception as e:
        logger.debug(f"Failed to extract HTML metadata: {e}")

    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")

    return text, metadata


def parse_file(path: Path, enable_ocr: bool = False) -> Tuple[str, Dict[str, Any]]:
    """
    Parse a file based on its extension.

    Args:
        path: Path to file
        enable_ocr: Enable OCR for image-based PDFs

    Returns:
        Tuple of (text content, metadata dict)
    """
    ext = path.suffix.lower()

    if ext in {".txt", ".rst"}:
        return parse_txt(path)
    if ext == ".md":
        return parse_md(path)
    if ext == ".pdf":
        return parse_pdf(path, enable_ocr=enable_ocr)
    if ext == ".docx":
        return parse_docx(path)
    if ext in {".html", ".htm"}:
        return parse_html(path)

    # Fallback best-effort (treat as text)
    return parse_txt(path)
