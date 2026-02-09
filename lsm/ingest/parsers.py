from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import logging

from lsm.ingest.models import PageSegment
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


def _open_pdf_with_repair(path: Path) -> fitz.Document:
    """Open a PDF with progressive repair strategies for common corruption.

    Tries increasingly aggressive repair strategies:

    1. Direct file open (normal path).
    2. In-memory stream with garbage collection (``garbage=4``, ``deflate=True``,
       ``clean=True``) to rebuild the xref table and recompress streams — fixes
       most zlib and xref corruption.
    3. Plain in-memory stream open (minimal intervention fallback).

    Args:
        path: Path to the PDF file.

    Returns:
        Opened ``fitz.Document``.

    Raises:
        Exception: If all repair strategies fail.
    """
    # Strategy 1: direct file open
    try:
        return fitz.open(str(path))
    except Exception as e:
        msg = str(e).lower()
        repairable = (
            "syntax error", "zlib error", "xref", "trailer",
            "startxref", "corrupt", "malformed",
        )
        if not any(marker in msg for marker in repairable):
            raise
        logger.warning(
            "PDF open failed for %s (%s), attempting repair strategies", path, e,
        )

    data = path.read_bytes()

    # Strategy 2: garbage-collection repair (rebuild xref, recompress streams)
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        cleaned = doc.tobytes(garbage=4, deflate=True, clean=True)
        doc.close()
        return fitz.open(stream=cleaned, filetype="pdf")
    except Exception as e2:
        logger.debug("Garbage-collection repair failed for %s: %s", path, e2)

    # Strategy 3: plain stream open (minimal intervention)
    try:
        return fitz.open(stream=data, filetype="pdf")
    except Exception as e3:
        logger.error(
            "All PDF repair strategies failed for %s. Last error: %s", path, e3,
        )
        raise


def parse_pdf(
    path: Path,
    enable_ocr: bool = False,
    skip_errors: bool = True,
) -> Tuple[str, Dict[str, Any], Optional[List[PageSegment]]]:
    """
    Parse PDF file with optional OCR for image-based PDFs.

    Args:
        path: Path to PDF file
        enable_ocr: If True, use OCR for pages with little text
        skip_errors: If True, continue on per-page failures

    Returns:
        Tuple of (text content, metadata dict, page segments).
        Page segments are 1-based PageSegment objects for each page with text.
    """
    page_segments: List[PageSegment] = []
    metadata: Dict[str, Any] = {}

    parse_errors: List[Dict[str, Any]] = []

    try:
        with _open_pdf_with_repair(path) as doc:
            # Extract metadata
            metadata = extract_pdf_metadata(doc)

            # Extract text from each page
            for page_num in range(doc.page_count):
                try:
                    page = doc.load_page(page_num)
                except Exception as e:
                    parse_errors.append({
                        "page": page_num + 1,
                        "stage": "load_page",
                        "error": str(e),
                    })
                    logger.warning(f"Failed to load PDF page {page_num + 1} from {path}: {e}")
                    if not skip_errors:
                        raise
                    continue

                text = ""
                try:
                    text = page.get_text()
                except Exception as e:
                    parse_errors.append({
                        "page": page_num + 1,
                        "stage": "extract_text",
                        "error": str(e),
                    })
                    logger.warning(f"Failed to extract text from PDF page {page_num + 1} in {path}: {e}")
                    if not skip_errors:
                        raise

                needs_ocr = False
                if not text:
                    needs_ocr = True
                else:
                    try:
                        needs_ocr = is_page_image_based(page)
                    except Exception as e:
                        parse_errors.append({
                            "page": page_num + 1,
                            "stage": "ocr_detection",
                            "error": str(e),
                        })
                        logger.warning(f"Failed OCR detection on PDF page {page_num + 1} in {path}: {e}")
                        if not skip_errors:
                            raise
                        needs_ocr = True

                if enable_ocr and needs_ocr:
                    logger.debug(f"Page {page_num + 1} appears image-based or failed text, using OCR")
                    ocr_text = ocr_page(page)
                    if ocr_text:
                        text = ocr_text
                    else:
                        parse_errors.append({
                            "page": page_num + 1,
                            "stage": "ocr",
                            "error": "OCR returned empty text",
                        })

                if text and text.strip():
                    page_segments.append(PageSegment(
                        text=text,
                        page_number=page_num + 1,
                    ))

    except Exception as e:
        logger.error(f"Failed to parse PDF {path}: {e}")
        if not skip_errors:
            raise
        if "error" not in metadata:
            metadata["error"] = str(e)
        if parse_errors:
            metadata["_parse_errors"] = parse_errors
        return "", metadata, None

    if parse_errors:
        metadata["_parse_errors"] = parse_errors

    combined = "\n\n".join(seg.text for seg in page_segments)
    return combined, metadata, page_segments if page_segments else None


def _docx_has_page_break_before(paragraph) -> bool:
    """Check if a DOCX paragraph has a page break before it.

    Detects both explicit page breaks (``<w:br w:type="page"/>``) in runs and
    rendered page breaks (``<w:lastRenderedPageBreak/>``) in the paragraph XML.
    """
    from lxml import etree

    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

    xml = paragraph._element

    # Check for <w:lastRenderedPageBreak/> anywhere in the paragraph
    if xml.findall(f".//{ns}lastRenderedPageBreak"):
        return True

    # Check for explicit <w:br w:type="page"/> in runs
    for br in xml.findall(f".//{ns}br"):
        if br.get(f"{ns}type") == "page" or br.get("type") == "page":
            return True

    return False


def _docx_heading_level(paragraph) -> Optional[int]:
    """Return heading level (1-6) for DOCX heading-styled paragraphs."""
    try:
        style_name = (paragraph.style.name or "").strip().lower()
    except Exception:
        style_name = ""

    if not style_name.startswith("heading"):
        return None

    parts = style_name.split()
    if len(parts) < 2:
        return None

    try:
        level = int(parts[1])
    except ValueError:
        return None

    if 1 <= level <= 6:
        return level
    return None


def parse_docx(
    path: Path,
) -> Tuple[str, Dict[str, Any], Optional[List[PageSegment]]]:
    """
    Parse DOCX file, extract metadata, and track page boundaries.

    Detects page breaks via ``<w:lastRenderedPageBreak/>`` and
    ``<w:br w:type="page"/>`` elements in the document XML.

    Returns:
        Tuple of (text content, metadata dict, page segments).
        Page segments are 1-based PageSegment objects when page breaks are
        detected; ``None`` when no page break information is available.
    """
    try:
        doc = Document(str(path))

        # Extract metadata
        metadata = extract_docx_metadata(doc)

        # Build page segments by detecting page breaks
        current_page = 1
        page_parts: Dict[int, List[str]] = {1: []}

        for para in doc.paragraphs:
            if _docx_has_page_break_before(para):
                current_page += 1
                page_parts[current_page] = []

            if para.text:
                heading_level = _docx_heading_level(para)
                if heading_level is not None:
                    heading_prefix = "#" * heading_level
                    page_parts[current_page].append(f"{heading_prefix} {para.text}")
                else:
                    page_parts[current_page].append(para.text)

        # Build page segments
        page_segments: List[PageSegment] = []
        all_parts: List[str] = []

        for page_num in sorted(page_parts.keys()):
            parts = page_parts[page_num]
            if parts:
                page_text = "\n".join(parts)
                page_segments.append(PageSegment(
                    text=page_text,
                    page_number=page_num,
                ))
                all_parts.extend(parts)

        combined = "\n".join(all_parts)

        # Only return page segments if we actually detected page breaks
        has_page_breaks = current_page > 1
        return combined, metadata, page_segments if has_page_breaks else None

    except Exception as e:
        logger.error(f"Failed to parse DOCX {path}: {e}")
        return "", {}, None


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

    # Preserve heading structure as markdown-style heading lines so
    # structure chunking can detect heading boundaries.
    lines: List[str] = []
    body = soup.body or soup
    for node in body.descendants:
        if not getattr(node, "name", None):
            continue
        name = str(node.name).lower()

        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            heading_text = normalize_whitespace(node.get_text(" ", strip=True))
            if heading_text:
                level = int(name[1])
                lines.append(f"{'#' * level} {heading_text}")
                lines.append("")
            continue

        if name == "p":
            para_text = normalize_whitespace(node.get_text(" ", strip=True))
            if para_text:
                lines.append(para_text)
                lines.append("")
            continue

        if name == "li":
            item_text = normalize_whitespace(node.get_text(" ", strip=True))
            if item_text:
                lines.append(item_text)
                lines.append("")

    if lines:
        text = "\n".join(lines).strip()
    else:
        text = soup.get_text("\n")

    return text, metadata


def parse_file(
    path: Path,
    enable_ocr: bool = False,
    skip_errors: bool = True,
) -> Tuple[str, Dict[str, Any], Optional[List[PageSegment]]]:
    """
    Parse a file based on its extension.

    Args:
        path: Path to file
        enable_ocr: Enable OCR for image-based PDFs
        skip_errors: If True, continue on per-page failures

    Returns:
        Tuple of (text content, metadata dict, page segments).
        Page segments are provided for formats that support page tracking
        (PDF, DOCX); ``None`` for other formats.
    """
    ext = path.suffix.lower()

    if ext in {".txt", ".rst"}:
        text, meta = parse_txt(path)
        return text, meta, None
    if ext == ".md":
        text, meta = parse_md(path)
        return text, meta, None
    if ext == ".pdf":
        return parse_pdf(path, enable_ocr=enable_ocr, skip_errors=skip_errors)
    if ext == ".docx":
        return parse_docx(path)
    if ext in {".html", ".htm"}:
        text, meta = parse_html(path)
        return text, meta, None

    # Fallback best-effort (treat as text)
    text, meta = parse_txt(path)
    return text, meta, None
