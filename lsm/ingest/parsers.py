from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import List

from lsm.ingest.utils import normalize_whitespace

import fitz  # PyMuPDF
from docx import Document
from bs4 import BeautifulSoup

# -----------------------------
# Parse (by extension)
# -----------------------------
def parse_txt(path: Path) -> str:
    # Try utf-8, then fallback
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")

def parse_pdf(path: Path) -> str:
    parts: List[str] = []
    try:
        with fitz.open(str(path)) as doc:
            for page in doc:
                blocks = page.get_text("blocks") or []
                for b in blocks:
                    txt = b[4]
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt)
    except Exception:
        return ""
    return "\n\n".join(parts)

def parse_docx(path: Path) -> str:
    doc = Document(str(path))
    parts: List[str] = []
    for para in doc.paragraphs:
        if para.text:
            parts.append(para.text)
    return "\n".join(parts)

def parse_html(path: Path) -> str:
    raw = parse_txt(path)
    soup = BeautifulSoup(raw, "lxml")
    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    return text

def parse_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md", ".rst"}:
        return parse_txt(path)
    if ext == ".pdf":
        return parse_pdf(path)
    if ext == ".docx":
        return parse_docx(path)
    if ext in {".html", ".htm"}:
        return parse_html(path)
    # Fallback best-effort
    return parse_txt(path)