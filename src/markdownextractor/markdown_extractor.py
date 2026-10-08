"""
src/markdownextractor/markdown_extractor.py
-------------------------------------
In-memory Markdown extraction from a PDF BLOB using pymupdf4llm.

Public API
    extract_markdown_from_blob(blob_data)  -> MarkdownResult

MarkdownResult fields:
    markdown_text : str   — combined markdown text across all pages
    page_texts    : list  — per-page text strings
    is_scanned    : bool  — True when text is too short (scanned PDF)
    error         : str   — non-empty on exception during extraction
"""

from __future__ import annotations

import logging
import os
import re
import traceback
from dataclasses import dataclass, field
from typing import List

from src.markdownextractor.markdown_extractor_utils.cleaning import clean_markdown

log = logging.getLogger(__name__)

# Minimum character count to consider a PDF as text-based.
# Configurable via SCANNED_TEXT_THRESHOLD env var (default 50).
_DEFAULT_THRESHOLD = 250

# Fraction of U+FFFD replacement characters above which extraction is
# considered garbled (non-standard / unembedded font encoding in the PDF).
GARBLED_CHAR_THRESHOLD = 0.05   # >5% → bad font encoding


@dataclass
class MarkdownResult:
    """Holds the markdown extraction output for a single PDF blob."""

    markdown_text: str = ""
    page_texts: List[str] = field(default_factory=list)
    raw_text: str = ""           # plain text via fitz get_text — no Markdown formatting
    is_scanned: bool = False
    error: str = ""
    extraction_failure_reason: str = ""  # non-empty when extraction quality check fails


def garbled_ratio(pages: list[str]) -> float:
    """
    Returns the fraction of characters that are the Unicode replacement character
    U+FFFD (\ufffd), which pymupdf emits when it cannot map a glyph to Unicode.
    A ratio > GARBLED_CHAR_THRESHOLD indicates a font-encoding problem in the PDF.
    """
    combined = ''.join(pages)
    total = len(combined)
    if total == 0:
        return 0.0
    return combined.count('\ufffd') / total


def extract_markdown_from_blob(blob_data: bytes) -> MarkdownResult:
    """
    Extract LLM-friendly Markdown from a PDF blob entirely in memory.

    Uses ``fitz.Document(stream=blob_data)`` so no temporary files are
    ever written to disk.  pymupdf4llm's ``to_markdown`` is then called
    on the in-memory fitz document.

    Args:
        blob_data: Raw PDF bytes from the database.

    Returns:
        MarkdownResult with the combined markdown text, per-page texts,
        and an ``is_scanned`` flag.
    """
    result = MarkdownResult()
    threshold = int(os.environ.get("SCANNED_TEXT_THRESHOLD", _DEFAULT_THRESHOLD))

    try:
        import fitz  # PyMuPDF
        import pymupdf4llm

        # Open in-memory — no temp file needed
        doc = fitz.Document(stream=blob_data, filetype="pdf")

        # Extract plain text while doc is already open (no second fitz open needed).
        # get_text("text") returns unformatted text: no table pipes, no ** bold markers,
        # no Markdown syntax — ideal for regex-based header extraction.
        result.raw_text = "\n".join(
            doc[i].get_text("text") for i in range(len(doc))
        )

        # to_markdown returns a list of page-chunk dicts when page_chunks=True
        md_pages = pymupdf4llm.to_markdown(doc, page_chunks=True, margins=0, show_progress=False) or []

        # Collect per-page text strings.
        # pymupdf4llm page_chunks=True returns dicts where "text" holds the
        # full rendered Markdown (tables, headings, footers, etc.).
        result.page_texts = [page.get("text") for page in md_pages]

        # ── Extraction quality checks ──────────────────────────────────────────
        ratio       = garbled_ratio(result.page_texts)
        total_words = sum(len(re.findall(r'\b\w+\b', p)) for p in result.page_texts)
        if ratio > GARBLED_CHAR_THRESHOLD:
            result.extraction_failure_reason = (
                f'Garbled ratio {ratio:.2%} exceeds {GARBLED_CHAR_THRESHOLD:.0%} '
                f'— non-standard font encoding ({total_words} words extracted)'
            )
            result.is_scanned = True   # fall back to blob / image path
            log.warning(
                "Markdown extraction: GARBLED — %s. Flagging as scanned so caller "
                "falls back to PDF blob.",
                result.extraction_failure_reason,
            )
            return result
        # ── End extraction quality checks ──────────────────────────────────────

        # Deduplicate copy labels (Original / Duplicate / Triplicate …)
        # before the text is sent to Gemini.
        result.markdown_text = clean_markdown(result.page_texts)
        # Decide if the document is scanned (no/too little selectable text)
        result.is_scanned = not (result.markdown_text and len(result.markdown_text.strip()) > threshold)

        if result.is_scanned:
            log.info(
                "Markdown extraction: document flagged as SCANNED "
                "(text length %d ≤ threshold %d). Will fall back to PDF blob.",
                len(result.markdown_text.strip()),
                threshold,
            )
        else:
            log.info(
                "Markdown extraction: TEXT_MODE — %d chars across %d page(s).",
                len(result.markdown_text),
                len(md_pages),
            )

    except Exception as exc:
        result.error = str(exc)
        result.is_scanned = True  # treat as scanned so the caller falls back
        log.warning("Markdown extraction failed: %s\n%s", exc, traceback.format_exc())

    return result
