"""
pipeline_text.py  —  ONE place that builds the text regexes run against.

The PRIMARY invoice and its SIBLINGS must be validated on the exact same text
representation the production pipeline (header_extraction_v2.py) feeds its
regexes. Previously the primary got per-page text with '# Page N' headers while
siblings got a plain join (and scanned PDFs took yet another path), so a regex
could pass on the primary and "fail to generalize" on a sibling purely because the
two texts were built differently. Both now come from here.

SPEED: the markdown/OCR converter (pymupdf4llm, may invoke Tesseract) is slow and
is only needed for SCANNED or GARBLED PDFs. For a normal text PDF the plain PyMuPDF
text decides that on its own, so the converter is skipped entirely.
"""
import os


def pipeline_text(blob: bytes):
    """Return (raw_text, is_scanned, markdown_result_or_None) for a PDF blob.

    Non-scanned: per-page PyMuPDF get_text('text') joined with '\\n\\n# Page N\\n\\n'
    headers (matches header_extraction_v2.py). Scanned/garbled: the markdown/OCR text."""
    import pymupdf
    from src.markdownextractor.markdown_extractor import (
        extract_markdown_from_blob, garbled_ratio, GARBLED_CHAR_THRESHOLD, _DEFAULT_THRESHOLD)

    pages = []
    try:
        doc = pymupdf.open(stream=blob, filetype="pdf")
        try:
            pages = [doc[i].get_text("text") for i in range(doc.page_count)]
        finally:
            doc.close()
    except Exception:
        pages = []

    threshold = int(os.environ.get("SCANNED_TEXT_THRESHOLD", _DEFAULT_THRESHOLD))
    plain_len = len("".join(pages).strip())
    try:
        garbled = garbled_ratio(pages) > GARBLED_CHAR_THRESHOLD if pages else True
    except Exception:
        garbled = False

    # FAST PATH: enough clean selectable text -> no markdown/OCR conversion needed
    if pages and plain_len > threshold and not garbled:
        raw = "".join(f"\n\n# Page {i+1}\n\n{p}" for i, p in enumerate(pages))
        return raw, False, None

    # SLOW PATH (scanned / garbled / unreadable): the full extractor decides
    md = extract_markdown_from_blob(blob)
    if md.is_scanned:
        return (md.markdown_text or md.raw_text or ""), True, md
    raw = "".join(f"\n\n# Page {i+1}\n\n{p}" for i, p in enumerate(pages))
    return raw, False, md
