"""
src/markdownextractor/markdown_extractor_utils/cleaning.py
----------------------------------------------------
Deduplicates per-page markdown extracted from invoice PDFs.

Invoice PDFs often contain the same invoice content printed multiple times
(Original / Duplicate / Triplicate copies).  This module detects copy labels,
groups consecutive same-label pages, and removes near-duplicate copies using
cosine similarity so that only unique content is sent to Gemini.

Public API
    clean_markdown(pages: List[str]) -> str
        Accepts one raw markdown string per PDF page.
        Returns a single deduplicated markdown string.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

# ── Config ────────────────────────────────────────────────────────────────────
LABEL_PATTERNS: List[Tuple[str, str]] = [
    # NOTE: More specific patterns MUST come before generic ones.
    # 'Customer Extra Copy' must be before 'Copy' — otherwise \bcopy\b
    # matches first and 'Customer Extra Copy' is never reached.
    # 'orginal' handles a real-world typo missing the first 'i'.
    ("Original", r"\bori?ginal\b"),
    ("Duplicate", r"\bdu[pb]licate\b"),
    ("Triplicate", r"\btriplicate\b"),
    ("Extra Copy", r"\bextra\s+copy\b"),
    ("Customer Extra Copy", r"customer\s+extra\s+copy"),
    ("Quadruplicate", r"\bquadruplicate\b|\bqudruplate\b"),
    ("Extra Copy 1", r"\bextra\s+copy\s*[-]?\s*1\b"),
    ("Extra Copy 2", r"\bexatra\s+copy\s*[-]?\s*2\b|\bextra\s+copy\s*[-]?\s*2\b"),
    ("Extra Copy 3", r"\bextra\s+copy\s*[-]?\s*3\b"),
    ("Extra Copy 4", r"\bextra\s+copy\s*[-]?\s*4\b"),
    ("Despatch slip / Intra GATE", r"despatch\s*/\s*slip|intra\s+gate"),
    ("Security Gate Pass", r"security\s+gate\s+pass"),
    # 'Copy' is intentionally last and restricted to standalone-line occurrences.
    # \bcopy\b alone is too broad — it matches body text like
    # "computer generated copy" or "retain a copy for your records".
    ("Copy", r"(?mi)^\s*(?:\w+\s+)?copy\s*$"),
]

ORIGINAL_LABELS: set = {"Original"}
THRESHOLD: float = 0.80


# ── Internal helpers ──────────────────────────────────────────────────────────


def _detect_label(block_text: str) -> Optional[str]:
    """Return the first matching copy label for *block_text*, or None."""
    for name, pat in LABEL_PATTERNS:
        if re.search(pat, block_text, re.IGNORECASE):
            return name
    return None


def _group_blocks_by_label(blocks: List[str]) -> List[Dict]:
    """
    Group consecutive pages that belong to the same copy label.

    Label carry-forward rule:
      - Explicit label found  → switch to that label.
      - No label found        → inherit previous label (continuation page).
    """
    groups: List[Dict] = []
    current_label: Optional[str] = None
    current_pages: List[str] = []

    for block in blocks:
        detected = _detect_label(block)
        label = detected if detected is not None else current_label

        if label != current_label:
            if current_pages:
                groups.append(
                    {
                        "label": current_label,
                        "text": "\n\n".join(current_pages),
                        "page_count": len(current_pages),
                    }
                )
            current_label, current_pages = label, [block]
        else:
            current_pages.append(block)

    if current_pages:
        groups.append(
            {
                "label": current_label,
                "text": "\n\n".join(current_pages),
                "page_count": len(current_pages),
            }
        )

    return groups


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _cosine_similarity(a: str, b: str) -> float:
    fa, fb = Counter(_tokenize(a)), Counter(_tokenize(b))
    if not fa or not fb:
        return 0.0
    all_words = set(fa) | set(fb)
    dot = sum(fa[w] * fb[w] for w in all_words)
    ma = math.sqrt(sum(v * v for v in fa.values()))
    mb = math.sqrt(sum(v * v for v in fb.values()))
    return dot / (ma * mb) if (ma and mb) else 0.0


# ── Public API ────────────────────────────────────────────────────────────────


def clean_markdown(pages: List[str], threshold: float = THRESHOLD) -> str:
    """
    Deduplicate per-page markdown and return a single cleaned string.

    Steps:
      1. Group pages by copy label (Original / Duplicate / Triplicate …).
      2. Compare each non-Original group against the Original via cosine
         similarity; trim groups that exceed *threshold*.
      3. Join and return the kept groups.

    If no copy labels are found, all pages are joined and returned as-is.

    Args:
        pages:     List of raw markdown strings, one per PDF page.
        threshold: Cosine-similarity cutoff above which a copy is trimmed.

    Returns:
        Deduplicated markdown string.
    """
    raw_len = sum(len(p) for p in pages)

    groups = _group_blocks_by_label(pages)
    labeled = [g for g in groups if g["label"] is not None]

    if not labeled:
        # No copy labels detected — return everything unchanged.
        cleaned = "\n\n".join(pages)
        print(
            f"   [CLEAN] Raw markdown: {raw_len} chars | Cleaned markdown: {len(cleaned)} chars (no labels)"
        )
        return cleaned

    original_group = next((g for g in groups if g["label"] in ORIGINAL_LABELS), None)
    original_text = original_group["text"] if original_group else None

    kept_parts: List[str] = []
    for g in groups:
        if g["label"] in ORIGINAL_LABELS or original_text is None:
            kept_parts.append(g["text"])
        else:
            sim = _cosine_similarity(original_text, g["text"])
            if sim <= threshold:
                kept_parts.append(g["text"])

    cleaned = "\n\n".join(kept_parts)
    print(f"   [CLEAN] Raw markdown: {raw_len} chars | Cleaned markdown: {len(cleaned)} chars")
    return cleaned
