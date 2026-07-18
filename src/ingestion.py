"""
Ingestion & Chunking
--------------------
Naive RAG chunks documents by fixed token/char windows, which cuts sentences
and sections in half and hurts retrieval precision. This module chunks by
document structure instead: it detects numbered section headers (e.g. "4.3
Rollback Procedure") and keeps each section as a coherent chunk, only
falling back to a sliding window for sections that are too long.

Each chunk carries metadata (source file, section id, section title) so
that generated answers can cite exactly where a claim came from.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List


SECTION_HEADER_RE = re.compile(r"^\s*(\d+\.\d+)\s+(.+)$")
MAX_CHUNK_CHARS = 900          # soft cap before we split a long section further
CHUNK_OVERLAP_CHARS = 120       # overlap for the fallback sliding window


@dataclass
class Chunk:
    chunk_id: str
    source: str
    section_id: str
    section_title: str
    text: str

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "text": self.text,
        }


def _split_into_sections(raw_text: str):
    """Split a document's body into (section_id, section_title, body_text) tuples
    using numbered headers like '4.3 Rollback Procedure'. Lines before the first
    header (e.g. a document title line) are attached to a synthetic '0.0 Header' section.
    """
    lines = raw_text.splitlines()
    sections = []
    current_id, current_title, current_body = "0.0", "Document Header", []

    for line in lines:
        m = SECTION_HEADER_RE.match(line)
        if m:
            if current_body:
                sections.append((current_id, current_title, "\n".join(current_body).strip()))
            current_id, current_title = m.group(1), m.group(2).strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_id, current_title, "\n".join(current_body).strip()))

    # Drop empty header-only sections (e.g. a lone title line at the very top)
    return [s for s in sections if s[2]]


def _sliding_window(text: str, max_chars=MAX_CHUNK_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    if len(text) <= max_chars:
        return [text]
    windows = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        windows.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return windows


def load_corpus(corpus_dir: str) -> List[Chunk]:
    """Read every .txt file in corpus_dir and return a flat list of Chunks."""
    chunks: List[Chunk] = []
    filenames = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".txt"))

    for fname in filenames:
        path = os.path.join(corpus_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        sections = _split_into_sections(raw)
        for sec_id, sec_title, body in sections:
            windows = _sliding_window(body)
            for i, w in enumerate(windows):
                suffix = f"-{i}" if len(windows) > 1 else ""
                chunk_id = f"{fname}::{sec_id}{suffix}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    source=fname,
                    section_id=sec_id,
                    section_title=sec_title,
                    text=w.strip(),
                ))
    return chunks


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    corpus_dir = os.path.join(here, "..", "data", "corpus")
    result = load_corpus(corpus_dir)
    print(f"Loaded {len(result)} chunks from {corpus_dir}")
    for c in result[:5]:
        print(f"  [{c.chunk_id}] {c.section_title}: {c.text[:80]}...")
