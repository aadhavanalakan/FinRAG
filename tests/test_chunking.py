"""Chunker invariants: fixed shreds tables, semantic keeps them whole (no API for fixed)."""

from pathlib import Path

import pytest

from finrag.config import load_config
from finrag.chunking import chunk_fixed
from finrag.parsing import parse_document

DATA = Path("data/verizon_10k.html")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="corpus not present")


def test_fixed_produces_chunks_within_size():
    cfg = load_config()
    chunks = chunk_fixed(parse_document(str(DATA)), cfg.chunking.fixed, cfg.chunking.shared)
    assert len(chunks) > 500
    # respects the size cap (+ a little slack for the recursive splitter)
    assert max(c.char_count for c in chunks) <= cfg.chunking.fixed.chunk_size + 50


def test_fixed_shreds_tables():
    """The whole point of the baseline: fixed produces NO atomic table chunks."""
    cfg = load_config()
    chunks = chunk_fixed(parse_document(str(DATA)), cfg.chunking.fixed, cfg.chunking.shared)
    assert all(c.is_table is False for c in chunks)
    # and no protective markers should survive into chunk text
    assert all("<!--TABLE-->" not in c.text for c in chunks)


def test_fixed_passes_junk_filter():
    cfg = load_config()
    chunks = chunk_fixed(parse_document(str(DATA)), cfg.chunking.fixed, cfg.chunking.shared)
    assert all(c.char_count >= cfg.chunking.shared.min_chunk_chars for c in chunks)


def test_chunk_ids_unique_and_prefixed():
    cfg = load_config()
    chunks = chunk_fixed(parse_document(str(DATA)), cfg.chunking.fixed, cfg.chunking.shared)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.id.startswith("verizon-") for c in chunks)
