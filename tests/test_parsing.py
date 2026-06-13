"""Parser invariants — the table-preservation guarantee (no API needed)."""

from pathlib import Path

import pytest

from finrag.parsing import parse_document

DATA = Path("data/verizon_10k.html")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="corpus not present")


def test_parses_into_blocks():
    doc = parse_document(str(DATA))
    assert doc.company == "verizon"
    assert doc.fiscal_year == 2025
    assert len(doc.blocks) > 100
    assert len(doc.tables()) > 50          # real filings have many data tables


def test_tables_are_marked_and_whole():
    doc = parse_document(str(DATA))
    for t in doc.tables():
        assert "<!--TABLE-->" in t.content and "<!--/TABLE-->" in t.content
    # markers must never leak into prose blocks
    for p in doc.prose():
        assert "<!--TABLE-->" not in p.content


def test_known_financial_figure_preserved_in_a_table():
    doc = parse_document(str(DATA))
    # the operating-revenues figure must survive inside a single table block
    assert any("138,191" in t.content for t in doc.tables())


def test_sections_detected():
    doc = parse_document(str(DATA))
    sections = {b.section for b in doc.blocks if b.section}
    assert {"Item 1", "Item 7", "Item 8"} <= sections
