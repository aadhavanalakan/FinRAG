"""Unit tests for the corpus registry + company-detection hints (no network).

Guards the single-letter-ticker bug: AT&T's ticker "T" must NOT become a hint, or it
matches the substring "t" in every question and misroutes retrieval.
"""

from finrag.corpus import company_hints, load_registry


def test_registry_has_seed_telecoms():
    reg = load_registry()
    for sym in ("verizon", "att", "tmobile"):
        assert sym in reg
        assert reg[sym].get("eval") is True       # protected from removal


def test_single_letter_ticker_excluded():
    hints = company_hints()
    # AT&T's ticker is "T" — must be excluded (len < 2), or it poisons detection.
    assert "t" not in hints
    assert "vz" in hints and hints["vz"] == "verizon"      # 2-char ticker kept
    assert "tmus" in hints and hints["tmus"] == "tmobile"


def test_symbol_and_name_words_are_hints():
    # Registry-agnostic: the seed telecoms' symbols + salient name words must always
    # be hints, regardless of any companies added at runtime.
    hints = company_hints()
    assert hints.get("verizon") == "verizon"
    assert hints.get("att") == "att"            # symbol
    assert hints.get("at&t") == "att"           # name word from "AT&T Inc."
    assert hints.get("vz") == "verizon"         # ticker


def test_stopwords_not_hints():
    hints = company_hints()
    for stop in ("inc", "communications", "us", "the", "company"):
        assert stop not in hints
