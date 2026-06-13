"""Retrieval-metric unit tests (pure functions, no API, no corpus)."""

from eval.metrics import hit_at_k, mrr, ndcg_at_k, relevant_flags, score_one


def test_relevant_flags():
    texts = ["total revenues were 138,191 million", "unrelated prose", "see note 5"]
    assert relevant_flags(texts, ["138,191"]) == [True, False, False]


def test_hit_at_k():
    flags = [False, False, True, False, False]
    assert hit_at_k(flags, 1) == 0.0
    assert hit_at_k(flags, 3) == 1.0
    assert hit_at_k(flags, 5) == 1.0


def test_mrr():
    assert mrr([False, True, False]) == 0.5      # first relevant at rank 2
    assert mrr([True, False]) == 1.0
    assert mrr([False, False]) == 0.0


def test_ndcg_rewards_higher_rank():
    top = ndcg_at_k([True, False, False, False, False], 5)
    low = ndcg_at_k([False, False, False, False, True], 5)
    assert top == 1.0                            # ideal: single relevant at rank 1
    assert 0.0 < low < top


def test_score_one_keys():
    s = score_one(["revenue 138,191"], ["138,191"], k=5)
    assert s["hit@1"] == 1.0 and s["mrr"] == 1.0 and s["ndcg@5"] == 1.0
