from domain_emergence.multiple_comparisons import (
    bonferroni_correct, benjamini_hochberg_correct, compare_corrections,
)


def test_bonferroni_matches_manual_calc():
    raw = {"a": 0.01, "b": 0.02, "c": 0.5}
    corrected = bonferroni_correct(raw)
    assert corrected["a"] == 0.03  # 0.01 * 3
    assert corrected["b"] == 0.06  # 0.02 * 3
    assert corrected["c"] == 1.0   # 0.5 * 3 = 1.5, capped at 1.0


def test_bonferroni_empty_input():
    assert bonferroni_correct({}) == {}


def test_bh_matches_textbook_example():
    # classic BH example: 5 p-values, sorted 0.01, 0.02, 0.03, 0.04, 0.5
    # BH adjusted: min_{j>=i}(p_(j) * n / j)
    raw = {"p1": 0.01, "p2": 0.02, "p3": 0.03, "p4": 0.04, "p5": 0.5}
    adjusted = benjamini_hochberg_correct(raw)
    # rank 5 (p5=0.5): 0.5*5/5 = 0.5
    assert abs(adjusted["p5"] - 0.5) < 1e-9
    # rank 4 (p4=0.04): min(0.04*5/4, adjusted[rank5]) = min(0.05, 0.5) = 0.05
    assert abs(adjusted["p4"] - 0.05) < 1e-9
    # rank 1 (p1=0.01): min(0.01*5/1, adjusted[rank2]) chain -> should be <= 0.05
    assert adjusted["p1"] <= adjusted["p2"] <= adjusted["p3"] <= adjusted["p4"] <= adjusted["p5"]


def test_bh_is_monotonic_non_decreasing_by_rank():
    raw = {"a": 0.001, "b": 0.3, "c": 0.02, "d": 0.15, "e": 0.04}
    adjusted = benjamini_hochberg_correct(raw)
    ranked = sorted(raw.items(), key=lambda kv: kv[1])
    adj_in_rank_order = [adjusted[k] for k, _ in ranked]
    for i in range(len(adj_in_rank_order) - 1):
        assert adj_in_rank_order[i] <= adj_in_rank_order[i + 1] + 1e-12


def test_bh_empty_input():
    assert benjamini_hochberg_correct({}) == {}


def test_bh_less_conservative_than_bonferroni():
    """The core Day 18 comparison claim: BH should find >= as many
    significant results as Bonferroni on the same data."""
    raw = {f"pair_{i}": p for i, p in enumerate(
        [0.001, 0.002, 0.01, 0.02, 0.03, 0.04, 0.3, 0.5, 0.7, 0.9]
    )}
    result = compare_corrections(raw, alpha=0.05)
    assert result.bh_significant >= result.bonferroni_significant
    assert result.only_bonferroni_finds == set()


def test_compare_corrections_agree_on_strong_signal():
    raw = {"a": 0.0001, "b": 0.9, "c": 0.8}
    result = compare_corrections(raw, alpha=0.05)
    assert "a" in result.agree or "a" in result.only_bh_finds


def test_compare_corrections_all_nonsignificant():
    raw = {"a": 0.9, "b": 0.8, "c": 0.99}
    result = compare_corrections(raw, alpha=0.05)
    assert result.bonferroni_significant == set()
    assert result.bh_significant == set()
    assert result.only_bh_finds == set()