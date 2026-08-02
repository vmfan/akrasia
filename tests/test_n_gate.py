"""Extend-gate tests (LOCKED_DECISIONS § N LOCKED): the stdlib Student-t
quantile and the pre-registered EXTEND/STOP rule."""

from __future__ import annotations

from akrasia.scaffold.n_gate import _t_cdf, _t_ppf, decide_extend, paired_diffs


def _results(base: list, c1c2: list, model: str = "gpt-5.5") -> dict:
    return {
        "models": {
            model: {
                "baseline": {"episode_mean_ev": base},
                "c1c2": {"episode_mean_ev": c1c2},
            }
        }
    }


# --- Student-t quantile ----------------------------------------------------
def test_t_ppf_matches_published_tables():
    # Two-sided 95% critical values.
    assert abs(_t_ppf(0.975, 1) - 12.706) < 1e-2
    assert abs(_t_ppf(0.975, 28) - 2.048) < 1e-3
    assert abs(_t_ppf(0.975, 56) - 2.003) < 1e-3
    assert abs(_t_ppf(0.975, 1000) - 1.962) < 1e-3


def test_t_cdf_ppf_roundtrip():
    for df in (5, 28, 56):
        for p in (0.9, 0.975, 0.99):
            assert abs(_t_cdf(_t_ppf(p, df), df) - p) < 1e-6


# --- paired_diffs ----------------------------------------------------------
def test_paired_diffs_pairs_by_index_and_drops_none():
    diffs = paired_diffs(_results([0.5, None, 0.4], [0.6, 0.7, None]), "gpt-5.5")
    assert diffs == [0.6 - 0.5]  # only episode 0 defined in both arms


# --- the gate rule ---------------------------------------------------------
def test_extend_when_ci_contains_both_zero_and_delta():
    # Small effect, high variance, small n -> wide CI spanning 0..0.05.
    d = decide_extend(_results([0.5, 0.4, 0.6, 0.5, 0.45], [0.55, 0.35, 0.7, 0.4, 0.6]), "gpt-5.5")
    assert d.decision == "EXTEND"
    assert d.contains_zero and d.contains_delta


def test_stop_when_ci_excludes_zero():
    # Large, tight positive effect -> CI well above 0.
    base = [0.30 + 0.001 * i for i in range(20)]
    c1c2 = [0.55 + 0.001 * i for i in range(20)]
    d = decide_extend(_results(base, c1c2), "gpt-5.5")
    assert d.decision == "STOP"
    assert not d.contains_zero
    assert d.ci_low > 0.0


def test_stop_when_ci_excludes_delta():
    # Effect ~0, tight -> CI near 0, excludes Delta=0.05.
    base = [0.5 + 0.001 * i for i in range(20)]
    c1c2 = [0.5 + 0.001 * i for i in range(20)]
    d = decide_extend(_results(base, c1c2), "gpt-5.5")
    assert d.decision == "STOP"
    assert not d.contains_delta


def test_extend_when_too_few_paired_episodes():
    d = decide_extend(_results([0.5, None], [None, 0.6]), "gpt-5.5")
    assert d.decision == "EXTEND"
    assert d.n_paired == 0


def test_contrast_is_c1c2_minus_baseline():
    d = decide_extend(_results([0.4] * 10, [0.5] * 10), "gpt-5.5")
    assert d.contrast == "c1c2 - baseline"
    assert abs(d.effect - 0.1) < 1e-9
