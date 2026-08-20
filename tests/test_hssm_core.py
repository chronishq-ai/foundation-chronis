"""
Checklist items 4-8: K selection by BIC, random-init behavior (failed init
ignored), reproducibility, label-switching canonicalization, and recovery
of an obvious synthetic 3-regime structure.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest

from backbone.hssm.model import GaussianHSMM
from backbone.hssm.fitting import fit_with_random_restarts, select_k_by_bic
from backbone.hssm.label_switching import canonicalize_labels, activity_levels


def _obvious_three_regime_data(seed=42):
    rng = np.random.default_rng(seed)
    state0 = rng.normal(0, 0.2, (15, 3))
    state1 = rng.normal(3, 0.2, (15, 3))
    state2 = rng.normal(6, 0.2, (15, 3))
    return np.vstack([state0, state1, state2])


# ---------- #4: K selection by BIC ----------

def test_bic_selects_true_k_on_obvious_structure():
    X = _obvious_three_regime_data()
    model, report = select_k_by_bic(X, n_features=3, k_candidates=(2, 3, 4), n_init=10, base_seed=0)
    assert report["selected_k"] == 3, f"BIC picked K={report['selected_k']}, expected 3. bic_by_k={report['bic_by_k']}"
    assert report["bic_by_k"][3] == min(report["bic_by_k"].values())


def test_k_selection_never_hardcodes_a_favorite():
    # Selection rule must literally be argmin(BIC), regardless of which K
    # "looks" most structured -- verified by re-deriving selected_k from the
    # reported bic_by_k dict independently of the function's internal choice.
    X = _obvious_three_regime_data()
    model, report = select_k_by_bic(X, n_features=3, k_candidates=(2, 3, 4), n_init=10, base_seed=0)
    recomputed_best_k = min(report["bic_by_k"], key=report["bic_by_k"].get)
    assert report["selected_k"] == recomputed_best_k


# ---------- #5: random initialization ----------

def test_hard_minimum_10_inits_enforced():
    X = _obvious_three_regime_data()
    with pytest.raises(AssertionError):
        fit_with_random_restarts(X, n_regimes=2, n_features=3, n_init=5, base_seed=0)


def test_selected_model_is_highest_ll_among_converged():
    X = _obvious_three_regime_data()
    model, run_log = fit_with_random_restarts(X, n_regimes=3, n_features=3, n_init=10, base_seed=7)
    converged_lls = [r["log_likelihood"] for r in run_log if r["converged"]]
    assert len(converged_lls) > 0
    assert np.isclose(model.log_likelihood_, max(converged_lls), atol=1e-6), (
        "selected model's LL must equal the max LL among converged runs, not just "
        "any converged run"
    )


def test_run_log_records_every_attempt():
    X = _obvious_three_regime_data()
    _, run_log = fit_with_random_restarts(X, n_regimes=2, n_features=3, n_init=10, base_seed=1)
    assert len(run_log) == 10
    for r in run_log:
        assert set(r.keys()) >= {"init", "seed", "converged", "n_iter", "log_likelihood", "monotonic_ll"}


def test_non_converged_run_never_selected_even_if_ll_looks_high():
    # Build a fake run_log-style scenario using the actual selection logic by
    # monkeypatching GaussianHSMM.fit to force one run to "fail" with an
    # artificially high (but non-converged) log-likelihood, and confirm it's
    # not chosen.
    X = _obvious_three_regime_data()
    real_fit = GaussianHSMM.fit
    call_count = {"n": 0}

    def rigged_fit(self, X, n_iter=100, tol=1e-4, verbose=False):
        call_count["n"] += 1
        result = real_fit(self, X, n_iter=n_iter, tol=tol, verbose=verbose)
        if call_count["n"] == 3:
            # force this run to look like it has a great LL but never converged
            self.converged_ = False
            self.log_likelihood_ = 1e9
        return result

    GaussianHSMM.fit = rigged_fit
    try:
        model, run_log = fit_with_random_restarts(X, n_regimes=2, n_features=3, n_init=10, base_seed=3)
    finally:
        GaussianHSMM.fit = real_fit

    assert model.log_likelihood_ != 1e9, "a non-converged run must never be selected regardless of its reported LL"
    rigged_entry = run_log[2]
    assert rigged_entry["converged"] is False
    assert rigged_entry["log_likelihood"] == 1e9


def test_all_inits_failing_raises_not_silently_returns_none():
    X = _obvious_three_regime_data()
    real_fit = GaussianHSMM.fit

    def always_fail(self, X, n_iter=100, tol=1e-4, verbose=False):
        real_fit(self, X, n_iter=n_iter, tol=tol, verbose=verbose)
        self.converged_ = False
        return self

    GaussianHSMM.fit = always_fail
    try:
        with pytest.raises(RuntimeError):
            fit_with_random_restarts(X, n_regimes=2, n_features=3, n_init=10, base_seed=3)
    finally:
        GaussianHSMM.fit = real_fit


# ---------- #6: reproducibility ----------

def test_same_seed_same_config_is_deterministic():
    X = _obvious_three_regime_data()
    m1, _ = fit_with_random_restarts(X, n_regimes=3, n_features=3, n_init=10, base_seed=5)
    m2, _ = fit_with_random_restarts(X, n_regimes=3, n_features=3, n_init=10, base_seed=5)
    assert np.allclose(m1.mu, m2.mu)
    assert np.allclose(m1.var, m2.var)
    assert np.isclose(m1.log_likelihood_, m2.log_likelihood_)
    assert m1._label_order_applied == m2._label_order_applied


def test_different_seed_can_change_init_but_stays_sane():
    X = _obvious_three_regime_data()
    m1, _ = fit_with_random_restarts(X, n_regimes=3, n_features=3, n_init=10, base_seed=42)
    m2, _ = fit_with_random_restarts(X, n_regimes=3, n_features=3, n_init=10, base_seed=100)
    # different seeds are allowed to differ, but both must still recover
    # three well-separated regimes (canonicalized ascending activity)
    assert activity_levels(m1).tolist() == sorted(activity_levels(m1).tolist())
    assert activity_levels(m2).tolist() == sorted(activity_levels(m2).tolist())
    assert np.allclose(sorted(np.linalg.norm(m1.mu, axis=1)), sorted(np.linalg.norm(m2.mu, axis=1)), atol=1.0)


# ---------- #7: label switching ----------

def test_canonicalization_orders_ascending_activity():
    X = _obvious_three_regime_data()
    model, _ = fit_with_random_restarts(X, n_regimes=3, n_features=3, n_init=10, base_seed=11)
    levels = activity_levels(model)
    assert list(levels) == sorted(levels), f"regimes not canonically ordered ascending: {levels}"


def test_canonicalization_stable_across_runs_with_different_raw_label_order():
    # Fit twice with different seeds (raw EM label assignment order is
    # arbitrary/seed-dependent) and confirm that after canonicalization the
    # *semantic* ordering (low/mid/high activity regime identity) matches.
    X = _obvious_three_regime_data()
    m1, _ = fit_with_random_restarts(X, n_regimes=3, n_features=3, n_init=10, base_seed=11)
    m2, _ = fit_with_random_restarts(X, n_regimes=3, n_features=3, n_init=10, base_seed=999)
    # regime 0 in both should be the near-origin cluster, regime 2 the ~6.0 cluster
    assert np.linalg.norm(m1.mu[0]) < np.linalg.norm(m1.mu[1]) < np.linalg.norm(m1.mu[2])
    assert np.linalg.norm(m2.mu[0]) < np.linalg.norm(m2.mu[1]) < np.linalg.norm(m2.mu[2])
    assert np.allclose(m1.mu[0], m2.mu[0], atol=1.0)
    assert np.allclose(m1.mu[2], m2.mu[2], atol=1.0)


def test_canonicalize_labels_permutation_is_recorded_for_audit():
    model = GaussianHSMM(n_regimes=3, n_features=2, seed=0)
    model.pi = np.array([0.3, 0.3, 0.4])
    model.A = np.eye(3)
    model.mu = np.array([[5.0, 5.0], [0.0, 0.0], [2.0, 2.0]])  # deliberately out of order
    model.var = np.ones((3, 2))
    model.dur_mu = np.zeros(3)
    model.dur_sigma = np.ones(3)

    canonicalize_labels(model)
    assert model._label_order_applied == [1, 2, 0]
    assert np.allclose(model.mu[0], [0.0, 0.0])
    assert np.allclose(model.mu[2], [5.0, 5.0])


# ---------- #8: obvious synthetic pattern recovery ----------

def test_hssm_recovers_approximately_three_regimes_on_obvious_data():
    X = _obvious_three_regime_data()
    model, run_log = fit_with_random_restarts(X, n_regimes=3, n_features=3, n_init=10, base_seed=0)
    assert model.converged_
    # each fitted regime mean should land near one of the three true clusters
    true_means = np.array([[0, 0, 0], [3, 3, 3], [6, 6, 6]])
    for true_mu in true_means:
        dists = np.linalg.norm(model.mu - true_mu, axis=1)
        assert dists.min() < 1.0, f"no fitted regime close to true mean {true_mu}: fitted mu={model.mu}"


def test_hssm_convergence_rate_above_ninety_percent():
    from backbone.shared.synthetic_data import generate_surrogate_user
    from backbone.shared.feature_reduction import per_person_zscore, reduce_dimensionality

    # Fit 3 distinct surrogate users and verify convergence rate is >= 90%
    for seed in [101, 102, 103]:
        user = generate_surrogate_user(n_sessions=120, n_features=15, n_regimes=2, seed=seed)
        Z, _, _ = per_person_zscore(user["X"])
        feat_names = [f"f{i}" for i in range(15)]
        Z_red, _, _ = reduce_dimensionality(Z, feat_names, target_dims=8)
        model, run_log = fit_with_random_restarts(Z_red, n_regimes=2, n_features=8, n_init=10, base_seed=seed)
        assert model.convergence_rate_ >= 0.9, f"Convergence rate {model.convergence_rate_} < 90% for seed {seed}"

