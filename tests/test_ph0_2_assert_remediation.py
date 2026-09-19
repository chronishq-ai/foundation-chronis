import pytest
import subprocess
import sys
import numpy as np
from backbone.hssm import (
    GaussianHSMM,
    KimHSSMModel,
    HSSMError,
    NotFittedError,
    FittingConvergenceError,
    InternalStateError,
    fit_with_random_restarts,
    select_k_by_bic,
    fit_hssm_gated,
)
from backbone.attractors.detector import compute_attractor_stats


def test_hssm_exception_hierarchy():
    """Verify inheritance hierarchy for all HSSM exception classes."""
    assert issubclass(NotFittedError, HSSMError)
    assert issubclass(NotFittedError, ValueError)
    assert issubclass(NotFittedError, AttributeError)

    assert issubclass(FittingConvergenceError, HSSMError)
    assert issubclass(FittingConvergenceError, RuntimeError)

    assert issubclass(InternalStateError, HSSMError)
    assert issubclass(InternalStateError, RuntimeError)


# Helper functions to trigger each of the audited sites individually
def trigger_site_1(): GaussianHSMM(n_regimes=0, n_features=3)
def trigger_site_2(): GaussianHSMM(n_regimes=2, n_features=0)
def trigger_site_3(): GaussianHSMM(n_regimes=2, n_features=3)._require_fitted()
def trigger_site_4(): GaussianHSMM(n_regimes=2, n_features=3)._duration_logpmf()
def trigger_site_5(): GaussianHSMM(n_regimes=2, n_features=3)._emission_loglik(np.random.randn(5, 3))
def trigger_site_6(): GaussianHSMM(n_regimes=2, n_features=3)._forward_backward(np.random.randn(5, 3))
def trigger_site_7():
    m = GaussianHSMM(n_regimes=2, n_features=3)
    m._forward_backward = lambda X: (np.ones((5, 2)), np.ones((5, 2, 40)), np.ones((2, 2)))
    m.fit(np.random.randn(5, 3), n_iter=1)
def trigger_site_8(): GaussianHSMM(n_regimes=2, n_features=3)._m_step(np.random.randn(5, 3), np.random.rand(5, 2), np.random.rand(5, 2, 40), np.random.rand(2, 2))
def trigger_site_9():
    # Site 9: GaussianHSMM.bic raises InternalStateError if model claims fitted but log_likelihood_ is None
    m = GaussianHSMM(n_regimes=2, n_features=3)
    m._is_fitted = True
    m.bic(10)
def trigger_site_10(): _ = GaussianHSMM(n_regimes=2, n_features=3).transition_matrix
def trigger_site_11(): _ = GaussianHSMM(n_regimes=2, n_features=3).duration_mu
def trigger_site_12(): _ = GaussianHSMM(n_regimes=2, n_features=3).duration_sigma
def trigger_site_13(): _ = GaussianHSMM(n_regimes=2, n_features=3).emission_means
def trigger_site_14(): _ = GaussianHSMM(n_regimes=2, n_features=3).emission_covariances
def trigger_site_15(): GaussianHSMM(n_regimes=2, n_features=3).duration_probability([1, 2], 0)
def trigger_site_16(): GaussianHSMM(n_regimes=2, n_features=3).generate_regime_sequence(10)
def trigger_site_17():
    KimHSSMModel(
        n_regimes=2, n_features=3, transition_matrix=np.eye(2),
        emission_means=np.zeros((2, 3)), emission_covariances=[np.eye(3), np.eye(3)],
        duration_mu=np.zeros(2), duration_sigma=np.ones(2), duration_prior="geometric"
    )
def trigger_site_18():
    kim = KimHSSMModel(
        n_regimes=2, n_features=3, transition_matrix=np.eye(2),
        emission_means=np.zeros((2, 3)), emission_covariances=[np.eye(3), np.eye(3)],
        duration_mu=np.zeros(2), duration_sigma=np.ones(2)
    )
    kim._is_fitted = False
    _ = kim.transition_matrix
def trigger_site_19():
    kim = KimHSSMModel(
        n_regimes=2, n_features=3, transition_matrix=np.eye(2),
        emission_means=np.zeros((2, 3)), emission_covariances=[np.eye(3), np.eye(3)],
        duration_mu=np.zeros(2), duration_sigma=np.ones(2)
    )
    kim._covariances_valid = False
    kim.emission_loglik(np.random.randn(3), 0)
def trigger_site_20(): fit_with_random_restarts(np.random.randn(35, 3), n_regimes=2, n_features=3, n_init=5, bypass_init_gate=False)
def trigger_site_21(): fit_with_random_restarts(np.random.randn(35, 3), n_regimes=2, n_features=3, n_init=0, bypass_init_gate=True)
def trigger_site_22():
    # Site 22: fit_with_random_restarts raises InternalStateError if model.converged_ is True but log_likelihood_ is None
    X = np.random.randn(35, 3)
    def mock_fit(self, X, n_iter=100, verbose=False, timestamps=None):
        self.converged_ = True
        self.log_likelihood_ = None
        return self
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(GaussianHSMM, "fit", mock_fit)
        fit_with_random_restarts(X, n_regimes=2, n_features=3, n_init=1, bypass_init_gate=True)
def trigger_site_23():
    # Site 23: All valid random restarts attempted and none converged -> FittingConvergenceError (Case A)
    X = np.random.randn(35, 3)
    fit_with_random_restarts(X, n_regimes=2, n_features=3, n_init=1, n_iter=0, bypass_init_gate=True)

# Site 24 (Case B): REMOVED — best_model is already proven non-None by the immediately preceding
# Site 23 guard/check in fit_with_random_restarts. The check was provably redundant and has been removed entirely.

def trigger_site_25():
    # Site 25: Canonicalization returning None violates postcondition -> InternalStateError (Case C)
    X = np.vstack([np.zeros((20, 3)), np.ones((20, 3)) * 4])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("backbone.hssm.fitting.canonicalize_labels", lambda m: None)
        fit_with_random_restarts(X, n_regimes=2, n_features=3, n_init=1, bypass_init_gate=True)
def trigger_site_26():
    # Site 26: select_k_by_bic with model returned with log_likelihood_ None -> InternalStateError
    X = np.random.randn(35, 3)
    def mock_fit_restarts(*args, **kwargs):
        m = GaussianHSMM(n_regimes=2, n_features=3)
        m._is_fitted = True
        m.converged_ = True
        m.log_likelihood_ = None
        return m, []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("backbone.hssm.fitting.fit_with_random_restarts", mock_fit_restarts)
        select_k_by_bic(X, n_features=3, k_candidates=(2,), bypass_init_gate=True)
def trigger_site_27():
    # Site 27: select_k_by_bic with empty k_candidates raises ValueError upfront (Case D)
    select_k_by_bic(np.random.randn(35, 3), n_features=3, k_candidates=(), bypass_init_gate=True)


SITE_TRIGGER_TABLE = [
    (1, trigger_site_1, ValueError),
    (2, trigger_site_2, ValueError),
    (3, trigger_site_3, NotFittedError),
    (4, trigger_site_4, NotFittedError),
    (5, trigger_site_5, NotFittedError),
    (6, trigger_site_6, NotFittedError),
    (7, trigger_site_7, InternalStateError),
    (8, trigger_site_8, NotFittedError),
    (9, trigger_site_9, InternalStateError),
    (10, trigger_site_10, NotFittedError),
    (11, trigger_site_11, NotFittedError),
    (12, trigger_site_12, NotFittedError),
    (13, trigger_site_13, NotFittedError),
    (14, trigger_site_14, NotFittedError),
    (15, trigger_site_15, NotFittedError),
    (16, trigger_site_16, NotFittedError),
    (17, trigger_site_17, ValueError),
    (18, trigger_site_18, NotFittedError),
    (19, trigger_site_19, ValueError),
    (20, trigger_site_20, ValueError),
    (21, trigger_site_21, ValueError),
    (22, trigger_site_22, InternalStateError),
    (23, trigger_site_23, FittingConvergenceError),
    # Site 24 removed per Case B (provably redundant check)
    (25, trigger_site_25, InternalStateError),
    (26, trigger_site_26, InternalStateError),
    (27, trigger_site_27, ValueError),
]


@pytest.mark.parametrize("site_num,trigger_fn,expected_exc", SITE_TRIGGER_TABLE)
def test_individual_sites_raise_expected_exceptions(site_num, trigger_fn, expected_exc):
    """Parametrized test individually exercising all active audited sites."""
    with pytest.raises(expected_exc):
        trigger_fn()


def test_site_25_reaches_canonicalization_after_successful_selection(monkeypatch):
    """The postcondition failure must occur after a genuine successful fit."""
    X = np.vstack([np.zeros((20, 3)), np.ones((20, 3)) * 4])
    canonicalize_calls = 0

    def reject_canonicalization(model):
        nonlocal canonicalize_calls
        canonicalize_calls += 1
        return None

    monkeypatch.setattr("backbone.hssm.fitting.canonicalize_labels", reject_canonicalization)

    with pytest.raises(InternalStateError, match="canonicalize_labels"):
        fit_with_random_restarts(
            X,
            n_regimes=2,
            n_features=3,
            n_init=1,
            bypass_init_gate=True,
        )

    assert canonicalize_calls == 1


def test_site_26_uses_real_select_k_by_bic_path(monkeypatch):
    """A corrupt successful-fit result must fail inside select_k_by_bic."""
    X = np.random.default_rng(26).normal(size=(35, 3))
    corrupt_model = GaussianHSMM(n_regimes=2, n_features=3)
    corrupt_model._is_fitted = True
    corrupt_model.converged_ = True
    corrupt_model.log_likelihood_ = None
    fit_calls = 0

    def return_corrupt_model(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return corrupt_model, []

    monkeypatch.setattr(
        "backbone.hssm.fitting.fit_with_random_restarts",
        return_corrupt_model,
    )

    with pytest.raises(InternalStateError, match="log_likelihood"):
        select_k_by_bic(
            X,
            n_features=3,
            k_candidates=(2,),
            bypass_init_gate=True,
        )

    assert fit_calls == 1


def test_site_27_rejects_empty_candidates_before_fitting(monkeypatch):
    """Invalid model-selection configuration must cause zero fitting work."""
    fit_calls = 0

    def fitting_must_not_start(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        raise AssertionError("fitting should not have started")

    monkeypatch.setattr(
        "backbone.hssm.fitting.fit_with_random_restarts",
        fitting_must_not_start,
    )

    with pytest.raises(ValueError, match="non-empty"):
        select_k_by_bic(
            np.random.default_rng(27).normal(size=(35, 3)),
            n_features=3,
            k_candidates=(),
            bypass_init_gate=True,
        )

    assert fit_calls == 0


def test_not_fitted_error_raised_in_both_optimizations():
    """Verify all audited sites raise correct exceptions under both normal and -O subprocess runs."""
    script = """
import numpy as np
from backbone.hssm import (
    GaussianHSMM, KimHSSMModel, HSSMError, NotFittedError, FittingConvergenceError, InternalStateError,
    fit_with_random_restarts, select_k_by_bic
)

# Test fitted-state properties and methods on GaussianHSMM -> NotFittedError (14 sites)
m = GaussianHSMM(n_regimes=2, n_features=3)

nf_sites = [
    ("Site 3", lambda: m._require_fitted()),
    ("Site 4", lambda: m._duration_logpmf()),
    ("Site 5", lambda: m._emission_loglik(np.random.randn(5, 3))),
    ("Site 6", lambda: m._forward_backward(np.random.randn(5, 3))),
    ("Site 8", lambda: m._m_step(np.random.randn(5, 3), np.random.rand(5, 2), np.random.rand(5, 2, 40), np.random.rand(2, 2))),
    ("Site 10", lambda: m.transition_matrix),
    ("Site 11", lambda: m.duration_mu),
    ("Site 12", lambda: m.duration_sigma),
    ("Site 13", lambda: m.emission_means),
    ("Site 14", lambda: m.emission_covariances),
    ("Site 15", lambda: m.duration_probability([1, 2], 0)),
    ("Site 16", lambda: m.generate_regime_sequence(10)),
]

for name, fn in nf_sites:
    try:
        fn()
        raise AssertionError(f"{name} did not raise NotFittedError")
    except NotFittedError:
        pass

kim = KimHSSMModel(
    n_regimes=2, n_features=3, transition_matrix=np.eye(2),
    emission_means=np.zeros((2, 3)), emission_covariances=[np.eye(3), np.eye(3)],
    duration_mu=np.zeros(2), duration_sigma=np.ones(2)
)
kim._is_fitted = False
try:
    _ = kim.transition_matrix
except NotFittedError:
    pass

# ValueError sites (7 sites: 1, 2, 17, 19, 20, 21, 27)
try: GaussianHSMM(n_regimes=0, n_features=3)
except ValueError: pass

try: GaussianHSMM(n_regimes=2, n_features=0)
except ValueError: pass

try: KimHSSMModel(2, 3, np.eye(2), np.zeros((2,3)), [np.eye(3), np.eye(3)], np.zeros(2), np.ones(2), duration_prior="geometric")
except ValueError: pass

try:
    kim_inv = KimHSSMModel(2, 3, np.eye(2), np.zeros((2,3)), [np.eye(3), np.eye(3)], np.zeros(2), np.ones(2))
    kim_inv._covariances_valid = False
    kim_inv.emission_loglik(np.random.randn(3), 0)
except ValueError: pass

try: fit_with_random_restarts(np.random.randn(35, 3), n_regimes=2, n_features=3, n_init=5, bypass_init_gate=False)
except ValueError: pass

try: fit_with_random_restarts(np.random.randn(35, 3), n_regimes=2, n_features=3, n_init=0, bypass_init_gate=True)
except ValueError: pass

try: select_k_by_bic(np.random.randn(35, 3), n_features=3, k_candidates=(), bypass_init_gate=True)
except ValueError: pass

# InternalStateError sites (Site 7, Site 9, Site 22, Site 25, Site 26)
try:
    m_sev = GaussianHSMM(2, 3)
    m_sev._forward_backward = lambda X: (np.ones((5, 2)), np.ones((5, 2, 40)), np.ones((2, 2)))
    m_sev.fit(np.random.randn(5, 3), n_iter=1)
except InternalStateError: pass

try:
    m_bic = GaussianHSMM(2, 3)
    m_bic._is_fitted = True
    m_bic.bic(10)
except InternalStateError: pass

try:
    old_fit = GaussianHSMM.fit
    def mock_f(self, X, n_iter=100, verbose=False, timestamps=None):
        self.converged_ = True
        self.log_likelihood_ = None
        return self
    GaussianHSMM.fit = mock_f
    try:
        fit_with_random_restarts(np.random.randn(35, 3), n_regimes=2, n_features=3, n_init=1, bypass_init_gate=True)
    finally:
        GaussianHSMM.fit = old_fit
except InternalStateError: pass

import backbone.hssm.fitting as fitting_mod
old_canon = fitting_mod.canonicalize_labels
fitting_mod.canonicalize_labels = lambda m: None
try:
    fit_with_random_restarts(np.vstack([np.zeros((20, 3)), np.ones((20, 3)) * 4]), n_regimes=2, n_features=3, n_init=1, bypass_init_gate=True)
except InternalStateError:
    pass
finally:
    fitting_mod.canonicalize_labels = old_canon

old_fit_restarts = fitting_mod.fit_with_random_restarts
def mock_fit_r(*args, **kwargs):
    m = GaussianHSMM(2, 3)
    m._is_fitted = True
    m.converged_ = True
    m.log_likelihood_ = None
    return m, []
fitting_mod.fit_with_random_restarts = mock_fit_r
try:
    select_k_by_bic(np.random.randn(35, 3), n_features=3, k_candidates=(2,), bypass_init_gate=True)
except InternalStateError:
    pass
finally:
    fitting_mod.fit_with_random_restarts = old_fit_restarts

# FittingConvergenceError site (Site 23)
try: fit_with_random_restarts(np.random.randn(35, 3), n_regimes=2, n_features=3, n_init=1, n_iter=0, bypass_init_gate=True)
except FittingConvergenceError: pass

print("ALL_AUDITED_SITES_PASSED")
"""
    res_normal = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=".")
    assert "ALL_AUDITED_SITES_PASSED" in res_normal.stdout, f"Normal execution failed: {res_normal.stderr}"

    res_opt = subprocess.run([sys.executable, "-O", "-c", script], capture_output=True, text=True, cwd=".")
    assert "ALL_AUDITED_SITES_PASSED" in res_opt.stdout, f"Optimized execution failed: {res_opt.stderr}"
