import numpy as np
# RELOCATED (S56.6): fixture moved out of the production import path
# (domain_emergence/) -- test-fixture-only now.
from tests.fixtures.synthetic_hssm_fixture import generate_synthetic_hssm_output


def test_shapes_match_contract():
    out = generate_synthetic_hssm_output(T=200, K=3, F=5, seed=0)
    assert out.regime_sequence.shape == (200,)
    assert out.observations.shape == (200, 5)
    assert out.transition_matrix.shape == (3, 3)
    assert out.emission_means.shape == (3, 5)
    assert len(out.emission_covariances) == 3
    assert out.emission_covariances[0].shape == (5, 5)
    assert out.duration_mu.shape == (3,)
    assert out.duration_sigma.shape == (3,)


def test_regime_sequence_values_in_range():
    out = generate_synthetic_hssm_output(T=200, K=4, F=3, seed=1)
    assert out.regime_sequence.min() >= 0
    assert out.regime_sequence.max() <= 3


def test_transition_matrix_zero_diagonal_row_stochastic():
    out = generate_synthetic_hssm_output(K=4, seed=2)
    A = out.transition_matrix
    assert np.allclose(np.diag(A), 0.0)
    row_sums = A.sum(axis=1)
    assert np.allclose(row_sums, 1.0)


def test_missing_rate_produces_nan_rows():
    out = generate_synthetic_hssm_output(T=500, missing_rate=0.1, seed=3)
    n_missing = np.isnan(out.observations).any(axis=1).sum()
    # loose bound, stochastic
    assert 20 <= n_missing <= 90


def test_zero_missing_rate_produces_no_nan():
    out = generate_synthetic_hssm_output(T=100, missing_rate=0.0, seed=4)
    assert not np.isnan(out.observations).any()


def test_seed_reproducibility():
    out1 = generate_synthetic_hssm_output(seed=42)
    out2 = generate_synthetic_hssm_output(seed=42)
    assert np.array_equal(out1.regime_sequence, out2.regime_sequence)
    assert np.allclose(out1.observations, out2.observations, equal_nan=True)


def test_k_equals_1_edge_case():
    out = generate_synthetic_hssm_output(T=50, K=1, F=2, seed=5)
    assert np.all(out.regime_sequence == 0)
    assert out.transition_matrix.shape == (1, 1)