import pytest
import numpy as np
from unittest.mock import patch
from nssm_pipeline.nssm_calibration import fit_nssm_for_j

def test_fit_nssm_assert_guard():
    """
    Test that the fitted-state guard in fit_nssm_for_j raises an explicit
    AssertionError instead of a bare assert, ensuring the pipeline fails
    loudly even when run under `python -O`.
    """
    obs = np.zeros((10, 8))
    var = np.ones((10, 8))
    
    # By forcing n_random_inits=0 and mocking the warm start to return None,
    # we ensure the candidates list is empty and best_params remains None.
    with patch("nssm_pipeline.nssm_calibration._statsmodels_warm_start", return_value=None):
        with pytest.raises(AssertionError, match="best_params is not None"):
            fit_nssm_for_j(obs, var, j_count=2, n_random_inits=0)
