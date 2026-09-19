# HSSM Result Schema & Specification Contract

**Document ID**: `CHRONIS-HSSM-SCHEMA-001`  
**Version**: `1.0.0`  
**Acceptance State**: `SEALED_YELLOW` (FULL_MODEL_SCAFFOLD_READY)

---

## 1. Overview

The `HSSMResult` dataclass in `backbone.hssm.fitting` is the single canonical return type for all HSSM fitting procedures. It enforces complete metadata tracking across 40+ fields required by Section Q of the Chronis Behavioral Backbone Directive.

---

## 2. Field Definitions

| Field | Type | Description | Baseline Model Value | Full Dual Model Value |
|---|---|---|---|---|
| `model` | `Any` | Fitted model instance | `GaussianHSMM` | `DualTimescaleHSSM` |
| `user_id` | `str` | User identifier | String | String |
| `model_identity` | `str` | Formal class identifier | `"ExplicitDurationSwitchingBaselineV1"` | `"ChronisDualTimescaleHSMMV1"` |
| `model_family` | `str` | Algorithmic family | `"slow_regime_only"` / `"GaussianHSMM"` | `"fast_slow_dual_timescale"` |
| `model_capability` | `dict` | Capability dictionary | `{"fast_state_supported": False, "is_baseline_model": True}` | `{"fast_state_supported": True, ...}` |
| `is_baseline_model` | `bool` | Flag indicating baseline | `True` | `False` |
| `fast_state_supported`| `bool` | Whether $m_t$ is supported | `False` | `True` |
| `p_t_posterior` | `ndarray (T, K)` | Posterior regime probabilities | Sums to 1.0 across rows | Sums to 1.0 across rows |
| `p_t_estimate` | `ndarray (T,)` | MAP regime sequence | Array of ints $\{0..K-1\}$ | Array of ints $\{0..K-1\}$ |
| `m_t_posterior_mean`| `ndarray (T, L)`| Continuous latent mean | `None` (Never faked) | Array of floats |
| `m_t_uncertainty` | `ndarray (T, L)`| Latent state std error | `None` (Never faked) | Array of floats $\ge 0$ |
| `duration_parameters`| `dict` | $\mu, \sigma$ arrays | Dict of lists | Dict of lists |
| `duration_model_identity`| `str` | Duration distribution | `"explicit_truncated_gaussian"` | `"explicit_truncated_lognormal"` |
| `Dmax` | `int` | Maximum duration truncation | Integer ($> 0$) | Integer ($> 0$) |
| `duration_unit_source`| `str` | Source of duration time | `"session_index"` or `"calendar_days"` | `"calendar_days"` |
| `calendar_start` | `float` | Start timestamp | Float or `None` | Float or `None` |
| `calendar_end` | `float` | End timestamp | Float or `None` | Float or `None` |
| `observation_density`| `float` | Fraction of present sessions | Float in $[0, 1]$ | Float in $[0, 1]$ |
| `model_version` | `str` | Code version | `"1.0.0"` | `"1.0.0"` |
| `feature_schema_version`| `str` | Feature schema version | `"1.0.0"` | `"1.0.0"` |
| `alignment_version` | `str` | Label switching version | `"1.0.0"` | `"1.0.0"` |
| `eligible_session_count`| `int` | Count of eligible rows | Integer $\ge 30$ | Integer $\ge 30$ |
| `min_present_sessions`| `int` | Minimum required sessions | `30` | `30` |
| `fit_dataset_hash` | `str` | Deterministic SHA-256 | 64-char hex string | 64-char hex string |
| `K` | `int` | Selected regime count | Integer in $\{2, 3, 4\}$ | Integer in $\{2, 3, 4\}$ |
| `BIC` | `float` | Bayesian Information Criterion| Float | Float |
| `AIC` | `float` | Akaike Information Criterion | Float | Float |
| `heldout_predictive_log_likelihood` | `float` | Chronological test LL | Float or `None` | Float or `None` |
| `fit_state` | `FitState` | Terminal state enum | `FitState.CONVERGED` etc. | `FitState.CONVERGED` etc. |
| `optimizer_success` | `bool` | All optimizers converged | Boolean | Boolean |
| `fallback_used` | `bool` | Truncation fallback flag | Boolean | Boolean |
| `run_log` | `list[dict]` | Restart audit trail | List of dicts | List of dicts |
| `warnings` | `list[str]` | Fitting warnings | List of strings | List of strings |
| `limitations` | `list[str]` | Model boundary statements | List of strings | List of strings |
| `neural_residual_enabled`| `bool` | Neural residual flag | `False` | `False` (by default) |
| `residual_mode` | `str` | Mode of neural residual | `"DISABLED"` | `"DISABLED"` / `"SHADOW"` |
| `residual_weight` | `float` | Additive blending factor | `0.0` | `0.0` |

---

## 3. Backward Compatibility Aliases

The following read properties maintain 100% backward compatibility with legacy Sprint 3/4 consumers:
- `result.p_t` $\rightarrow$ returns `result.p_t_estimate`
- `result.regime_posterior` $\rightarrow$ returns `result.p_t_posterior`
- `result.m_t` $\rightarrow$ returns `result.m_t_posterior_mean`
- `result.k_selected` $\rightarrow$ returns `result.K`
- `result.bic_by_k` $\rightarrow$ returns BIC map from convergence metadata
- `result.duration_info` $\rightarrow$ returns duration parameter dictionary
