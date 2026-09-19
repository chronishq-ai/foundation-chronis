# Chronis Dual-Timescale HSSM: Architecture & Design Note

**Document ID**: `CHRONIS-HSSM-DN-001`  
**Version**: `1.0.0`  
**Acceptance State**: `SEALED_YELLOW` (FULL_MODEL_SCAFFOLD_READY)  
**Status**: APPROVED

---

## 1. Executive Summary & Problem Context

The Chronis behavioral backbone models human habit and behavioral phase transitions over time. In earlier development sprints, the system relied on an incomplete slow-regime-only baseline model (`GaussianHSMM`). While that baseline handled discrete macro-regimes ($p_t$), it lacked the fast continuous latent state trajectory ($m_t$) demanded by the Chronis Bible (Bible 5.1 & 5.2).

This design note documents the implementation of the full, Bible-grade **Dual-Timescale Hidden Semi-Markov State-Space Model (`DualTimescaleHSSM` / `ChronisDualTimescaleHSMMV1`)**, which couples:
1. A **fast continuous latent state** $m_t \in \mathbb{R}^L$ evolving at momentary/session resolution under regime-dependent linear dynamics ($A_k, b_k, Q_k$).
2. A **slow discrete macro-regime state** $p_t \in \{0, \dots, K-1\}$ governing life-phase transitions under semi-Markov explicit duration log-normal distributions ($\mu_k, \sigma_k, D_{\max}$).
3. A **canonical data contract** (`HSSMFitDataset`) enforcing calendar-time semantics, missingness/confidence masks, and cold-start gates (30-session rule).
4. A **controlled neural residual adapter** (`NeuralResidualAdapter`) operating in `DISABLED` / `SHADOW` modes with zero-weight ablation guarantees.

---

## 2. Core Architectural Principles

### 2.1 Explicit Model Identity & Non-Faking Invariant
To prevent silent degradation and false claims:
- **Baseline Model (`ExplicitDurationSwitchingBaselineV1`)**: Implemented by `GaussianHSMM`. Only supports slow discrete regimes. Its outputs strictly enforce `m_t = None`, `m_t_uncertainty = None`, and `fast_state_supported = False`. Under no circumstances are placeholder coordinates emitted.
- **Full Model (`ChronisDualTimescaleHSMMV1`)**: Implemented by `DualTimescaleHSSM`. Provides genuine Kalman-smoothed $m_t$, $m_t$ covariance uncertainty, discrete regime posterior $p_t$, and explicit duration parameters.

### 2.2 Dual-Timescale Coupling & Filtering
Exact inference in switching linear dynamical systems suffers from exponential mode explosion ($K^T$ Gaussian mixtures). We implement **Kim's Filter with Gaussian Pseudo-Bayes 2 (GPB2) collapse**:
- At each timestep $t$, the state is represented as $K$ conditional Gaussian distributions $\mathcal{N}(\hat{m}_{t|t}^{(j)}, P_{t|t}^{(j)})$.
- Forward transitions produce $K \times K$ candidate mixtures, which are moment-matched and collapsed back into $K$ densities based on the semi-Markov regime posterior.
- The backward smoother runs Rauch-Tung-Striebel recursion across regime paths to produce the smoothed continuous trajectory $m_{t|T}$ and standard deviations.

### 2.3 Calendar-Time Units & Explicit Semi-Markov Durations
Human behavioral regimes operate on real-world calendar time (days, minutes), not discrete session index counts.
- Durations are parameterized via truncated log-normal distributions up to $D_{\max}$.
- When timestamps are supplied, duration parameters evaluate physical day differences $\Delta t$.
- In the M-step, duration optimization uses Generalized EM with monotonic backtracking rollback: if an L-BFGS-B or Newton update drops overall log-likelihood, the step is rolled back and flagged as `FitState.REJECTED_UPDATE`.

### 2.4 Controlled Neural Residual Adapter
Neural networks must never author ground-truth latent state identities or break physical interpretability:
- The base HSSM is purely generative and identifiable.
- The neural residual adapter outputs strictly bounded additive corrections $\Delta y_t, \Delta \pi, \Delta Q$.
- In `SHADOW` mode, neural predictions are logged as diagnostic metrics while live outputs remain 100% generative base HSSM.
- An ablation guarantee guarantees that setting `residual_weight = 0.0` or mode `DISABLED` yields bit-for-bit identity with the pure base model.

---

## 3. Downstream Interface Contract

All outputs are unified in the `HSSMResult` dataclass:
- Attractor Detection (`backbone.attractors.detector`): Directly consumes `result.m_t` and `result.p_t` to calculate dwell times, revisit counts, and transition stability.
- Divergence & Drift Detection: Operates on parameter shifts across chronological splits.
- Provenance & Audit: Every result records the SHA-256 `fit_dataset_hash`, random initialization seeds, AIC, BIC, and full run log.
