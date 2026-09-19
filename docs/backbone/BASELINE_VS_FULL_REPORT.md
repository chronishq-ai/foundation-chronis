# Comparative Report: Baseline vs. Full Dual-Timescale HSSM

**Document ID**: `CHRONIS-HSSM-COMP-001`  
**Version**: `1.1.0` (Empirical Audit Revision)  
**Acceptance State**: `SEALED_YELLOW` (FULL_MODEL_SCAFFOLD_READY)

---

## 1. Architectural & Theoretical Comparison

| Dimension | Baseline Model (`GaussianHSMM`) | Full Model (`DualTimescaleHSSM`) |
|---|---|---|
| **Model Identity** | `ExplicitDurationSwitchingBaselineV1` | `ChronisDualTimescaleHSMMV1` |
| **Model Family** | `slow_regime_only` | `fast_slow_dual_timescale` |
| **Latent Structure** | 1-level: discrete $p_t \in \{0..K-1\}$ | 2-level: continuous $m_t \in \mathbb{R}^L$ + discrete $p_t$ |
| **Temporal Dynamics** | Jump process with duration | Switching vector autoregressive AR(1) |
| **Fast State $m_t$** | `None` (Strictly non-faking) | Kalman-filtered and RTS-smoothed mean & cov |
| **Downstream Compatibility**| Macro-regime classification only | Attractor dynamics, dwell statistics, continuous velocity |
| **Inference Method** | Expanded-state Baum-Welch | Kim's Filter (GPB2 collapse) + RTS smoother |
| **Neural Residual Mode** | Disabled (`NOT_APPLICABLE`) | Supported in `SHADOW` or `ACTIVE` mode |

---

## 2. Empirical Performance & Multi-Seed Replication

The empirical recovery claims from initial development ($R^2=0.842$, $\text{ARI}=0.965$) were subjected to zero-trust replication. Below are the verified, multi-seed empirical recovery metrics on synthetic benchmark sequences ($T \approx 150, K=3, F=4, L=2$, $N_{\text{iter}}=25$):

### 2.1 Multi-Seed Replication Table

| Seed | Sequence Length ($T$) | Baseline LL | Baseline ARI | Dual LL | Dual ARI | Fast-State Subspace $R^2$ |
|---|---|---|---|---|---|---|
| **Seed 42** | $T=153$ | -926.35 | 1.0000 | -825.93 | 0.2865 | 0.0341 |
| **Seed 101** | $T=165$ | -1008.30 | 1.0000 | -805.57 | 0.0083 | 0.4256 |
| **Seed 202** | $T=173$ | -1146.11 | 1.0000 | -712.61 | 0.4021 | 0.1006 |
| **Seed 303** | $T=155$ | -956.37 | 1.0000 | -651.39 | -0.0220 | 0.2121 |
| **Seed 1** | $T=144$ | -774.62 | 1.0000 | -594.81 | 0.0874 | 0.2440 |
| **Seed 0** | $T=141$ | -920.09 | 1.0000 | -611.37 | 0.5769 | 0.3962 |
| **Mean $\pm$ Std** | — | **-955.31 $\pm$ 111.4** | **1.0000 $\pm$ 0.0** | **-700.28 $\pm$ 94.6** | **0.2232 $\pm$ 0.218** | **0.2354 $\pm$ 0.147** |

### 2.2 Dedicated Test Fixture Metrics (Unit Test Suite)
- **Test 38 (`test_dual_model_verification.py`, Seed 101, $N_{\text{init}}=10$ restarts)**:
  - Accuracy: **0.8333**
  - Adjusted Rand Index (ARI): **0.4352**
- **Test 39 (`test_dual_model_verification.py`, Seed 202, $N_{\text{init}}=1$)**:
  - Fast-State Subspace $R^2$: **0.5650**

### 2.3 Diagnostic Finding: Initialization Sensitivity
The dual-timescale model consistently achieves higher log-likelihood than the baseline (mean $+255.0$ nats improvement in likelihood fit), confirming that continuous state dynamics better account for observed variance. However, fast-state subspace $R^2$ and discrete regime recovery (ARI) exhibit high variance across single-seed initializations without multi-restart selection and identifiability constraints.

---

## 3. Conclusions & Acceptance State

The mathematical foundation of `DualTimescaleHSSM` (Kim filtering, GPB2 moment matching, RTS smoothing, and generalized-EM backtracking) is verified and functional. However, due to observed empirical sensitivity across random seeds and unconstrained rotational identifiability during EM, acceptance remains at **`SEALED_YELLOW` (FULL_MODEL_SCAFFOLD_READY)** until multi-restart convergence tuning and coordinate-anchoring are integrated.
