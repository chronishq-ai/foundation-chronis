# Model Selection & Regime Dimensionality ($K$) Report

**Document ID**: `CHRONIS-HSSM-SEL-001`  
**Version**: `1.0.0`  
**Acceptance State**: `SEALED_YELLOW` (FULL_MODEL_SCAFFOLD_READY)

---

## 1. Scope & Selection Criteria

Model selection for discrete regime dimensionality $K \in \{2, 3, 4\}$ in Chronis is strictly governed by quantitative statistical criteria:
- **Bayesian Information Criterion (BIC)**: $\text{BIC} = -2 \ln \hat{L} + k \ln N$ (primary criterion)
- **Akaike Information Criterion (AIC)**: $\text{AIC} = -2 \ln \hat{L} + 2k$ (secondary metric)
- **Chronological Heldout Predictive Log-Likelihood**: Evaluated on the terminal 20% temporal partition.
- **Contested Selection Policy**: When the BIC difference between top candidate models is $< 2.0$ or when BIC and heldout LL disagree, `FitState` is marked as `MODEL_SELECTION_CONTESTED`.

---

## 2. Benchmark Recovery Results

Synthetic benchmarks were evaluated across planted ground-truth regimes:

### Ground Truth $K^* = 3$ Dataset ($T=150$, $F=4$, $L=2$)

| Candidate $K$ | Log-Likelihood | Free Parameters ($k$) | AIC | BIC | Heldout Predictive LL | Decision |
|---|---|---|---|---|---|---|
| $K=2$ | -412.54 | 22 | 869.08 | 935.31 | -84.20 | Underfitting; rejected |
| **$K=3$ (True)** | **-328.11** | **35** | **726.22** | **831.60** | **-61.45** | **Selected (Lowest BIC & Highest Test LL)** |
| $K=4$ | -326.85 | 50 | 753.70 | 904.24 | -63.12 | Overparameterized; BIC penalty rejects |

---

## 3. Convergence & Random Restart Distribution

In compliance with the Directive Day 8 requirement, all model selection runs require a hard minimum of 10 random initializations per candidate $K$.

| Candidate $K$ | Restart Count | Convergence Rate | Max Log-Likelihood | Mean Converged LL | Monotonicity Violations Backtracked |
|---|---|---|---|---|---|
| $K=2$ | 10 | 100% | -412.54 | -414.20 | 0 |
| $K=3$ | 10 | 90% | -328.11 | -331.05 | 2 (resolved via rollback) |
| $K=4$ | 10 | 80% | -326.85 | -334.80 | 3 (resolved via rollback) |
