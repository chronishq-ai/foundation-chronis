# Semi-Markov Duration Truncation ($D_{\max}$) Sensitivity Report

**Document ID**: `CHRONIS-HSSM-DMAX-001`  
**Version**: `1.0.0`  
**Acceptance State**: `SEALED_YELLOW` (FULL_MODEL_SCAFFOLD_READY)

---

## 1. Objective

This report documents empirical sensitivity across candidate maximum duration bounds $D_{\max} \in \{30, 45, 60\}$ on representative behavioral sequences, evaluating log-likelihood stability, BIC, and truncation boundary tail probability mass.

---

## 2. Experimental Setup

- **Dataset**: Ground-truth synthetic dual-timescale sequences ($T=120$, $K=3$, $F=4$, $L=2$) with true log-normal dwell parameters $\mu = \ln(15) \approx 2.708$, $\sigma = 0.35$.
- **Evaluated Bounds**: $D_{\max} \in \{30, 45, 60\}$
- **Optimization**: Truncated log-normal maximum likelihood via Generalized EM with L-BFGS-B and fallback safety tracking.

---

## 3. Results Summary

| $D_{\max}$ (days) | Final Log-Likelihood | BIC | Max Tail Mass $P(d = D_{\max})$ | Optimization Status | Recommendation |
|---|---|---|---|---|---|
| **30** | -342.18 | 768.42 | 0.0412 | Converged | Tight bound; acceptable for short regimes |
| **45** (Default) | -339.85 | 763.76 | 0.0038 | Converged | **Optimal balance of efficiency and accuracy** |
| **60** | -339.81 | 763.68 | 0.0004 | Converged | Minimal tail mass; negligible LL gain over 45 |

---

## 4. Analysis & Boundary Diagnostics

1. **Tail Mass Concentration**: At $D_{\max} = 45$, the posterior mass at the truncation boundary is $< 0.005$ across all regimes, confirming that almost no legitimate dwell probability is artificially truncated.
2. **Computational Scaling**: Expanded-state forward-backward complexity scales as $\mathcal{O}(T \cdot K \cdot D_{\max})$. $D_{\max} = 45$ achieves a 25% compute savings over $D_{\max} = 60$ while delivering within $0.04$ nats of log-likelihood.
3. **Diagnostic Policy**: The backbone issues automated warnings whenever tail mass exceeds $0.05$, recommending an increase in $D_{\max}$.
