# Chronis HSSM: Known Limitations & Operational Boundaries

**Document ID**: `CHRONIS-HSSM-LIMITATIONS-001`  
**Version**: `1.1.0` (Empirical Audit Revision)  
**Acceptance State**: `SEALED_YELLOW` (FULL_MODEL_SCAFFOLD_READY)

---

## 1. Explicit Non-Claims & Boundary Conditions

To maintain scientific integrity and prevent overclaiming across product and clinical contexts:

1. **Baseline Model (`ExplicitDurationSwitchingBaselineV1`)**:
   - **Does NOT track momentary or continuous latent state ($m_t$).** Any downstream consumer expecting fast continuous trajectories must check `fast_state_supported == True`.
   - Discrete regime sequences reflect macro cluster shifts, not continuous behavioral velocity.

2. **Full Dual Model (`ChronisDualTimescaleHSMMV1`)**:
   - Continuous latent trajectories $m_t$ are identifiable **up to an affine / orthogonal coordinate transformation**. Cross-user coordinate comparisons must use canonicalized invariant metrics (such as attractor dwell times, revisit counts, and transition graphs), not raw coordinate values.
   - Requires a minimum of 30 eligible sessions (`min_present_sessions = 30`). Running fits with fewer observations triggers `ColdStartError`.

3. **Semi-Markov Duration Truncation ($D_{\max}$)**:
   - Durations are truncated at $D_{\max}$ (default 45 or 60 days/timesteps). Regimes exceeding $D_{\max}$ in reality will experience probability mass accumulation at the boundary. The `compute_dmax_tail_diagnostic()` function monitors tail mass at $D_{\max}$ and emits warnings if mass exceeds 0.05.

4. **Missingness & Sparse Data**:
   - Emission evaluations support missing coordinates through Gaussian marginalization. However, if a timestep has $< 50\%$ feature presence, it is excluded via `eligibility_mask`.
   - Feature confidence values $< 0.5$ are treated as unobserved to prevent noisy sensor artifacts from biasing continuous trajectory estimates.

5. **Neural Residual Boundary**:
   - The neural residual adapter is an additive corrector only. It **cannot** author regime states, continuous state space coordinates, or override convergence failure flags.
   - In production deployments, neural residuals remain in `SHADOW` or `DISABLED` modes until empirical ablation reports demonstrate non-degradation on out-of-sample holdouts.

6. **Fast-State Identifiability & Initialization Instability (Audit Finding)**:
   - **Empirical Sensitivity**: As verified in adversarial audit testing, single-restart EM fitting of the continuous state-space model exhibits significant initialization sensitivity on synthetic sequences ($R^2$ ranging from 0.03 to 0.56, ARI ranging from -0.02 to 0.58).
   - **Identifiability Constraint Needed**: The continuous loading matrix $C_{\text{obs}}$ currently lacks an explicit column-orthogonalization or sign-anchoring constraint during the analytical M-step. Consequently, different random seeds can lead the optimizer into distinct local coordinate rotations.
   - **Operational Mitigation**: Multi-restart fitting ($\ge 10$ initializations via `fit_dual_with_random_restarts`) is mandatory for non-trivial datasets. Sprint 3–4 status is held at `FULL_MODEL_SCAFFOLD_READY` / `SEALED_YELLOW` pending formal identifiability anchoring in the next sprint cycle.

7. **Sprint 5–6 Downstream Adapter Integration Boundary**:
   - Tests 121-123 verify this side of the contract only. Real end-to-end Sprint 5-6 adapter integration has NOT been tested, because the Sprint 5-6 codebase is not present in this workspace. This remains unverified until that integration test can be run against the real adapter.

