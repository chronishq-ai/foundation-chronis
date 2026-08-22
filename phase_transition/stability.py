import numpy as np

# HONESTY FLAG (S56.1, per audit ownership model -- same pattern as
# divergence_engine's granger.py): doctrine (Bible Part 5.1 / phase-
# transition condition 3) specifies this metric as regime-posterior
# UNCERTAINTY (entropy or posterior variance of the regime-probability
# vector). The implementation below computes plain variance over raw
# data values instead -- NOT the same statistical object. Swapping the
# production metric is a model/statistical-estimator decision and is
# Senior-owned (Ownership Model: "Harnesses" tier -- intern builds the
# validation harness, senior decides/implements the metric itself). Do
# not silently treat raw-variance output as equivalent to posterior
# entropy. See `regime_posterior_entropy` and
# `validate_stability_metric_against_entropy` below for the harness that
# documents the gap; see KNOWN_LIMITATIONS.md.


def regime_posterior_entropy(probabilities) -> float:
    """Shannon entropy (nats) of a regime-probability vector, the
    doctrine-correct 'uncertainty' object for condition 3. Zero-probability
    entries contribute 0 (0*log(0) := 0), not NaN."""
    p = np.asarray(probabilities, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)))


def validate_stability_metric_against_entropy(
    known_regime_probs: list, expected_entropies: list | None = None,
) -> dict:
    """Harness (S56.1, Harnesses tier): for a synthetic fixture with a
    KNOWN regime-posterior distribution per timestep, computes the
    doctrine-correct entropy and reports it alongside what the current
    raw-variance metric would need to match to be equivalent. Diagnostic
    only -- does not decide or implement the production metric swap;
    a senior interprets this output. See S56.1 Test Sheet T1."""
    computed = [regime_posterior_entropy(p) for p in known_regime_probs]
    result = {"computed_entropy": computed}
    if expected_entropies is not None:
        result["expected_entropy"] = list(expected_entropies)
        result["matches_within_tolerance"] = all(
            abs(c - e) < 1e-6 for c, e in zip(computed, expected_entropies)
        )
    return result


class RegimeStability:
    """
    Condition 3 of 3. Monitors regime posterior variance >=min_days
    post-candidate. Splits post-candidate window in half, compares raw
    variance. Needs enough samples per half (>=10) for the estimate to
    not be dominated by small-sample noise.

    HONESTY FLAG: the metric here is raw-value variance, not the
    regime-posterior entropy/variance doctrine specifies -- see module
    docstring above. Metric swap is senior-owned; not done here.
    """

    def __init__(self, min_days: int = 20):
        self.min_days = min_days

    def is_stabilizing(self, data: list[float], candidate_t: int,
                         min_days: int | None = None,
                         drop_ratio: float = 0.75,
                         timestamps: list[float] | None = None) -> dict:
        """
        drop_ratio: second-half variance must be <= first-half * drop_ratio
        to count as 'decreasing' (meaningful drop, not noise-level wobble).

        timestamps: optional, same length as `data`. When provided, the
        post-candidate window is selected by actual CALENDAR-DAY span
        (candidate timestamp, candidate timestamp + min_days days) rather
        than by sample count -- required for irregular/sparse-session
        users where N samples != N calendar days (S56.1 fix, mechanical
        data-handling change, not a statistical-estimator change). When
        omitted, falls back to the original sample-offset behavior
        (backward compatible).
        """
        min_days = min_days or self.min_days
        data = np.asarray(data)

        if min_days < 10:
            return {"valid": False, "reason": "insufficient post-candidate data",
                    "met": False}

        if timestamps is not None:
            timestamps = np.asarray(timestamps, dtype=float)
            if len(timestamps) != len(data):
                raise ValueError("timestamps must be same length as data")
            t0 = timestamps[candidate_t]
            window_mask = (timestamps >= t0) & (timestamps < t0 + min_days)
            post = data[window_mask]
            if len(post) < 10:
                return {"valid": False,
                        "reason": "insufficient post-candidate data in calendar window",
                        "met": False}
        else:
            available = len(data) - candidate_t
            if available < min_days:
                return {"valid": False, "reason": "insufficient post-candidate data",
                        "met": False}
            post = data[candidate_t:candidate_t + min_days]

        half = len(post) // 2
        first_half_var = float(np.var(post[:half]))
        second_half_var = float(np.var(post[half:]))

        decreasing = second_half_var <= (first_half_var * drop_ratio)

        return {
            "valid": True,
            "met": bool(decreasing),
            "first_half_var": first_half_var,
            "second_half_var": second_half_var,
            "reset": not decreasing,
        }

    def is_stabilizing_entropy(self, regime_probabilities: list,
                                 candidate_t: int,
                                 min_days: int | None = None,
                                 drop_ratio: float = 0.75,
                                 timestamps: list[float] | None = None) -> dict:
        """
        S56.1 METRIC SWAP -- implemented per user request, ships without
        the Mandatory senior sign-off the pack requires for this ID.
        DO NOT MERGE without that review; flag it explicitly in the PR.

        Doctrine-correct condition-3 check: operates on the regime-
        POSTERIOR-PROBABILITY vector per timestep (shape (T, K), K =
        number of regimes -- this is what backbone.hssm.fit_hssm's
        HSSMResult is expected to expose per S34.7/S34.3, NOT raw
        behavioral signal values) rather than raw_variance over `data`.
        Computes Shannon entropy of the posterior at each timestep via
        `regime_posterior_entropy`, splits the post-candidate window in
        half, and requires entropy to DECREASE (posterior concentrating
        on one regime => genuinely more certain/stable), using the same
        split-half / drop_ratio / >=10-samples-per-half / calendar-day
        window machinery as `is_stabilizing`.

        This is NOT validated against real HSSM output -- Sprint 3-4's
        `backbone` package (the only real source of a regime-posterior
        vector) is not part of this delivered zip, so this method is
        exercised only against synthetic known-distribution fixtures
        (see `validate_stability_metric_against_entropy` and
        `tests/test_stability.py`). The raw-variance `is_stabilizing`
        above is left in place, unremoved, for backward compatibility
        and until this entropy path is confirmed against real HSSM
        posteriors.
        """
        min_days = min_days or self.min_days
        regime_probabilities = list(regime_probabilities)
        T = len(regime_probabilities)

        if min_days < 10:
            return {"valid": False, "reason": "insufficient post-candidate data",
                    "met": False}

        entropies = np.array([regime_posterior_entropy(p) for p in regime_probabilities])

        if timestamps is not None:
            timestamps = np.asarray(timestamps, dtype=float)
            if len(timestamps) != T:
                raise ValueError("timestamps must be same length as regime_probabilities")
            t0 = timestamps[candidate_t]
            window_mask = (timestamps >= t0) & (timestamps < t0 + min_days)
            post = entropies[window_mask]
            if len(post) < 10:
                return {"valid": False,
                        "reason": "insufficient post-candidate data in calendar window",
                        "met": False}
        else:
            available = T - candidate_t
            if available < min_days:
                return {"valid": False, "reason": "insufficient post-candidate data",
                        "met": False}
            post = entropies[candidate_t:candidate_t + min_days]

        half = len(post) // 2
        first_half_entropy = float(np.mean(post[:half]))
        second_half_entropy = float(np.mean(post[half:]))

        # Entropy DECREASING (posterior more concentrated / less
        # uncertain) is the "stabilizing" direction -- inverse sense
        # from variance, same drop_ratio semantics: second half must be
        # <= first half * drop_ratio to count as a meaningful drop.
        decreasing = second_half_entropy <= (first_half_entropy * drop_ratio)

        return {
            "valid": True,
            "met": bool(decreasing),
            "first_half_entropy": first_half_entropy,
            "second_half_entropy": second_half_entropy,
            "reset": not decreasing,
            "metric": "regime_posterior_entropy",
        }