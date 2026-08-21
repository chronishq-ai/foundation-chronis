import numpy as np


class RegimeStability:
    """
    Condition 3 of 3. Monitors regime posterior variance >=min_days
    post-candidate. Splits post-candidate window in half, compares raw
    variance. Needs enough samples per half (>=10) for the estimate to
    not be dominated by small-sample noise.
    """

    def __init__(self, min_days: int = 20):
        self.min_days = min_days

    def is_stabilizing(self, data: list[float], candidate_t: int,
                         min_days: int | None = None,
                         drop_ratio: float = 0.75) -> dict:
        """
        drop_ratio: second-half variance must be <= first-half * drop_ratio
        to count as 'decreasing' (meaningful drop, not noise-level wobble).
        """
        min_days = min_days or self.min_days
        data = np.asarray(data)
        available = len(data) - candidate_t

        if available < min_days or min_days < 10:
            return {"valid": False, "reason": "insufficient post-candidate data",
                    "met": False}

        post = data[candidate_t:candidate_t + min_days]
        half = min_days // 2
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