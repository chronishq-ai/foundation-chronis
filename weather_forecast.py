"""
Sprint 11 - Behavioral Weather Forecast.

Weather is an evidence-composition layer.

It consumes already-produced upstream behavioral evidence:
    - m_t
    - p_t
    - optional domain attractor state
    - optional PPG trend
    - optional 7-day social pattern

It does not train a new behavioral model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import sqrt, isfinite
from statistics import mean
from typing import Optional, Sequence


MIN_SESSIONS = 45


class WeatherForecastError(ValueError):
    """Invalid Weather Forecast input."""


@dataclass(frozen=True)
class RegimeState:
    regime_label: int
    regime_posterior: Sequence[float]


@dataclass(frozen=True)
class WeatherInput:
    user_id: str
    timestamp: datetime
    m_t: Sequence[float]
    p_t: RegimeState

    energy: Optional[float] = None
    social_engagement: Optional[float] = None
    stress: Optional[float] = None
    productivity: Optional[float] = None

    # Optional upstream evidence.
    domain_attractor: Optional[float] = None
    ppg_stress: Optional[float] = None
    social_pattern: Optional[float] = None


@dataclass(frozen=True)
class WeatherForecast:
    forecast_date: datetime
    energy_level: str
    energy_confidence_interval: tuple[float, float]
    social_engagement: str
    stress_trajectory: str
    productivity_context: str
    confidence: float
    confidence_qualifier: str
    transition_warning: Optional[str] = None

    # Explicit Weather-style difficult/high-focus flag.
    historical_focus_flag: str = "uncertain"


class WeatherForecastEngine:
    """
    Historical analogue Weather Forecast.

    The engine:
      1. enforces the 45-session gate;
      2. respects the cold-start gate;
      3. validates m_t/p_t;
      4. matches tomorrow's weekday and current regime;
      5. ranks historical analogues using m_t cosine similarity;
      6. incorporates available upstream Weather evidence;
      7. lowers confidence during transition;
      8. never invents missing measurements.
    """

    def __init__(
        self,
        *,
        min_sessions: int = MIN_SESSIONS,
        similarity_candidates: int = 10,
    ) -> None:
        if min_sessions <= 0:
            raise ValueError("min_sessions must be positive")

        if similarity_candidates <= 0:
            raise ValueError("similarity_candidates must be positive")

        self.min_sessions = min_sessions
        self.similarity_candidates = similarity_candidates

    def forecast(
        self,
        *,
        current: WeatherInput,
        history: Sequence[WeatherInput],
        cold_start_can_surface_claims: bool = True,
        transition_in_progress: bool = False,
    ) -> WeatherForecast | str:

        self._validate_record(current)

        if not cold_start_can_surface_claims:
            return (
                "Behavioral Weather Forecast is not available yet because "
                "the current cold-start evidence gate has not passed."
            )

        if len(history) < self.min_sessions:
            return (
                f"Behavioral Weather Forecast requires at least "
                f"{self.min_sessions} historical sessions; "
                f"{len(history)} are currently available."
            )

        # Validate EVERY historical record before filtering. Mixed-user
        # batches are rejected rather than silently discarding evidence.
        for record in history:
            self._validate_record(record)

            if len(record.m_t) != len(current.m_t):
                raise WeatherForecastError(
                    "m_t dimension mismatch between current and historical record"
                )

        if any(record.user_id != current.user_id for record in history):
            raise WeatherForecastError("history contains records for another user")

        user_history = list(history)
        timestamps = [record.timestamp for record in user_history]
        if len(timestamps) != len(set(timestamps)):
            raise WeatherForecastError("history contains duplicate timestamps")

        if len(user_history) < self.min_sessions:
            return (
                f"Behavioral Weather Forecast requires at least "
                f"{self.min_sessions} historical sessions for this user; "
                f"{len(user_history)} are currently available."
            )

        tomorrow = current.timestamp + timedelta(days=1)

        candidates = [
            record
            for record in user_history
            if record.timestamp.date() != current.timestamp.date()
            and record.timestamp.weekday() == tomorrow.weekday()
            and record.p_t.regime_label == current.p_t.regime_label
        ]

        if not candidates:
            return (
                "Behavioral Weather Forecast is unavailable because no "
                "historical analogue matches tomorrow's weekday and "
                "current behavioral regime."
            )

        ranked = sorted(
            (
                (
                    self._cosine_similarity(
                        current.m_t,
                        record.m_t,
                    ),
                    record,
                )
                for record in candidates
            ),
            key=lambda item: item[0],
            reverse=True,
        )[: self.similarity_candidates]

        if not ranked:
            return (
                "Behavioral Weather Forecast is unavailable because "
                "no valid historical analogue could be matched."
            )

        similarities = [score for score, _ in ranked]

        confidence = self._confidence_from_similarity(similarities)

        if transition_in_progress:
            confidence *= 0.5

        # Optional upstream evidence can increase confidence slightly,
        # but never manufacture a forecast when the evidence is absent.
        evidence_values = self._available_evidence(current)

        if evidence_values:
            confidence = min(
                1.0,
                confidence + (0.05 * len(evidence_values)),
            )

        transition_warning = None

        if transition_in_progress:
            transition_warning = (
                "Currently in behavioral transition. Forecast reliability "
                "reduced. Patterns from prior phase may not apply."
            )

        energy_values = self._values(ranked, "energy")
        social_values = self._values(ranked, "social_engagement")
        stress_values = self._values(ranked, "stress")
        productivity_values = self._values(ranked, "productivity")

        # PPG is the preferred historical stress evidence when available.
        ppg_values = self._values(ranked, "ppg_stress")

        if ppg_values:
            stress_values = ppg_values

        # Social pattern is the preferred historical social evidence when
        # available.
        social_pattern_values = self._values(
            ranked,
            "social_pattern",
        )

        if social_pattern_values:
            social_values = social_pattern_values

        return WeatherForecast(
            forecast_date=tomorrow,
            energy_level=self._energy_level(energy_values),
            energy_confidence_interval=self._energy_interval(
                energy_values
            ),
            social_engagement=self._social_level(
                social_values
            ),
            stress_trajectory=self._stress_trajectory(
                ranked,
                stress_values,
            ),
            productivity_context=self._productivity_context(
                productivity_values
            ),
            confidence=confidence,
            confidence_qualifier=self._confidence_qualifier(
                confidence
            ),
            transition_warning=transition_warning,
            historical_focus_flag=self._historical_focus_flag(
                productivity_values
            ),
        )

    @staticmethod
    def _validate_record(record: WeatherInput) -> None:
        if not isinstance(record.timestamp, datetime):
            raise WeatherForecastError(
                "timestamp must be a datetime"
            )

        if not record.user_id:
            raise WeatherForecastError(
                "user_id must not be empty"
            )

        if not record.m_t:
            raise WeatherForecastError(
                "m_t must contain at least one dimension"
            )

        for value in record.m_t:
            if not isinstance(value, (int, float)):
                raise WeatherForecastError(
                    "m_t must contain only numeric values"
                )

            if not isfinite(value):
                raise WeatherForecastError(
                    "m_t must contain only finite numeric values"
                )

        if not isinstance(record.p_t.regime_label, int):
            raise WeatherForecastError(
                "regime_label must be an integer"
            )

        posterior = list(record.p_t.regime_posterior)

        if not posterior:
            raise WeatherForecastError(
                "regime_posterior must not be empty"
            )

        if record.p_t.regime_label < 0:
            raise WeatherForecastError(
                "regime_label must be >= 0"
            )

        if record.p_t.regime_label >= len(posterior):
            raise WeatherForecastError(
                "regime_label must index regime_posterior"
            )

        if any(
            not isinstance(value, (int, float))
            or not isfinite(value)
            for value in posterior
        ):
            raise WeatherForecastError(
                "regime_posterior must contain finite numeric values"
            )

        if any(value < 0 for value in posterior):
            raise WeatherForecastError(
                "regime_posterior cannot contain negative probabilities"
            )

        if abs(sum(posterior) - 1.0) > 1e-6:
            raise WeatherForecastError(
                "regime_posterior must sum to 1"
            )

        optional_fields = (
            record.energy,
            record.social_engagement,
            record.stress,
            record.productivity,
            record.domain_attractor,
            record.ppg_stress,
            record.social_pattern,
        )

        for value in optional_fields:
            if value is not None:
                if not isinstance(value, (int, float)):
                    raise WeatherForecastError(
                        "Weather measurements must be numeric or None"
                    )

                if not isfinite(value):
                    raise WeatherForecastError(
                        "Weather measurements must be finite or None"
                    )

    @staticmethod
    def _available_evidence(
        record: WeatherInput,
    ) -> list[float]:
        return [
            value
            for value in (
                record.domain_attractor,
                record.ppg_stress,
                record.social_pattern,
            )
            if value is not None
        ]

    @staticmethod
    def _cosine_similarity(
        a: Sequence[float],
        b: Sequence[float],
    ) -> float:

        if len(a) != len(b):
            raise WeatherForecastError(
                "m_t dimension mismatch between current and historical record"
            )

        dot = sum(x * y for x, y in zip(a, b))

        norm_a = sqrt(
            sum(x * x for x in a)
        )

        norm_b = sqrt(
            sum(y * y for y in b)
        )

        if norm_a == 0 or norm_b == 0:
            raise WeatherForecastError(
                "cosine similarity is undefined for a zero-vector m_t"
            )

        return dot / (norm_a * norm_b)

    @staticmethod
    def _values(
        ranked: Sequence[tuple[float, WeatherInput]],
        field: str,
    ) -> list[float]:

        values = []

        for _, record in ranked:
            value = getattr(record, field)

            if value is not None:
                values.append(float(value))

        return values

    @staticmethod
    def _energy_level(
        values: Sequence[float],
    ) -> str:

        if not values:
            return "uncertain"

        value = mean(values)

        if value < 0.33:
            return "low"

        if value < 0.67:
            return "moderate"

        return "high"

    @staticmethod
    def _social_level(
        values: Sequence[float],
    ) -> str:

        if not values:
            return "uncertain"

        value = mean(values)

        if value < 0.33:
            return "likely solo"

        if value >= 0.67:
            return "likely social"

        return "uncertain"

    @staticmethod
    def _stress_trajectory(
        ranked: Sequence[tuple[float, WeatherInput]],
        values: Sequence[float],
    ) -> str:

        if len(values) < 2:
            return "uncertain"

        chronological = sorted(
            (
                record.timestamp,
                record.stress
                if record.stress is not None
                else record.ppg_stress,
            )
            for _, record in ranked
            if (
                record.stress is not None
                or record.ppg_stress is not None
            )
        )

        if len(chronological) < 2:
            return "uncertain"

        midpoint = len(chronological) // 2

        first_half = [
            value
            for _, value in chronological[:midpoint]
            if value is not None
        ]

        second_half = [
            value
            for _, value in chronological[midpoint:]
            if value is not None
        ]

        if not first_half or not second_half:
            return "uncertain"

        difference = (
            mean(second_half)
            - mean(first_half)
        )

        if difference > 0.05:
            return "trending up"

        if difference < -0.05:
            return "trending down"

        return "stable"

    @staticmethod
    def _productivity_context(
        values: Sequence[float],
    ) -> str:

        if not values:
            return "uncertain"

        value = mean(values)

        if value >= 0.67:
            return "focused"

        if value < 0.33:
            return "fragmented"

        return "transitional"

    @staticmethod
    def _historical_focus_flag(
        productivity_values: Sequence[float],
    ) -> str:

        if not productivity_values:
            return "uncertain"

        value = mean(productivity_values)

        if value >= 0.67:
            return "historically high-focus"

        if value < 0.33:
            return "historically difficult"

        return "historically mixed"

    @staticmethod
    def _energy_interval(
        values: Sequence[float],
    ) -> tuple[float, float]:

        if not values:
            return (0.0, 1.0)

        average = mean(values)

        if len(values) == 1:
            spread = 0.25
        else:
            spread = sqrt(
                mean(
                    (value - average) ** 2
                    for value in values
                )
            )

        return (
            max(0.0, average - spread),
            min(1.0, average + spread),
        )

    @staticmethod
    def _confidence_from_similarity(
        similarities: Sequence[float],
    ) -> float:

        if not similarities:
            return 0.0

        normalized = [
            max(
                0.0,
                min(
                    1.0,
                    (value + 1.0) / 2.0,
                ),
            )
            for value in similarities
        ]

        return mean(normalized)

    @staticmethod
    def _confidence_qualifier(
        confidence: float,
    ) -> str:

        if confidence >= 0.75:
            return "high confidence"

        if confidence >= 0.50:
            return "moderate confidence"

        return "low confidence"