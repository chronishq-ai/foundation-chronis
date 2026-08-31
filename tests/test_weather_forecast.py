from datetime import datetime, timedelta

import pytest

from weather_forecast import (
    MIN_SESSIONS,
    RegimeState,
    WeatherForecastEngine,
    WeatherForecastError,
    WeatherInput,
    WeatherForecast,
)


def regime(label=0, posterior=None):
    if posterior is None:
        posterior = [1.0]

    return RegimeState(
        regime_label=label,
        regime_posterior=posterior,
    )


def make_record(
    *,
    user_id="user_001",
    day=1,
    hour=12,
    m_t=None,
    regime_label=0,
    posterior=None,
    energy=0.7,
    social=0.7,
    stress=0.4,
    productivity=0.7,
):
    if m_t is None:
        m_t = [1.0, 0.0, 0.0]

    return WeatherInput(
        user_id=user_id,
        timestamp=datetime(2026, 8, day, hour, 0, 0),
        m_t=m_t,
        p_t=regime(regime_label, posterior),
        energy=energy,
        social_engagement=social,
        stress=stress,
        productivity=productivity,
    )


def make_history(count=MIN_SESSIONS, user_id="user_001"):
    records = []

    base = datetime(2026, 1, 1, 12, 0)

    for index in range(count):
        records.append(
            WeatherInput(
                user_id=user_id,
                timestamp=base + timedelta(days=index),
                m_t=[1.0, 0.0, 0.0],
                p_t=regime(0, [1.0]),
                energy=0.7,
                social_engagement=0.7,
                stress=0.4,
                productivity=0.7,
            )
        )

    return records


def test_weather_requires_45_sessions():
    engine = WeatherForecastEngine()

    current = make_record(day=24)
    history = make_history(MIN_SESSIONS - 1)

    result = engine.forecast(
        current=current,
        history=history,
    )

    assert isinstance(result, str)
    assert "45" in result


def test_weather_can_run_at_45_sessions():
    engine = WeatherForecastEngine()

    current = make_record(day=24)

    # Make sure tomorrow's weekday has historical analogues.
    history = make_history(MIN_SESSIONS)

    tomorrow = current.timestamp + timedelta(days=1)

    history[0] = make_record(
        day=3,
        m_t=[1.0, 0.0, 0.0],
    )

    # Find a date in August 2026 having tomorrow's weekday.
    for day in range(1, 25):
        candidate = make_record(
            day=day,
            m_t=[1.0, 0.0, 0.0],
        )

        if candidate.timestamp.weekday() == tomorrow.weekday():
            history[0] = candidate
            break

    result = engine.forecast(
        current=current,
        history=history,
    )

    assert isinstance(result, WeatherForecast)


def test_weather_respects_cold_start_gate():
    engine = WeatherForecastEngine()

    current = make_record(day=24)
    history = make_history(MIN_SESSIONS)

    result = engine.forecast(
        current=current,
        history=history,
        cold_start_can_surface_claims=False,
    )

    assert isinstance(result, str)
    assert "cold-start" in result.lower()


def test_weather_requires_same_weekday_and_regime():
    engine = WeatherForecastEngine()

    current = make_record(
        day=24,
        regime_label=1,
        posterior=[0.1, 0.9],
    )

    history = make_history(MIN_SESSIONS)

    # Deliberately make every record the wrong regime.
    history = [
        WeatherInput(
            user_id="user_001",
            timestamp=datetime(2026, 1, 1) + timedelta(days=index),
            m_t=[1.0, 0.0, 0.0],
            p_t=regime(0, [1.0]),
            energy=0.7, social_engagement=0.7, stress=0.4, productivity=0.7,
        )
        for index in range(MIN_SESSIONS)
    ]

    result = engine.forecast(
        current=current,
        history=history,
    )

    assert isinstance(result, str)
    assert "analogue" in result.lower()


def test_weather_uses_cosine_similarity():
    engine = WeatherForecastEngine(
        similarity_candidates=1,
    )

    current = make_record(
        day=24,
        m_t=[1.0, 0.0, 0.0],
    )

    history = make_history(MIN_SESSIONS)

    tomorrow = current.timestamp + timedelta(days=1)

    matching_day = None

    for day in range(1, 29):
        candidate = make_record(
            day=day,
            m_t=[1.0, 0.0, 0.0],
        )

        if candidate.timestamp.weekday() == tomorrow.weekday():
            matching_day = day
            break

    assert matching_day is not None

    history[0] = WeatherInput(
        user_id="user_001", timestamp=datetime(2026, 8, matching_day, 12), m_t=[1.0,0.0,0.0],
        p_t=regime(0,[1.0]), energy=0.9, social_engagement=0.9, stress=0.1, productivity=0.9
    )
    history[1] = WeatherInput(
        user_id="user_001", timestamp=datetime(2026, 8, matching_day, 13), m_t=[0.0,1.0,0.0],
        p_t=regime(0,[1.0]), energy=0.1, social_engagement=0.1, stress=0.9, productivity=0.1
    )

    result = engine.forecast(
        current=current,
        history=history,
    )

    assert isinstance(result, WeatherForecast)
    assert result.energy_level == "high"


def test_weather_reduces_confidence_during_transition():
    engine = WeatherForecastEngine()

    current = make_record(day=24)
    history = make_history(MIN_SESSIONS)

    stable = engine.forecast(
        current=current,
        history=history,
        transition_in_progress=False,
    )

    transitional = engine.forecast(
        current=current,
        history=history,
        transition_in_progress=True,
    )

    assert isinstance(stable, WeatherForecast)
    assert isinstance(transitional, WeatherForecast)

    assert transitional.confidence < stable.confidence

    assert transitional.transition_warning is not None
    assert "transition" in transitional.transition_warning.lower()


def test_weather_output_contains_all_four_dimensions():
    engine = WeatherForecastEngine()

    current = make_record(day=24)
    history = make_history(MIN_SESSIONS)

    result = engine.forecast(
        current=current,
        history=history,
    )

    assert isinstance(result, WeatherForecast)

    assert result.energy_level in {
        "low",
        "moderate",
        "high",
        "uncertain",
    }

    assert result.social_engagement in {
        "likely solo",
        "likely social",
        "uncertain",
    }

    assert result.stress_trajectory in {
        "trending up",
        "stable",
        "trending down",
        "uncertain",
    }

    assert result.productivity_context in {
        "focused",
        "fragmented",
        "transitional",
        "uncertain",
    }


def test_weather_confidence_is_bounded():
    engine = WeatherForecastEngine()

    current = make_record(day=24)
    history = make_history(MIN_SESSIONS)

    result = engine.forecast(
        current=current,
        history=history,
    )

    assert isinstance(result, WeatherForecast)

    assert 0.0 <= result.confidence <= 1.0

    lower, upper = result.energy_confidence_interval

    assert 0.0 <= lower <= 1.0
    assert 0.0 <= upper <= 1.0
    assert lower <= upper


def test_weather_rejects_mismatched_m_t_dimensions():
    engine = WeatherForecastEngine()

    current = make_record(
        day=24,
        m_t=[1.0, 0.0, 0.0],
    )

    history = make_history(MIN_SESSIONS)

    # Deliberately corrupt one historical record.
    history[0] = WeatherInput(
        user_id="user_001",
        timestamp=history[0].timestamp,
        m_t=[1.0, 0.0],
        p_t=regime(0, [1.0]),
        energy=0.7,
        social_engagement=0.7,
        stress=0.4,
        productivity=0.7,
    )

    with pytest.raises(
        WeatherForecastError,
        match="m_t dimension mismatch",
    ):
        engine.forecast(
            current=current,
            history=history,
        )


def test_weather_rejects_invalid_regime_posterior():
    engine = WeatherForecastEngine()

    current = WeatherInput(
        user_id="user_001",
        timestamp=datetime(2026, 8, 24, 12, 0),
        m_t=[1.0, 0.0, 0.0],
        p_t=RegimeState(
            regime_label=0,
            regime_posterior=[0.8, 0.8],
        ),
    )

    with pytest.raises(WeatherForecastError):
        engine.forecast(
            current=current,
            history=[],
        )


def test_weather_rejects_non_finite_m_t():
    engine = WeatherForecastEngine()

    current = make_record(
        day=24,
        m_t=[1.0, float("nan"), 0.0],
    )

    with pytest.raises(WeatherForecastError):
        engine.forecast(
            current=current,
            history=[],
        )


def test_weather_rejects_zero_vector_m_t():
    engine = WeatherForecastEngine()

    current = make_record(
        day=24,
        m_t=[0.0, 0.0, 0.0],
    )

    history = make_history(MIN_SESSIONS)

    with pytest.raises(WeatherForecastError):
        engine.forecast(
            current=current,
            history=history,
        )


def test_weather_does_not_mix_users():
    engine = WeatherForecastEngine()

    current = make_record(
        user_id="user_A",
        day=24,
    )

    history = [
        make_record(
            user_id="user_B",
            day=((index % 28) + 1),
        )
        for index in range(MIN_SESSIONS)
    ]

    with pytest.raises(WeatherForecastError):
        engine.forecast(
            current=current,
            history=history,
        )

    assert True

def test_weather_accepts_optional_v2_evidence():
    engine = WeatherForecastEngine()

    current = make_record(day=24)

    current = WeatherInput(
        user_id=current.user_id,
        timestamp=current.timestamp,
        m_t=current.m_t,
        p_t=current.p_t,
        energy=current.energy,
        social_engagement=current.social_engagement,
        stress=current.stress,
        productivity=current.productivity,
        domain_attractor=0.9,
        ppg_stress=0.6,
        social_pattern=0.8,
    )

    history = make_history(MIN_SESSIONS)

    result = engine.forecast(
        current=current,
        history=history,
    )

    assert isinstance(result, WeatherForecast)


def test_weather_rejects_non_finite_v2_evidence():
    engine = WeatherForecastEngine()

    current = make_record(day=24)

    current = WeatherInput(
        user_id=current.user_id,
        timestamp=current.timestamp,
        m_t=current.m_t,
        p_t=current.p_t,
        domain_attractor=float("nan"),
    )

    with pytest.raises(WeatherForecastError):
        engine.forecast(
            current=current,
            history=[],
        )


def test_weather_exposes_historical_focus_flag():
    engine = WeatherForecastEngine()

    current = make_record(day=24)

    history = make_history(MIN_SESSIONS)

    result = engine.forecast(
        current=current,
        history=history,
    )

    assert isinstance(result, WeatherForecast)

    assert result.historical_focus_flag in {
        "historically high-focus",
        "historically difficult",
        "historically mixed",
        "uncertain",
    }


def test_weather_ppg_can_drive_stress_evidence():
    engine = WeatherForecastEngine()

    current = make_record(day=24)

    history = make_history(MIN_SESSIONS)

    for index, record in enumerate(history):
        history[index] = WeatherInput(
            user_id=record.user_id,
            timestamp=record.timestamp,
            m_t=record.m_t,
            p_t=record.p_t,
            energy=record.energy,
            social_engagement=record.social_engagement,
            stress=None,
            productivity=record.productivity,
            ppg_stress=0.8 if index >= 23 else 0.2,
        )

    result = engine.forecast(
        current=current,
        history=history,
    )

    assert isinstance(result, WeatherForecast)


def test_weather_social_pattern_can_drive_social_forecast():
    engine = WeatherForecastEngine()

    current = make_record(day=24)

    history = make_history(MIN_SESSIONS)

    for index, record in enumerate(history):
        history[index] = WeatherInput(
            user_id=record.user_id,
            timestamp=record.timestamp,
            m_t=record.m_t,
            p_t=record.p_t,
            energy=record.energy,
            social_engagement=None,
            stress=record.stress,
            productivity=record.productivity,
            social_pattern=0.9,
        )

    result = engine.forecast(
        current=current,
        history=history,
    )

    assert isinstance(result, WeatherForecast)
    assert result.social_engagement == "likely social"


def test_weather_transition_warning_is_exact_required_copy():
    engine = WeatherForecastEngine()

    current = make_record(day=24)
    history = make_history(MIN_SESSIONS)

    result = engine.forecast(
        current=current,
        history=history,
        transition_in_progress=True,
    )

    assert isinstance(result, WeatherForecast)

    assert result.transition_warning == (
        "Currently in behavioral transition. Forecast reliability "
        "reduced. Patterns from prior phase may not apply."
    )
def test_weather_rejects_mixed_user_history():
    engine = WeatherForecastEngine()
    current = make_record(day=24)
    history = make_history(MIN_SESSIONS)
    history[-1] = make_record(user_id="other", day=23)
    with pytest.raises(WeatherForecastError):
        engine.forecast(current=current, history=history)

def test_weather_flags_high_and_low_focus_analogues():
    engine = WeatherForecastEngine(similarity_candidates=1)
    current = make_record(day=24, m_t=[1.0,0.0,0.0])
    history = make_history(MIN_SESSIONS)
    tomorrow = current.timestamp + timedelta(days=1)
    matching = next(day for day in range(1,29) if make_record(day=day).timestamp.weekday() == tomorrow.weekday())
    history[0] = make_record(day=matching, productivity=0.9)
    result = engine.forecast(current=current, history=history)
    assert result.historical_focus_flag == "historically high-focus"
    history[0] = make_record(day=matching, productivity=0.1)
    result = engine.forecast(current=current, history=history)
    assert result.historical_focus_flag == "historically difficult"


def test_weather_cosine_similarity_matches_hand_computed_value():
    similarity = WeatherForecastEngine._cosine_similarity(
        [1.0, 1.0],
        [1.0, 0.0],
    )
    assert similarity == pytest.approx(1.0 / (2.0 ** 0.5))


def test_weather_confidence_matches_hand_computed_similarity_average():
    confidence = WeatherForecastEngine._confidence_from_similarity([1.0, 0.0, -1.0])
    assert confidence == pytest.approx((1.0 + 0.5 + 0.0) / 3.0)
