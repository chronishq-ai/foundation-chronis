from datetime import datetime, timedelta

from silence_map import SilenceInput, SilenceMap
from social_graph import SocialGraph, VocalFingerprint
from weather_forecast import (
    RegimeState,
    WeatherForecastEngine,
    WeatherInput,
)


def test_sprint11_shared_surrogate_profile():
    user_id = "shared_surrogate"

    silence = SilenceMap().classify(
        SilenceInput(
            user_id=user_id,
            silence_seconds=8.0,
            turn_expected=False,
            typing_active=True,
            physiological_delta=0.1,
        )
    )

    assert silence.classification == "attentive"

    social = SocialGraph().build(
        user_id,
        [
            VocalFingerprint(
                user_id=user_id,
                session_id="session_1",
                values=[1.0, 0.0, 0.0],
            ),
            VocalFingerprint(
                user_id=user_id,
                session_id="session_2",
                values=[0.99, 0.05, 0.0],
            ),
        ],
    )

    assert social.user_id == user_id
    assert len(social.nodes) == 1

    current = WeatherInput(
        user_id=user_id,
        timestamp=datetime(2026, 8, 24, 12, 0),
        m_t=[1.0, 0.0, 0.0],
        p_t=RegimeState(
            regime_label=0,
            regime_posterior=[1.0],
        ),
        energy=0.7,
        social_engagement=0.7,
        stress=0.4,
        productivity=0.7,
    )

    history = [
        WeatherInput(
            user_id=user_id,
            timestamp=datetime(2026, 1, 1) + timedelta(days=index),
            m_t=[1.0, 0.0, 0.0],
            p_t=RegimeState(
                regime_label=0,
                regime_posterior=[1.0],
            ),
            energy=0.7,
            social_engagement=0.7,
            stress=0.4,
            productivity=0.7,
        )
        for index in range(45)
    ]

    weather = WeatherForecastEngine()

    result = weather.forecast(
        current=current,
        history=history,
    )

    assert result is not None