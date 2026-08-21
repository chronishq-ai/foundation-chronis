def calculate_effective_speed(
        variable_speed,
        confidence_score
):
    """
    Calculates how much influence an incoming signal should have.

    effective_speed = variable_speed * confidence_score

    Example:
    speed = 0.8
    confidence = 0.5

    effective_speed = 0.4
    """

    if not 0 <= variable_speed <= 1:
        raise ValueError(
            "Variable speed must be between 0 and 1"
        )

    if not 0 <= confidence_score <= 1:
        raise ValueError(
            "Confidence score must be between 0 and 1"
        )

    return variable_speed * confidence_score



def confidence_weighted_update(
        current_value: float,
        evidence_value: float,
        variable_speed: float,
        confidence_score: float,
        min_value: float = 0,
        max_value: float = 10
) -> float:
    """
    Updates a state variable using confidence-weighted transition.

    Formula:

    new_state =
    current_state +
    effective_speed * (evidence - current_state)


    Parameters:
        current_value:
            Current state value

        evidence_value:
            New suggested value from event signal

        variable_speed:
            How quickly this variable changes

        confidence_score:
            Reliability of incoming signal

        min_value:
            Minimum allowed state value

        max_value:
            Maximum allowed state value
    """


    # Validate state range

    if not min_value <= current_value <= max_value:
        raise ValueError(
            f"Current value must be between {min_value} and {max_value}"
        )


    if not min_value <= evidence_value <= max_value:
        raise ValueError(
            f"Evidence value must be between {min_value} and {max_value}"
        )


    # Calculate confidence adjusted speed

    effective_speed = calculate_effective_speed(
        variable_speed,
        confidence_score
    )


    # State transition

    new_value = (
        current_value +
        effective_speed *
        (evidence_value - current_value)
    )


    # Keep value within allowed range

    new_value = max(
        min_value,
        min(max_value, new_value)
    )


    return round(new_value, 2)
