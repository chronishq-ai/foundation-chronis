from core.config_loader import load_state_schema


DEFAULT_CONFIDENCE = 0.3
# Used when Pod B forgets to send a confidence score


def validate_signal(
        variable_name,
        signal
):
    """
    Checks ONE variable's signal against the shared schema.

    signal example:
    {
        "value": 8,
        "confidence": 0.9
    }

    Returns:
        is_valid, cleaned_signal, errors
    """

    errors = []

    schema = load_state_schema()
    variables = schema["variables"]

    if variable_name not in variables:
        errors.append(
            f"Unknown variable: {variable_name}"
        )
        return False, None, errors

    if "value" not in signal:
        errors.append(
            f"Missing 'value' for {variable_name}"
        )
        return False, None, errors

    value = signal["value"]
    min_value, max_value = variables[variable_name]["range"]

    if not min_value <= value <= max_value:
        errors.append(
            f"{variable_name} value {value} is out of range "
            f"({min_value}-{max_value})"
        )
        return False, None, errors

    confidence = signal.get(
        "confidence",
        DEFAULT_CONFIDENCE
    )

    if not 0 <= confidence <= 1:
        errors.append(
            f"{variable_name} confidence {confidence} is out of range (0-1)"
        )
        return False, None, errors

    cleaned = {
        "value": value,
        "confidence": confidence
    }

    return True, cleaned, errors


def validate_event_signals(signals):
    """
    Checks a FULL event's signals dict, e.g.:

    {
        "stress": {"value": 8, "confidence": 0.9},
        "focus": {"value": 3}
    }

    Returns:
        clean_signals, all_errors
    """

    clean_signals = {}
    all_errors = []

    for variable_name, signal in signals.items():

        is_valid, cleaned, errors = validate_signal(
            variable_name,
            signal
        )

        if is_valid:
            clean_signals[variable_name] = cleaned
        else:
            all_errors.extend(errors)

    return clean_signals, all_errors