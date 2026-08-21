from core.confidence_handler import confidence_weighted_update
from core.config_loader import load_state_schema

def update_state(current_state, incoming_event):

    if not isinstance(current_state, dict) or not isinstance(incoming_event, dict):
        raise TypeError("current_state and incoming_event must be dictionaries")

    # created a copy of current state
    updated_state = current_state.copy()

    # load state schema
    state_schema = load_state_schema()

    # loop through all variables affected by event
    for variable_name, signal in incoming_event.items():

        if variable_name not in state_schema["variables"]:
            raise ValueError(f"Unknown state variable: {variable_name}")
        if variable_name not in current_state:
            raise ValueError(f"Current state is missing variable: {variable_name}")
        if not isinstance(signal, dict):
            raise TypeError(f"Signal for {variable_name} must be a dictionary")

        # getting incoming value (or delta converted to absolute value) and confidence score
        if "value" in signal:
            incoming_value = signal["value"]
        elif "delta" in signal:
            incoming_value = current_state[variable_name] + signal["delta"]
        else:
            continue

        confidence = signal.get("confidence", 0.5)
        if not isinstance(incoming_value, (int, float)) or not isinstance(confidence, (int, float)):
            raise TypeError(f"Signal for {variable_name} must contain numeric value/delta and confidence")

        # get current value of the variable
        old_state = current_state[variable_name]

        # get variable metadata from schema
        variable_speed = state_schema["variables"][variable_name]["speed"]
        min_value = state_schema["variables"][variable_name]["range"][0]
        max_value = state_schema["variables"][variable_name]["range"][1]

        # calculate the updated value
        new_value = confidence_weighted_update(
            old_state,
            incoming_value,
            variable_speed,
            confidence,
            min_value,
            max_value
        )

        # store updated values
        updated_state[variable_name] = new_value

    # return the complete updated state
    return updated_state
