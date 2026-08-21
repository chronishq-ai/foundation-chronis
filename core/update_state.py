from core.confidence_handler import confidence_weighted_update
from core.config_loader import load_state_schema
from core.spread_handler import update_spread

def update_state(current_state, incoming_event):

    # Create a copy of the current state
    updated_state = current_state.copy()

    # Load the shared state schema
    state_schema = load_state_schema()

    # Loop through all variables affected by the event
    for variable_name in incoming_event:

        # Validate variable exists
        if variable_name not in current_state:
            raise ValueError(f"Unknown state variable: {variable_name}")

        # Get incoming value and confidence score
        incoming_value = incoming_event[variable_name]["value"]
        confidence = incoming_event[variable_name]["confidence"]

        # Get current value and spread
        current_value = current_state[variable_name]["value"]
        current_spread = current_state[variable_name]["spread"]

        # Get variable metadata from schema
        variable_info = state_schema["variables"][variable_name]

        variable_speed = variable_info["speed"]
        min_value, max_value = variable_info["range"]

        # Update the state value using existing v0.1 logic
        new_value = confidence_weighted_update(
            current_value,
            incoming_value,
            variable_speed,
            confidence,
            min_value,
            max_value
        )

        # Update uncertainty spread using v0.2 logic
        new_spread = update_spread(
            current_value,
            incoming_value,
            current_spread,
            confidence
        )

        # Store both value and spread
        updated_state[variable_name] = {
            "value": new_value,
            "spread": new_spread
        }

    # Return the complete updated state
    return updated_state