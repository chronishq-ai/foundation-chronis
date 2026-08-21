from core.update_state import update_state
from core.config_loader import load_state_schema


class StateManager:

    def __init__(self, initial_state):

        # Load shared state schema
        state_schema = load_state_schema()

        # Initialize state with value + spread representation
        self.current_state = {}

        for variable_name, value in initial_state.items():

            variable_info = state_schema["variables"][variable_name]

            self.current_state[variable_name] = {
                "value": value,
                "spread": variable_info["initial_spread"]
            }

    def process_event(self, incoming_event):

        # Update current state using incoming event
        self.current_state = update_state(
            self.current_state,
            incoming_event
        )

    def get_current_state(self):

        # Return latest state
        return self.current_state