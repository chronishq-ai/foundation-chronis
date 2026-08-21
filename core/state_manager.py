from core.update_state import update_state

class StateManager:

    def __init__(self, initial_state):
        # store current state of AI agent
        self.current_state = initial_state

    def process_event(self, incoming_event):
        # update current state using incoming event
        self.current_state = update_state(
            self.current_state,
            incoming_event
        )

    def get_current_state(self):
        # return latest state
        return self.current_state
