import json
import os


def load_state_schema():

    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    schema_path = os.path.join(
        current_dir,
        "..",
        "config",
        "state_schema.json"
    )


    with open(schema_path, "r") as file:
        return json.load(file)