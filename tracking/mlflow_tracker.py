from datetime import datetime
from typing import Dict, Any


class MLflowTracker:

    def __init__(self):
        self.experiments = []


    def log_experiment(
        self,
        name: str,
        dataset_hash: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float]
    ):

        experiment = {

            "name": name,

            "dataset_hash": dataset_hash,

            "parameters": parameters,

            "metrics": metrics,

            "timestamp": datetime.now().isoformat()

        }

        self.experiments.append(experiment)


    def get_experiments(self):

        return self.experiments