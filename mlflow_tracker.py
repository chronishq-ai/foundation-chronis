from datetime import datetime
from typing import Dict, Any, List

import mlflow


class MLflowTracker:
    """
    MLflow-based experiment tracker for the Chronis ML pipeline.
    """

    def __init__(self, experiment_name: str = "chronis-foundation"):
        self.experiment_name = experiment_name

        mlflow.set_experiment(self.experiment_name)

        # Keep a local record as well so the existing interface
        # remains easy to test.
        self.experiments: List[Dict[str, Any]] = []

    def log_experiment(
        self,
        name: str,
        dataset_hash: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> None:
        """
        Log an experiment to MLflow.

        Tracks:
        - Experiment/run name
        - Dataset hash
        - Parameters
        - Metrics
        - Timestamp
        """

        timestamp = datetime.now().isoformat()

        with mlflow.start_run(run_name=name):

            # Dataset/version information
            mlflow.set_tag(
                "dataset_hash",
                dataset_hash
            )

            # Experiment timestamp
            mlflow.set_tag(
                "experiment_timestamp",
                timestamp
            )

            # Hyperparameters / configuration
            if parameters:
                mlflow.log_params(parameters)

            # Evaluation metrics
            if metrics:
                mlflow.log_metrics(metrics)

            experiment = {
                "name": name,
                "dataset_hash": dataset_hash,
                "parameters": parameters,
                "metrics": metrics,
                "timestamp": timestamp
            }

            self.experiments.append(experiment)

    def get_experiments(self) -> List[Dict[str, Any]]:
        """
        Return experiments logged through this tracker instance.
        """

        return self.experiments