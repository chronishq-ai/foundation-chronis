from typing import Dict, Any, List


class FeatureStore:

    def __init__(self):
        self.storage = []


    def insert_feature(
        self,
        user_id: str,
        timestamp: str,
        feature_name: str,
        value: float
    ):

        self.storage.append(
            {
                "user_id": user_id,
                "timestamp": timestamp,
                "feature_name": feature_name,
                "value": value
            }
        )


    def get_features(
        self,
        user_id: str
    ) -> List[Dict[str, Any]]:

        return [
            item
            for item in self.storage
            if item["user_id"] == user_id
        ]