import os
from datetime import datetime
from typing import Dict, Any, List

import psycopg
from dotenv import load_dotenv


load_dotenv()


class FeatureStore:
    """
    PostgreSQL-backed feature storage interface for Chronis.

    The public interface remains compatible with the existing
    FeatureStore API used by the project tests.
    """

    def __init__(self):
        self.host = os.getenv(
            "CHRONIS_DB_HOST",
            "localhost"
        )

        self.port = int(
            os.getenv(
                "CHRONIS_DB_PORT",
                "1305"
            )
        )

        self.database = os.getenv(
            "CHRONIS_DB_NAME",
            "chronis"
        )

        self.user = os.getenv(
            "CHRONIS_DB_USER",
            "postgres"
        )

        self.password = os.getenv(
            "CHRONIS_DB_PASSWORD"
        )

        if not self.password:
            raise ValueError(
                "CHRONIS_DB_PASSWORD is not configured"
            )

    def _connect(self):
        """
        Create a PostgreSQL database connection.
        """

        return psycopg.connect(
            host=self.host,
            port=self.port,
            dbname=self.database,
            user=self.user,
            password=self.password
        )

    @staticmethod
    def _validate_timestamp(timestamp: str) -> None:
        """
        Validate that the timestamp is ISO-8601 compatible.
        """

        try:
            datetime.fromisoformat(timestamp)

        except ValueError as exc:

            raise ValueError(
                f"Invalid timestamp: {timestamp}"
            ) from exc

    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        """
        Validate user ID.
        """

        if not user_id:
            raise ValueError(
                "user_id cannot be empty"
            )

    @staticmethod
    def _validate_feature_name(feature_name: str) -> None:
        """
        Validate feature name.
        """

        if not feature_name:
            raise ValueError(
                "feature_name cannot be empty"
            )

    @staticmethod
    def _validate_value(value: float) -> None:
        """
        Validate feature value.
        """

        if value is not None and not isinstance(
            value,
            (int, float)
        ):
            raise TypeError(
                "Feature value must be numeric or None"
            )

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        """
        Convert a PostgreSQL row into the public FeatureStore format.
        """

        return {
            "user_id": row[0],
            "timestamp": row[1].isoformat(),
            "feature_name": row[2],
            "value": row[3]
        }

    def insert_feature(
        self,
        user_id: str,
        timestamp: str,
        feature_name: str,
        value: float
    ) -> None:
        """
        Insert one processed feature into PostgreSQL.
        """

        self._validate_user_id(user_id)

        self._validate_feature_name(
            feature_name
        )

        self._validate_timestamp(
            timestamp
        )

        self._validate_value(
            value
        )

        query = """
        INSERT INTO features (
            user_id,
            timestamp,
            feature_name,
            feature_value
        )
        VALUES (%s, %s, %s, %s);
        """

        with self._connect() as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    query,
                    (
                        user_id,
                        timestamp,
                        feature_name,
                        value
                    )
                )

            connection.commit()

    def get_features(
        self,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all features belonging to a user.
        """

        self._validate_user_id(
            user_id
        )

        query = """
        SELECT
            user_id,
            timestamp,
            feature_name,
            feature_value
        FROM features
        WHERE user_id = %s
        ORDER BY timestamp ASC;
        """

        with self._connect() as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    query,
                    (user_id,)
                )

                rows = cursor.fetchall()

        return [
            self._row_to_dict(row)
            for row in rows
        ]

    def get_features_by_time_range(
        self,
        user_id: str,
        start_time: str,
        end_time: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve a user's features within an inclusive time range.
        """

        self._validate_user_id(
            user_id
        )

        self._validate_timestamp(
            start_time
        )

        self._validate_timestamp(
            end_time
        )

        start = datetime.fromisoformat(
            start_time
        )

        end = datetime.fromisoformat(
            end_time
        )

        if start > end:

            raise ValueError(
                "start_time cannot be after end_time"
            )

        query = """
        SELECT
            user_id,
            timestamp,
            feature_name,
            feature_value
        FROM features
        WHERE user_id = %s
          AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp ASC;
        """

        with self._connect() as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    query,
                    (
                        user_id,
                        start_time,
                        end_time
                    )
                )

                rows = cursor.fetchall()

        return [
            self._row_to_dict(row)
            for row in rows
        ]

    def delete_features(
        self,
        user_id: str
    ) -> None:
        """
        Delete all stored features belonging to a user.

        This is primarily used for test cleanup and controlled
        user-level data removal.
        """

        self._validate_user_id(
            user_id
        )

        query = """
        DELETE FROM features
        WHERE user_id = %s;
        """

        with self._connect() as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    query,
                    (user_id,)
                )

            connection.commit()