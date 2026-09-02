"""
SQL queries used by the Chronis Feature Store.

These queries are kept separate from the storage implementation so that
the FeatureStore can later use a PostgreSQL/TimescaleDB backend without
mixing SQL with application logic.
"""

INSERT_FEATURE = """
INSERT INTO features (
    user_id,
    timestamp,
    feature_name,
    feature_value
)
VALUES (%s, %s, %s, %s);
"""


GET_FEATURES_BY_USER = """
SELECT
    user_id,
    timestamp,
    feature_name,
    feature_value
FROM features
WHERE user_id = %s
ORDER BY timestamp ASC;
"""


GET_FEATURES_BY_TIME_RANGE = """
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


CREATE_FEATURES_TABLE = """
CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


CREATE_FEATURES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_features_user_timestamp
ON features (user_id, timestamp);
"""


CREATE_HYPERTABLE = """
SELECT create_hypertable(
    'features',
    'timestamp',
    if_not_exists => TRUE
);
"""