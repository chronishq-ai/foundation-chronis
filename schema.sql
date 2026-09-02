-- Chronis Feature Store
-- PostgreSQL / TimescaleDB schema

CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,

    user_id VARCHAR(100) NOT NULL,

    timestamp TIMESTAMP NOT NULL,

    feature_name VARCHAR(100) NOT NULL,

    feature_value DOUBLE PRECISION,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Index for efficient user + time-range queries

CREATE INDEX IF NOT EXISTS idx_features_user_timestamp
ON features (user_id, timestamp);


-- TimescaleDB hypertable
--
-- Run this statement when the TimescaleDB extension
-- is installed and enabled on the PostgreSQL database.
--
-- SELECT create_hypertable(
--     'features',
--     'timestamp',
--     if_not_exists => TRUE
-- );


-- Optional continuous aggregate foundation
--
-- This should only be enabled when the production
-- aggregation requirements are finalized.
--
-- Example:
--
-- CREATE MATERIALIZED VIEW feature_hourly
-- WITH (timescaledb.continuous) AS
-- SELECT
--     user_id,
--     feature_name,
--     time_bucket('1 hour', timestamp) AS bucket,
--     AVG(feature_value) AS average_value
-- FROM features
-- GROUP BY user_id, feature_name, bucket;