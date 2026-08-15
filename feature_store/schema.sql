CREATE TABLE IF NOT EXISTS features (

    id SERIAL PRIMARY KEY,

    user_id VARCHAR(100) NOT NULL,

    timestamp TIMESTAMP NOT NULL,

    feature_name VARCHAR(100) NOT NULL,

    feature_value FLOAT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- TimescaleDB conversion (production)
-- SELECT create_hypertable(
--     'features',
--     'timestamp'
-- );