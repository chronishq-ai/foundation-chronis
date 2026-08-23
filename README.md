# HARDENERS (this branch)
Sprint 13: per-user model isolation  
Sprint 14: policy engine + gated ML I/O + TILES e2e  
Sprint 15: observer-effect flags — mitigates MP-13, does **not** close it  
Observer-effect writeup lives in `observer_effect/README.md`, not this file.
---
# Chronis Foundation
## Overview
Chronis Foundation is the core foundation layer for the Chronis system.
The project provides a structured pipeline for processing raw user and sensor data into cleaned, normalized, extracted, temporally aligned, and stored features.
The current implementation focuses on building a reliable foundation for:
* Data preprocessing
* Feature extraction
* Temporal feature alignment
* PostgreSQL-based feature storage
* ML experiment tracking
* End-to-end pipeline validation
* Automated testing
The architecture is designed to evolve toward a larger production-oriented feature and intelligence platform.
---
## Architecture
The current processing pipeline follows this flow:
```text
Raw Input
    |
    v
+----------------------+
| Data Cleaning        |
| preprocessing/       |
+----------------------+
    |
    v
+----------------------+
| Feature Normalization|
| preprocessing/       |
+----------------------+
    |
    v
+----------------------+
| Feature Extraction   |
| features/            |
+----------------------+
    |
    v
+----------------------+
| Temporal Alignment   |
| alignment/           |
+----------------------+
    |
    v
+----------------------+
| Feature Store        |
| PostgreSQL           |
+----------------------+
    |
    v
+----------------------+
| ML Experiment        |
| Tracking             |
+----------------------+
Repository Structure
chronis-foundation/
│
├── alignment/
│   └── temporal_alignment.py
│
├── feature_store/
│   ├── database.py
│   ├── queries.py
│   └── schema.sql
│
├── features/
│   └── feature_pipeline.py
│
├── preprocessing/
│   ├── cleaner.py
│   ├── normalizer.py
│   └── validator.py
│
├── tracking/
│   └── mlflow_tracker.py
│
├── tests/
│   ├── test_alignment.py
│   ├── test_feature_store.py
│   ├── test_features.py
│   ├── test_integration.py
│   ├── test_mlflow.py
│   └── test_preprocessing.py
│
├── requirements.txt
└── README.md
Core Components
1. Data Preprocessing
The preprocessing layer prepares incoming records before they are passed to downstream components.

Responsibilities
Validate incoming records
Clean input data
Handle invalid or missing data
Normalize feature values
Prepare structured data for feature extraction
Files
preprocessing/
├── cleaner.py
├── normalizer.py
└── validator.py
2. Feature Extraction
The feature extraction layer converts processed sensor and input data into useful numerical features.

The current feature pipeline supports inputs such as:

IMU data
Heart-rate measurements
Audio-related information
Example feature categories include:

Movement
Heart Rate
Speech Activity
Audio Energy
Pause Information
The extracted features can then be passed to the temporal alignment layer and eventually stored in the feature store.

File
features/feature_pipeline.py
3. Temporal Alignment
Chronis processes information originating from different sources.

These sources may produce observations at different timestamps or intervals.

The temporal alignment layer provides a common representation so that downstream processing can work with temporally aligned data.

File
alignment/temporal_alignment.py
The current implementation includes automated tests covering temporal alignment behavior.

4. Feature Store
The feature store provides persistent storage for processed Chronis features.

The current backend is:

PostgreSQL
The database schema is designed with future TimescaleDB support in mind.

Feature Table
The primary table is:

features
Column	Type	Description
id
SERIAL
Primary key
user_id
VARCHAR(100)
User identifier
timestamp
TIMESTAMP
Feature timestamp
feature_name
VARCHAR(100)
Name of the feature
feature_value
DOUBLE PRECISION
Numerical feature value
created_at
TIMESTAMP
Record creation timestamp
Database Index
An index is created on:

(user_id, timestamp)
This supports efficient queries involving:

A specific user
Time-range feature retrieval
PostgreSQL Configuration
The application reads database configuration from environment variables.

Example:

CHRONIS_DB_HOST=localhost
CHRONIS_DB_PORT=5432
CHRONIS_DB_NAME=chronis
CHRONIS_DB_USER=postgres
CHRONIS_DB_PASSWORD=<your-password>
The password should be stored locally in .env and must not be committed to Git.

PostgreSQL Setup
1. Install PostgreSQL
Install PostgreSQL locally and make sure the PostgreSQL service is running.

Verify that PostgreSQL is available:

psql --version
2. Create the Database
Connect to PostgreSQL:

psql -h localhost -p 5432 -U postgres
Create the Chronis database:

CREATE DATABASE chronis;
Connect to it:

\c chronis
3. Create the Feature Table
The schema is available in:

feature_store/schema.sql
The core table can be created with:

CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
Create the performance index:

CREATE INDEX IF NOT EXISTS idx_features_user_timestamp
ON features (user_id, timestamp);
TimescaleDB Support
The feature store schema has been prepared with future TimescaleDB support in mind.

The schema provides the foundation for converting the features table into a TimescaleDB hypertable:

SELECT create_hypertable(
    'features',
    'timestamp',
    if_not_exists => TRUE
);
This is optional and requires TimescaleDB to be installed and enabled.

The current implementation works with standard PostgreSQL.

Feature Store API
The FeatureStore class provides the application-level interface to the PostgreSQL feature store.

Location
feature_store/database.py
Insert a Feature
store.insert_feature(
    "user_001",
    "2026-08-16T10:00:00",
    "heart_rate",
    75
)
This creates a feature record containing:

user_id       -> user_001
timestamp     -> 2026-08-16T10:00:00
feature_name  -> heart_rate
feature_value -> 75
Retrieve User Features
store.get_features("user_001")
The result is returned as a list of dictionaries:

[
    {
        "user_id": "user_001",
        "timestamp": "2026-08-16T10:00:00",
        "feature_name": "heart_rate",
        "value": 75.0
    }
]
Retrieve Features by Time Range
store.get_features_by_time_range(
    "user_001",
    "2026-08-16T10:00:00",
    "2026-08-16T11:00:00"
)
The time range is inclusive.

For example:

10:00 -> included
11:00 -> included
12:00 -> excluded
Validation
The feature store validates important inputs before writing or querying data.

User ID
An empty user_id is rejected.

ValueError
Feature Name
An empty feature name is rejected.

ValueError
Timestamp
Timestamps must be ISO-8601 compatible.

Example:

2026-08-16T10:00:00
Invalid timestamps are rejected.

Time Range
The start timestamp cannot occur after the end timestamp.

For example, this is invalid:

start = 12:00
end   = 10:00
ML Experiment Tracking
The project includes an ML experiment tracking layer.

File
tracking/mlflow_tracker.py
The tracking component provides an interface for recording experiment information such as:

Experiment name
Code/version identifier
Parameters
Metrics
Example:

tracker = MLflowTracker()
tracker.log_experiment(
    "pipeline_test",
    "hash123",
    {},
    {
        "accuracy": 0.9
    }
)
Experiments can then be retrieved using the tracking interface.

End-to-End Pipeline
The project includes integration testing that validates the complete processing flow.

The pipeline covers:

Input Record
     ↓
Cleaning
     ↓
Normalization
     ↓
Feature Extraction
     ↓
Temporal Alignment
     ↓
Feature Store
     ↓
Experiment Tracking
This ensures that individual components are tested independently while also verifying that they can work together as an end-to-end system.

Testing
The project uses pytest.

Run the complete test suite:

pytest
The repository contains tests covering:

Alignment
Feature Store
Feature Extraction
Integration
MLflow Tracking
Preprocessing
Test Validation
All tests in the current repository should pass before submitting changes.

Running the Project
1. Clone the Repository
git clone https://github.com/chronishq-ai/foundation-chronis.git
Move into the project directory:

cd foundation-chronis
2. Install Dependencies
pip install -r requirements.txt
3. Configure Environment Variables
Create a .env file in the project root.

Example:

CHRONIS_DB_HOST=localhost
CHRONIS_DB_PORT=5432
CHRONIS_DB_NAME=chronis
CHRONIS_DB_USER=postgres
CHRONIS_DB_PASSWORD=your_password
Do not commit the .env file.

4. Start PostgreSQL
Make sure the PostgreSQL service is running.

pg_isready -h localhost -p 5432
Expected result:

localhost:5432 - accepting connections
5. Run Tests
pytest
All tests should pass before submitting changes.

Development Guidelines
When modifying the project:

Understand the existing implementation before making changes.
Preserve existing public interfaces unless a change is required.
Add tests for new behavior.
Run the complete test suite before committing.
Do not commit passwords, API keys, or .env files.
Keep database-related configuration environment-based.
Avoid unnecessary changes to unrelated components.
Database Design
The feature store follows a simple feature-event model:

User
 |
 +---- Feature Event
          |
          +---- Timestamp
          +---- Feature Name
          +---- Feature Value
Performance Considerations
The current database design includes an index on:

(user_id, timestamp)
This is important because one of the primary access patterns is:

Retrieve features for a user within a time range
Error Handling
The system performs validation at the application layer before database operations.

Examples of invalid input include:

Empty user ID
Empty feature name
Invalid timestamp
Invalid time range
Invalid feature value type
Missing database password
These conditions raise appropriate Python exceptions rather than silently failing.

Project Status
Completed
Data cleaning
Data validation
Feature normalization
Feature extraction
Temporal alignment
PostgreSQL feature store
Feature insertion
Feature retrieval
Time-range feature retrieval
Feature-store validation
PostgreSQL indexing
ML experiment tracking
End-to-end integration testing
Automated test coverage for current modules
PostgreSQL schema foundation
TimescaleDB-ready schema foundation
Security Notes
Database credentials must not be hard-coded into source files.

Use environment variables. Never commit:

.env
database passwords
API keys
private credentials
access tokens
Technology Stack
Python
PostgreSQL
TimescaleDB-compatible schema
pytest
MLflow
psycopg
python-dotenv
Git / GitHub
Repository
https://github.com/chronishq-ai/foundation-chronis

License
This project is currently maintained as part of the Chronis Foundation development work.
