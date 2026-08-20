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
```

---

## Repository Structure

```text
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
```

---

# Core Components

## 1. Data Preprocessing

The preprocessing layer prepares incoming records before they are passed to downstream components.

### Responsibilities

* Validate incoming records
* Clean input data
* Handle invalid or missing data
* Normalize feature values
* Prepare structured data for feature extraction

### Files

```text
preprocessing/
├── cleaner.py
├── normalizer.py
└── validator.py
```

---

## 2. Feature Extraction

The feature extraction layer converts processed sensor and input data into useful numerical features.

The current feature pipeline supports inputs such as:

* IMU data
* Heart-rate measurements
* Audio-related information

Example feature categories include:

```text
Movement
Heart Rate
Speech Activity
Audio Energy
Pause Information
```

The extracted features can then be passed to the temporal alignment layer and eventually stored in the feature store.

### File

```text
features/feature_pipeline.py
```

---

## 3. Temporal Alignment

Chronis processes information originating from different sources.

These sources may produce observations at different timestamps or intervals.

The temporal alignment layer provides a common representation so that downstream processing can work with temporally aligned data.

### File

```text
alignment/temporal_alignment.py
```

The current implementation includes automated tests covering temporal alignment behavior.

---

## 4. Feature Store

The feature store provides persistent storage for processed Chronis features.

The current backend is:

```text
PostgreSQL
```

The database schema is designed with future TimescaleDB support in mind.

### Feature Table

The primary table is:

```text
features
```

| Column          | Type             | Description               |
| --------------- | ---------------- | ------------------------- |
| `id`            | SERIAL           | Primary key               |
| `user_id`       | VARCHAR(100)     | User identifier           |
| `timestamp`     | TIMESTAMP        | Feature timestamp         |
| `feature_name`  | VARCHAR(100)     | Name of the feature       |
| `feature_value` | DOUBLE PRECISION | Numerical feature value   |
| `created_at`    | TIMESTAMP        | Record creation timestamp |

---

## Database Index

An index is created on:

```sql
(user_id, timestamp)
```

This supports efficient queries involving:

* A specific user
* Time-range feature retrieval

---

# PostgreSQL Configuration

The application reads database configuration from environment variables.

Example:

```env
CHRONIS_DB_HOST=localhost
CHRONIS_DB_PORT=5432
CHRONIS_DB_NAME=chronis
CHRONIS_DB_USER=postgres
CHRONIS_DB_PASSWORD=<your-password>
```

The password should be stored locally in `.env` and must **not** be committed to Git.

---

# PostgreSQL Setup

## 1. Install PostgreSQL

Install PostgreSQL locally and make sure the PostgreSQL service is running.

Verify that PostgreSQL is available:

```bash
psql --version
```

---

## 2. Create the Database

Connect to PostgreSQL:

```bash
psql -h localhost -p 5432 -U postgres
```

Create the Chronis database:

```sql
CREATE DATABASE chronis;
```

Connect to it:

```sql
\c chronis
```

---

## 3. Create the Feature Table

The schema is available in:

```text
feature_store/schema.sql
```

The core table can be created with:

```sql
CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Create the performance index:

```sql
CREATE INDEX IF NOT EXISTS idx_features_user_timestamp
ON features (user_id, timestamp);
```

---

# TimescaleDB Support

The feature store schema has been prepared with future TimescaleDB support in mind.

The schema provides the foundation for converting the `features` table into a TimescaleDB hypertable:

```sql
SELECT create_hypertable(
    'features',
    'timestamp',
    if_not_exists => TRUE
);
```

This is optional and requires TimescaleDB to be installed and enabled.

The current implementation works with standard PostgreSQL.

---

# Feature Store API

The `FeatureStore` class provides the application-level interface to the PostgreSQL feature store.

### Location

```text
feature_store/database.py
```

---

## Insert a Feature

```python
store.insert_feature(
    "user_001",
    "2026-08-16T10:00:00",
    "heart_rate",
    75
)
```

This creates a feature record containing:

```text
user_id       -> user_001
timestamp     -> 2026-08-16T10:00:00
feature_name  -> heart_rate
feature_value -> 75
```

---

## Retrieve User Features

```python
store.get_features("user_001")
```

The result is returned as a list of dictionaries:

```python
[
    {
        "user_id": "user_001",
        "timestamp": "2026-08-16T10:00:00",
        "feature_name": "heart_rate",
        "value": 75.0
    }
]
```

---

## Retrieve Features by Time Range

```python
store.get_features_by_time_range(
    "user_001",
    "2026-08-16T10:00:00",
    "2026-08-16T11:00:00"
)
```

The time range is inclusive.

For example:

```text
10:00 -> included
11:00 -> included
12:00 -> excluded
```

---

# Validation

The feature store validates important inputs before writing or querying data.

### User ID

An empty `user_id` is rejected.

```python
ValueError
```

### Feature Name

An empty feature name is rejected.

```python
ValueError
```

### Timestamp

Timestamps must be ISO-8601 compatible.

Example:

```text
2026-08-16T10:00:00
```

Invalid timestamps are rejected.

### Time Range

The start timestamp cannot occur after the end timestamp.

For example, this is invalid:

```text
start = 12:00
end   = 10:00
```

---

# ML Experiment Tracking

The project includes an ML experiment tracking layer.

### File

```text
tracking/mlflow_tracker.py
```

The tracking component provides an interface for recording experiment information such as:

* Experiment name
* Code/version identifier
* Parameters
* Metrics

Example:

```python
tracker = MLflowTracker()

tracker.log_experiment(
    "pipeline_test",
    "hash123",
    {},
    {
        "accuracy": 0.9
    }
)
```

Experiments can then be retrieved using the tracking interface.

---

# End-to-End Pipeline

The project includes integration testing that validates the complete processing flow.

The pipeline covers:

```text
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
```

This ensures that individual components are tested independently while also verifying that they can work together as an end-to-end system.

---

# Testing

The project uses `pytest`.

Run the complete test suite:

```bash
pytest
```

The repository contains tests covering:

```text
Alignment
Feature Store
Feature Extraction
Integration
MLflow Tracking
Preprocessing
```

### Test Validation

All tests in the current repository should pass before submitting changes.

Example:

```text
=========================== test session starts ===========================

collected tests

tests/test_alignment.py
tests/test_feature_store.py
tests/test_features.py
tests/test_integration.py
tests/test_mlflow.py
tests/test_preprocessing.py

=========================== tests passed ============================
```

---

# Running the Project

## 1. Clone the Repository

```bash
git clone https://github.com/chronishq-ai/foundation-chronis.git
```

Move into the project directory:

```bash
cd foundation-chronis
```

---

## 2. Install Dependencies

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Configure Environment Variables

Create a `.env` file in the project root.

Example:

```env
CHRONIS_DB_HOST=localhost
CHRONIS_DB_PORT=5432
CHRONIS_DB_NAME=chronis
CHRONIS_DB_USER=postgres
CHRONIS_DB_PASSWORD=your_password
```

Do not commit the `.env` file.

---

## 4. Start PostgreSQL

Make sure the PostgreSQL service is running.

Verify the database connection:

```bash
pg_isready -h localhost -p 5432
```

Expected result:

```text
localhost:5432 - accepting connections
```

---

## 5. Run Tests

```bash
pytest
```

All tests should pass before submitting changes.

---

# Development Guidelines

When modifying the project:

1. Understand the existing implementation before making changes.
2. Preserve existing public interfaces unless a change is required.
3. Add tests for new behavior.
4. Run the complete test suite before committing.
5. Do not commit passwords, API keys, or `.env` files.
6. Keep database-related configuration environment-based.
7. Avoid unnecessary changes to unrelated components.

---

# Database Design

The feature store follows a simple feature-event model:

```text
User
 |
 +---- Feature Event
          |
          +---- Timestamp
          +---- Feature Name
          +---- Feature Value
```

Example:

```text
user_001
   |
   +-- 10:00 -> heart_rate = 75
   |
   +-- 11:00 -> heart_rate = 80
   |
   +-- 12:00 -> heart_rate = 85
```

This structure allows Chronis to retrieve the evolution of individual features over time.

---

# Performance Considerations

The current database design includes an index on:

```sql
(user_id, timestamp)
```

This is important because one of the primary access patterns is:

```text
Retrieve features for a user within a time range
```

The schema is also structured so that TimescaleDB can be introduced later if the volume and time-series workload require it.

---

# Error Handling

The system performs validation at the application layer before database operations.

Examples of invalid input include:

```text
Empty user ID
Empty feature name
Invalid timestamp
Invalid time range
Invalid feature value type
Missing database password
```

These conditions raise appropriate Python exceptions rather than silently failing.

---

# Project Status

## Completed

* Data cleaning
* Data validation
* Feature normalization
* Feature extraction
* Temporal alignment
* PostgreSQL feature store
* Feature insertion
* Feature retrieval
* Time-range feature retrieval
* Feature-store validation
* PostgreSQL indexing
* ML experiment tracking
* End-to-end integration testing
* Automated test coverage for current modules
* PostgreSQL schema foundation
* TimescaleDB-ready schema foundation

---

# Current Sprint Result

The PostgreSQL-backed Feature Store has been integrated into the Chronis Foundation.

The implementation includes:

```text
PostgreSQL Connection
        ↓
Feature Insertion
        ↓
Feature Retrieval
        ↓
Time-Range Retrieval
        ↓
Input Validation
        ↓
Integration Testing
```

The implementation has been validated through the project's automated test suite.

---

# Future Extensions

Potential future improvements include:

* TimescaleDB hypertable activation
* Continuous aggregates for time-based feature statistics
* Additional feature types
* Larger-scale time-series optimization
* Improved experiment tracking
* Production database connection management
* Database migrations
* Feature versioning
* Feature quality monitoring
* Additional integration tests
* Production deployment configuration

These extensions should be implemented based on the requirements of later Chronis sprints.

---

# Security Notes

Database credentials must not be hard-coded into source files.

Use environment variables:

```env
CHRONIS_DB_PASSWORD=your_password
```

The `.env` file should remain local and should be excluded from version control.

Never commit:

```text
.env
database passwords
API keys
private credentials
access tokens
```

---

# Technology Stack

### Programming Language

* Python

### Database

* PostgreSQL

### Time-Series Database Support

* TimescaleDB-compatible schema

### Testing

* pytest

### ML Experiment Tracking

* MLflow

### Database Driver

* psycopg

### Configuration

* python-dotenv

### Version Control

* Git
* GitHub

---

# Repository

Chronis Foundation repository:

https://github.com/chronishq-ai/foundation-chronis

---

# Team Development

Chronis Foundation is developed collaboratively.

Changes should be made within the appropriate sprint or feature scope while preserving existing functionality developed by other team members.

Before pushing changes:

```bash
pytest
git status
git diff
```

After confirming the implementation:

```bash
git add <files>
git commit -m "Description of change"
git push
```

---

# Verification Checklist

Before considering a change complete:

* [ ] Code implemented
* [ ] Existing functionality preserved
* [ ] Tests added or updated where required
* [ ] `pytest` passes
* [ ] No credentials committed
* [ ] `git diff` reviewed
* [ ] `git status` reviewed
* [ ] Commit created
* [ ] Changes pushed to GitHub

---

# License

This project is currently maintained as part of the Chronis Foundation development work.
