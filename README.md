# Chronis Foundation — Part 2

## ML Pipeline, Feature Extraction & Feature Storage

Chronis Foundation — Part 2 implements the **ML pipeline layer** responsible for transforming standardized time-series data into structured, ML-ready feature representations.

The module provides the foundation for downstream AI/ML systems by handling:

* Data preprocessing and validation
* Multimodal feature extraction
* Personal baseline normalization
* Temporal synchronization
* Feature storage and retrieval
* Experiment tracking
* End-to-end pipeline validation

The primary objective is to transform raw behavioral and sensor-derived data into **consistent, structured features** that can be consumed by future Chronis intelligence modules.

---

## Pipeline Architecture

```text
Standardized Input Data
        │
        ▼
┌─────────────────────┐
│ Data Preprocessing  │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Feature Extraction  │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Temporal Alignment  │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│    Feature Store    │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Experiment Tracking │
└─────────────────────┘
        │
        ▼
 ML-Ready Feature Representation
```

---

## Project Structure

```text
chronis-foundation/
│
├── preprocessing/
│   ├── validator.py
│   ├── cleaner.py
│   └── normalizer.py
│
├── features/
│   ├── imu_features.py
│   ├── ppg_features.py
│   ├── audio_features.py
│   └── feature_pipeline.py
│
├── alignment/
│   └── temporal_alignment.py
│
├── feature_store/
│   ├── schema.sql
│   ├── database.py
│   └── queries.py
│
├── tracking/
│   └── mlflow_tracker.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_alignment.py
│   ├── test_feature_store.py
│   └── test_integration.py
│
└── docs/
    ├── architecture.md
    └── data_flow.md
```

---

# Modules

## 1. Preprocessing Layer

The preprocessing layer prepares standardized time-series data for downstream feature extraction.

### Responsibilities

#### Data Validation

* Validates incoming data structures
* Checks for required fields
* Ensures consistent feature formats
* Detects invalid or malformed inputs

#### Data Cleaning

* Removes invalid feature entries
* Maintains consistent data representation
* Preserves missing values rather than replacing them with artificial values
* Prepares clean inputs for feature extraction

#### Personal Baseline Normalization

Chronis uses **user-specific normalization** to account for individual behavioral differences.

Instead of comparing a feature against a generic population average:

```text
feature - population_average
```

the pipeline uses the user's personal baseline:

```text
feature - user's_personal_baseline
```

This allows downstream systems to identify deviations relative to an individual's normal behavioral pattern.

---

# 2. Multimodal Feature Extraction

The feature extraction layer converts sensor and behavioral signals into structured numerical features.

Currently supported modalities include:

* IMU
* PPG / Heart
* Audio / Prosody

---

## IMU Features

The IMU feature extractor processes movement-related sensor data.

### Extracted Features

* Movement intensity
* Movement variance

### Example Output

```json
{
  "movement_mean": 0.45,
  "movement_variance": 0.02
}
```

---

## PPG / Heart Features

The PPG feature extractor processes physiological signals.

### Extracted Features

* Average heart rate
* Heart-rate variability

### Example Output

```json
{
  "heart_rate_mean": 75,
  "heart_rate_variability": 4.5
}
```

---

## Audio / Prosody Features

The audio feature extractor processes speech-related characteristics.

### Extracted Features

* Speaking rate
* Average pause duration
* Audio energy

### Example Output

```json
{
  "speaking_rate": 0.2,
  "average_pause_duration": 1.5,
  "average_energy": 0.6
}
```

---

## Unified Feature Pipeline

The feature pipeline combines features from multiple modalities into a unified representation.

```text
        IMU Features
             │
             │
        PPG Features
             │
             ├──────► Feature Pipeline
             │
       Audio Features
             │
             ▼
   Unified Multimodal
     Feature Vector
```

This unified representation provides a consistent interface for downstream ML components.

---

# 3. Temporal Alignment Layer

Different sensors may operate at different sampling frequencies and produce observations at different timestamps.

The temporal alignment module synchronizes multimodal feature streams onto a common timeline.

### Example

```text
Audio Timeline
│   │    │      │
│   │    │      │
└───┴────┴──────┴──────

IMU Timeline
│ │ │ │ │ │ │ │ │
└─┴─┴─┴─┴─┴─┴─┴─┴────

PPG Timeline
│    │    │    │
└────┴────┴────┴──────

              │
              ▼

     Unified Chronological
            Timeline
```

### Purpose

Temporal alignment ensures that features generated by different modalities correspond to the same behavioral time window.

---

# 4. Feature Store

The feature store provides storage and retrieval capabilities for processed ML features.

### Current Capabilities

* Feature insertion
* Feature retrieval
* Time-series feature representation
* Structured feature storage

The storage layer is designed to support future integration with **TimescaleDB**.

### Feature Schema

The feature representation includes:

| Field           | Description                             |
| --------------- | --------------------------------------- |
| `user_id`       | Identifier associated with the user     |
| `timestamp`     | Time at which the feature was generated |
| `feature_name`  | Name of the extracted feature           |
| `feature_value` | Numerical feature value                 |
| `created_at`    | Feature creation timestamp              |

---

# 5. Experiment Tracking

The experiment tracking module provides a consistent mechanism for recording ML pipeline experiments.

### Tracks

* Experiment name
* Dataset hash
* Parameters
* Metrics
* Experiment timestamp

### Example

```json
{
  "experiment": "feature_pipeline_v1",
  "dataset_hash": "abc123",
  "metrics": {
    "missing_rate": 0.02
  }
}
```

This allows experiments and feature-processing changes to be tracked and reproduced more reliably.

---

# Testing

The project includes automated tests covering the major pipeline components.

### Run Tests

```bash
python -m pytest
```

### Current Test Status

```text
14 tests passed
```

### Test Coverage

Tests currently cover:

* Preprocessing validation
* Data cleaning
* Personal baseline normalization
* Feature extraction
* Temporal alignment
* Feature storage
* Experiment tracking
* End-to-end pipeline integration

---

# Development Setup

## 1. Clone the Repository

```bash
git clone <repository-url>
```

## 2. Navigate to the Project

```bash
cd chronis-foundation
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Test Suite

```bash
python -m pytest
```

---

# Design Principles

The pipeline follows several principles to maintain consistency and extensibility.

### Modularity

Each pipeline responsibility is separated into an independent module.

### Consistent Data Representation

All extracted features follow structured and predictable formats.

### Personalization

User-specific baselines are used where applicable instead of relying exclusively on population-level statistics.

### Testability

Core functionality is covered by automated tests.

### Extensibility

The architecture is designed to support additional modalities, storage systems, and ML infrastructure in future iterations.

---

# Future Extensions

Planned improvements include:

* [ ] Direct TimescaleDB integration
* [ ] Production MLflow server integration
* [ ] Advanced multimodal feature extraction
* [ ] Production dataset connectors
* [ ] Real-time feature streaming
* [ ] Additional sensor modalities
* [ ] Scalable feature retrieval APIs

---

# Contribution Guidelines

All changes to this module should maintain:

* Clear module boundaries
* Consistent data formats
* Appropriate test coverage
* Backward compatibility where possible
* Compatibility with downstream Chronis AI/ML components

Before submitting changes, run the complete test suite:

```bash
python -m pytest
```

---

# Role in the Chronis System

Chronis Foundation — Part 2 serves as the **ML data processing and feature engineering layer** between standardized input data and downstream intelligence systems.

```text
                Chronis Foundation
                       │
                       ▼
              Standardized Data
                       │
                       ▼
          ┌─────────────────────────┐
          │      Part 2: ML         │
          │     Foundation Layer    │
          ├─────────────────────────┤
          │ • Preprocessing         │
          │ • Feature Extraction    │
          │ • Normalization         │
          │ • Temporal Alignment    │
          │ • Feature Storage       │
          │ • Experiment Tracking   │
          └─────────────────────────┘
                       │
                       ▼
              ML-Ready Features
                       │
                       ▼
          Downstream AI/ML Systems
```

This layer establishes the foundation required for building more advanced Chronis intelligence and prediction capabilities.
