"""S1.3 re-verification — original test sheet, run fresh.

Test Sheet — S1.3 (from the original Intern Remediation & Test Pack):
  T1: Duplicate (user, timestamp, feature, modality) record, unversioned
      -> Rejected
  T2: Impossible physiological value (e.g. negative heart rate) ->
      Rejected
  T3: Record from user A queried under user B's session -> Rejected /
      empty
  T4: Record tagged with an unknown schema version -> Explicit
      incompatibility error, not silently accepted
"""

from datetime import UTC, datetime

import pytest

from chronis_ml.schema.models import ChronisDataset, FeatureRecord, MeasurementStatus
from chronis_ml.schema.validation import SchemaValidationError, validate_dataset


def make_record(**overrides) -> FeatureRecord:
    defaults = dict(
        user_id="user_001",
        timestamp=datetime(2026, 8, 16, 12, 0, tzinfo=UTC),
        feature_name="heart_rate",
        value=72.0,
        modality="ppg",
        status=MeasurementStatus.OBSERVED,
    )
    defaults.update(overrides)
    return FeatureRecord(**defaults)


def test_s13_t1_duplicate_record_rejected() -> None:
    """T1: duplicate (user, timestamp, feature, modality) -> Rejected."""
    timestamp = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)
    record_a = make_record(timestamp=timestamp)
    record_b = make_record(timestamp=timestamp, value=73.0)  # same key, different value

    dataset = ChronisDataset.from_records([record_a, record_b])

    with pytest.raises(SchemaValidationError, match="duplicate observation"):
        validate_dataset(dataset)


def test_s13_t2_impossible_physiological_value_rejected() -> None:
    """T2: negative heart rate -> Rejected."""
    record = make_record(feature_name="heart_rate", value=-10.0)

    with pytest.raises(SchemaValidationError, match="plausible physiological range"):
        validate_dataset(ChronisDataset.from_records([record]))


def test_s13_t3_user_a_record_queried_as_user_b() -> None:
    """T3: record from user A queried under user B's identity -> empty.

    HONEST CAVEAT (unchanged from the original build): there is no
    session/auth layer in this codebase. "User B's session" is tested
    here as the closest real equivalent -- ChronisDataset.by_user()
    correctly returning only the requested user's records, and never
    another user's. This is NOT a full session-boundary/auth test.
    """
    record_a = make_record(user_id="user_a")
    record_b = make_record(user_id="user_b")

    dataset = ChronisDataset.from_records([record_a, record_b])

    result_under_b = dataset.by_user("user_b")

    assert record_a not in result_under_b
    assert result_under_b == (record_b,)


def test_s13_t4_unknown_schema_version_rejected() -> None:
    """T4: unknown schema version -> explicit incompatibility error,
    never silently accepted."""
    record = make_record(schema_version="0.1")

    with pytest.raises(SchemaValidationError, match="unsupported schema_version"):
        validate_dataset(ChronisDataset.from_records([record]))
