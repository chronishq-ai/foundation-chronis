# tests/test_audit_tamper.py — Sprint 14 Day 41.
#
# Directive requirement: "Simulate audit-log tampering and verify the
# hash-chained, append-only structure detects it." Every test here mutates
# a clean, verified chain in some way an attacker (or a buggy migration
# script) might, then asserts AuditLog.verify() catches it — and, where
# relevant, that it names the right index.
from __future__ import annotations

import dataclasses

import pytest

from policy_engine.audit_log import AuditAction, AuditLog, AuditOutcome, GENESIS_HASH
from policy_engine.errors import AuditTamperError


def _clean_log(n: int = 5) -> AuditLog:
    log = AuditLog()
    for i in range(n):
        log.record(
            action=AuditAction.INFERENCE if i % 2 == 0 else AuditAction.CLAIM_ACCESS,
            outcome=AuditOutcome.GRANTED if i % 3 else AuditOutcome.DENIED,
            principal_id=f"user-{i % 3}",
            reason=f"reason {i}",
            detail={"seq": i},
        )
    return log


class TestCleanChain:
    def test_empty_log_verifies(self):
        AuditLog().verify()  # must not raise

    def test_single_entry_verifies(self):
        log = AuditLog()
        log.record(action=AuditAction.INFERENCE, outcome=AuditOutcome.GRANTED,
                    principal_id="u1", reason="ok")
        log.verify()

    def test_many_entries_verify(self):
        _clean_log(50).verify()

    def test_first_entry_prev_hash_is_genesis(self):
        log = _clean_log(1)
        assert log._entries[0].prev_hash == GENESIS_HASH

    def test_head_hash_updates_on_append(self):
        log = AuditLog()
        h0 = log.head_hash
        log.record(action=AuditAction.INFERENCE, outcome=AuditOutcome.GRANTED,
                    principal_id="u1", reason="ok")
        assert log.head_hash != h0


class TestContentTamper:
    """Mutating a stored entry's content in place, without recomputing hashes
    (exactly what a direct DB row edit would do)."""

    @pytest.mark.parametrize("field_name,new_value", [
        ("reason", "a different reason entirely"),
        ("outcome", AuditOutcome.GRANTED),  # flip denied -> granted
        ("principal_id", "someone-else"),
        ("detail", {"seq": 9999}),
        ("action", AuditAction.MODEL_WRITE),
    ])
    def test_single_field_tamper_detected(self, field_name, new_value):
        log = _clean_log(6)
        target_index = 3
        original = log._entries[target_index]
        tampered = dataclasses.replace(original, **{field_name: new_value})
        log._entries[target_index] = tampered
        with pytest.raises(AuditTamperError) as exc:
            log.verify()
        assert exc.value.at_index == target_index

    def test_tamper_on_first_entry_detected(self):
        log = _clean_log(6)
        original = log._entries[0]
        log._entries[0] = dataclasses.replace(original, reason="forged")
        with pytest.raises(AuditTamperError) as exc:
            log.verify()
        assert exc.value.at_index == 0

    def test_tamper_on_last_entry_detected(self):
        log = _clean_log(6)
        last = len(log) - 1
        original = log._entries[last]
        log._entries[last] = dataclasses.replace(original, reason="forged")
        with pytest.raises(AuditTamperError) as exc:
            log.verify()
        assert exc.value.at_index == last

    def test_forged_entry_hash_without_recompute_still_caught(self):
        """Attacker edits content AND slaps a fake-but-wrong-looking hash on
        it — still must be caught, since the hash won't match recomputation."""
        log = _clean_log(4)
        original = log._entries[2]
        tampered = dataclasses.replace(
            original, reason="forged", entry_hash="0" * 64  # fake hash
        )
        log._entries[2] = tampered
        with pytest.raises(AuditTamperError) as exc:
            log.verify()
        assert exc.value.at_index == 2


class TestStructuralTamper:
    """Deletion, reordering, and insertion — attacks on the chain's shape,
    not just one entry's content."""

    def test_deletion_from_middle_detected(self):
        log = _clean_log(6)
        del log._entries[2]
        with pytest.raises(AuditTamperError) as exc:
            log.verify()
        # entry originally at position 3 now sits at position 2 but still
        # claims index 3 -> caught by the index/position check first.
        assert exc.value.at_index == 2

    def test_deletion_of_first_entry_detected(self):
        log = _clean_log(5)
        del log._entries[0]
        with pytest.raises(AuditTamperError):
            log.verify()

    def test_deletion_of_last_entry_is_NOT_flagged_as_tamper(self):
        """Important boundary case: truncating the tail (removing the most
        recent entries) does not break the hash chain of what remains — a
        prefix of a valid chain is still a valid chain. This is expected
        and correct: it means 'verify()' proves what's present hasn't been
        altered, not that nothing has ever been removed from the tail. A
        real deployment must pair this with a separately-tracked expected
        entry COUNT (e.g. a periodically-published head-hash + count
        checkpoint) to catch tail truncation — that checkpoint mechanism is
        explicitly out of scope for this in-memory reference class; see
        the package README's 'What this does NOT do' section.
        """
        log = _clean_log(6)
        del log._entries[-1]
        log.verify()  # must NOT raise — this is the documented limitation

    def test_reordering_two_entries_detected(self):
        log = _clean_log(6)
        log._entries[2], log._entries[3] = log._entries[3], log._entries[2]
        with pytest.raises(AuditTamperError):
            log.verify()

    def test_prev_hash_rewired_to_skip_an_entry_detected(self):
        """Attacker removes entry 2 AND patches entry 3's prev_hash to
        point at entry 1's hash, trying to make the chain look continuous.
        Still must be caught, because entry 3's own index (3) no longer
        matches its new position (2)."""
        log = _clean_log(6)
        entry1_hash = log._entries[1].entry_hash
        entry3 = log._entries[3]
        patched_entry3 = dataclasses.replace(entry3, prev_hash=entry1_hash)
        del log._entries[2]
        log._entries[2] = patched_entry3
        with pytest.raises(AuditTamperError) as exc:
            log.verify()
        assert exc.value.at_index == 2

    def test_duplicate_entry_inserted_detected(self):
        log = _clean_log(4)
        dup = log._entries[1]
        log._entries.insert(2, dup)
        with pytest.raises(AuditTamperError):
            log.verify()


class TestAppendOnlyEnforcement:
    """There is no update/delete method on AuditLog at all — this is the
    primary defense, structural rather than detective. These tests assert
    that absence, not just that verify() catches manual list surgery."""

    def test_no_update_method_exists(self):
        assert not hasattr(AuditLog, "update")
        assert not hasattr(AuditLog, "edit")
        assert not hasattr(AuditLog, "amend")

    def test_no_delete_method_exists(self):
        assert not hasattr(AuditLog, "delete")
        assert not hasattr(AuditLog, "remove")
        assert not hasattr(AuditLog, "purge")

    def test_public_api_surface_is_record_and_read_only(self):
        public_methods = {
            name for name in dir(AuditLog)
            if not name.startswith("_") and callable(getattr(AuditLog, name))
        }
        assert public_methods == {"record", "verify", "entries_for"}


class TestDenialsAuditedLikeGrants:
    """Not a tamper test, but the same directive line item — 'a denied
    access is audited exactly like a granted one' — belongs in this file
    since it's a property of the log's structure, not of any particular
    caller's behavior."""

    def test_denied_and_granted_entries_are_the_same_schema(self):
        log = AuditLog()
        granted = log.record(action=AuditAction.INFERENCE, outcome=AuditOutcome.GRANTED,
                               principal_id="u1", reason="ok")
        denied = log.record(action=AuditAction.INFERENCE, outcome=AuditOutcome.DENIED,
                              principal_id="u1", reason="blocked")
        assert set(granted.to_dict().keys()) == set(denied.to_dict().keys())

    def test_denied_entries_included_in_entries_for(self):
        log = AuditLog()
        log.record(action=AuditAction.INFERENCE, outcome=AuditOutcome.GRANTED,
                    principal_id="u1", reason="ok")
        log.record(action=AuditAction.INFERENCE, outcome=AuditOutcome.DENIED,
                    principal_id="u1", reason="blocked")
        entries = log.entries_for("u1")
        outcomes = {e.outcome for e in entries}
        assert outcomes == {AuditOutcome.GRANTED, AuditOutcome.DENIED}