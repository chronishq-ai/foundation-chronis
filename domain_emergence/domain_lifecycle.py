"""
Day 18 -- Domain split/merge logic (Bible Part 5.8).

Split: triggered when within-cluster variance (behavioral or narrative)
exceeds between-cluster variance, SUSTAINED across >=2 distinct windows.
On confirmed split: parent domain preserved in the canonical record
(marked inactive, never deleted); child domains created with inherited
history; any Level 1+ claims on the parent marked "pre-split" and held --
the claims-engine hook is out of scope here (Sprint 6 doesn't build the
claims engine), but DomainRegistry exposes `pre_split_hold` on the parent
record so a future claims-engine integration has the flag to check.

Merge: triggered when two domains show increasing co-occurrence in BOTH
behavioral and narrative space, SUSTAINED, with cross-domain behavioral
transition probability AND narrative co-mention rate both above threshold.

Registry is append-only (same doctrine as BifurcationLog, Day 15): no
delete/remove methods. Inactive domains stay in the record forever, just
flagged inactive.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class DomainRecord:
    domain_id: int
    active: bool = True
    parent_ids: list = field(default_factory=list)     # non-empty if born from split/merge
    child_ids: list = field(default_factory=list)
    pre_split_hold: bool = False                         # set True when this domain is split's parent
    history: list = field(default_factory=list)          # inherited + own window stats, append-only


def _longest_contiguous_qualifying_run(qualifies: list) -> tuple:
    """One explicit temporal rule shared by should_split and should_merge
    (R2-S56.7 fix): 'sustained' means an unbroken run of qualifying
    windows, not a scattered total count anywhere in history. Returns
    (longest_run_length, run_start_idx, run_end_idx) for the longest
    contiguous stretch of True values in `qualifies` (run_end_idx
    inclusive). Returns (0, -1, -1) if nothing qualifies."""
    longest_run = 0
    best_start = best_end = -1
    current_run = 0
    current_start = 0
    for i, q in enumerate(qualifies):
        if q:
            if current_run == 0:
                current_start = i
            current_run += 1
            if current_run > longest_run:
                longest_run = current_run
                best_start = current_start
                best_end = i
        else:
            current_run = 0
    return longest_run, best_start, best_end


def _check_timestamp_args(n_windows: int, timestamps: list | None,
                           min_sustained_duration: float | None) -> None:
    if timestamps is not None and len(timestamps) != n_windows:
        raise ValueError("timestamps must be the same length as the window series")
    if timestamps is not None and min_sustained_duration is None:
        raise ValueError("min_sustained_duration is required when timestamps is supplied")
    if timestamps is None and min_sustained_duration is not None:
        raise ValueError(
            "min_sustained_duration requires timestamps -- a window COUNT "
            "alone cannot be converted into a real duration for "
            "irregularly-spaced windows"
        )


def should_split(within_var_by_window: list, between_var_by_window: list,
                  min_sustained_windows: int = 2,
                  timestamps: list | None = None,
                  min_sustained_duration: float | None = None) -> bool:
    """A split is confirmed only if within-cluster variance exceeds
    between-cluster variance for a CONTIGUOUS run of qualifying windows
    (S56.8 fix: doctrine says 'sustained', which means an unbroken run,
    not scattered total occurrences over the whole history -- a
    qualifying window in month 1 and another in month 6 is not
    'sustained').

    Two modes:
      - fixed-duration windows (default, timestamps=None): the longest
        contiguous qualifying run must contain >= min_sustained_windows
        windows.
      - real-timestamp mode (timestamps supplied, R2-S56.7): windows may
        be irregularly spaced, so window COUNT is not a valid duration
        proxy. Pass `timestamps` (one per window, non-decreasing) and
        `min_sustained_duration`; the longest contiguous qualifying run
        must SPAN at least that much real time
        (timestamps[run_end] - timestamps[run_start]), not merely
        contain a certain number of windows."""
    if len(within_var_by_window) != len(between_var_by_window):
        raise ValueError("within_var_by_window and between_var_by_window must be same length")
    _check_timestamp_args(len(within_var_by_window), timestamps, min_sustained_duration)

    qualifies = [w > b for w, b in zip(within_var_by_window, between_var_by_window)]
    longest_run, start, end = _longest_contiguous_qualifying_run(qualifies)

    if timestamps is None:
        return longest_run >= min_sustained_windows
    if longest_run == 0:
        return False
    return (timestamps[end] - timestamps[start]) >= min_sustained_duration


def should_merge(
    cross_transition_prob_by_window: list,
    narrative_comention_rate_by_window: list,
    transition_threshold: float = 0.3,
    comention_threshold: float = 0.3,
    min_sustained_windows: int = 2,
    timestamps: list | None = None,
    min_sustained_duration: float | None = None,
) -> bool:
    """Merge confirmed only if BOTH behavioral transition probability AND
    narrative co-mention rate exceed their thresholds in the SAME window,
    for a CONTIGUOUS run of such windows (R2-S56.7 fix: this previously
    counted total qualifying windows anywhere in the history, giving
    'sustained' an asymmetric meaning from should_split's -- windows 1
    and 7 both qualifying used to be enough; that is not a sustained
    run. Now uses the same `_longest_contiguous_qualifying_run` rule
    should_split uses.

    Two modes, same as should_split:
      - fixed-duration windows (default, timestamps=None): the longest
        contiguous qualifying run must contain >= min_sustained_windows
        windows.
      - real-timestamp mode (timestamps supplied): pass `timestamps`
        (one per window, non-decreasing) and `min_sustained_duration`;
        decision is based on the actual real-time SPAN of the longest
        contiguous qualifying run, not the number of windows in it --
        matters when windows are irregularly spaced."""
    if len(cross_transition_prob_by_window) != len(narrative_comention_rate_by_window):
        raise ValueError("cross_transition_prob_by_window and narrative_comention_rate_by_window must be same length")
    _check_timestamp_args(len(cross_transition_prob_by_window), timestamps, min_sustained_duration)

    qualifies = [
        t > transition_threshold and c > comention_threshold
        for t, c in zip(cross_transition_prob_by_window, narrative_comention_rate_by_window)
    ]
    longest_run, start, end = _longest_contiguous_qualifying_run(qualifies)

    if timestamps is None:
        return longest_run >= min_sustained_windows
    if longest_run == 0:
        return False
    return (timestamps[end] - timestamps[start]) >= min_sustained_duration


class DomainRegistry:
    """Append-only domain lifecycle registry. No delete/remove methods by
    design -- inactive domains remain in the record forever (matches
    BifurcationLog's append-only doctrine from Day 15)."""

    def __init__(self):
        self._domains: dict = {}
        self._next_id = 0

    def register_domain(self, history: list | None = None) -> int:
        """Create a fresh root domain (no parents). Returns its id."""
        domain_id = self._next_id
        self._next_id += 1
        self._domains[domain_id] = DomainRecord(
            domain_id=domain_id, history=list(history or []),
        )
        return domain_id

    def get(self, domain_id: int) -> DomainRecord:
        return self._domains[domain_id]

    def active_domains(self) -> list:
        return [d for d in self._domains.values() if d.active]

    def split_domain(self, parent_id: int, n_children: int = 2) -> list:
        """Mark parent inactive + pre_split_hold, create n_children new
        domains inheriting the parent's history. Returns child ids."""
        parent = self._domains[parent_id]
        parent.active = False
        parent.pre_split_hold = True

        child_ids = []
        for _ in range(n_children):
            child_id = self._next_id
            self._next_id += 1
            child = DomainRecord(
                domain_id=child_id,
                parent_ids=[parent_id],
                history=list(parent.history),   # inherited history
            )
            self._domains[child_id] = child
            child_ids.append(child_id)
            parent.child_ids.append(child_id)

        return child_ids

    def merge_domains(self, domain_id_a: int, domain_id_b: int) -> int:
        """Mark both parents inactive, create one merged domain inheriting
        both histories (concatenated, both preserved). Returns merged id."""
        a = self._domains[domain_id_a]
        b = self._domains[domain_id_b]
        a.active = False
        b.active = False

        merged_id = self._next_id
        self._next_id += 1
        merged = DomainRecord(
            domain_id=merged_id,
            parent_ids=[domain_id_a, domain_id_b],
            history=list(a.history) + list(b.history),
        )
        self._domains[merged_id] = merged
        a.child_ids.append(merged_id)
        b.child_ids.append(merged_id)

        return merged_id