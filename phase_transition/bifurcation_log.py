from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class BifurcationEvent:
    """One rupture event, immutable once created -- append-only doctrine."""
    timestamp: float
    logged_at: str
    cond_voice_energy: bool
    cond_ppg_hr: bool
    cond_cse_salience: bool
    cond_imu_motion: bool
    rupture_declared: bool


class BifurcationLog:
    """
    Separate, append-only log for rupture events (Module 4.11 output).
    NEVER the ordinary recurrence pipeline. Nothing is ever deleted or
    overwritten here -- Global Standard non-negotiable (Layer 0 doctrine
    extended to this log).
    """

    def __init__(self):
        self._events: list[BifurcationEvent] = []

    def append(self, rupture_result: dict) -> BifurcationEvent:
        """Append one rupture-check result. Only meaningful to append
        events where rupture_declared is True -- this log tracks actual
        bifurcations, not every non-event check."""
        if not rupture_result["rupture_declared"]:
            raise ValueError(
                "Refusing to append a non-declared rupture check. "
                "Bifurcation log only holds actual declared events."
            )
        event = BifurcationEvent(
            timestamp=rupture_result["timestamp"],
            logged_at=datetime.now(timezone.utc).isoformat(),
            cond_voice_energy=rupture_result["cond_voice_energy"],
            cond_ppg_hr=rupture_result["cond_ppg_hr"],
            cond_cse_salience=rupture_result["cond_cse_salience"],
            cond_imu_motion=rupture_result["cond_imu_motion"],
            rupture_declared=rupture_result["rupture_declared"],
        )
        self._events.append(event)
        return event

    def all_events(self) -> list[BifurcationEvent]:
        """Read-only view. Returns a copy -- callers cannot mutate the log."""
        return list(self._events)

    def events_in_window(self, start: float, end: float) -> list[BifurcationEvent]:
        """Events with timestamp in [start, end] -- used to feed phase-
        transition condition 2 as additional evidence."""
        return [e for e in self._events if start <= e.timestamp <= end]

    def as_condition2_evidence(self, candidate_t: float, window: float = 5.0) -> bool:
        """
        True if a declared bifurcation occurred within `window` of
        candidate_t -- additional evidence feeding condition 2
        (predictive fit degradation) of the phase-transition gate.
        """
        nearby = self.events_in_window(candidate_t - window, candidate_t + window)
        return len(nearby) > 0