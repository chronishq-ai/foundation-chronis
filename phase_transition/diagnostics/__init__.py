"""Quarantined, non-production diagnostic implementations for
phase_transition (item 4 / R2-S56.4 mechanical sub-fix, this pass).

Anything here is explicitly NOT the doctrine-correct production path --
see each module's own HONESTY FLAG. Nothing under phase_transition/
(outside this package) may import from here; the reverse direction
(re-exporting for backward compat) is fine and used by
phase_transition/degradation.py.
"""