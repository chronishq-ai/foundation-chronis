# consent tiers + operational modes (Bible Part 6.1, A/B/C).
#
# Mode A = cloud-assisted processing, full pipeline available.
# Mode B = local-only processing path (same codebase, different execution
#          target — Sprint 16 wires the actual branching; here we only need
#          to know it's a legitimate mode, not a degraded one).
# Mode C = Raw Vault. Encrypted-at-rest cold storage the user can access
#          directly. The ML layer must NEVER read from Mode C. Not "needs
#          high consent_tier" — structurally blocked, full stop, no
#          consent_tier value unblocks it. That's the "hard block ... not
#          just at the gateway layer" requirement from Sprint 14 Day 40.
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .errors import ConsentTierError, ModeCBlocked


class ConsentTier(IntEnum):
    """
    Ordered so `consent_tier >= 2` (the inference-eligibility floor) is a
    plain integer comparison, not a set-membership check.
    """

    NONE = 0          # no consent on file — nothing ML-layer touches this user
    STORAGE_ONLY = 1  # user allows encrypted storage, not model inference
    INFERENCE = 2      # minimum tier for any ML inference — the Day 40 floor
    FULL = 3           # inference + product-surface features (Mirror, etc.)


# The one hard number from the directive: "inference is only permitted
# where consent_tier >= 2." Named so nobody re-derives or hardcodes `2`
# somewhere else — Global Standard's no-silent-magic-number rule.
MIN_INFERENCE_CONSENT_TIER = ConsentTier.INFERENCE


class OperationalMode(IntEnum):
    MODE_A = 0  # cloud-assisted
    MODE_B = 1  # local-only processing path
    MODE_C = 2  # Raw Vault — direct-access cold storage, ML layer never reads this


# Modes the ML layer is structurally permitted to read from. Mode C is
# deliberately absent from this set — not filtered out by a runtime check,
# absent from the allowed set itself, so a future edit that "adds a
# consent_tier override" for Mode C still can't accidentally unblock it.
ML_LAYER_READABLE_MODES = frozenset({OperationalMode.MODE_A, OperationalMode.MODE_B})


@dataclass(frozen=True)
class ConsentRecord:
    """A user's current consent state, as the model principal sees it."""

    user_id: str
    tier: ConsentTier
    mode: OperationalMode


def check_mode_c_block(mode: OperationalMode) -> None:
    """
    Raise ModeCBlocked if `mode` is Mode C. This function has no consent_tier
    parameter on purpose: Mode C is not tier-gated, it is mode-gated. Any
    caller tempted to pass a tier here to "allow it through" is proof the
    call site is wrong, not this function.
    """
    if mode not in ML_LAYER_READABLE_MODES:
        raise ModeCBlocked(
            f"Mode {mode.name} is not ML-layer-readable — Raw Vault hard block, no override.",
        )


def check_inference_consent(record: ConsentRecord) -> None:
    """
    Raise ConsentTierError if `record.tier` is below the inference floor.
    Also re-asserts the Mode C block, since an inference request always
    implies a read, and every read must clear both checks — callers should
    not be able to satisfy this function while skipping check_mode_c_block
    by calling this one instead.
    """
    check_mode_c_block(record.mode)
    if record.tier < MIN_INFERENCE_CONSENT_TIER:
        raise ConsentTierError(
            f"user {record.user_id!r} has consent_tier={int(record.tier)} "
            f"({record.tier.name}), need >= {int(MIN_INFERENCE_CONSENT_TIER)} "
            f"({MIN_INFERENCE_CONSENT_TIER.name}) for inference.",
            principal_id=record.user_id,
        )


def is_inference_permitted(record: ConsentRecord) -> bool:
    """Non-raising probe form of check_inference_consent, for UI/logging paths
    that want a bool rather than a try/except. Never used to actually gate
    a real read/write — those call check_inference_consent and let it raise."""
    try:
        check_inference_consent(record)
    except (ConsentTierError, ModeCBlocked):
        return False
    return True