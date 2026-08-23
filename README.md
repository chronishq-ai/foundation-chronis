# policy_engine

Sprint 14, Days 40–41. The constitutional layer's model principal — the
single choke point every ML data read/write is required to route through
(directive Day 40: "No bypass, not even in dev").

**Bible traceability:** Part 6 (Constitutional Layer), G1–G4 front-matter
guarantees as they bind at the ML layer specifically.

## What this does

- **`consent.py`** — defines `ConsentTier` (0–3) and `OperationalMode`
  (A/B/C). Enforces two independent, unconditional floors:
  - `consent_tier >= 2` (`MIN_INFERENCE_CONSENT_TIER`) for any inference.
  - Mode C (Raw Vault) is **structurally unreachable** by the ML layer —
    not tier-gated, not overridable. `OperationalMode.MODE_C` cannot even
    be constructed into a `PolicyRule`'s `allowed_modes` (see
    `policy_rule.py::__post_init__`).
- **`policy_rule.py`** — `PolicyRule` / `Scope` schema. One rule primitive
  reused everywhere access needs to be granted, scoped, and time-boxed.
  Built generic enough that Sprint 17's `emergency_access_grant` should be
  a new `PolicyRule` *instance*, not a new schema — unverified until
  Sprint 17 actually lands; flagged as an assumption, not a guarantee.
- **`principal.py`** — `ModelPrincipal.check(request)`. The actual choke
  point. Raises `PolicyDenied` (or a subtype) or returns `None`. Every
  call — granted or denied — produces exactly one `AuditLog` entry.
- **`audit_log.py`** — hash-chained, append-only log. `AuditLog.record()`
  is the only way entries are added; no update/delete method exists.
  `AuditLog.verify()` walks the full chain and raises `AuditTamperError`
  at the first broken link — catches in-place edits, deletions, and
  reordering.
- **`errors.py`** — typed exceptions (`ConsentTierError`, `ModeCBlocked`,
  `AuditTamperError`, etc.) so callers and tests can assert on exact
  failure mode, not parse message strings.

## What this does NOT do

- **Does not implement Mode B's local-processing execution branch itself**
  — that's Sprint 16. This module only guarantees Mode B is a legal,
  ML-layer-readable mode; it doesn't build the branching pipeline.
- **Does not implement the isolated processing container** (microVM/gVisor,
  24h session-key TTL, RAM zero-fill) — that's Sprint 16 (Bible 5.24).
  This module assumes decrypt-in-RAM isolation happens upstream of it and
  only gates what the ML layer is *permitted* to do with the result.
- **`AuditLog` here is an in-memory reference implementation.** It defines
  the hash-chaining/verification *contract* a real backend must satisfy —
  it is not itself a durability guarantee. A real deployment needs an
  actually-append-only store (write-once table / object-lock storage)
  behind this interface. Do not ship the in-memory class as-is to
  production and call the audit requirement met.
- **Does not resolve who counts as the "subject" for a multi-party session**
  (e.g. bystander audio). `ConsentRecord`/`AccessRequest` are single-user
  scoped; bystander governance is Bible 5.27 / Sprint 16 territory, out of
  this module's scope.
- **`min_consent_tier` on `PolicyRule` is stored but not yet enforced.**
  `ModelPrincipal` currently enforces the tier floor via
  `check_inference_consent()` against the subject's own `ConsentRecord`
  directly, independent of which rule matches. A rule can only narrow
  scope, never raise or lower that floor. This is intentional per the
  current design, but the unused field is a known loose end — flagged so
  it isn't mistaken for dead code to delete later.

## Still open / provisional

- `_RULE_TO_AUDIT_ACTION` in `principal.py` manually maps `RuleAction` →
  `AuditAction` (two separately defined enums). A mismatch fails at
  runtime (`KeyError`), not at lint/import time. Day 41's boundary-test
  suite should include a test asserting both enums have identical member
  sets, so this can't silently drift.
- No integration yet with a real `consent_tier` source of truth (e.g. a
  user-preferences service) — `ConsentRecord` is currently constructed
  directly by callers. `integration/` wrappers (next files) are where this
  gets wired to Sprint 13's model store and MLflow registry; a real
  consent-lookup service is still an open dependency beyond that.
- Audit log storage backend (durable, actually-append-only) is
  unspecified — see "What this does NOT do" above.

## Testing

Day 41 requires 100+ policy-boundary test cases run against real ML
pipeline entry points (not the generic policy-engine suite alone) — see
`tests/test_policy_boundary_cases.py` (upcoming) and
`tests/test_audit_tamper.py` (upcoming). Manual smoke tests run during
development (consent-tier gate, Mode C block at construction and at
runtime, mixed grant/deny chain integrity, tamper detection via in-place
edit and mid-log deletion) are documented inline in this conversation but
are **not** a substitute for the Day 41 suite or for independent
verification by whoever owns this sprint, per the directive's AI-assistant
policy.