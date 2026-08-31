"""
frontier/interfaces/policy.py

Sprint 1 / R2-F18.1

Policy / Access Engine interface.

Contract:
  check_access(requesting_user_id, resource, action, required_tier) -> bool
    ALLOW  → return True  → caller may proceed
    DENY   → return False → caller must raise PermissionError / return error

  IMPORTANT: consent_tier is determined by the policy layer, never derived
  from caller-supplied arguments.  The PolicyEngine receives a
  requesting_user_id that the platform has already authenticated; it is NOT
  the caller's responsibility to supply or validate the tier integer.

  Trusted-principal boundary: the CRC and all downstream components receive
  'user_id' from their callers, but this is treated as an IDENTIFIER only,
  not as authentication.  Real deployment MUST authenticate the principal
  at the API gateway layer and pass a validated principal context to the
  PolicyEngine.  The PolicyEngine is the ONLY component authorised to grant
  or deny access based on that context.

  If authentication is not yet wired (e.g., during Sprint 17-20 development),
  the PolicyEngine should be configured conservatively.  It MUST NOT default
  to allow-all in production.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class PolicyEngine(ABC):
    """
    Thin interface for Policy/Access Engine (Sprint 1).

    check_access contract:
      requesting_user_id: identifier validated at the API gateway
      resource_type: e.g. "personal_retrieval", "identity_graph", "claims"
      action: e.g. "retrieve", "write", "explain"  (optional, for fine-grained)
      required_tier: minimum consent tier required by the calling subsystem

      Returns True (ALLOW) or False (DENY).
      Callers must raise PermissionError or return an error response on DENY.
    """

    @abstractmethod
    def check_access(
        self,
        user_id: str,
        resource_type: str,
        required_tier: int = 0,
        action: str = "access",
    ) -> bool:
        pass

    @abstractmethod
    def grant_emergency_access(
        self,
        granter_id: str,
        principal: str,
        scope: List[str],
        duration_hours: int,
    ) -> bool:
        """Sprint 17 extension for Tier 4/5 emergency access."""
        pass


class MockPolicyEngine(PolicyEngine):
    """
    Deterministic mock for testing.
    Defaults to allow-all so existing tests work.
    Real production engine must never default to allow-all.
    """
    def __init__(self):
        self._emergency_grants = []

    def check_access(
        self,
        user_id: str,
        resource_type: str,
        required_tier: int = 0,
        action: str = "access",
    ) -> bool:
        return True  # test default — conservative in production

    def grant_emergency_access(
        self,
        granter_id: str,
        principal: str,
        scope: List[str],
        duration_hours: int,
    ) -> bool:
        self._emergency_grants.append({
            "granter": granter_id,
            "principal": principal,
            "scope": scope,
            "duration": duration_hours,
        })
        return True


class DenyForPrincipalMismatchPolicy(PolicyEngine):
    """
    Test policy engine that models authenticated principal vs requested user_id.
    Denies any request where requested user_id != authenticated_principal.
    """
    def __init__(self, authenticated_principal: str):
        self.authenticated_principal = authenticated_principal

    def check_access(
        self,
        user_id: str,
        resource_type: str,
        required_tier: int = 0,
        action: str = "access",
    ) -> bool:
        return user_id == self.authenticated_principal

    def grant_emergency_access(
        self,
        granter_id: str,
        principal: str,
        scope: List[str],
        duration_hours: int,
    ) -> bool:
        return True
