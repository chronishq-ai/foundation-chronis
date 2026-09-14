# ADR S19: Global vs. Personal Learning Boundary

**Status:** Approved  
**Date:** 2026-09-14  
**Reviewers:** Security Architecture Board

## Context
Sprint 19 explored federated and global learning capabilities to improve Chronis representations across users. We need to define the strict architectural boundary between personal isolation and global model updates, particularly how it affects storage economics, privacy budget, and data retention.

## Decision
1. **Global vs Personal Learning Boundary**: No personal data, gradients, or model weights derived from a specific user's private data may be merged into the shared global model without explicit, differential-privacy-compliant authorization and an established privacy budget.
2. **Cross-User Learning**: Cross-user learning (e.g., federated averaging of personal adapters) is currently **FORBIDDEN**. 
3. **Storage Economics**: Personal Language Models (adapters/LoRA weights) are stored strictly within the user's isolated `models/<user_id>/` directory. The shared Class A base model is stored in `models/_shared/`.
4. **Privacy Budget & Data Retention**: 
   - No user data may be retained longer than the user's explicit retention policy.
   - Any future federated learning must implement a formal privacy budget (epsilon/delta) tracking mechanism.
5. **No-Code-Before-Justification**: No production code for global model aggregation or federated averaging may be merged without a prior approved architectural justification and security audit.
6. **Production Global Learning**: Production global learning is currently **UNAUTHORIZED** and disabled.

## Consequences
- The system maintains strict unlinkability and bystander privacy (S16/J0).
- Developers must use isolated personal models for any fine-tuning.
- The `put_base` API for updating the shared base model is restricted to authenticated CI systems only.
