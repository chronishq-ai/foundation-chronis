# Long-Horizon Storage Economics (Sprint 19)

## Cost Model
Based on measurements from Sprint 1-17 surrogate runs:
- Average daily usage per active user: ~4.75 GB/day.
- Estimated cost per GB/month: $0.02.
- Projected cost per user/year: ~$34.67.

### Projections
- **10K Users**: ~$346,700 / year
- **100K Users**: ~$3,467,000 / year
- **1M Users**: ~$34,670,000 / year

## Retrieval Cache vs. Layer 0
Tests on the retrieval-tier cache (Sprint 17 Day 54) confirm that moving older Layer 0 data to cheaper cold storage adds a ~1500ms rehydration latency for cache misses, but maintains 100% visual and temporal accuracy once rehydrated.

## Conclusion & Doctrine Status
Any tiering approach that quietly reduces fidelity violates canonical-record doctrine. Moving to cold storage is valid, but the associated latency/cost tradeoff is a business decision.
**Status**: MP-15 is updated. We have measured the cost, but the decision on whether to thin data or pay the cost rests outside this engineering sprint.
