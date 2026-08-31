# Global Learning Architecture (Sprint 19)

## Iteration Notes Checklist (§7.2)
*Answers for the vision encoder Class A model.*

1. **Does this model require cross-user data to generalize?** Yes, visual environments are highly diverse.
2. **What is the privacy budget?** Epsilon 3.0, Delta 1e-5.
*(Answers 3-20 elided for brevity in this simulation document...)*

## Candidate Architecture Comparison
We simulated three approaches:
1. **Federated Averaging**: Accuracy cost 5%, Bandwidth cost 50MB/round.
2. **Secure Aggregation**: Accuracy cost 8%, Bandwidth cost 150MB/round.
3. **DP Centralized (Opt-in)**: Accuracy cost 25%, High privacy guarantee.
4. **Baseline (Do nothing)**: 0% cost, but model stagnates.

## Recommendation
**Vision Encoder**: GO for Federated Averaging. The 5% accuracy drop is acceptable given the privacy guarantees, and bandwidth is manageable.
**Wake-word**: NO-GO. Current baseline is sufficient.
**Base ASR**: NO-GO. Do nothing baseline wins.
