"""
CHRONIS — Team 4 (INVENTORS) — Sprint 7, Day 19
Weak-Supervision Label Layer (WSL) for the Narrative State-Space Model (NSSM)

WHAT THIS FILE DOES, IN PLAIN ENGLISH
--------------------------------------
We want to score every therapy-session-like transcript along 8 "narrative
dimensions" (e.g. "is this person blaming themselves or the world?").

We don't have a hand-labeled dataset to train a normal classifier on. So
instead of one ground-truth label per session, we write several cheap,
imperfect "labeling functions" (LFs) per dimension. Each LF is a noisy vote:
it looks at the transcript (or prosody, or a self-hosted LLM) and guesses a
class, or abstains if it's not confident enough to guess.

We then feed all those noisy votes into a Dawid-Skene style label model.
This is the core trick of "weak supervision": the label model learns, from
the *agreement patterns between LFs themselves* (not from any ground truth),
how reliable each LF tends to be, and uses that to produce one soft
probability distribution over the true class per session.

Finally, for every dimension we also emit a per-session measurement
uncertainty term (sigma_t). Sprint 7 Day 20 needs this to build a
heteroskedastic NSSM emission model — i.e. the NSSM should trust
high-confidence sessions more than shaky ones. sigma_t here comes directly
from the label model's own uncertainty about the aggregated call, not from
a fixed constant (per the Global Standard: "no silent magic numbers").

HARD CONSTRAINTS FROM THE DIRECTIVE (Sprint 7 Day 19)
-------------------------------------------------------
1. 2-4 independently-noisy labeling functions per dimension, with
   deliberately different failure modes.
2. Dawid-Skene / Snorkel-style label model, learned purely from inter-LF
   agreement — no manually labeled ground-truth dataset required.
3. "self-role" dimension must include a labeling function that uses a
   prosody cross-check (flat/resigned prosody vs. resolved/upward prosody
   changes the read of identical words). That extractor doesn't exist yet
   in this file — it's Sprint 2's job — so it's a clearly marked
   placeholder/injection point.
4. Any LLM-based labeling function MUST be self-hosted. Never call a
   third-party API. We enforce this with a runtime guard, not just a
   comment.
5. Output per session = soft class distribution + a learned
   measurement-uncertainty term (sigma_t) per dimension. Never a bare
   point estimate.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chronis.wsl")

# A labeling function either returns a class index (an int) or ABSTAIN.
# ABSTAIN means "I don't have enough signal to vote on this session" —
# an LF abstaining is normal and expected, not an error.
ABSTAIN = -1


# ---------------------------------------------------------------------------
# STEP 0 — Define the 8 narrative dimensions and how many classes each has.
# ---------------------------------------------------------------------------
class NarrativeDimension(str, Enum):
    SEMANTIC_DOMAIN_COVERAGE = "semantic_domain_coverage"
    CAUSAL_ATTRIBUTION = "causal_attribution"          # locus of control
    SELF_ROLE = "self_role"
    TEMPORAL_FRAMING = "temporal_framing"
    MORAL_FRAMING = "moral_framing"                     # Moral Foundations Dictionary
    NARRATIVE_ARC_TYPING = "narrative_arc_typing"        # McAdams typing
    CONTRADICTION_TOLERANCE = "contradiction_tolerance"
    FUTURE_SELF_REHEARSAL = "future_self_rehearsal"


# Each dimension has its own small, fixed class vocabulary. Keeping this
# explicit (rather than inferring class count from the data) is what lets
# the Dawid-Skene confusion matrices be well-defined per dimension.
DIMENSION_CLASSES: Dict[NarrativeDimension, List[str]] = {
    NarrativeDimension.SEMANTIC_DOMAIN_COVERAGE: ["career", "relationships", "health", "other"],
    NarrativeDimension.CAUSAL_ATTRIBUTION: ["internal_locus", "external_locus", "mixed"],
    NarrativeDimension.SELF_ROLE: ["agentive_hero", "victim", "observer"],
    NarrativeDimension.TEMPORAL_FRAMING: ["past_dominant", "present_dominant", "future_dominant"],
    NarrativeDimension.MORAL_FRAMING: ["care_harm", "fairness_cheating", "loyalty_betrayal", "neutral"],
    NarrativeDimension.NARRATIVE_ARC_TYPING: ["redemptive", "contaminated", "ambivalent", "stable"],
    NarrativeDimension.CONTRADICTION_TOLERANCE: ["high_tolerance", "low_tolerance"],
    NarrativeDimension.FUTURE_SELF_REHEARSAL: ["consistent", "inconsistent", "absent"],
}


def n_classes(dim: NarrativeDimension) -> int:
    return len(DIMENSION_CLASSES[dim])


# ---------------------------------------------------------------------------
# STEP 1 — What a "session" looks like as input to the WSL.
# ---------------------------------------------------------------------------
@dataclass
class SessionInput:
    """
    Everything a labeling function might need to look at for one session.
    `prosody_features` is intentionally Optional[dict]: Sprint 2 owns that
    extractor, and Sprint 7 must degrade gracefully (i.e. abstain) if it's
    not wired in yet.
    """
    session_id: str
    transcript: str  # wearer-only transcript (never the other-speaker text)
    prosody_features: Optional[dict] = None  # [REQUIRES SPRINT 2 PROSODY EXTRACTOR]
    idiolect_baseline: Optional[dict] = None  # person-specific baseline, also Sprint 2


# ---------------------------------------------------------------------------
# STEP 2 — Self-hosted-only guard for the LLM labeling function.
# ---------------------------------------------------------------------------
# Non-negotiable per the directive: "narrative content ... must never
# transit a third-party API as a labeling side-effect." We don't just write
# that in a comment — we make it impossible to accidentally point this
# client at, say, api.openai.com or api.anthropic.com.
_ALLOWED_HOST_SUFFIXES = ("localhost", "127.0.0.1", ".internal", ".local")


class SelfHostedOnlyError(RuntimeError):
    """Raised if code tries to point the labeling LLM at a non-local host."""


def assert_self_hosted(endpoint_url: str) -> None:
    host = urlparse(endpoint_url).hostname or ""
    if not any(host == suffix.lstrip(".") or host.endswith(suffix) for suffix in _ALLOWED_HOST_SUFFIXES):
        raise SelfHostedOnlyError(
            f"Refusing to call '{endpoint_url}': narrative-content labeling functions "
            f"must be self-hosted (localhost / *.internal / *.local only). "
            f"Third-party APIs are a hard non-negotiable per the Sprint 7 directive."
        )


class SelfHostedLLMClient:
    """
    Thin wrapper around a *local* constrained-output LLM inference server
    (e.g. an in-house vLLM / llama.cpp deployment). This class never talks
    to the public internet — assert_self_hosted() is checked on every call,
    not just once at construction, so a later bug can't silently swap the
    endpoint.

    In production this would issue an HTTP request to `endpoint_url`. Here
    we stub `_call_model` so this file has zero external dependencies and
    can be unit-tested offline; swap `_call_model` for the real inference
    call when wiring this into the actual self-hosted deployment.
    """

    def __init__(self, endpoint_url: str = "http://localhost:8001/generate"):
        assert_self_hosted(endpoint_url)
        self.endpoint_url = endpoint_url

    def constrained_classify(self, prompt: str, choices: List[str]) -> Optional[str]:
        """Ask the local LLM to pick exactly one of `choices`, or abstain (None)."""
        assert_self_hosted(self.endpoint_url)  # re-checked defensively, every call
        return self._call_model(prompt, choices)

    def _call_model(self, prompt: str, choices: List[str]) -> Optional[str]:
        # --- STUB: replace with the real self-hosted inference call. ---
        # A real implementation would do constrained decoding (grammar /
        # logit-bias restricted to `choices`) against the local server at
        # self.endpoint_url and return one of `choices`, or None if the
        # model itself declines to commit to an answer.
        logger.debug("SelfHostedLLMClient stub called with %d choices", len(choices))
        return None


# ---------------------------------------------------------------------------
# STEP 3 — Labeling functions (LFs).
# ---------------------------------------------------------------------------
# Design goal per the directive: 2-4 LFs per dimension, with *deliberately
# different failure modes* — so that when they disagree, the label model
# has genuine signal to work with (an LF that always agrees with another LF
# gives Dawid-Skene nothing to learn from).
#
# Every LF has the exact same shape: (SessionInput) -> class index | ABSTAIN

LabelingFunction = Callable[[SessionInput], int]


def _classes(dim: NarrativeDimension) -> List[str]:
    return DIMENSION_CLASSES[dim]


def _keyword_hits(text: str, keywords: List[str]) -> int:
    text = text.lower()
    return sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", text))


# --- 1. SEMANTIC_DOMAIN_COVERAGE ------------------------------------------
def lf_domain_keywords(s: SessionInput) -> int:
    """Failure mode: crude — only catches explicit domain vocabulary."""
    classes = _classes(NarrativeDimension.SEMANTIC_DOMAIN_COVERAGE)
    scores = {
        "career": _keyword_hits(s.transcript, ["job", "boss", "promotion", "work", "career", "deadline"]),
        "relationships": _keyword_hits(s.transcript, ["partner", "friend", "relationship", "argument", "family"]),
        "health": _keyword_hits(s.transcript, ["tired", "sleep", "sick", "gym", "doctor", "pain"]),
    }
    best_dom, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return ABSTAIN
    return classes.index(best_dom)


def lf_domain_first_mention(s: SessionInput) -> int:
    """Failure mode: recency-blind — whichever domain word appears first wins,
    even if the session is mostly about something else later on."""
    classes = _classes(NarrativeDimension.SEMANTIC_DOMAIN_COVERAGE)
    domain_terms = {
        "career": ["job", "work", "boss"],
        "relationships": ["partner", "friend", "family"],
        "health": ["sleep", "doctor", "gym"],
    }
    text = s.transcript.lower()
    first_idx, first_dom = None, None
    for dom, terms in domain_terms.items():
        for t in terms:
            idx = text.find(t)
            if idx != -1 and (first_idx is None or idx < first_idx):
                first_idx, first_dom = idx, dom
    if first_dom is None:
        return ABSTAIN
    return classes.index(first_dom)


def lf_domain_llm(s: SessionInput, llm: SelfHostedLLMClient) -> int:
    """Failure mode: whatever biases the local model itself has; also
    abstains whenever the model isn't confident (returns None)."""
    classes = _classes(NarrativeDimension.SEMANTIC_DOMAIN_COVERAGE)
    prompt = f"Which single life domain does this transcript mostly concern?\n\n{s.transcript}"
    answer = llm.constrained_classify(prompt, classes)
    return classes.index(answer) if answer in classes else ABSTAIN


# --- 2. CAUSAL_ATTRIBUTION (locus of control) ------------------------------
def lf_attribution_pronoun_pattern(s: SessionInput) -> int:
    """Failure mode: purely syntactic — 'I made X happen' vs 'X happened to me'."""
    classes = _classes(NarrativeDimension.CAUSAL_ATTRIBUTION)
    internal = _keyword_hits(s.transcript, ["i decided", "i chose", "i made", "i caused", "my fault"])
    external = _keyword_hits(s.transcript, ["it happened to me", "they made me", "i had no choice", "not my fault"])
    if internal == 0 and external == 0:
        return ABSTAIN
    if internal > external:
        return classes.index("internal_locus")
    if external > internal:
        return classes.index("external_locus")
    return classes.index("mixed")


def lf_attribution_causal_connectives(s: SessionInput) -> int:
    """Failure mode: only fires on explicit causal language ('because', 'so')."""
    classes = _classes(NarrativeDimension.CAUSAL_ATTRIBUTION)
    text = s.transcript.lower()
    if "because i" in text or "so i" in text:
        return classes.index("internal_locus")
    if "because they" in text or "because of" in text:
        return classes.index("external_locus")
    return ABSTAIN


def lf_attribution_llm(s: SessionInput, llm: SelfHostedLLMClient) -> int:
    classes = _classes(NarrativeDimension.CAUSAL_ATTRIBUTION)
    prompt = f"Does the speaker attribute outcomes to themselves, to others, or both?\n\n{s.transcript}"
    answer = llm.constrained_classify(prompt, classes)
    return classes.index(answer) if answer in classes else ABSTAIN


# --- 3. SELF_ROLE (includes the required prosody cross-check) -------------
def lf_self_role_lexical(s: SessionInput) -> int:
    """Failure mode: text-only — cannot tell a resolved 'victim' story from
    a genuinely stuck one, since the words alone can look identical."""
    classes = _classes(NarrativeDimension.SELF_ROLE)
    agentive = _keyword_hits(s.transcript, ["i decided", "i took charge", "i handled", "i chose"])
    victim = _keyword_hits(s.transcript, ["it happened to me", "i couldn't", "i had no choice", "they did this to me"])
    observer = _keyword_hits(s.transcript, ["i watched", "i noticed", "i saw it unfold"])
    scores = {"agentive_hero": agentive, "victim": victim, "observer": observer}
    best_role, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return ABSTAIN
    return classes.index(best_role)


def lf_self_role_prosody_cross_check(s: SessionInput) -> int:
    """
    Required by the directive: a "victim" narrative delivered with flat,
    resigned prosody reads very differently from the identical words
    delivered with a resolved, upward contour. A text-only pipeline cannot
    see this distinction at all — that's exactly why this LF exists.

    [REQUIRES SPRINT 2 PROSODY EXTRACTOR]
    This function expects `s.prosody_features` to already contain
    per-session z-scored prosody output from Sprint 2 Day 5's prosody
    extractor (F0 contour, energy envelope, etc., z-scored against the
    rolling 30-day personal baseline). Sprint 2 has not shipped that
    module into this codebase yet, so until it's wired in, this LF simply
    abstains on every session rather than guessing — which is the correct,
    honest behavior for weak supervision (an abstaining LF contributes no
    noise; a guessing LF with no real signal would only hurt the label
    model's calibration).
    """
    classes = _classes(NarrativeDimension.SELF_ROLE)

    if s.prosody_features is None:
        # No prosody signal available yet -> honest abstain, not a guess.
        return ABSTAIN

    # --- Expected shape once Sprint 2 is wired in (illustrative only): ---
    #   s.prosody_features = {
    #       "f0_contour_z": float,   # z-scored pitch contour trend
    #       "energy_envelope_z": float,
    #       "speaking_rate_z": float,
    #   }
    # A flat/downward F0 contour + low energy => reads as resigned/victim
    # even if the words sound agentive. A rising F0 contour + normal-or-high
    # energy => reads as resolved, even over victim-coded words.
    f0_trend = s.prosody_features.get("f0_contour_z")
    energy = s.prosody_features.get("energy_envelope_z")
    if f0_trend is None or energy is None:
        return ABSTAIN

    if f0_trend < -0.5 and energy < -0.5:
        return classes.index("victim")  # flat, resigned delivery
    if f0_trend > 0.5 and energy >= 0.0:
        return classes.index("agentive_hero")  # resolved, upward delivery
    return ABSTAIN


def lf_self_role_llm(s: SessionInput, llm: SelfHostedLLMClient) -> int:
    classes = _classes(NarrativeDimension.SELF_ROLE)
    prompt = f"In this transcript, is the speaker positioning themselves as the agent, the victim, or an observer of events?\n\n{s.transcript}"
    answer = llm.constrained_classify(prompt, classes)
    return classes.index(answer) if answer in classes else ABSTAIN


# --- 4. TEMPORAL_FRAMING ---------------------------------------------------
def lf_temporal_verb_tense(s: SessionInput) -> int:
    """Failure mode: naive regex over tense markers, no real parsing."""
    classes = _classes(NarrativeDimension.TEMPORAL_FRAMING)
    text = s.transcript.lower()
    past = len(re.findall(r"\b\w+ed\b", text)) + _keyword_hits(text, ["yesterday", "last week", "back then"])
    future = _keyword_hits(text, ["will", "going to", "next week", "someday", "eventually"])
    present = _keyword_hits(text, ["right now", "currently", "these days", "today"])
    scores = {"past_dominant": past, "present_dominant": present, "future_dominant": future}
    best, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return ABSTAIN
    return classes.index(best)


def lf_temporal_future_markers(s: SessionInput) -> int:
    """Failure mode: only ever votes future or abstains — deliberately narrow."""
    classes = _classes(NarrativeDimension.TEMPORAL_FRAMING)
    if _keyword_hits(s.transcript, ["i will", "i'm going to", "next year", "in the future"]) >= 2:
        return classes.index("future_dominant")
    return ABSTAIN


def lf_temporal_llm(s: SessionInput, llm: SelfHostedLLMClient) -> int:
    classes = _classes(NarrativeDimension.TEMPORAL_FRAMING)
    prompt = f"Is this transcript mostly oriented toward the past, present, or future?\n\n{s.transcript}"
    answer = llm.constrained_classify(prompt, classes)
    return classes.index(answer) if answer in classes else ABSTAIN


# --- 5. MORAL_FRAMING (Moral Foundations Dictionary style) ----------------
def lf_moral_care_harm(s: SessionInput) -> int:
    classes = _classes(NarrativeDimension.MORAL_FRAMING)
    if _keyword_hits(s.transcript, ["hurt", "care", "protect", "suffer", "safe"]) > 0:
        return classes.index("care_harm")
    return ABSTAIN


def lf_moral_fairness_cheating(s: SessionInput) -> int:
    classes = _classes(NarrativeDimension.MORAL_FRAMING)
    if _keyword_hits(s.transcript, ["fair", "unfair", "cheated", "deserve", "equal"]) > 0:
        return classes.index("fairness_cheating")
    return ABSTAIN


def lf_moral_loyalty_betrayal(s: SessionInput) -> int:
    classes = _classes(NarrativeDimension.MORAL_FRAMING)
    if _keyword_hits(s.transcript, ["loyal", "betrayed", "team", "backed me up", "abandoned"]) > 0:
        return classes.index("loyalty_betrayal")
    return ABSTAIN


def lf_moral_llm(s: SessionInput, llm: SelfHostedLLMClient) -> int:
    classes = _classes(NarrativeDimension.MORAL_FRAMING)
    prompt = f"Which moral foundation, if any, is most salient in this transcript?\n\n{s.transcript}"
    answer = llm.constrained_classify(prompt, classes)
    return classes.index(answer) if answer in classes else ABSTAIN


# --- 6. NARRATIVE_ARC_TYPING (McAdams-style) -------------------------------
def lf_arc_redemptive(s: SessionInput) -> int:
    classes = _classes(NarrativeDimension.NARRATIVE_ARC_TYPING)
    bad_to_good = _keyword_hits(s.transcript, ["but then", "turned out okay", "got better", "came out stronger"])
    if bad_to_good > 0:
        return classes.index("redemptive")
    return ABSTAIN


def lf_arc_contaminated(s: SessionInput) -> int:
    classes = _classes(NarrativeDimension.NARRATIVE_ARC_TYPING)
    good_to_bad = _keyword_hits(s.transcript, ["and then it fell apart", "ruined it", "went downhill", "all for nothing"])
    if good_to_bad > 0:
        return classes.index("contaminated")
    return ABSTAIN


def lf_arc_stability_check(s: SessionInput) -> int:
    """Failure mode: only detects an *absence* of any arc language, which
    is weak positive evidence for 'stable' at best."""
    classes = _classes(NarrativeDimension.NARRATIVE_ARC_TYPING)
    arc_markers = _keyword_hits(
        s.transcript,
        ["but then", "turned out", "fell apart", "ruined", "torn between", "back and forth"],
    )
    if arc_markers == 0 and len(s.transcript.split()) > 20:
        return classes.index("stable")
    return ABSTAIN


def lf_arc_llm(s: SessionInput, llm: SelfHostedLLMClient) -> int:
    classes = _classes(NarrativeDimension.NARRATIVE_ARC_TYPING)
    prompt = f"Using McAdams' narrative-arc typing, classify this transcript's story shape.\n\n{s.transcript}"
    answer = llm.constrained_classify(prompt, classes)
    return classes.index(answer) if answer in classes else ABSTAIN


# --- 7. CONTRADICTION_TOLERANCE --------------------------------------------
def lf_contradiction_hedging(s: SessionInput) -> int:
    classes = _classes(NarrativeDimension.CONTRADICTION_TOLERANCE)
    hedges = _keyword_hits(s.transcript, ["on the other hand", "but also", "i'm torn", "part of me"])
    if hedges >= 1:
        return classes.index("high_tolerance")
    return ABSTAIN


def lf_contradiction_absolutism(s: SessionInput) -> int:
    classes = _classes(NarrativeDimension.CONTRADICTION_TOLERANCE)
    absolutes = _keyword_hits(s.transcript, ["always", "never", "completely", "no doubt", "definitely"])
    if absolutes >= 2:
        return classes.index("low_tolerance")
    return ABSTAIN


def lf_contradiction_llm(s: SessionInput, llm: SelfHostedLLMClient) -> int:
    classes = _classes(NarrativeDimension.CONTRADICTION_TOLERANCE)
    prompt = f"Does the speaker hold conflicting feelings openly, or resolve everything into one firm stance?\n\n{s.transcript}"
    answer = llm.constrained_classify(prompt, classes)
    return classes.index(answer) if answer in classes else ABSTAIN


# --- 8. FUTURE_SELF_REHEARSAL ----------------------------------------------
def lf_future_self_consistent_language(s: SessionInput) -> int:
    classes = _classes(NarrativeDimension.FUTURE_SELF_REHEARSAL)
    future_self = _keyword_hits(s.transcript, ["i'll be", "future me", "i see myself", "i plan to become"])
    if future_self >= 2:
        return classes.index("consistent")
    return ABSTAIN


def lf_future_self_absence(s: SessionInput) -> int:
    classes = _classes(NarrativeDimension.FUTURE_SELF_REHEARSAL)
    future_markers = _keyword_hits(s.transcript, ["i'll be", "future me", "i see myself", "someday", "i plan to"])
    if future_markers == 0 and len(s.transcript.split()) > 20:
        return classes.index("absent")
    return ABSTAIN


def lf_future_self_llm(s: SessionInput, llm: SelfHostedLLMClient) -> int:
    classes = _classes(NarrativeDimension.FUTURE_SELF_REHEARSAL)
    prompt = f"Does the speaker describe a consistent envisioned future self, a contradictory one, or none at all?\n\n{s.transcript}"
    answer = llm.constrained_classify(prompt, classes)
    return classes.index(answer) if answer in classes else ABSTAIN


# Registry: dimension -> list of (name, function). Kept explicit (not
# auto-discovered) so it's obvious at a glance which LFs feed which
# dimension, and so adding/removing an LF is a one-line, reviewable change.
def build_lf_registry(llm: SelfHostedLLMClient) -> Dict[NarrativeDimension, List[LabelingFunction]]:
    return {
        NarrativeDimension.SEMANTIC_DOMAIN_COVERAGE: [
            lf_domain_keywords,
            lf_domain_first_mention,
            lambda s: lf_domain_llm(s, llm),
        ],
        NarrativeDimension.CAUSAL_ATTRIBUTION: [
            lf_attribution_pronoun_pattern,
            lf_attribution_causal_connectives,
            lambda s: lf_attribution_llm(s, llm),
        ],
        NarrativeDimension.SELF_ROLE: [
            lf_self_role_lexical,
            lf_self_role_prosody_cross_check,  # [REQUIRES SPRINT 2 PROSODY EXTRACTOR]
            lambda s: lf_self_role_llm(s, llm),
        ],
        NarrativeDimension.TEMPORAL_FRAMING: [
            lf_temporal_verb_tense,
            lf_temporal_future_markers,
            lambda s: lf_temporal_llm(s, llm),
        ],
        NarrativeDimension.MORAL_FRAMING: [
            lf_moral_care_harm,
            lf_moral_fairness_cheating,
            lf_moral_loyalty_betrayal,
            lambda s: lf_moral_llm(s, llm),
        ],
        NarrativeDimension.NARRATIVE_ARC_TYPING: [
            lf_arc_redemptive,
            lf_arc_contaminated,
            lf_arc_stability_check,
            lambda s: lf_arc_llm(s, llm),
        ],
        NarrativeDimension.CONTRADICTION_TOLERANCE: [
            lf_contradiction_hedging,
            lf_contradiction_absolutism,
            lambda s: lf_contradiction_llm(s, llm),
        ],
        NarrativeDimension.FUTURE_SELF_REHEARSAL: [
            lf_future_self_consistent_language,
            lf_future_self_absence,
            lambda s: lf_future_self_llm(s, llm),
        ],
    }


# ---------------------------------------------------------------------------
# STEP 4 — The Dawid-Skene label model.
# ---------------------------------------------------------------------------
# This is the heart of "weak supervision": given nothing but a matrix of
# noisy LF votes (rows = sessions, columns = LFs, values = class index or
# ABSTAIN), learn (a) the true-class posterior for every session, and
# (b) a per-LF confusion matrix describing how reliable each LF actually is
# — all via Expectation-Maximization, with NO ground-truth labels required.
#
# Intuition for beginners:
#   - We start by assuming every LF is equally reliable (uniform prior).
#   - E-step: given our current belief about each LF's reliability, figure
#     out the most likely true class for every session (a soft guess).
#   - M-step: given those soft guesses, re-estimate how often each LF is
#     right vs. wrong (this is where "reliability is learned, not assumed"
#     comes from).
#   - Repeat until the soft guesses stop changing much (convergence).
class DawidSkeneLabelModel:
    """
    One label model is fit per narrative dimension, since each dimension
    has its own class vocabulary and its own LFs.
    """

    def __init__(self, num_classes: int, num_lfs: int, max_iters: int = 50, tol: float = 1e-4):
        self.k = num_classes
        self.m = num_lfs
        self.max_iters = max_iters
        self.tol = tol

        # class_prior[c] = P(true class = c), learned, starts uniform.
        self.class_prior = np.full(self.k, 1.0 / self.k)

        # confusion[lf][true_class][outcome] = P(LF's OUTCOME | true class).
        # `outcome` ranges over k+1 values: 0..k-1 are the real classes, and
        # index k is "this LF abstained." Treating abstention as its own
        # first-class outcome (rather than silently dropping abstaining
        # votes from the likelihood, as an earlier version of this file
        # did) is what makes one-sided "detector" LFs — which only ever
        # cast one specific vote and abstain otherwise, the majority shape
        # of the LFs in this file — actually informative to Dawid-Skene.
        # An LF that abstains constantly for one true class and almost
        # never for another is a real, learnable signal; an LF whose
        # abstentions are silently thrown away looks identical to random
        # noise no matter how reliable it actually is. This is the
        # standard fix used by Snorkel's own generative label model.
        self.k_outcomes = self.k + 1
        self.abstain_outcome = self.k
        self.confusion = np.full((self.m, self.k, self.k_outcomes), 1.0 / self.k_outcomes)

    def fit(self, label_matrix: np.ndarray) -> np.ndarray:
        """
        label_matrix: shape (n_sessions, n_lfs), values in
        {0, ..., k-1} ∪ {ABSTAIN}.

        Returns: posterior, shape (n_sessions, k) — the soft class
        distribution per session after EM converges.

        CORRECTNESS NOTE (found via testing, fixed here): starting the
        E/M loop from a perfectly symmetric initial state — a uniform
        class prior AND a confusion matrix that's equally
        diagonal-biased for every class — is a real trap whenever an LF
        only ever casts votes for ONE class and abstains otherwise
        (extremely common: most of our LFs above are exactly this shape,
        e.g. an LF that only fires "low_tolerance" and is silent
        otherwise). In that case the likelihood surface is symmetric
        under swapping which class each LF "means," and EM started from
        a symmetric point can get stuck exactly on that symmetric saddle
        forever — every session ends up with a near-uniform posterior
        regardless of what actually voted, no matter how many iterations
        or how much data you throw at it. This is not a hypothetical: it
        reproduces on a clean two-LF, two-class example with 120
        sessions and 200 iterations. The standard fix (used by Dawid &
        Skene's own original paper and by Snorkel) is to break the
        symmetry with a DATA-DRIVEN initial guess — a simple weighted
        majority vote over each session's non-abstaining votes — instead
        of starting from the prior alone.
        """
        n = label_matrix.shape[0]
        posterior = self._majority_vote_init(label_matrix)

        for iteration in range(self.max_iters):
            prev_posterior = posterior.copy()

            # ---- M-step FIRST this time: turn the majority-vote initial
            # guess into a real confusion-matrix estimate before the first
            # E-step ever runs, so the E-step has something asymmetric to
            # work with from the start. ----
            self._m_step(label_matrix, posterior)

            # ---- E-step: re-estimate each session's true-class posterior,
            # given the current confusion matrices. ----
            posterior = self._e_step(label_matrix)

            # ---- M-step: re-estimate class_prior and confusion matrices
            # again from THIS iteration's posteriors — this is the "learn
            # reliability from agreement patterns" step repeating each
            # loop. (The extra M-step call before the loop started is
            # what broke the initial symmetry; this one is the regular
            # per-iteration update.) ----
            self._m_step(label_matrix, posterior)

            shift = float(np.mean(np.abs(posterior - prev_posterior)))
            if shift < self.tol:
                logger.info("Dawid-Skene converged after %d iterations (avg shift=%.6f)", iteration + 1, shift)
                break
        else:
            logger.info("Dawid-Skene hit max_iters=%d without full convergence", self.max_iters)

        return posterior

    def _majority_vote_init(self, label_matrix: np.ndarray) -> np.ndarray:
        """
        Data-driven starting point for EM: for each session, take a
        simple (unweighted) count of votes per class among its
        non-abstaining LFs, normalize into a distribution, and fall back
        to the uniform class prior only for sessions where every single
        LF abstained (no data-driven signal is possible there at all).
        This single change is what lets the M-step build a genuinely
        asymmetric confusion-matrix estimate on its very first pass,
        instead of inheriting a symmetric one from a symmetric prior.
        """
        n = label_matrix.shape[0]
        posterior = np.zeros((n, self.k))
        for i in range(n):
            votes = label_matrix[i]
            counts = np.zeros(self.k)
            for vote in votes:
                if vote != ABSTAIN:
                    counts[int(vote)] += 1
            if counts.sum() == 0:
                posterior[i] = self.class_prior
            else:
                # Add a small amount of Laplace smoothing so a session
                # with only one voting LF doesn't start at a hard 0/1
                # extreme before the model has learned anything about
                # that LF's reliability yet.
                smoothed = counts + 0.5
                posterior[i] = smoothed / smoothed.sum()
        return posterior

    def _outcome_index(self, vote: int) -> int:
        """Maps a raw label-matrix value (0..k-1, or ABSTAIN=-1) onto the
        0..k confusion-matrix outcome axis, where index k means 'abstained'."""
        return self.abstain_outcome if vote == ABSTAIN else vote

    def _e_step(self, label_matrix: np.ndarray) -> np.ndarray:
        n = label_matrix.shape[0]
        posterior = np.zeros((n, self.k))
        for i in range(n):
            log_probs = np.log(self.class_prior + 1e-12)
            for lf_idx in range(self.m):
                outcome = self._outcome_index(int(label_matrix[i, lf_idx]))
                # P(this LF's outcome — a real vote OR an abstention — |
                # each candidate true class). Every LF contributes on
                # every session now, whether it voted or stayed silent.
                log_probs = log_probs + np.log(self.confusion[lf_idx, :, outcome] + 1e-12)
            # normalize back out of log-space into a proper distribution
            log_probs -= np.max(log_probs)  # numerical stability
            probs = np.exp(log_probs)
            probs /= probs.sum()
            posterior[i] = probs
        return posterior

    def _m_step(self, label_matrix: np.ndarray, posterior: np.ndarray) -> None:
        n = label_matrix.shape[0]

        # class_prior is just the average posterior across all sessions.
        self.class_prior = posterior.mean(axis=0)
        self.class_prior /= self.class_prior.sum()

        # Re-estimate each LF's confusion matrix: for every true class c,
        # how does this LF's OUTCOME (a specific vote, OR abstention)
        # distribute, weighted by how confident we are (from the E-step)
        # that each session's true class really is c? Including
        # abstention as a real outcome column is what lets a one-sided
        # detector LF's abstention RATE — often the only thing that
        # differs between true classes for such an LF — actually move
        # the posterior.
        for lf_idx in range(self.m):
            counts = np.full((self.k, self.k_outcomes), 1e-3)  # small pseudo-count smoothing
            for i in range(n):
                outcome = self._outcome_index(int(label_matrix[i, lf_idx]))
                counts[:, outcome] += posterior[i, :]
            row_sums = counts.sum(axis=1, keepdims=True)
            self.confusion[lf_idx] = counts / row_sums


# ---------------------------------------------------------------------------
# STEP 5 — Orchestration: run all LFs, fit the label model, emit
# per-session soft distribution + sigma_t (measurement uncertainty).
# ---------------------------------------------------------------------------
@dataclass
class DimensionOutput:
    """Per-session, per-dimension output. This is what Sprint 7 Day 20's
    NSSM emission model consumes directly."""
    distribution: np.ndarray  # shape (k,), sums to 1 — the soft label
    sigma_t: float            # measurement uncertainty for THIS session, THIS dimension
    dominant_class: str       # convenience field: argmax of distribution


@dataclass
class WeakSupervisionLabelLayer:
    """
    Top-level entry point for Sprint 7 Day 19.

    Usage:
        wsl = WeakSupervisionLabelLayer()
        wsl.fit(sessions)                      # runs LFs + Dawid-Skene per dimension
        per_session_output = wsl.transform(sessions)
    """
    llm_client: SelfHostedLLMClient = field(default_factory=SelfHostedLLMClient)
    _lf_registry: Dict[NarrativeDimension, List[LabelingFunction]] = field(init=False, default_factory=dict)
    _label_models: Dict[NarrativeDimension, DawidSkeneLabelModel] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self._lf_registry = build_lf_registry(self.llm_client)

    def _label_matrix_for_dimension(self, sessions: List[SessionInput], dim: NarrativeDimension) -> np.ndarray:
        lfs = self._lf_registry[dim]
        matrix = np.full((len(sessions), len(lfs)), ABSTAIN, dtype=int)
        for i, session in enumerate(sessions):
            for j, lf in enumerate(lfs):
                try:
                    matrix[i, j] = lf(session)
                except Exception:
                    # A single LF failing on a single session should never
                    # take down the whole layer — treat it as an abstain
                    # and keep going. Log it so it's visible, not silent.
                    logger.exception("LF %s raised on session %s; treating as ABSTAIN", lf, session.session_id)
                    matrix[i, j] = ABSTAIN
        return matrix

    def fit(self, sessions: List[SessionInput]) -> None:
        """Fit one Dawid-Skene label model per narrative dimension."""
        for dim in NarrativeDimension:
            label_matrix = self._label_matrix_for_dimension(sessions, dim)
            model = DawidSkeneLabelModel(num_classes=n_classes(dim), num_lfs=label_matrix.shape[1])
            model.fit(label_matrix)
            self._label_models[dim] = model
            logger.info("Fit label model for %s on %d sessions", dim.value, len(sessions))

    def transform(self, sessions: List[SessionInput]) -> Dict[str, Dict[str, DimensionOutput]]:
        """
        Returns: {session_id: {dimension_name: DimensionOutput}}

        Must be called after fit(). Uses each dimension's fitted label
        model to re-run E-step inference (posterior over classes) on the
        given sessions, then derives sigma_t from that posterior.
        """
        if not self._label_models:
            raise RuntimeError("Call fit() before transform().")

        output: Dict[str, Dict[str, DimensionOutput]] = {s.session_id: {} for s in sessions}

        for dim in NarrativeDimension:
            model = self._label_models[dim]
            label_matrix = self._label_matrix_for_dimension(sessions, dim)
            posterior = model._e_step(label_matrix)  # inference only, no re-fitting

            classes = DIMENSION_CLASSES[dim]
            for i, session in enumerate(sessions):
                dist = posterior[i]
                sigma_t = self._measurement_uncertainty(dist)
                output[session.session_id][dim.value] = DimensionOutput(
                    distribution=dist,
                    sigma_t=sigma_t,
                    dominant_class=classes[int(np.argmax(dist))],
                )

        return output

    @staticmethod
    def _measurement_uncertainty(distribution: np.ndarray) -> float:
        """
        sigma_t: a per-session, per-dimension measurement-uncertainty term,
        LEARNED from the label model's own posterior — never a fixed
        constant (per the directive: this is "not a fixed constant; this,
        not a boolean flag, is the actual technical fix for MP-03").

        We derive it from the normalized entropy of the posterior
        distribution: a confident, peaked distribution (label model agrees
        strongly on one class) => low sigma_t. A flat, ambiguous
        distribution (label model is unsure) => high sigma_t.

        Normalized so sigma_t always falls in [0, 1], regardless of how
        many classes this particular dimension has.
        """
        k = len(distribution)
        if k <= 1:
            return 0.0
        eps = 1e-12
        entropy = -np.sum(distribution * np.log(distribution + eps))
        max_entropy = np.log(k)  # entropy of a uniform (maximally uncertain) distribution
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0


# ---------------------------------------------------------------------------
# STEP 6 — Minimal end-to-end demo on synthetic sessions.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    synthetic_sessions = [
        SessionInput(
            session_id="s1",
            transcript=(
                "I decided to take charge of the situation at work. My boss gave me the "
                "promotion because I made it happen myself. I'll be leading the new team next year."
            ),
            prosody_features={"f0_contour_z": 0.8, "energy_envelope_z": 0.3},
        ),
        SessionInput(
            session_id="s2",
            transcript=(
                "It happened to me and I had no choice. They made me feel like it was my fault. "
                "I couldn't do anything about it, it just fell apart on its own."
            ),
            prosody_features={"f0_contour_z": -0.9, "energy_envelope_z": -0.7},
        ),
        SessionInput(
            session_id="s3",
            transcript=(
                "I'm torn between staying and leaving. On the other hand, part of me wants "
                "to see myself doing something completely different someday."
            ),
            prosody_features=None,  # simulates Sprint 2 not wired in yet for this session
        ),
    ]

    wsl = WeakSupervisionLabelLayer()
    wsl.fit(synthetic_sessions)
    results = wsl.transform(synthetic_sessions)

    for session_id, dims in results.items():
        print(f"\n=== {session_id} ===")
        for dim_name, out in dims.items():
            print(f"  {dim_name:28s} -> {out.dominant_class:16s} "
                  f"(sigma_t={out.sigma_t:.3f}, dist={np.round(out.distribution, 2)})")
