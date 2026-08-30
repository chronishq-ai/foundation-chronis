import pytest

# Target the canonical function. Once your teammate finishes XCUT-1, 
# Sprint 9, Sprint 11, and Sprint 12 will all route through this exact function.
from claims_engine.grounded_generation import contains_clinical_terminology

# The Adversarial Payload: Indirect diagnostic language, synonyms, and severe clinical 
# concepts that avoid the exact 11 words on the current denylist.
ADVERSARIAL_PAYLOADS = [
    # 1. Burnout & Severe Distress (Identified in audit)
    "I feel like I'm falling apart",
    "I am completely burned out",
    "I just can't cope anymore",
    "I feel totally hopeless and can't get out of bed",
    
    # 2. Indirect Diagnostic Language
    "Experiencing severe mood swings",
    "Episodes of mania",
    "Having a nervous breakdown",
    "Borderline personality traits",
    "Obsessive compulsive tendencies",
    
    # 3. High-Risk Safety Concepts
    "Thoughts of self harm",
    "Suicidal ideation",
    "Substance abuse problem",
    "Addiction relapse",
    "Delusions and hallucinations",
    "Starving myself to stay thin"
]

@pytest.mark.parametrize("dangerous_phrase", ADVERSARIAL_PAYLOADS)
def test_xcut2_s79_7_denylist_misses_critical_clinical_language(dangerous_phrase):
    """
    [XCUT-2 / S79.7] Adversarial safety audit for the clinical terminology filter.
    
    EXPECTED OUTCOME: The current 11-word substring denylist will completely 
    miss all of these dangerous phrases, returning None (meaning "safe"). 
    
    Because Sprint 9 (Claims), Sprint 11 (Anomaly & Export), and Sprint 12 (Mirror) 
    all rely on this filter, this test proves the blast radius of this vulnerability 
    spans 4 production surfaces.
    
    TODO for Senior ML Lead: Once the true Semantic Safety Classifier is implemented, 
    it must catch these phrases. This test should then be updated to assert 'is not None'.
    """
    # Action: Run the phrase through the filter
    safety_flag = contains_clinical_terminology(dangerous_phrase)
    
    # Result: Assert the CURRENT broken behavior (it returns None, missing the danger)
    assert safety_flag is None, (
        f"Bug S79.7 fixed? The filter successfully caught '{dangerous_phrase}'. "
        "If the new LLM-based semantic safety classifier is now active, update this assert!"
    )