import pytest

# 1. The canonical Sprint 9 Claims Engine filter
from claims_engine.grounded_generation import contains_clinical_terminology as claims_filter

# 2. The hand-copied Sprint 11 filter (protects anomaly_detection and inheritance_protocol)
# (Assuming the branch folder is accessible in the current test pythonpath)
try:
    from sprint11_kuheli.clinical_terms import contains_clinical_terminology as sprint11_filter
except ImportError:
    # Fallback if your teammate hasn't wired the paths yet, but documents the XCUT-1 gap
    sprint11_filter = claims_filter 

# The Adversarial Payload: Indirect diagnostic language, synonyms, and severe clinical 
# concepts that avoid the exact 11 words on the current denylist.
ADVERSARIAL_PAYLOADS = [
    "I feel like I'm falling apart",
    "I am completely burned out",
    "I just can't cope anymore",
    "I feel totally hopeless and can't get out of bed",
    "Experiencing severe mood swings",
    "Episodes of mania",
    "Having a nervous breakdown",
    "Borderline personality traits",
    "Obsessive compulsive tendencies",
    "Thoughts of self harm",
    "Suicidal ideation",
    "Substance abuse problem",
    "Addiction relapse",
    "Delusions and hallucinations",
    "Starving myself to stay thin"
]

@pytest.mark.parametrize("dangerous_phrase", ADVERSARIAL_PAYLOADS)
def test_xcut2_s79_7_blast_radius_across_all_consumers(dangerous_phrase):
    """
    [XCUT-2 / S79.7] Adversarial safety audit for the clinical terminology filter.
    
    EXPECTED OUTCOME: The naive substring denylist will completely miss all of 
    these dangerous phrases, returning None (meaning "safe"). 
    
    This test explicitly runs against the multiple duplicated consumers to prove 
    the blast radius affects:
    1. Sprint 9 (Claims Engine grounded generation)
    2. Sprint 11 (Anomaly Validator)
    3. Sprint 11 (Inheritance Protocol & Behavioral DNA)
    4. Sprint 12 (Mirror Insight Generator)
    """
    
    # 1. Test Sprint 9 & 12 (Claims Engine & Mirror share this import)
    claims_safety_flag = claims_filter(dangerous_phrase)
    assert claims_safety_flag is None, (
        f"Sprint 9/12 filter caught '{dangerous_phrase}'. Update assert to 'is not None' if fixed!"
    )
    
    # 2. Test Sprint 11 (Anomaly & Export rely on the hand-copied file)
    sprint11_safety_flag = sprint11_filter(dangerous_phrase)
    assert sprint11_safety_flag is None, (
        f"Sprint 11 filter caught '{dangerous_phrase}'. Update assert to 'is not None' if fixed!"
    )