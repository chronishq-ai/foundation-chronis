from datetime import datetime, timedelta

from anomaly_detection import detect_acute_anomalies, validate_anomaly_copy
from behavioral_dna import build_behavioral_dna_export
from echo_detection import find_echoes
from inheritance_protocol import build_inheritance_letter
from second_brain import build_decision_replication_snapshot
from silence_map import SilenceInput, SilenceMap
from social_graph import SocialGraph, VocalFingerprint
from upstream_interfaces import SocialContext
from weather_forecast import RegimeState, WeatherForecastEngine, WeatherInput
from signing import DeviceSigner
from tests.fixtures.synthetic_user_profile import build_behavioral_state_records, build_claims, build_session_excerpts, USER_ID

class Generated:
    def __init__(self, text, citation_chain):
        self.text = text
        self.citation_chain = citation_chain

def generator(claim, divergence_state, candidate_excerpts, llm_client):
    return Generated(f"You often describe {claim.claim_id} in a grounded way.", [candidate_excerpts[0].session_id])

def test_sprint11_shared_surrogate_profile_all_eight_modules():
    user_id = USER_ID
    records = build_behavioral_state_records()
    assert all(r.user_id == user_id for r in records)

    echoes = find_echoes(records)
    assert any(e.echo_type == "conversation" for e in echoes)

    silence = SilenceMap().classify(SilenceInput(user_id, 8.0, False, True, 0.1))
    assert silence.classification == "attentive"

    social = SocialGraph().build(user_id, [
        VocalFingerprint(user_id, "session_1", [1.0,0.0,0.0]),
        VocalFingerprint(user_id, "session_2", [0.99,0.05,0.0]),
    ])
    assert len(social.nodes) == 1

    current = WeatherInput(user_id, datetime(2026,8,24,12), [1.0,0.0,0.0], RegimeState(0,[1.0]), energy=.7, social_engagement=.7, stress=.4, productivity=.7)
    history = [WeatherInput(user_id, datetime(2026,1,1)+timedelta(days=i), [1.0,0.0,0.0], RegimeState(0,[1.0]), energy=.7, social_engagement=.7, stress=.4, productivity=.7) for i in range(45)]
    weather = WeatherForecastEngine().forecast(current=current, history=history)
    assert weather is not None

    anomalies = detect_acute_anomalies(records)
    assert isinstance(anomalies, list)
    validate_anomaly_copy("Your recent pattern differed from your usual routine.")

    claims = build_claims()
    signer = DeviceSigner.generate()
    lexicon = {"recurring_words": ["overwhelmed", "trying"], "style": "concise"}
    export = build_behavioral_dna_export(user_id, claims, lexicon, {"cluster_count": len(social.nodes)}, signer)
    assert export.is_signed and export.verify_signature(signer.public_key_bytes())

    snapshot = build_decision_replication_snapshot(user_id, claims)
    assert {c.claim_id for c in snapshot.all_claims} >= {"claim-002", "claim-003"}

    excerpts = build_session_excerpts()
    letter = build_inheritance_letter(export, None, excerpts, generator, None, signer)
    assert letter.is_signed and letter.verify_signature(signer.public_key_bytes())
    assert not any(ex.text in letter.letter_text for ex in excerpts)
