import pytest

from social_graph import (
    SocialGraph,
    SocialGraphError,
    SocialGraphResult,
    VocalFingerprint,
)


def fp(user, session, values):
    return VocalFingerprint(
        user_id=user,
        session_id=session,
        values=values,
    )


def test_builds_social_graph():
    graph = SocialGraph()

    result = graph.build(
        "user_001",
        [
            fp("user_001", "s1", [1.0, 0.0]),
            fp("user_001", "s2", [0.99, 0.1]),
        ],
    )

    assert isinstance(result, SocialGraphResult)
    assert len(result.nodes) == 1


def test_similar_sessions_cluster():
    graph = SocialGraph(similarity_threshold=0.8)

    result = graph.build(
        "user_001",
        [
            fp("user_001", "s1", [1.0, 0.0]),
            fp("user_001", "s2", [1.0, 0.0]),
        ],
    )

    assert result.nodes[0].session_ids == ("s1", "s2")


def test_different_fingerprints_remain_separate():
    graph = SocialGraph(similarity_threshold=0.8)

    result = graph.build(
        "user_001",
        [
            fp("user_001", "s1", [1.0, 0.0]),
            fp("user_001", "s2", [0.0, 1.0]),
        ],
    )

    assert len(result.nodes) == 2


def test_cross_user_records_are_not_included():
    graph = SocialGraph()

    result = graph.build(
        "user_001",
        [
            fp("user_001", "s1", [1.0, 0.0]),
            fp("user_002", "s2", [1.0, 0.0]),
        ],
    )

    assert len(result.nodes) == 1
    assert result.nodes[0].session_ids == ("s1",)


def test_node_is_user_scoped():
    graph = SocialGraph()

    result = graph.build(
        "user_001",
        [
            fp("user_001", "s1", [1.0, 0.0]),
        ],
    )

    assert result.nodes[0].user_id == "user_001"
    assert result.nodes[0].node_id.startswith("user_001:")


def test_unknown_person_has_no_inferred_name():
    graph = SocialGraph()

    result = graph.build(
        "user_001",
        [
            fp("user_001", "s1", [1.0, 0.0]),
        ],
    )

    assert not hasattr(result.nodes[0], "name")


def test_cosine_similarity():
    similarity = SocialGraph.cosine_similarity(
        [1.0, 0.0],
        [1.0, 0.0],
    )

    assert similarity == pytest.approx(1.0)


def test_rejects_dimension_mismatch():
    with pytest.raises(SocialGraphError):
        SocialGraph.cosine_similarity(
            [1.0, 0.0],
            [1.0],
        )


def test_rejects_zero_vector():
    with pytest.raises(SocialGraphError):
        SocialGraph.cosine_similarity(
            [0.0, 0.0],
            [1.0, 0.0],
        )


def test_rejects_non_finite_fingerprint():
    graph = SocialGraph()

    with pytest.raises(SocialGraphError):
        graph.build(
            "user_001",
            [
                fp(
                    "user_001",
                    "s1",
                    [1.0, float("nan")],
                )
            ],
        )


def test_rejects_empty_session_id():
    graph = SocialGraph()

    with pytest.raises(SocialGraphError):
        graph.build(
            "user_001",
            [
                fp(
                    "user_001",
                    "",
                    [1.0, 0.0],
                )
            ],
        )
def test_social_graph_uses_connected_components_and_is_order_independent():
    graph = SocialGraph(similarity_threshold=0.8)
    records=[fp("u","a",[1,0]),fp("u","b",[0.99,0.1]),fp("u","c",[0.8,0.6])]
    first=graph.build("u",records)
    second=graph.build("u",list(reversed(records)))
    assert [n.session_ids for n in first.nodes] == [n.session_ids for n in second.nodes]

def test_social_graph_rejects_duplicate_session_ids():
    import pytest
    with pytest.raises(SocialGraphError):
        SocialGraph().build("u", [fp("u","same",[1,0]), fp("u","same",[1,0])])

def test_social_graph_does_not_export_raw_fingerprints():
    result=SocialGraph().build("u", [fp("u","s",[1,0])])
    assert not hasattr(result.nodes[0], "values")
