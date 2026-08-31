# Test Results Sprints 17-20

`	ext
============================= test session starts =============================
platform win32 -- Python 3.14.6, pytest-9.1.1, pluggy-1.6.0 -- C:\Python314\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\TirthPatel\Documents\projects\updated chronis\foundation-chronis
collecting ... collected 26 items

tests/frontier/test_sprint17.py::test_class_a_b_boundary_enforcement PASSED [  3%]
tests/frontier/test_sprint17.py::test_voice_routing PASSED               [  7%]
tests/frontier/test_sprint17.py::test_explainability_multi_hop_citation PASSED [ 11%]
tests/frontier/test_sprint17.py::test_explainability_clinical_filter PASSED [ 15%]
tests/frontier/test_sprint18.py::test_central_retrieval_routing PASSED   [ 19%]
tests/frontier/test_sprint18.py::test_mixed_query_composition PASSED     [ 23%]
tests/frontier/test_sprint18.py::test_no_bypass_of_central_retrieval_core PASSED [ 26%]
tests/frontier/test_sprint18.py::test_evidence_package_contract_low_confidence PASSED [ 30%]
tests/frontier/test_sprint18.py::test_evidence_package_contract_contradictions PASSED [ 34%]
tests/frontier/test_sprint19.py::test_no_production_ml PASSED            [ 38%]
tests/frontier/test_sprint20.py::test_provenance_pipeline PASSED         [ 42%]
tests/frontier/test_sprint20.py::test_provenance_explain_retrofitted PASSED [ 46%]
tests/frontier/test_sprint20.py::test_conflict_resolution PASSED         [ 50%]
tests/frontier/test_sprint20.py::test_conflict_resolution_targeted PASSED [ 53%]
tests/frontier/test_sprint20.py::test_identity_graph_isolation PASSED    [ 57%]
tests/frontier/test_sprint20.py::test_claims_engine_integration PASSED   [ 61%]
tests/frontier/test_sprint20.py::test_claims_engine_adapter_persistence PASSED [ 65%]
tests/frontier/test_sprint20.py::test_identity_confidence_floor PASSED   [ 69%]
tests/frontier/test_sprint20.py::test_identity_competing_inference PASSED [ 73%]
tests/frontier/test_sprint20.py::test_identity_unresolved_node PASSED    [ 76%]
tests/frontier/test_sprint20.py::test_systemic_cross_user_isolation PASSED [ 80%]
tests/frontier/test_sprint20.py::test_visual_embedding_randomness PASSED [ 84%]
tests/frontier/test_sprint20.py::test_visual_embedding_consistency XFAIL [ 88%]
tests/frontier/test_sprint20.py::test_visual_embedding_namespace_and_version PASSED [ 92%]
tests/frontier/test_sprint20.py::test_visual_memory_deletion PASSED      [ 96%]
tests/frontier/test_sprint20.py::test_user_scope_propagation PASSED      [100%]

============================== warnings summary ===============================
tests/frontier/test_sprint20.py::test_claims_engine_integration
tests/frontier/test_sprint20.py::test_claims_engine_adapter_persistence
  <string>:9: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================== 25 passed, 1 xfailed, 2 warnings in 0.29s ==================

``n
