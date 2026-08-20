from enum import Enum
from typing import Dict, Any, List

class ModelClass(Enum):
    CLASS_A = "A" # Shared, improvable, non-identity-bearing (vision, ASR, wake-word)
    CLASS_B = "B" # Permanently per-user, never aggregated (HSSM, Personal LM, Claims)

class BoundaryValidator:
    """
    Validates Global/Personal (Class A/B) model boundary constraints (Sprint 17).
    """

    @staticmethod
    def validate_registration(artifact_metadata: Dict[str, Any]) -> bool:
        """
        A model version cannot be registered without declaring its class.
        """
        model_class = artifact_metadata.get("model_class")
        if not model_class:
            raise ValueError("Model registration failed: 'model_class' tag is required.")
        
        try:
            ModelClass(model_class)
        except ValueError:
            raise ValueError(f"Model registration failed: invalid model_class '{model_class}'. Must be 'A' or 'B'.")
            
        return True

    @staticmethod
    def check_ci_isolation(training_manifest: Dict[str, Any], all_artifacts: Dict[str, Dict[str, Any]]) -> bool:
        """
        CI isolation check: fails if a Class A training manifest contains any Class B artifact path.
        """
        target_class = training_manifest.get("target_class")
        if target_class != ModelClass.CLASS_A.value:
            return True # Check only applies to Class A manifests
            
        dependencies = training_manifest.get("artifact_dependencies", [])
        for dep_path in dependencies:
            dep_metadata = all_artifacts.get(dep_path, {})
            dep_class = dep_metadata.get("model_class")
            if dep_class == ModelClass.CLASS_B.value:
                raise PermissionError(f"CI ISOLATION FAILURE: Class A manifest cannot depend on Class B artifact '{dep_path}'")
                
        return True
