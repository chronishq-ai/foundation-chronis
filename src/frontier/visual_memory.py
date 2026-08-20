import os
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

# In a real environment, we'd use faiss. Using a mock class here for the interface.
class MockFAISS:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = []
        self.metadata = []

    def add(self, vectors: np.ndarray):
        self.vectors.append(vectors)

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray):
        self.vectors.append(vectors)

    def search(self, query: np.ndarray, k: int):
        # Dummy search
        return np.array([[0.1]]), np.array([[0]])

class SelfHostedCLIPEncoder:
    """Mock for the self-hosted CLIP-class vision-language encoder."""
    def __init__(self):
        self.dimension = 512

    def encode(self, frame_data: Any) -> np.ndarray:
        # Return a dummy random vector
        return np.random.rand(self.dimension).astype('float32')

class VisualMemoryIndex:
    """
    Per-user visual/episodic memory index.
    """
    def __init__(self, user_id: str, encoder: SelfHostedCLIPEncoder):
        self.user_id = user_id
        self.encoder = encoder
        self.index = MockFAISS(dimension=self.encoder.dimension)
        self.entries = [] # To store metadata alongside the FAISS index
        
    def _is_salience_l2_or_above(self, salience: str) -> bool:
        order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}
        return order.get(salience, 0) >= 2

    def process_and_store(self, cse_frames: List[Dict[str, Any]]):
        """
        Processes CSE frames, running the encoder and storing in the index.
        """
        for frame in cse_frames:
            salience = frame.get("salience_level", "L0")
            if self._is_salience_l2_or_above(salience):
                embedding = self.encoder.encode(frame.get("frame_data"))
                
                entry = {
                    "canonical_record_pointer": frame.get("canonical_record_pointer"),
                    "timestamp_ntp": frame.get("timestamp_ntp"),
                    "salience_level": salience,
                    "gps_if_present": frame.get("gps_if_present"),
                    "cse_inputs": frame.get("cse_inputs")
                }
                
                self.index.add(np.expand_dims(embedding, axis=0))
                self.entries.append(entry)

    def delete_index(self):
        """
        Deletes the retrieval index. By design, this has ZERO effect on Layer 0.
        """
        self.index = MockFAISS(dimension=self.encoder.dimension)
        self.entries = []

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves top k entries."""
        if not self.entries:
            return []
        distances, indices = self.index.search(np.expand_dims(query_embedding, axis=0), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.entries):
                res = dict(self.entries[idx])
                res["ann_distance"] = float(dist)
                results.append(res)
        return results
