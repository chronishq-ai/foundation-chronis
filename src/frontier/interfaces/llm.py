from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """
    Thin interface for the general-purpose, self-hosted LLM (Sprint 18).
    Explicitly sandboxed to general knowledge; zero default access to per-user storage.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generates a text response for the given prompt.
        """
        pass
