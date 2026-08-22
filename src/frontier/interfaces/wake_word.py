from abc import ABC, abstractmethod

class WakeWordProvider(ABC):
    """
    Thin interface for on-device wake-word detection.
    """

    @abstractmethod
    def listen(self) -> bool:
        """
        Listens for the wake word and returns True when detected.
        """
        pass
