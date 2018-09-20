"""Base class for analyzing models."""

from abc import ABC, abstractmethod


class BaseAnalyzer(ABC):

    """Base class for analyzing models."""

    def __init__(self):
        """Initialize base analyzer."""

    @abstractmethod
    def load(self):
        """Load previously trained model."""
        raise NotImplementedError("load is an abstract method.")

    @abstractmethod
    def save(self):
        """Save the analysis."""
        raise NotImplementedError("save is an abstract method.")

    @abstractmethod
    def analyze(self):
        """Perform analysis."""
        raise NotImplementedError("analyze is an abstract method.")
