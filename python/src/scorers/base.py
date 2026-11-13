"""Base classes and data models for deterministic benchmark scoring."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EvaluationResult:
    """Standardized result returned by all deterministic scorers."""
    passed: bool
    score: float
    extracted_value: Any
    expected_value: Any
    details: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": self.score,
            "extracted_value": self.extracted_value,
            "expected_value": self.expected_value,
            "details": self.details,
            "metrics": self.metrics,
        }


class BaseScorer(ABC):
    """Abstract base class for all deterministic scorers."""

    @abstractmethod
    def score(self, completion: str, expected: Any, metadata: Optional[dict] = None) -> EvaluationResult:
        """Evaluate a model completion against an expected value.

        Args:
            completion: Raw text completion from the model.
            expected: Ground truth expectation from the benchmark item.
            metadata: Additional validation parameters (e.g., tolerance, case_sensitive).

        Returns:
            EvaluationResult indicating pass/fail status and diagnostic details.
        """
        pass
