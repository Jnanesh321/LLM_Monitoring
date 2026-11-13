"""Scorers package and registry factory."""

from typing import Dict, Type
from src.scorers.base import BaseScorer, EvaluationResult
from src.scorers.exact_match import ExactMatchScorer
from src.scorers.numerical import NumericalScorer
from src.scorers.json_validator import JSONValidatorScorer
from src.scorers.refusal import SafetyRefusalScorer
from src.scorers.constraint_checker import ConstraintCheckScorer
from src.scorers.unit_tests import UnitTestScorer

_SCORER_REGISTRY: Dict[str, Type[BaseScorer]] = {
    "exact_match": ExactMatchScorer,
    "numerical": NumericalScorer,
    "numeric_tolerance": NumericalScorer,
    "json_schema": JSONValidatorScorer,
    "json_exact": JSONValidatorScorer,
    "safety_refusal": SafetyRefusalScorer,
    "constraint_check": ConstraintCheckScorer,
    "unit_tests": UnitTestScorer,
}


def get_scorer(eval_type: str) -> BaseScorer:
    """Retrieve an instantiated scorer based on the eval_type identifier."""
    scorer_cls = _SCORER_REGISTRY.get(eval_type.lower())
    if not scorer_cls:
        raise ValueError(
            f"Unsupported eval_type '{eval_type}'. Available scorers: {list(_SCORER_REGISTRY.keys())}"
        )
    return scorer_cls()


__all__ = [
    "BaseScorer",
    "EvaluationResult",
    "ExactMatchScorer",
    "NumericalScorer",
    "JSONValidatorScorer",
    "get_scorer",
]

