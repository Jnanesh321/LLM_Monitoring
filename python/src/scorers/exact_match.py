"""Exact match scorer for deterministic text validation."""

import re
from typing import Any, Optional
from src.scorers.base import BaseScorer, EvaluationResult


class ExactMatchScorer(BaseScorer):
    """Evaluates whether the completion matches the expected text exactly or normalized."""

    def score(self, completion: str, expected: Any, metadata: Optional[dict] = None) -> EvaluationResult:
        metadata = metadata or {}
        case_sensitive = metadata.get("case_sensitive", False)

        clean_completion = completion.strip()
        clean_expected = str(expected).strip()

        # Remove surrounding quotes if the model quoted the answer
        if (clean_completion.startswith('"') and clean_completion.endswith('"')) or \
           (clean_completion.startswith("'") and clean_completion.endswith("'")):
            clean_completion = clean_completion[1:-1].strip()

        # Check direct match
        if case_sensitive:
            passed = clean_completion == clean_expected
        else:
            passed = clean_completion.lower() == clean_expected.lower()

        # If not passed, check if the first line or final line is an exact match (common model formatting)
        if not passed:
            lines = [l.strip() for l in clean_completion.splitlines() if l.strip()]
            for line in lines:
                candidate = line.strip("`'\".,! ")
                if case_sensitive:
                    if candidate == clean_expected:
                        clean_completion = candidate
                        passed = True
                        break
                else:
                    if candidate.lower() == clean_expected.lower():
                        clean_completion = candidate
                        passed = True
                        break

        details = f"Expected: '{clean_expected}', Got: '{clean_completion}'"
        return EvaluationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            extracted_value=clean_completion,
            expected_value=clean_expected,
            details=details,
        )

