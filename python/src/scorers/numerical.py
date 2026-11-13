"""Numerical comparison scorer with comprehensive extraction and tolerance support."""

import re
from typing import Any, Optional
from src.scorers.base import BaseScorer, EvaluationResult


class NumericalScorer(BaseScorer):
    """Extracts numerical values from model completions and compares with expected values.

    Handles integers, decimals, negative values, scientific notation,
    comma-separated numbers, and numbers embedded within prose or markdown.
    """

    # Comprehensive regex matching integer, decimal, and scientific notation numbers
    NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

    # Target conclusion markers
    EXPLICIT_ANSWER_PATTERN = re.compile(
        r"(?:final\s+answer|answer\s+is|result\s+is|equals|equal\s+to|answer:|result:)\s*[:=]?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )
    BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")

    def extract_number(self, completion: str) -> Optional[float]:
        """Extract the most relevant candidate number from completion text."""
        if not completion:
            return None

        # Clean thousand-separator commas between digits (e.g., 1,234,567 -> 1234567)
        cleaned = re.sub(r"(?<=\d),(?=\d)", "", completion.strip())

        # 1. Check for LaTeX boxed notation: \boxed{...}
        boxed_match = self.BOXED_PATTERN.search(cleaned)
        if boxed_match:
            boxed_content = boxed_match.group(1).strip()
            boxed_num = self.NUMBER_PATTERN.search(boxed_content)
            if boxed_num:
                try:
                    return float(boxed_num.group(0))
                except ValueError:
                    pass

        # 2. Check for explicit answer statements (e.g., "The final answer is: 1210")
        explicit_matches = self.EXPLICIT_ANSWER_PATTERN.findall(cleaned)
        if explicit_matches:
            try:
                return float(explicit_matches[-1])
            except ValueError:
                pass

        # 3. Fallback to scanning all numbers and taking the final one (model conclusion)
        all_numbers = self.NUMBER_PATTERN.findall(cleaned)
        if all_numbers:
            try:
                return float(all_numbers[-1])
            except ValueError:
                pass

        return None

    def score(self, completion: str, expected: Any, metadata: Optional[dict] = None) -> EvaluationResult:
        metadata = metadata or {}
        tolerance = float(metadata.get("tolerance", 0.0))

        try:
            expected_num = float(expected)
        except (ValueError, TypeError):
            return EvaluationResult(
                passed=False,
                score=0.0,
                extracted_value=None,
                expected_value=expected,
                details=f"Invalid expected numerical value: {expected}",
            )

        extracted_float = self.extract_number(completion)

        if extracted_float is None:
            return EvaluationResult(
                passed=False,
                score=0.0,
                extracted_value=None,
                expected_value=expected_num,
                details="No valid numerical value found in completion.",
            )

        diff = abs(extracted_float - expected_num)
        passed = diff <= tolerance

        # Format cleanly as integer if whole number for readable reporting
        display_extracted = int(extracted_float) if extracted_float.is_integer() else extracted_float
        display_expected = int(expected_num) if expected_num.is_integer() else expected_num
        display_diff = round(diff, 6)

        details = (
            f"Expected: {display_expected}, Extracted: {display_extracted}, "
            f"Tolerance: {tolerance}, Diff: {display_diff}"
        )

        return EvaluationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            extracted_value=display_extracted,
            expected_value=display_expected,
            details=details,
            metrics={
                "extracted_number": extracted_float,
                "expected_number": expected_num,
                "diff": diff,
                "tolerance": tolerance,
            },
        )
