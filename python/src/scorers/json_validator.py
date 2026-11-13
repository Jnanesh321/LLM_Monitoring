"""Deterministic JSON syntax, schema, and instruction compliance validators."""

import json
import re
from typing import Any, Optional, Tuple
from src.scorers.base import BaseScorer, EvaluationResult


class JSONValidatorScorer(BaseScorer):
    """Validates that completions parse as JSON and adhere to required schema or exact values.

    Separates structural JSON validity from prompt instruction compliance
    (e.g., detecting unwanted markdown code fences when raw JSON was explicitly requested).
    """

    def _inspect_format_compliance(self, text: str) -> Tuple[bool, bool, Optional[str]]:
        """Inspects if response contains markdown fences and extracts the candidate JSON string.

        Returns:
            (raw_json_compliant, has_markdown_fences, candidate_json_string)
        """
        trimmed = text.strip()

        # Check for markdown code fences (``` or ```json)
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", trimmed, re.IGNORECASE)
        has_markdown_fences = fence_match is not None

        if has_markdown_fences:
            # Contains markdown fences, so it violates strict raw-JSON instruction
            candidate_str = fence_match.group(1).strip()
            raw_json_compliant = False
            return raw_json_compliant, has_markdown_fences, candidate_str

        # If no markdown fences, check whether text is strictly raw JSON (starts with { or [ and ends with } or ])
        starts_valid = trimmed.startswith("{") or trimmed.startswith("[")
        ends_valid = trimmed.endswith("}") or trimmed.endswith("]")

        if starts_valid and ends_valid:
            raw_json_compliant = True
            candidate_str = trimmed
        else:
            # Model may have included conversational commentary outside braces
            raw_json_compliant = False
            start_brace = trimmed.find("{")
            end_brace = trimmed.rfind("}")
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                candidate_str = trimmed[start_brace : end_brace + 1].strip()
            else:
                candidate_str = trimmed

        return raw_json_compliant, has_markdown_fences, candidate_str

    def score(self, completion: str, expected: Any, metadata: Optional[dict] = None) -> EvaluationResult:
        metadata = metadata or {}
        eval_type = metadata.get("eval_type", "json_schema")
        strict_raw = metadata.get("strict_raw", True)

        raw_json_compliant, has_markdown_fences, json_str = self._inspect_format_compliance(completion)

        if not json_str:
            return EvaluationResult(
                passed=False,
                score=0.0,
                extracted_value=None,
                expected_value=expected,
                details="Could not locate JSON content in completion.",
                metrics={
                    "json_valid": False,
                    "schema_valid": False,
                    "raw_json_compliant": False,
                    "has_markdown_fences": has_markdown_fences,
                },
            )

        # 1. Test JSON parse validity
        try:
            parsed = json.loads(json_str)
            json_valid = True
        except json.JSONDecodeError as err:
            return EvaluationResult(
                passed=False,
                score=0.0,
                extracted_value=json_str[:120],
                expected_value=expected,
                details=f"Invalid JSON syntax: {str(err)}",
                metrics={
                    "json_valid": False,
                    "schema_valid": False,
                    "raw_json_compliant": raw_json_compliant,
                    "has_markdown_fences": has_markdown_fences,
                },
            )

        # 2. Test schema / exact match
        schema_valid = False
        failure_reason = ""

        if eval_type == "json_exact":
            schema_valid = parsed == expected
            if not schema_valid:
                failure_reason = f"Content mismatch: expected {expected}, got {parsed}"
        else:
            # Default: json_schema validation
            required_keys = expected.get("required_keys", []) if isinstance(expected, dict) else []
            key_types = expected.get("key_types", {}) if isinstance(expected, dict) else {}

            type_map = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "float": float,
                "boolean": bool,
                "list": list,
                "array": list,
                "dict": dict,
                "object": dict,
            }

            if not isinstance(parsed, dict):
                failure_reason = "Expected JSON object (dictionary), but received array or primitive."
            else:
                missing_keys = [k for k in required_keys if k not in parsed]
                if missing_keys:
                    failure_reason = f"Missing required keys: {missing_keys}"
                else:
                    type_failed = False
                    for key, expected_type_str in key_types.items():
                        if key in parsed:
                            val = parsed[key]
                            target_type = type_map.get(expected_type_str.lower())
                            if target_type:
                                if expected_type_str.lower() in ("integer", "int") and isinstance(val, bool):
                                    failure_reason = f"Key '{key}' expected integer but got boolean."
                                    type_failed = True
                                    break
                                if not isinstance(val, target_type):
                                    failure_reason = (
                                        f"Key '{key}' expected {expected_type_str}, "
                                        f"got {type(val).__name__}."
                                    )
                                    type_failed = True
                                    break
                    if not type_failed:
                        schema_valid = True

        # 3. Determine overall passed status
        if strict_raw:
            passed = json_valid and schema_valid and raw_json_compliant
        else:
            passed = json_valid and schema_valid

        # Construct diagnostic explanation
        reasons = []
        if not json_valid:
            reasons.append("Invalid JSON")
        if not schema_valid:
            reasons.append(failure_reason or "Schema validation failed")
        if not raw_json_compliant:
            if has_markdown_fences:
                reasons.append("Used markdown code fences instead of raw JSON")
            else:
                reasons.append("Contained surrounding non-JSON text")

        if passed:
            details = "Valid JSON conforming to schema and raw output constraints."
        else:
            details = "; ".join(reasons)

        return EvaluationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            extracted_value=parsed,
            expected_value=expected,
            details=details,
            metrics={
                "json_valid": json_valid,
                "schema_valid": schema_valid,
                "raw_json_compliant": raw_json_compliant,
                "has_markdown_fences": has_markdown_fences,
            },
        )
