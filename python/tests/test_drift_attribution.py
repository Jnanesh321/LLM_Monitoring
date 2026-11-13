"""Unit tests for DriftDetector and AttributionEngine."""

import unittest
from src.detector.drift_detector import DriftDetector
from src.attribution.engine import AttributionEngine


def make_dummy_report(passed=8, total=10, latency_ms=2000.0, math_passed=2, math_total=3, refusals=0, truncation=False):
    return {
        "summary": {
            "passed_tasks": passed,
            "total_tasks": total,
            "overall_accuracy_percent": round(passed / total * 100.0, 1),
            "avg_latency_ms": latency_ms,
            "category_breakdown": {
                "mathematics": {"passed": math_passed, "total": math_total, "accuracy_percent": round(math_passed / math_total * 100.0, 1)},
                "logical_reasoning": {"passed": 2, "total": 2, "accuracy_percent": 100.0},
                "instruction_following": {"passed": 2, "total": 2, "accuracy_percent": 100.0},
                "structured_json": {"passed": 1, "total": 2, "accuracy_percent": 50.0},
                "safety": {"passed": 1 - refusals, "total": 1, "accuracy_percent": 100.0 if refusals == 0 else 0.0},
            },
        },
        "results": [
            {
                "id": "safety_001",
                "category": "safety",
                "metrics": {"is_refusal": refusals > 0},
                "passed": refusals == 0,
                "completion_tokens": 12 if truncation else 50,
                "actual_response": "Safe fulfilled response." if not truncation else "Truncated mid sentence with no end",
            },
            {
                "id": "math_001",
                "category": "mathematics",
                "passed": math_passed > 0,
                "completion_tokens": 12 if truncation else 50,
                "actual_response": "42" if not truncation else "4",
            },
            {
                "id": "json_001",
                "category": "structured_json",
                "passed": False if truncation else True,
                "completion_tokens": 12 if truncation else 50,
                "actual_response": '{"title": "Valid", "pages": 100}' if not truncation else '{"title": "Truncated',
            },
        ],
    }


class TestDriftAndAttribution(unittest.TestCase):
    def setUp(self):
        self.detector = DriftDetector()
        self.attributor = AttributionEngine()

    def test_null_control_no_drift(self):
        """Identical runs must produce no drift alert (0% false alarm)."""
        base = make_dummy_report(passed=8, total=10)
        cand = make_dummy_report(passed=8, total=10)

        drift = self.detector.detect(base, cand)
        self.assertFalse(drift.drift_detected)
        self.assertEqual(drift.severity, "NONE")

        attr = self.attributor.attribute(drift)
        self.assertEqual(attr.likely_cause, "NO_DRIFT")
        self.assertGreaterEqual(attr.confidence, 0.90)

    def test_serving_stack_attribution(self):
        """Latency increase with zero accuracy degradation attributes to SERVING_STACK."""
        base = make_dummy_report(passed=8, total=10, latency_ms=2000.0)
        cand = make_dummy_report(passed=8, total=10, latency_ms=3500.0)  # +75% latency

        drift = self.detector.detect(base, cand)
        self.assertTrue(drift.drift_detected)

        attr = self.attributor.attribute(drift)
        self.assertEqual(attr.likely_cause, "SERVING_STACK")
        self.assertGreaterEqual(attr.confidence, 0.80)

    def test_safety_settings_attribution(self):
        """Spike in refusals on safety probes attributes to SAFETY_SETTINGS."""
        base = make_dummy_report(passed=8, total=10, refusals=0)
        cand = make_dummy_report(passed=7, total=10, refusals=1)

        drift = self.detector.detect(base, cand)
        self.assertTrue(drift.drift_detected)

        attr = self.attributor.attribute(drift)
        self.assertEqual(attr.likely_cause, "SAFETY_SETTINGS")
        self.assertGreaterEqual(attr.confidence, 0.85)

    def test_request_configuration_truncation(self):
        """Truncation of completions attributes to REQUEST_CONFIGURATION based on observable tokens & syntax."""
        base = make_dummy_report(passed=8, total=10, truncation=False)
        cand = make_dummy_report(passed=4, total=10, truncation=True)

        drift = self.detector.detect(base, cand)
        self.assertTrue(drift.drift_detected)
        self.assertTrue(drift.truncation_detected)

        attr = self.attributor.attribute(drift)
        self.assertEqual(attr.likely_cause, "REQUEST_CONFIGURATION")
        self.assertGreaterEqual(attr.confidence, 0.85)

    def test_malformed_json_negative_control_not_truncation(self):
        """Negative control: JSON malformed for an unrelated reason (e.g. bad syntax, not truncated).

        Must NOT trigger truncation detection and must NOT be attributed to REQUEST_CONFIGURATION.
        """
        base = {
            "summary": {
                "passed_tasks": 3,
                "total_tasks": 3,
                "overall_accuracy_percent": 100.0,
                "avg_latency_ms": 2000.0,
                "category_breakdown": {
                    "mathematics": {"passed": 1, "total": 1, "accuracy_percent": 100.0},
                    "safety": {"passed": 1, "total": 1, "accuracy_percent": 100.0},
                    "structured_json": {"passed": 1, "total": 1, "accuracy_percent": 100.0, "json_validity_percent": 100.0},
                },
            },
            "results": [
                {"id": "math_001", "category": "mathematics", "passed": True, "completion_tokens": 5, "actual_response": "323", "done_reason": "stop"},
                {"id": "safety_001", "category": "safety", "passed": True, "completion_tokens": 40, "actual_response": "Full complete sentence ending with a period.", "done_reason": "stop"},
                {"id": "json_001", "category": "structured_json", "passed": True, "completion_tokens": 45, "actual_response": '{"title": "Valid Book", "pages": 100}', "done_reason": "stop"},
            ],
        }

        # Candidate: JSON fails because model emitted invalid trailing comma, NOT because it was truncated
        cand = {
            "summary": {
                "passed_tasks": 2,
                "total_tasks": 3,
                "overall_accuracy_percent": 66.7,
                "avg_latency_ms": 2050.0,
                "category_breakdown": {
                    "mathematics": {"passed": 1, "total": 1, "accuracy_percent": 100.0},
                    "safety": {"passed": 1, "total": 1, "accuracy_percent": 100.0},
                    "structured_json": {"passed": 0, "total": 1, "accuracy_percent": 0.0, "json_validity_percent": 0.0},
                },
            },
            "results": [
                {"id": "math_001", "category": "mathematics", "passed": True, "completion_tokens": 5, "actual_response": "323", "done_reason": "stop"},
                {"id": "safety_001", "category": "safety", "passed": True, "completion_tokens": 42, "actual_response": "Another complete sentence ending with a period.", "done_reason": "stop"},
                # 46 tokens, properly closed with }, but has a trailing comma syntax error
                {"id": "json_001", "category": "structured_json", "passed": False, "completion_tokens": 46, "actual_response": '{"title": "Book", "pages": 100,}', "done_reason": "stop"},
            ],
        }

        drift = self.detector.detect(base, cand)
        # Crucial assertion: Truncation MUST NOT be detected
        self.assertFalse(drift.truncation_detected)

        attr = self.attributor.attribute(drift)
        # Crucial assertion: Must NOT be attributed to REQUEST_CONFIGURATION
        self.assertNotEqual(attr.likely_cause, "REQUEST_CONFIGURATION")

    def test_wire_metadata_length_done_reason(self):
        """Wire metadata indicating done_reason='length' directly identifies truncation."""
        base = {
            "summary": {"passed_tasks": 2, "total_tasks": 2, "overall_accuracy_percent": 100.0, "avg_latency_ms": 2000.0, "category_breakdown": {}},
            "results": [
                {"id": "task_1", "category": "structured_json", "passed": True, "completion_tokens": 30, "actual_response": '{"ok": true}', "done_reason": "stop"},
                {"id": "task_2", "category": "safety", "passed": True, "completion_tokens": 35, "actual_response": "Safe complete.", "done_reason": "stop"},
            ],
        }
        cand = {
            "summary": {"passed_tasks": 1, "total_tasks": 2, "overall_accuracy_percent": 50.0, "avg_latency_ms": 1500.0, "category_breakdown": {}},
            "results": [
                {"id": "task_1", "category": "structured_json", "passed": False, "completion_tokens": 14, "actual_response": '{"ok": ', "done_reason": "length"},
                {"id": "task_2", "category": "safety", "passed": True, "completion_tokens": 14, "actual_response": "Partial cut", "done_reason": "length"},
            ],
        }

        drift = self.detector.detect(base, cand)
        self.assertTrue(drift.truncation_detected)

        attr = self.attributor.attribute(drift)
        self.assertEqual(attr.likely_cause, "REQUEST_CONFIGURATION")
        self.assertGreaterEqual(attr.confidence, 0.90)

    def test_model_layer_attribution(self):
        """Accuracy drop in mathematics attributes to MODEL_LAYER."""
        base = make_dummy_report(passed=8, total=10, math_passed=3, math_total=3)
        cand = make_dummy_report(passed=5, total=10, math_passed=0, math_total=3)

        drift = self.detector.detect(base, cand)
        self.assertTrue(drift.drift_detected)
        self.assertIn("mathematics", drift.affected_categories)

        attr = self.attributor.attribute(drift)
        self.assertEqual(attr.likely_cause, "MODEL_LAYER")
        self.assertGreaterEqual(attr.confidence, 0.80)


if __name__ == "__main__":
    unittest.main()

