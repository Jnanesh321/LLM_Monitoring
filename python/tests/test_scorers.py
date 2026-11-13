"""Unit and regression tests for deterministic scorers."""

import unittest
from src.scorers.exact_match import ExactMatchScorer
from src.scorers.numerical import NumericalScorer
from src.scorers.json_validator import JSONValidatorScorer


class TestNumericalScorerRegression(unittest.TestCase):
    """Regression tests verifying numerical extraction for integers, decimals, negative numbers, and text."""

    def setUp(self):
        self.scorer = NumericalScorer()

    def test_regression_1210_not_extracted_as_zero(self):
        """CRITICAL FIX: '1210' must be extracted as 1210, never 0.0."""
        res = self.scorer.score("1210", 1550, {"tolerance": 0.0})
        self.assertEqual(res.extracted_value, 1210)
        self.assertFalse(res.passed)
        self.assertEqual(res.metrics["diff"], 340.0)

    def test_integers(self):
        # Direct integer
        res1 = self.scorer.score("42", 42)
        self.assertTrue(res1.passed)
        self.assertEqual(res1.extracted_value, 42)

        # Zero
        res2 = self.scorer.score("0", 0)
        self.assertTrue(res2.passed)
        self.assertEqual(res2.extracted_value, 0)

    def test_decimals(self):
        # Decimals
        res1 = self.scorer.score("3.1415", 3.14, {"tolerance": 0.01})
        self.assertTrue(res1.passed)
        self.assertAlmostEqual(res1.extracted_value, 3.1415)

        # Leading decimal point (.5)
        res2 = self.scorer.score(".5", 0.5)
        self.assertTrue(res2.passed)
        self.assertAlmostEqual(res2.extracted_value, 0.5)

    def test_negative_numbers(self):
        res1 = self.scorer.score("-42", -42)
        self.assertTrue(res1.passed)
        self.assertEqual(res1.extracted_value, -42)

        res2 = self.scorer.score("-3.14", -3.14)
        self.assertTrue(res2.passed)
        self.assertAlmostEqual(res2.extracted_value, -3.14)

    def test_scientific_notation(self):
        res1 = self.scorer.score("1.5e-3", 0.0015, {"tolerance": 1e-6})
        self.assertTrue(res1.passed)
        self.assertAlmostEqual(res1.extracted_value, 0.0015)

        res2 = self.scorer.score("2e4", 20000)
        self.assertTrue(res2.passed)
        self.assertEqual(res2.extracted_value, 20000)

    def test_numbers_embedded_in_prose(self):
        # Sentence ending with number and period
        res1 = self.scorer.score("The calculated total is 1210.", 1210)
        self.assertTrue(res1.passed)
        self.assertEqual(res1.extracted_value, 1210)

        # Comma-separated currency
        res2 = self.scorer.score("Total cost is $1,210.50 after tax.", 1210.5, {"tolerance": 0.01})
        self.assertTrue(res2.passed)
        self.assertAlmostEqual(res1.extracted_value, 1210)

        # Explicit answer statement
        res3 = self.scorer.score("First we get 1200. The final answer is: 1550", 1550)
        self.assertTrue(res3.passed)
        self.assertEqual(res3.extracted_value, 1550)

        # LaTeX boxed notation
        res4 = self.scorer.score("Therefore, \\boxed{1550}.", 1550)
        self.assertTrue(res4.passed)
        self.assertEqual(res4.extracted_value, 1550)


class TestJSONValidatorScorerRegression(unittest.TestCase):
    """Regression tests verifying separation of JSON validity from strict raw instruction compliance."""

    def setUp(self):
        self.scorer = JSONValidatorScorer()
        self.schema = {
            "required_keys": ["title", "pages"],
            "key_types": {"title": "string", "pages": "integer"},
        }

    def test_strict_raw_json_passed(self):
        """Raw JSON without markdown fences passes both syntax and strict compliance."""
        raw_completion = '{"title": "Clean Code", "pages": 464}'
        res = self.scorer.score(raw_completion, self.schema, {"eval_type": "json_schema", "strict_raw": True})
        self.assertTrue(res.passed)
        self.assertTrue(res.metrics["json_valid"])
        self.assertTrue(res.metrics["schema_valid"])
        self.assertTrue(res.metrics["raw_json_compliant"])
        self.assertFalse(res.metrics["has_markdown_fences"])

    def test_fenced_json_fails_strict_raw_compliance(self):
        """JSON wrapped in markdown fences parses validly but FAILS strict raw formatting."""
        fenced_completion = "```json\n{\n  \"title\": \"Clean Code\",\n  \"pages\": 464\n}\n```"
        res = self.scorer.score(fenced_completion, self.schema, {"eval_type": "json_schema", "strict_raw": True})
        self.assertFalse(res.passed)  # Failed because strict_raw is True
        self.assertTrue(res.metrics["json_valid"])
        self.assertTrue(res.metrics["schema_valid"])
        self.assertFalse(res.metrics["raw_json_compliant"])
        self.assertTrue(res.metrics["has_markdown_fences"])
        self.assertIn("Used markdown code fences instead of raw JSON", res.details)

    def test_fenced_json_passes_lenient_mode(self):
        """JSON wrapped in markdown fences passes when strict_raw is disabled."""
        fenced_completion = "```json\n{\n  \"title\": \"Clean Code\",\n  \"pages\": 464\n}\n```"
        res = self.scorer.score(fenced_completion, self.schema, {"eval_type": "json_schema", "strict_raw": False})
        self.assertTrue(res.passed)
        self.assertTrue(res.metrics["json_valid"])
        self.assertTrue(res.metrics["schema_valid"])
        self.assertFalse(res.metrics["raw_json_compliant"])

    def test_invalid_json_syntax(self):
        """Malformed JSON fails both valid syntax and compliance."""
        invalid_completion = "{'title': 'No closing quote, pages: 100}"
        res = self.scorer.score(invalid_completion, self.schema, {"eval_type": "json_schema"})
        self.assertFalse(res.passed)
        self.assertFalse(res.metrics["json_valid"])
        self.assertFalse(res.metrics["schema_valid"])


class TestExactMatchScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = ExactMatchScorer()

    def test_case_sensitivity(self):
        self.assertTrue(self.scorer.score("PARIS", "PARIS", {"case_sensitive": True}).passed)
        self.assertFalse(self.scorer.score("paris", "PARIS", {"case_sensitive": True}).passed)
        self.assertTrue(self.scorer.score("paris", "PARIS", {"case_sensitive": False}).passed)


if __name__ == "__main__":
    unittest.main()
