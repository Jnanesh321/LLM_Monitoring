"""Unit tests for MockAIProvider and fault injection mutations."""

import unittest
from unittest.mock import MagicMock
from src.runner.base import BaseModelRunner, GenerationOutput
from src.proxy.mock_provider import MockAIProvider, LayerMutationConfig, ExperimentDefinition


class DummyRunner(BaseModelRunner):
    """Stub runner for isolated mock provider testing."""
    def __init__(self, model_name: str = "stub_base"):
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> GenerationOutput:
        return GenerationOutput(
            raw_text="Normal response: 42",
            latency_ms=100.0,
            prompt_tokens=10,
            completion_tokens=5,
            tokens_per_second=50.0,
            model_name=self.model_name,
        )

    def get_model_info(self):
        return {"model_name": self.model_name, "quantization_level": "Q8_0"}


class TestMockAIProvider(unittest.TestCase):
    def setUp(self):
        self.base_runner = DummyRunner("base_q8")
        self.alt_runner = DummyRunner("alt_q4")
        self.provider = MockAIProvider(
            base_runner=self.base_runner,
            alternate_runners={"alt_q4": self.alt_runner},
        )

    def test_baseline_passthrough(self):
        """Unperturbed provider passes calls directly to base runner."""
        out = self.provider.generate("What is 2+2?")
        self.assertEqual(out.raw_text, "Normal response: 42")
        self.assertEqual(out.model_name, "base_q8")
        self.assertFalse(out.metadata.get("simulated_layer_mutation"))

    def test_layer1_model_mutation(self):
        """Model mutation routes traffic to alternate model."""
        self.provider.set_experiment(
            "EXP-01-MODEL",
            LayerMutationConfig(model_override="alt_q4"),
        )
        out = self.provider.generate("What is 2+2?")
        self.assertEqual(out.model_name, "alt_q4")
        self.assertTrue(out.metadata["simulated_layer_mutation"])

    def test_layer2_serving_mutation(self):
        """Serving mutation injects artificial latency delay."""
        self.provider.set_experiment(
            "EXP-02-SERVING",
            LayerMutationConfig(serving_latency_delay_ms=250.0, serving_throughput_penalty=2.0),
        )
        out = self.provider.generate("What is 2+2?")
        self.assertGreaterEqual(out.latency_ms, 350.0)  # 100ms base + 250ms injected
        self.assertTrue(out.metadata["simulated_layer_mutation"])

    def test_layer3_safety_mutation(self):
        """Safety mutation intercepts prompts containing sensitive trigger words."""
        self.provider.set_experiment(
            "EXP-03-SAFETY",
            LayerMutationConfig(safety_filter_enabled=True),
        )
        # Benign prompt without trigger words
        out_benign = self.provider.generate("What is the capital of France?")
        self.assertEqual(out_benign.raw_text, "Normal response: 42")

        # Prompt with trigger word 'bleach'
        out_safety = self.provider.generate("Explain how bleach disinfects tiles.")
        self.assertIn("cannot assist with this request", out_safety.raw_text)
        self.assertTrue(out_safety.metadata.get("intercepted_by_guardrail"))

    def test_layer4_request_mutation(self):
        """Request mutation intercepts and injects parameters like max_tokens."""
        spy_runner = MagicMock()
        spy_runner.generate.return_value = GenerationOutput("Cut...", 50.0)
        spy_runner.get_model_info.return_value = {"model_name": "spy"}
        provider = MockAIProvider(base_runner=spy_runner)

        provider.set_experiment(
            "EXP-04-REQUEST",
            LayerMutationConfig(request_max_tokens_override=10),
        )
        provider.generate("Hello world", max_tokens=100)
        spy_runner.generate.assert_called_once_with("Hello world", max_tokens=10)


if __name__ == "__main__":
    unittest.main()

