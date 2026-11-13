"""Mock AI Provider & Local Proxy supporting controlled multi-layer fault injections (PS-03)."""

import time
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.runner.base import BaseModelRunner, GenerationOutput
from src.runner.ollama_runner import OllamaRunner


@dataclass
class LayerMutationConfig:
    """Configuration for controlled fault injections across the four operational layers."""

    # Layer 1: Model Mutation (e.g., quantization swap)
    model_override: Optional[str] = None  # e.g., 'qwen2.5:1.5b' (Q4) vs 'qwen2.5:1.5b-instruct-q8_0' (Q8)

    # Layer 2: Serving Stack Mutation (e.g., infrastructure latency / TTFT delay)
    serving_latency_delay_ms: float = 0.0  # Artificial latency delay added to generation
    serving_throughput_penalty: float = 1.0  # Multiplier on latency to simulate slow engine

    # Layer 3: Safety Settings Mutation (e.g., moderation filter tightening)
    safety_filter_enabled: bool = False
    safety_trigger_keywords: List[str] = field(
        default_factory=lambda: [
            "bleach",
            "chemical",
            "peroxide",
            "medical",
            "burn",
            "dosage",
            "poison",
            "drug",
            "weapon",
            "hack",
            "bypass",
        ]
    )
    safety_refusal_message: str = (
        "I cannot assist with this request as it involves sensitive domain policies."
    )

    # Layer 4: Request Configuration Mutation (e.g., token truncation or parameter override)
    request_max_tokens_override: Optional[int] = None  # e.g., 12 to induce mid-sentence cutoff
    request_temperature_override: Optional[float] = None  # e.g., 0.95 to break determinism
    request_system_prompt_override: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_override": self.model_override,
            "serving_latency_delay_ms": self.serving_latency_delay_ms,
            "serving_throughput_penalty": self.serving_throughput_penalty,
            "safety_filter_enabled": self.safety_filter_enabled,
            "safety_trigger_keywords": self.safety_trigger_keywords,
            "request_max_tokens_override": self.request_max_tokens_override,
            "request_temperature_override": self.request_temperature_override,
            "request_system_prompt_override": self.request_system_prompt_override,
        }


@dataclass
class ExperimentDefinition:
    """Encapsulates a controlled experiment comparing a baseline against a modified layer configuration."""

    experiment_id: str
    target_layer: str  # 'model' | 'serving' | 'safety' | 'request' | 'baseline'
    description: str
    baseline_config: LayerMutationConfig
    modified_config: LayerMutationConfig

    @classmethod
    def create_baseline_control(cls, model_name: str = "qwen2.5:1.5b-instruct-q8_0") -> "ExperimentDefinition":
        """Null experiment: Baseline vs identical Baseline (measures false positive rate)."""
        base = LayerMutationConfig()
        return cls(
            experiment_id="EXP-00-BASELINE-CONTROL",
            target_layer="baseline",
            description="Null control run: Identical unperturbed baseline configuration",
            baseline_config=base,
            modified_config=base,
        )

    @classmethod
    def create_model_experiment(
        cls,
        reference_model: str = "qwen2.5:1.5b-instruct-q8_0",
        quantized_model: str = "qwen2.5:1.5b",
    ) -> "ExperimentDefinition":
        """Layer 1: Switch model precision from Q8_0 to Q4_K_M."""
        return cls(
            experiment_id="EXP-01-MODEL-QUANT",
            target_layer="model",
            description=f"Model Layer Mutation: Switch weights/quantization from {reference_model} to {quantized_model}",
            baseline_config=LayerMutationConfig(),
            modified_config=LayerMutationConfig(model_override=quantized_model),
        )

    @classmethod
    def create_serving_experiment(cls, added_latency_ms: float = 1200.0) -> "ExperimentDefinition":
        """Layer 2: Inject serving infrastructure latency delay and TTFT degradation."""
        return cls(
            experiment_id="EXP-02-SERVING-LATENCY",
            target_layer="serving",
            description=f"Serving Stack Mutation: Inject +{added_latency_ms}ms infrastructure queuing delay",
            baseline_config=LayerMutationConfig(),
            modified_config=LayerMutationConfig(
                serving_latency_delay_ms=added_latency_ms,
                serving_throughput_penalty=1.35,
            ),
        )

    @classmethod
    def create_safety_experiment(cls) -> "ExperimentDefinition":
        """Layer 3: Tighten safety guardrails to reject benign domain prompts."""
        return cls(
            experiment_id="EXP-03-SAFETY-REFUSAL",
            target_layer="safety",
            description="Safety Settings Mutation: Strict guardrail filter rejecting domain probes",
            baseline_config=LayerMutationConfig(safety_filter_enabled=False),
            modified_config=LayerMutationConfig(safety_filter_enabled=True),
        )

    @classmethod
    def create_request_experiment(cls, max_tokens: int = 15) -> "ExperimentDefinition":
        """Layer 4: Override request max_tokens to induce completion truncation and syntax failures."""
        return cls(
            experiment_id="EXP-04-REQUEST-TRUNCATE",
            target_layer="request",
            description=f"Request Configuration Mutation: Truncate max_tokens to {max_tokens} inducing syntax cutoff",
            baseline_config=LayerMutationConfig(),
            modified_config=LayerMutationConfig(request_max_tokens_override=max_tokens),
        )


class MockAIProvider(BaseModelRunner):
    """Proxy runner that wraps an underlying model runner and injects controlled layer mutations."""

    def __init__(
        self,
        base_runner: BaseModelRunner,
        experiment_id: str = "EXP-BASELINE",
        mutations: Optional[LayerMutationConfig] = None,
        alternate_runners: Optional[Dict[str, BaseModelRunner]] = None,
    ):
        self.base_runner = base_runner
        self.experiment_id = experiment_id
        self.mutations = mutations or LayerMutationConfig()
        self.alternate_runners = alternate_runners or {}

    def set_experiment(self, experiment_id: str, mutations: LayerMutationConfig) -> None:
        """Set active experiment ID and mutation configuration."""
        self.experiment_id = experiment_id
        self.mutations = mutations

    def clear_mutations(self) -> None:
        """Reset all mutations back to transparent pass-through baseline."""
        self.experiment_id = "EXP-BASELINE"
        self.mutations = LayerMutationConfig()

    def get_model_info(self) -> Dict[str, Any]:
        """Fetch model metadata from currently active target runner."""
        target_runner = self._resolve_runner()
        info = copy.deepcopy(target_runner.get_model_info())
        info["proxy_wrapped"] = True
        info["experiment_id"] = self.experiment_id
        info["active_mutations"] = self._get_active_mutation_summary()
        return info

    def _resolve_runner(self) -> BaseModelRunner:
        """Determine whether to use base runner or an alternate model due to Layer 1 mutation."""
        if self.mutations.model_override:
            alt_name = self.mutations.model_override
            if alt_name in self.alternate_runners:
                return self.alternate_runners[alt_name]
            base_url = getattr(self.base_runner, "base_url", "http://localhost:11434")
            new_runner = OllamaRunner(model_name=alt_name, base_url=base_url)
            self.alternate_runners[alt_name] = new_runner
            return new_runner

        return self.base_runner

    def _get_active_mutation_summary(self) -> Dict[str, Any]:
        """Summarize active mutations across all 4 layers."""
        active = {}
        if self.mutations.model_override:
            active["layer_1_model"] = f"Swapped target to {self.mutations.model_override}"
        if self.mutations.serving_latency_delay_ms > 0:
            active["layer_2_serving"] = f"+{self.mutations.serving_latency_delay_ms}ms artificial latency"
        if self.mutations.safety_filter_enabled:
            active["layer_3_safety"] = f"Filter active on {len(self.mutations.safety_trigger_keywords)} keywords"
        if self.mutations.request_max_tokens_override is not None:
            active["layer_4_request"] = f"max_tokens truncated to {self.mutations.request_max_tokens_override}"
        elif self.mutations.request_temperature_override is not None:
            active["layer_4_request"] = f"temperature overridden to {self.mutations.request_temperature_override}"
        return active

    def generate(self, prompt: str, **kwargs) -> GenerationOutput:
        """Execute inference, applying active layer mutations where configured."""
        active_mutations = self._get_active_mutation_summary()

        # -------------------------------------------------------------
        # Layer 3: Safety Settings Mutation (Guardrail Refusal Interception)
        # -------------------------------------------------------------
        if self.mutations.safety_filter_enabled:
            lower_prompt = prompt.lower()
            triggered_keywords = [
                kw for kw in self.mutations.safety_trigger_keywords if kw in lower_prompt
            ]
            if triggered_keywords:
                refusal_text = self.mutations.safety_refusal_message
                simulated_latency = 120.0  # Guardrail filter latency in ms
                return GenerationOutput(
                    raw_text=refusal_text,
                    latency_ms=simulated_latency,
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(refusal_text.split()),
                    tokens_per_second=18.5,
                    model_name=getattr(self.base_runner, "model_name", "mock_provider"),
                    metadata={
                        "experiment_id": self.experiment_id,
                        "intercepted_by_guardrail": True,
                        "triggered_keywords": triggered_keywords,
                        "active_mutations": active_mutations,
                    },
                )

        # -------------------------------------------------------------
        # Layer 4: Request Configuration Mutation (Parameter Overrides)
        # -------------------------------------------------------------
        gen_kwargs = copy.deepcopy(kwargs)
        if self.mutations.request_max_tokens_override is not None:
            gen_kwargs["max_tokens"] = self.mutations.request_max_tokens_override
        if self.mutations.request_temperature_override is not None:
            gen_kwargs["temperature"] = self.mutations.request_temperature_override

        # -------------------------------------------------------------
        # Layer 1: Model Mutation (Model Checkpoint / Quantization Swap)
        # -------------------------------------------------------------
        target_runner = self._resolve_runner()
        start_time = time.perf_counter()
        output = target_runner.generate(prompt, **gen_kwargs)

        # -------------------------------------------------------------
        # Layer 2: Serving Stack Mutation (Injected Latency & Jitter)
        # -------------------------------------------------------------
        added_latency_ms = self.mutations.serving_latency_delay_ms
        if added_latency_ms > 0:
            time.sleep(added_latency_ms / 1000.0)
            output.latency_ms += added_latency_ms

        if self.mutations.serving_throughput_penalty != 1.0:
            output.latency_ms *= self.mutations.serving_throughput_penalty
            if output.tokens_per_second > 0:
                output.tokens_per_second /= self.mutations.serving_throughput_penalty

        # Attach telemetry metadata
        output.metadata["experiment_id"] = self.experiment_id
        output.metadata["active_mutations"] = active_mutations
        output.metadata["simulated_layer_mutation"] = bool(active_mutations)
        output.metadata["active_settings"] = self.mutations.to_dict()
        if "effective_request_params" not in output.metadata:
            output.metadata["effective_request_params"] = {}
        output.metadata["effective_request_params"].update(gen_kwargs)

        return output
