"""Abstract interface and models for inference runners."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GenerationOutput:
    """Output generated from an inference run."""
    raw_text: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tokens_per_second": self.tokens_per_second,
            "model_name": self.model_name,
            "metadata": self.metadata,
        }


class BaseModelRunner(ABC):
    """Abstract base class for model execution backends."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationOutput:
        """Execute inference against the target model.

        Args:
            prompt: Text prompt to provide to the model.
            **kwargs: Generation overrides (temperature, seed, max_tokens, etc.).

        Returns:
            GenerationOutput containing text and performance telemetry.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata (family, parameter size, quantization level, etc.)."""
        pass

