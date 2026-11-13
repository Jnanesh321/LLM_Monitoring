"""Ollama inference runner implementation."""

import time
import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional
from src.runner.base import BaseModelRunner, GenerationOutput


class OllamaRunner(BaseModelRunner):
    """Inference runner connecting to local or remote Ollama server API."""

    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 60,
        default_temperature: float = 0.0,
        default_seed: int = 42,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.default_temperature = default_temperature
        self.default_seed = default_seed
        self._cached_model_info: Optional[Dict[str, Any]] = None

    @classmethod
    def list_local_models(cls, base_url: str = "http://localhost:11434") -> List[str]:
        """Return list of model names currently installed in local Ollama."""
        try:
            req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []

    def unload(self) -> bool:
        """Explicitly unload model from GPU/RAM using Ollama keep_alive=0."""
        payload = json.dumps({"model": self.model_name, "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("done_reason") == "unload"
        except Exception:
            return False

    def check_health(self) -> bool:
        """Verify that the Ollama service is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Fetch model metadata from Ollama /api/show."""
        if self._cached_model_info:
            return self._cached_model_info

        payload = json.dumps({"name": self.model_name}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/show",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                details = data.get("details", {})
                self._cached_model_info = {
                    "model_name": self.model_name,
                    "family": details.get("family", "unknown"),
                    "parameter_size": details.get("parameter_size", "unknown"),
                    "quantization_level": details.get("quantization_level", "unknown"),
                    "format": details.get("format", "gguf"),
                }
                return self._cached_model_info
        except Exception as e:
            return {
                "model_name": self.model_name,
                "error": f"Failed to fetch model info: {str(e)}",
                "quantization_level": "unknown",
            }

    def generate(self, prompt: str, **kwargs) -> GenerationOutput:
        """Execute inference against Ollama /api/generate."""
        temperature = kwargs.get("temperature", self.default_temperature)
        seed = kwargs.get("seed", self.default_seed)
        max_tokens = kwargs.get("max_tokens", 512)

        payload_dict = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_predict": max_tokens,
            },
        }

        # Enforce strict greedy argmax decoding when temperature is 0
        if temperature == 0.0:
            payload_dict["options"]["top_k"] = kwargs.get("top_k", 1)
            payload_dict["options"]["top_p"] = kwargs.get("top_p", 1.0)

        # Allow extra raw options passed in
        if "options" in kwargs and isinstance(kwargs["options"], dict):
            payload_dict["options"].update(kwargs["options"])

        data_bytes = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start_time = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                duration_sec = time.perf_counter() - start_time
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Failed to communicate with Ollama at {self.base_url}. Error: {str(e)}"
            )

        raw_response = body.get("response", "")
        prompt_tokens = body.get("prompt_eval_count", 0)
        completion_tokens = body.get("eval_count", 0)

        # Server-side duration reported in nanoseconds (convert to ms)
        server_total_duration_ms = body.get("total_duration", 0) / 1e6
        latency_ms = duration_sec * 1000.0

        eval_duration_sec = body.get("eval_duration", 0) / 1e9
        tokens_per_second = (
            completion_tokens / eval_duration_sec if eval_duration_sec > 0 else 0.0
        )

        return GenerationOutput(
            raw_text=raw_response,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=round(tokens_per_second, 2),
            model_name=self.model_name,
            metadata={
                "server_total_duration_ms": round(server_total_duration_ms, 2),
                "load_duration_ms": round(body.get("load_duration", 0) / 1e6, 2),
                "done_reason": body.get("done_reason", "unknown"),
                "effective_request_params": {
                    "temperature": temperature,
                    "seed": seed,
                    "max_tokens": max_tokens,
                },
            },
        )

