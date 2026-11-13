# Contributing Guidelines

Thank you for contributing to the **AI Model Behaviour Drift Detection System** (PS-03: *"The Model Changed Without Changing Its Name"*). This document outlines development standards, branch conventions, and testing workflows.

---

## 1. Development Workflow & Git Strategy

* **`main`:** Stable, demonstrable code only. All merges into `main` must pass unit tests and benchmark smoke runs.
* **`feature/<name>`:** Feature development branches (e.g., `feature/mock-provider`, `feature/drift-detector`, `feature/attribution-matrix`).
* **`fix/<name>`:** Urgent bug fixes or test repairs.

### Guidelines
* Keep pull requests small and focused on a single layer or component.
* Never force push (`git push --force`) to `main`.
* Ensure new components include automated unit tests under `tests/`.

---

## 2. Environment Setup & Tooling

```bash
# Clone the repository
git clone https://github.com/Jnanesh321/LLM_Monitoring.git
cd LLM_Monitoring

# Run unit tests using standard library unittest
python -m unittest discover -s tests

# Run benchmark runner against local Ollama model
python run_benchmark.py --model qwen2.5:1.5b --temperature 0.0 --seed 42
```

---

## 3. How to Add a New Canary Probe

All synthetic benchmark tasks reside in `evaluation/benchmarks/`.

### Probe Specification Format
```json
{
  "id": "math_multi_042",
  "category": "mathematics",
  "prompt": "Calculate: (15 * 18) - (48 / 6). Provide only the final integer answer.",
  "eval_type": "numerical",
  "expected": 262,
  "tolerance": 0.0
}
```

### Probe Rules:
1. **Zero Ambiguity:** Craft prompts requiring distinct, deterministic outputs.
2. **Explicit Output Constraints:** Instruct the model to output *only* the expected value, schema, or keyword without conversational filler.
3. **Privacy Compliance:** Never use proprietary or real customer data; use only synthetic questions.

---

## 4. How to Add a New Mock Provider Mutation

Mutations reside in `src/proxy/mock_provider.py` and must map to one of the four operational layers:
* **Model Layer:** Model checkpoint switches, quantization simulation.
* **Serving Stack Layer:** Artificial latency jitter, TTFT delays, tokenization whitespace tweaks.
* **Safety Settings Layer:** Refusal message interception for sensitive keywords.
* **Request Configuration Layer:** Parameter overrides (temperature, system prompt, `max_tokens` truncation).

Each mutation must be independently toggleable via a clean configuration dictionary or CLI flag.

---

## 5. How to Implement a Scorer

Custom scorers inherit from `BaseScorer` in `src/scorers/base.py`:

```python
from src.scorers.base import BaseScorer, EvaluationResult

class CustomScorer(BaseScorer):
    def score(self, completion: str, expected: Any, metadata: dict = None) -> EvaluationResult:
        passed = expected.lower() in completion.lower()
        return EvaluationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            extracted_value=completion.strip(),
            expected_value=expected,
            details=f"Match found: {passed}",
        )
```

Register your scorer in `src/scorers/__init__.py` and add test cases to `tests/test_scorers.py`.

---

## 6. Testing & Validation Checklist

Before opening a pull request, verify:
* [ ] All unit tests pass: `python -m unittest discover -s tests`.
* [ ] Benchmark runner executes without errors: `python run_benchmark.py --model qwen2.5:1.5b`.
* [ ] No production customer data is collected, logged, or committed.
* [ ] Code adheres to PEP 8 standards with descriptive type annotations.
