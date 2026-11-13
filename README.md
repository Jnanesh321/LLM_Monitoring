# AI Model Behaviour Drift Detection System

> A customer-side, privacy-preserving monitoring system designed to detect silent behavioural changes in AI models and attribute their root causes across four distinct layers: **Model**, **Serving Stack**, **Safety Settings**, and **Request Configuration**.

---

## 1. Problem Statement (PS-03: "The Model Changed Without Changing Its Name")

Modern software applications increasingly integrate Large Language Models (LLMs) via hosted API endpoints. The client application typically requests a static model alias (e.g., `gpt-4o`, `deepseek-r1`, `qwen-2.5`), expecting invariant behavior. However, the model name is only one part of the operational serving pipeline. Providers frequently update model weights, alter serving components, modify inference engines, adjust safety guardrails, or modify request handling—often without renaming the endpoint:

* **Model Layer:** Silently swapping checkpoints or applying aggressive quantization (e.g., from 8-bit to 4-bit) degrades mathematical precision and multi-step reasoning while maintaining superficial fluency.
* **Serving Stack Layer:** Upgrading backend inference engines (e.g., vLLM, TensorRT-LLM, llama.cpp), changing batching schedulers, or altering tokenization introduces latency variance, alters TTFT, or causes subtle whitespace formatting drifts.
* **Safety Settings Layer:** Silently tightening moderation guardrails causes benign customer requests containing sensitive vocabulary to be rejected unexpectedly.
* **Request Configuration Layer:** Parameter shifts (e.g., temperature overrides, system prompt modifications, context window truncation, or token limits) corrupt structured parsers and truncate completions.

Because client teams lack access to provider internals, they cannot isolate whether an application failure stems from an upstream model change, serving infrastructure modifications, safety adjustments, or client-side configuration errors.

---

## 2. Solution Overview

Our system provides an automated, customer-side monitoring harness that deploys **synthetic canary probes** to observe endpoint behavior, quantify degradation, and attribute root causes across all four operational layers—**without intercepting or storing production user prompts**.

```text
                                CANARY PROBE SUITE
                     (Synthetic, Deterministic, Zero Data Retention)
                                         │
                                         ▼
                               MOCK AI PROVIDER / PROXY
                     ┌───────────────────┴───────────────────┐
                     ▼                                       ▼
            BASELINE PIPELINE                       MUTATION INJECTION
          (Normal Configuration)             (Controlled Multi-Layer Changes)
                     │                                       │
                     ├───────────────────┬───────────────────┤
                     ▼                   ▼                   ▼
                [Layer 1: Model]   [Layer 2: Serving]  [Layer 3: Safety]  [Layer 4: Request]
                 (Q8 vs Q4 Quant)   (Latency/Engine)    (Filter Tighten)   (Prompt/Params)
                     │                   │                   │                   │
                     └───────────────────┼───────────────────┘                   │
                                         ▼                                       ▼
                                 EVALUATION ENGINE
                    (Deterministic Scoring: Math, Logic, JSON, Safety)
                                         │
                                         ▼
                              DRIFT DETECTION ENGINE
                     (Statistical Hypothesis Testing: z-test / Fisher)
                                         │
                                         ▼
                            ROOT-CAUSE ATTRIBUTION ENGINE
                     (Evidence Signatures -> Probable Layer Cause)
                                         │
                                         ▼
                                REPORTING DASHBOARD
                    (Drift Alerts, Evidence Citations & Trade-offs)
```

### Key Pillars:
1. **Four-Layer Root-Cause Attribution:** Distinguishes between Model (quantization/weights), Serving Stack (latency/infrastructure), Safety Settings (guardrail tightening), and Request Configuration (parameters/prompts).
2. **Mock AI Provider / Controlled Perturbation Harness:** A configurable local provider/proxy that introduces deliberate, controlled changes to individual layers to benchmark and evaluate attribution accuracy.
3. **Synthetic Canary Probes (Zero Production Data):** Probing relies exclusively on reproducible, deterministic synthetic queries. Production customer queries and responses are never captured or stored.
4. **Statistical Rigor:** Hypothesis testing (Two-proportion $z$-test, Fisher's exact test) and effect size measurements (Cohen's $h$) filter out normal stochastic generation noise.
5. **Real-Model & Quantization Testing:** Anchored on a local `qwen2.5:1.5b` (Q4_K_M) real-model deployment via Ollama, with Full Precision, Q8, and Q4 comparisons on the roadmap.

---

## 3. Four Controlled Benchmark Experiments

To validate detection and attribution efficacy, the system executes four controlled experiments:

| Layer | Controlled Change | Primary Observation | Expected Attribution |
|---|---|---|---|
| **1. Model** | Switch from Q8_0 to Q4_K_M quantization | Accuracy drop on multi-step math/logic; lower VRAM; higher TPS | Model Layer (Quantization Drift) |
| **2. Serving Stack** | Alter inference engine configuration / inject queue latency | TTFT and latency spike; invariant accuracy across all tasks | Serving Stack Layer (Infrastructure Shift) |
| **3. Safety Settings** | Tighten safety filters on benign probes with sensitive terms | Refusal rate surge on benign safety probes; stable math/logic | Safety Settings Layer (Guardrail Policy Shift) |
| **4. Request Config** | Override temperature or truncate `max_tokens` | Premature completion cutoffs or JSON parse syntax errors | Request Configuration Layer (Parameter Mismatch) |

---

## 4. Implementation Status

| Component | Responsibility | Status |
|---|---|---|
| **Deterministic Scorers** | Exact match, numerical comparison, JSON schema & exact match validation | **Implemented** |
| **Local Model Runner** | Ollama API client with telemetry (`qwen2.5:1.5b` Q4_K_M integration) | **Implemented** |
| **Smoke Benchmark Suite** | Initial 10-task test dataset across Math, Logic, Instruction, and JSON | **Implemented** |
| **Automated Benchmark Runner** | CLI entry point (`run_benchmark.py`) saving timestamped JSON logs | **Implemented** |
| **Mock AI Provider / Proxy** | Configurable proxy with layer-specific perturbation injection | Planned (Phase 2) |
| **Synthetic Canary Suite** | Extended 100+ task suite with continuous probing capability | Planned (Phase 3) |
| **Statistical Drift Detector** | Two-proportion $z$-test, Fisher's exact test, and Cohen's $h$ | Planned (Phase 4) |
| **Root-Cause Attribution** | Multi-factor evidence signature matching and confidence scoring | Planned (Phase 4) |
| **Reporting Dashboard** | Interactive CLI dashboard and audit log export | Planned (Phase 5) |

---

## 5. Repository Structure

```text
LLM_Monitoring/
├── README.md               # Project overview and current status
├── ARCHITECTURE.md         # Detailed 4-layer architecture and component specs
├── REQUIREMENTS.md         # Functional, non-functional, and verification requirements
├── IMPLEMENTATION_PLAN.md  # 36-hour hackathon roadmap across 6 milestones
├── EVALUATION.md           # Benchmark methodology, drift math, and attribution matrix
├── CONTRIBUTING.md         # Development guidelines and probe contribution specs
├── HARDWARE.md             # Workstation profile, VRAM feasibility, and guardrails
├── configs/                # Configuration files for benchmark runs and mock mutations
├── evaluation/             # Synthetic test datasets and prompt suites
│   └── benchmarks/
│       └── smoke_benchmark.json
├── src/                    # Core source code
│   ├── runner/             # Model runners (Ollama, mock proxy, llama.cpp)
│   ├── scorers/            # Deterministic answer extractors and validators
│   ├── proxy/              # Mock AI Provider with controlled layer fault injection
│   ├── detector/           # Statistical drift detection algorithms
│   ├── attribution/        # 4-layer root-cause diagnostic engine
│   └── benchmark.py        # Benchmark orchestration engine
├── results/                # Timestamped execution outputs and audit JSONs
├── tests/                  # Unit tests for scorers, runners, and detector
└── run_benchmark.py        # CLI entry point for benchmark execution
```

---

## 6. Quick Start

### Prerequisites
* Windows 11 or Linux
* Python 3.10+
* Ollama installed locally with `qwen2.5:1.5b` pulled (`ollama pull qwen2.5:1.5b`)

### Running the Benchmark
```bash
# Clone the repository
git clone https://github.com/Jnanesh321/LLM_Monitoring.git
cd LLM_Monitoring

# Run unit tests
python -m unittest discover -s tests

# Execute benchmark against local Ollama model
python run_benchmark.py --model qwen2.5:1.5b --temperature 0.0 --seed 42
```

---

## 7. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
