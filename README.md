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

The repository contains two complementary implementations designed for cross-verification:
1. **Go Implementation (Root):** High-throughput, concurrency-safe CLI and background monitor.
2. **Python Implementation (`python/`):** Statistical analysis, research benchmarks, and Ollama multi-precision fault-injection validation harness.

```text
LLM_Monitoring/
├── README.md               # Main project overview and dual-runtime quickstart
├── ARCHITECTURE.md         # Detailed 4-layer architecture and component specs
├── REQUIREMENTS.md         # Functional, non-functional, and verification requirements
├── IMPLEMENTATION_PLAN.md  # Hackathon roadmap and milestones
├── EVALUATION.md           # Benchmark methodology, statistical tests, and attribution matrix
├── CONTRIBUTING.md         # Development guidelines
├── HARDWARE.md             # Workstation profile, VRAM feasibility, and guardrails
├── .gitignore              # Repository git ignore rules
├── go.mod                  # Go module definition (Root Go Implementation)
├── cmd/
│   └── model-monitor/      # Go CLI application entry point
├── internal/
│   ├── canary/             # Go canary evaluation and checks
│   ├── evalspec/           # Go evaluation scenario specifications
│   ├── monitor/            # Go monitoring, mock provider, detector & attribution
│   └── report/             # Go reporting and presentation
├── data/                   # Go evaluation baseline JSON artifacts
├── docs/                   # Go developer documentation
├── canaries_fixed.json     # Go canary suite definitions
├── problem3_test_cases.json # Go problem test cases
└── python/                 # Python Implementation
    ├── requirements.txt    # Python dependencies
    ├── run_benchmark.py    # Python benchmark CLI runner
    ├── run_comparison.py   # Multi-precision comparative runner (FP16 vs Q8 vs Q4)
    ├── run_experiments.py  # 5-experiment controlled layer fault-injection & attribution suite
    ├── src/
    │   ├── runner/         # Model runners (Ollama API client with VRAM unload)
    │   ├── scorers/        # Deterministic scorers (ExactMatch, Numerical, JSON, Refusal)
    │   ├── proxy/          # Mock AI Provider for 4-layer fault injection
    │   ├── detector/       # Statistical drift detector (z-test, Cohen's h, token clustering)
    │   ├── attribution/    # 4-layer root cause attribution engine
    │   ├── telemetry/      # GPU VRAM & system RAM tracking
    │   └── validation/     # Controlled experiment test harness
    ├── evaluation/         # Synthetic canary datasets (canary_suite.json, smoke_benchmark.json)
    ├── results/            # Timestamped JSON execution logs and validation reports
    └── tests/              # Full Python unit test suite
```

---

## 6. Quick Start

### Prerequisites
* Windows 11 or Linux
* Go 1.23+
* Python 3.10+
* Ollama installed locally with target models:
  - `qwen2.5:1.5b-instruct-q8_0`
  - `qwen2.5:1.5b` (Q4_K_M)
  - `qwen2.5:1.5b-instruct-fp16`

### Running the Go Implementation
```bash
# Run Go unit tests
go test ./...

# Build Go CLI binary
go build -o model-monitor.exe ./cmd/model-monitor

# Run Go monitor CLI
./model-monitor.exe evaluate
```

### Running the Python Implementation
```bash
# Run Python unit tests
cd python
python -m unittest discover tests

# Run 5-experiment controlled fault-injection validation suite (Baseline, Model, Serving, Safety, Request)
python run_experiments.py --experiments all

# Run multi-precision comparison (FP16 vs Q8 vs Q4)
python run_comparison.py
```

---

## 7. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
