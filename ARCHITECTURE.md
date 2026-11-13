# System Architecture

## 1. Architectural Philosophy & Core Principles

The AI Model Behaviour Drift Detection System addresses the challenge: **"The Model Changed Without Changing Its Name"** (PS-03). The architecture is founded upon five core engineering tenets:

1. **Customer-Side Black-Box Monitoring:** The system requires zero visibility into provider infrastructure internals, proprietary weights, or backend host servers. All diagnostics are inferred strictly from observable request/response behavior.
2. **Four-Layer Diagnostic Scope:** Observability and root-cause attribution span four distinct operational layers:
   * **Layer 1: Model** (weights, checkpoints, quantization profiles).
   * **Layer 2: Serving Stack** (inference engines, batching schedulers, hardware backends).
   * **Layer 3: Safety Settings** (moderation filters, guardrail thresholds, refusal behaviors).
   * **Layer 4: Request Configuration** (client parameters, system prompts, token budgets).
3. **Controlled Controllability via Mock AI Provider:** To validate attribution algorithms scientifically, the architecture integrates a **Mock AI Provider / Local Proxy** that can inject isolated, controlled mutations into any of the four layers.
4. **Zero Production Data Retention:** User privacy is non-negotiable. Real customer prompts and completions are never captured, mirrored, or stored. Diagnostic telemetry is driven exclusively by synthetic canary probes.
5. **Deterministic Validation & Statistical Rigor:** Evaluations rely on deterministic ground-truth validators (regex, schema parsers, numerical tolerance) paired with two-proportion hypothesis testing to distinguish genuine behavioral shift from stochastic variance.

---

## 2. High-Level System Topology

```text
+---------------------------------------------------------------------------------------------------+
|                                     CANARY PROBING LAYER                                          |
|  - Synthetic Probe Suite (Math, Logic, JSON, Instruction, Safety)   - Dispatcher & Scheduler      |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                                  MOCK AI PROVIDER / LOCAL PROXY                                   |
|                                                                                                   |
|   +-------------------------------------------------------------------------------------------+   |
|   |                              CONTROLLED FAULT INJECTION MATRIX                            |   |
|   |                                                                                           |   |
|   |  [Layer 1: Model]       [Layer 2: Serving Stack]    [Layer 3: Safety]   [Layer 4: Request]  |   |
|   |  - Quantization swap    - Latency injection (TTFT)  - Filter tightening - Temp override       |   |
|   |  - Checkpoint switch    - Engine token formatting   - Benign refusals   - Token truncation    |   |
|   +-------------------------------------------------------------------------------------------+   |
|                                                 │                                                 |
|                                                 ▼                                                 |
|                      Target Endpoints: Local Ollama (qwen2.5:1.5b) / llama.cpp                    |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                                   EVALUATION & TELEMETRY LAYER                                    |
|                                                                                                   |
|   +-----------------------------------------+   +---------------------------------------------+   |
|   |          Deterministic Scorers          |   |             Resource Telemetry              |   |
|   | - Math (Numerical tolerance parsing)    |   | - Time to First Token (TTFT in ms)          |   |
|   | - Logic (Exact match / Deduction check) |   | - Generation Latency & Throughput (TPS)     |   |
|   | - Structured JSON (Schema validation)   |   | - Prompt & Completion Token Counts          |   |
|   | - Safety Probes (Refusal classification)|   | - Memory Footprint (VRAM via NVML / RAM)    |   |
|   +-----------------------------------------+   +---------------------------------------------+   |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                                    ANALYSIS & ATTRIBUTION LAYER                                   |
|                                                                                                   |
|   +-----------------------------------------+   +---------------------------------------------+   |
|   |         Drift Detection Engine          |   |         Root-Cause Attribution Engine       |   |
|   | - Two-proportion z-test / Fisher exact  |   | - Multi-factor signature matching           |   |
|   | - Effect size calculation (Cohen's h)   |   | - Confidence score & supporting evidence    |   |
|   | - Metric degradation categorization     |   | - Separation of Evidence vs. Inference      |   |
|   +-----------------------------------------+   +---------------------------------------------+   |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                                    REPORTING & DASHBOARD LAYER                                    |
|  - Interactive Terminal Summary   - Exportable JSON Audit Logs   - Markdown Diagnostic Reports    |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Component Deep Dive

### 3.1 Synthetic Canary Probe Suite
* **Function:** Serves as a customer-side external test harness dispatched against the target endpoint.
* **Characteristics:**
  * Contains reproducible, deterministic synthetic queries partitioned into five core domains: Mathematics, Logical Reasoning, Structured JSON, Instruction Following, and Safety Guardrails.
  * Completely divorced from live customer production data to ensure absolute data privacy and compliance.
  * Designed to be lightweight and cost-effective, running as quick smoke runs (10–30 tasks) or rigorous statistical batches (100–150 tasks).

### 3.2 Mock AI Provider / Local Proxy
* **Function:** Sits between the probe harness and the physical model backends, allowing developers to simulate real-world provider anomalies in a repeatable, scientifically controlled manner.
* **Injectable Mutation Controllers:**
  * **Model Layer Controller:** Toggles between physical model checkpoints or quantization levels (e.g., switches traffic from `qwen2.5:1.5b` Q8_0 to Q4_K_M).
  * **Serving Stack Controller:** Injects artificial queue latency, increases Time to First Token (TTFT) by a fixed factor, or alters tokenization whitespace.
  * **Safety Settings Controller:** Intercepts completions and triggers automated refusal templates (`"I cannot fulfill this request..."`) when prompts contain specific trigger terms (e.g., chemical, medical, financial).
  * **Request Configuration Controller:** Deliberately overrides client-supplied parameters (e.g., forcing temperature from `0.0` to `0.9`, altering the system prompt, or capping `max_tokens` at 15 to induce truncation).

### 3.3 Model Layer
* **Function:** Manages the actual physical LLM runtime.
* **Integration:**
  * Local Ollama engine running `qwen2.5:1.5b` (Q4_K_M).
  * `llama.cpp` runtime for GGUF paired executions (FP16, Q8_0, and Q4_K_M).
  * Enforces sequential model loading on memory-constrained hardware (e.g., 4 GB VRAM RTX 3050).

### 3.4 Evaluation Engine & Deterministic Scorers
* **Function:** Assesses response correctness without employing subjective LLM judges.
* **Scorers:**
  * `ExactMatchScorer`: Tests exact string equality with case-sensitivity and whitespace normalization toggles.
  * `NumericalScorer`: Regex extraction of integer and floating-point values with configurable tolerance ($\pm \epsilon$).
  * `JSONValidatorScorer`: Validates Draft-07 schema compliance, required key presence, and value type checks.
  * `SafetyRefusalScorer`: Classifies responses as successful completions vs. guardrail refusals.

### 3.5 Drift Detection Engine
* **Function:** Analyzes changes in accuracy and performance metrics between the established baseline and the candidate/perturbed state.
* **Statistical Methods:**
  * Formulates null hypothesis $H_0: p_{\text{candidate}} \ge p_{\text{baseline}}$.
  * Evaluates significance using two-proportion $z$-test ($N \ge 30$) or Fisher's Exact Test ($N < 30$).
  * Computes effect size using Cohen's $h$ to categorize degradation severity into Negligible, Minor, Moderate, or Severe.

### 3.6 Root-Cause Attribution Engine
* **Function:** Evaluates the aggregate vector of observed evidence to infer the most likely root-cause layer.
* **Attribution Rules Matrix:**

| Observed Evidence Signature | Primary Attribution | Supporting Evidence |
|---|---|---|
| Drop in `math` and `logic` accuracy; preserved `json` syntax; VRAM reduction $\ge 30\%$; higher throughput (TPS). | **Model Layer** (Quantization Drift) | Precision-sensitive capability loss correlates directly with memory drop. |
| Invariant capability accuracy; TTFT spikes $> 50\%$; total latency variance doubles; stable memory usage. | **Serving Stack Layer** (Infrastructure Shift) | Execution timing shifts drastically while model reasoning remains intact. |
| Spike in refusal rate on benign `safety` probes; invariant accuracy on `math`, `logic`, and `json`. | **Safety Settings Layer** (Guardrail Tightening) | Failures isolate exclusively to vocabulary-triggered safety probes. |
| Abrupt syntax errors across all categories; completions cut off mid-sentence at identical token lengths. | **Request Configuration Layer** (Token Truncation) | Global failure pattern matching token boundary limits. |
| Inconsistent outputs across repeated runs of the same task with fluctuating responses. | **Request Configuration Layer** (Temperature Override) | Non-deterministic generation observed under supposedly deterministic settings. |

* **Evidence vs. Inference Contract:** The engine explicitly categorizes every claim into **Observable Evidence** (e.g., *"Math accuracy dropped from 95% to 66%, p=0.012"*) and **Inferred Root Cause** (e.g., *"High confidence (88%) attribution to Model Quantization"*). It never asserts certainty without sufficient statistical proof.

### 3.7 Reporting Dashboard
* **Function:** Presents real-time and post-run diagnostic telemetry to developers.
* **Features:**
  * Terminal-based ASCII summary and rich tables.
  * Machine-readable JSON execution artifacts (`results/benchmark_<model>_<timestamp>.json`).
  * Accuracy vs. Resource (latency, VRAM, throughput) trade-off summaries.

---

## 4. End-to-End Data Flow

```text
[1. Baseline Establishment]
   Synthetic Probes ──► Normal Endpoint ──► Evaluation ──► Baseline Profile Stored
                                                                 │
[2. Controlled Perturbation]                                     │
   Mutation Config  ──► Mock AI Provider ──► Altered Behavior    │
                               │                                 │
[3. Canary Evaluation]         │                                 │
   Synthetic Probes ───────────┘                                 │
                               │                                 │
                               ▼                                 │
                     Evaluation Engine                           │
                               │                                 │
                               ▼                                 │
                     Telemetry & Scorers                         │
                               │                                 │
                               ▼                                 │
                     Drift Detection Engine ◄────────────────────┘
                               │
                               ▼
                     Attribution Engine
             (Signature Matching & Confidence)
                               │
                               ▼
                      Reporting Dashboard
```
