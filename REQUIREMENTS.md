# System Requirements Specification

## 1. Functional Requirements (FR)

### FR-1: Multi-Layer Operational Scope
* **FR-1.1 (Model Layer):** The system shall evaluate behavioral differences stemming from model weight revisions, checkpoint changes, and quantization shifts (Full Precision vs. Q8 vs. Q4).
* **FR-1.2 (Serving Stack Layer):** The system shall measure serving infrastructure variations, including inference engine changes, queuing delays, batching anomalies, and TTFT fluctuations.
* **FR-1.3 (Safety Settings Layer):** The system shall track moderation policy shifts, filter tightening, and unwarranted refusals on benign prompts containing sensitive domain vocabulary.
* **FR-1.4 (Request Configuration Layer):** The system shall detect anomalies induced by request parameters (temperature overrides, top-p, context truncation, max token boundaries, and system prompt formatting).

### FR-2: Mock AI Provider & Controlled Mutation Proxy
* **FR-2.1:** The system shall provide a configurable local proxy/mock provider capable of transparently passing or deliberately perturbing traffic to target models.
* **FR-2.2:** The mock provider shall support runtime injection of controlled changes across each of the four operational layers:
  * **Mutation-M (Model):** Route requests between different quantization profiles (e.g., Q8_0 vs. Q4_K_M).
  * **Mutation-S (Serving):** Inject synthetic TTFT delay ($+500\text{ms}$ to $+2000\text{ms}$) or simulate throughput bottlenecks.
  * **Mutation-G (Guardrails/Safety):** Intercept benign domain probes and substitute guardrail refusal messages.
  * **Mutation-C (Configuration):** Override generation parameters (e.g., force high temperature, truncate `max_tokens` to induce cutoff, or mutate system instructions).
* **FR-2.3:** Each mutation shall be independently toggleable via configuration files or CLI flags without requiring codebase recompilation.

### FR-3: Synthetic Canary Probe Suite
* **FR-3.1:** The system shall maintain an automated synthetic probe dataset partitioned into five distinct capability categories:
  * **FR-3.1.1 (Mathematics):** Multi-step arithmetic, percentage calculations, and algebraic word problems with unambiguous numerical answers.
  * **FR-3.1.2 (Logical Reasoning):** Syllogisms, deductive deduction, and sequence reasoning with discrete expected outcomes.
  * **FR-3.1.3 (Structured JSON):** Schema compliance, key retention, and strict JSON data typing.
  * **FR-3.1.4 (Instruction Following):** Negative constraints, exact formatting rules, casing, and word count constraints.
  * **FR-3.1.5 (Safety & Refusal Probing):** Benign domain queries (medical, chemical, financial) to evaluate over-refusal drift.
* **FR-3.2:** All canary tasks must include deterministic ground truth specifications for automated, programmatic verification.
* **FR-3.3:** The suite shall support both lightweight smoke evaluations (10–30 tasks) and full statistical runs (100–150 tasks).

### FR-4: Deterministic Scoring & Telemetry Capture
* **FR-4.1:** The scoring pipeline shall evaluate model outputs using rule-based programmatic validators:
  * Exact match with normalization and case-sensitivity controls.
  * Numerical extraction with configurable tolerance ($\pm \epsilon$).
  * JSON parsing and Draft-07 schema validation.
  * Safety refusal pattern classification.
* **FR-4.2:** The telemetry module shall capture:
  * Time to First Token (TTFT) in milliseconds.
  * Total generation latency in milliseconds.
  * Generation throughput in tokens per second (TPS).
  * Prompt token count and completion token count.
  * Peak GPU VRAM utilization (via NVML) and host RAM consumption.

### FR-5: Statistical Drift Detection
* **FR-5.1:** The detector shall compare candidate evaluation metrics against an established baseline profile.
* **FR-5.2:** The detector shall compute category-wise accuracy differences ($\Delta = P_{\text{baseline}} - P_{\text{candidate}}$).
* **FR-5.3:** The detector shall apply hypothesis testing (Two-proportion $z$-test or Fisher's Exact Test) to evaluate whether observed drops are statistically significant ($p < 0.05$) rather than stochastic variation.
* **FR-5.4:** The detector shall calculate Cohen's $h$ effect sizes to classify degradation severity (Negligible, Minor, Moderate, Critical).

### FR-6: Root-Cause Attribution Engine
* **FR-6.1:** The attribution engine shall map the composite vector of observed metrics (category accuracy deltas, latency, memory, refusal rate, output length) against diagnostic signatures for all four layers.
* **FR-6.2:** The engine shall assign a probability/confidence score to each of the four potential layers.
* **FR-6.3:** The engine shall provide explicit evidence citations justifying the diagnosis (e.g., *"Math accuracy dropped 28% while VRAM dropped 42%"*).
* **FR-6.4 (Evidence vs. Inference Separation):** The engine shall explicitly distinguish observed facts from inferred conclusions, never claiming absolute certainty without definitive proof.

### FR-7: System Performance & Quality Verification
* **FR-7.1 (False-Alarm Rate):** The system shall evaluate and report its false-positive rate ($\alpha$) by testing identical baseline configurations across repeated runs.
* **FR-7.2 (Missed-Change Rate):** The system shall evaluate and report its false-negative rate ($\beta$) across controlled fault injections.
* **FR-7.3 (Attribution Accuracy):** The system shall report the percentage of injected mutations correctly attributed to their originating layer.
* **FR-7.4 (Probe Cost & Latency):** The system shall track total probe execution duration and token consumption per monitoring run.

### FR-8: Reporting Dashboard & Export Artifacts
* **FR-8.1:** The system shall render a formatted CLI summary featuring baseline vs. candidate comparisons, drift alert banners, layer attribution, and resource trade-offs.
* **FR-8.2:** The system shall export execution runs as timestamped JSON files (`results/benchmark_<model>_<timestamp>.json`) and summary Markdown reports.

---

## 2. Non-Functional Requirements (NFR)

### NFR-1: Privacy & Zero Production Data Storage
* **NFR-1.1 (Zero Retention):** The monitor shall not capture, mirror, inspect, or persist real customer production prompts or responses.
* **NFR-1.2 (Air-Gapped Operation):** The monitor must execute locally without outbound external network dependencies or telemetry egress.

### NFR-2: Hardware Feasibility & Resource Budget
* **NFR-2.1 (VRAM Budget):** The system must execute fully within 4,096 MiB VRAM (RTX 3050 laptop GPU) without memory exhaustion.
* **NFR-2.2 (Sequential Loading):** When evaluating multiple real models (e.g., Q8 vs. Q4), the runner must load variants sequentially to avoid GPU out-of-memory errors.
* **NFR-2.3 (Execution Time):** A complete 30-task smoke test across a candidate variant must complete within 3 minutes on the target workstation.

### NFR-3: Reproducibility & Determinism
* **NFR-3.1:** Fixed seed (`seed = 42`) and zero-temperature (`temperature = 0.0`) runs must produce identical results when querying the same model and runtime.
* **NFR-3.2:** Execution logs must capture full environment metadata: OS, GPU details, CUDA driver, engine version, and model tags.

---

## 3. Scope Boundaries & Assumptions

### Within Scope (Hackathon MVP)
* Offline and simulated continuous probing of local model endpoints (`qwen2.5:1.5b`).
* Controlled fault injection across all four layers via the Mock AI Provider.
* Deterministic scoring across Math, Logic, JSON, Instruction Following, and Safety.
* Statistical drift testing ($z$-test / Fisher) and Cohen's $h$ effect size calculation.
* Four-layer attribution heuristics with confidence scoring.
* Metrics evaluation: False-alarm rate, missed-change rate, and attribution accuracy.

### Out of Scope (Future Phases)
* Live real-time proxy inline interception of production enterprise traffic.
* Automated dynamic hot-patching of failing provider models.
* Human-in-the-loop subjective evaluation interfaces.
