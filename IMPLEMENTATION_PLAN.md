# Implementation Plan & Roadmap

## 1. Project Context & Team Structure

* **Context:** 36-Hour Hackathon (DevHack) — Problem 3: *"The Model Changed Without Changing Its Name"* (PS-03).
* **Objective:** Customer-side AI monitoring system that detects silent model behaviour changes and attributes their root causes across four layers: **Model**, **Serving Stack**, **Safety Settings**, and **Request Configuration**.
* **Team:** 4 Software & ML Engineers.
* **Primary Target Hardware:** NVIDIA GeForce RTX 3050 (4 GB VRAM), Intel i5, 16 GB RAM, Windows 11.

---

## 2. Team Work Distribution

| Role | Primary Ownership | Key Deliverables |
|---|---|---|
| **Dev 1 (Lead / ML Systems)** | Runner & Mock AI Provider Harness | Ollama integration, Mock AI Provider with 4-layer mutation injection (Model, Serving, Safety, Request Config), sequential variant loading. |
| **Dev 2 (Evaluation Engineer)** | Canary Probe Suite & Deterministic Scorers | 5-category canary dataset (Math, Logic, JSON, Instruction, Safety); deterministic validator functions (regex, numerical tolerance, schema, refusal). |
| **Dev 3 (Data & Stats Engineer)** | Drift Detector & Attribution Engine | Two-proportion $z$-test, Fisher's exact test, Cohen's $h$ effect sizes, 4-layer heuristic attribution matrix, evidence generator. |
| **Dev 4 (Product & Platform Engineer)**| Metrics Evaluation & Reporting Dashboard | Verification of false-alarm / missed-change / attribution accuracy metrics; rich CLI dashboard, JSON audit logging. |

---

## 3. Phased 36-Hour Timeline & Milestones

```text
  Hours 00-06    Hours 06-12      Hours 12-18       Hours 18-26      Hours 26-32    Hours 32-36
┌─────────────┬────────────────┬─────────────────┬────────────────┬──────────────┬─────────────┐
│   Phase 1   │    Phase 2     │     Phase 3     │    Phase 4     │   Phase 5    │   Phase 6   │
│ Benchmark   │  Mock Provider │  Canary Probe   │  Drift Engine  │ 4 Controlled │ Dashboard & │
│ Runner MVP  │  & Mutations   │  & Telemetry    │  & Attribution │ Experiments  │ Live Demo   │
└─────────────┴────────────────┴─────────────────┴────────────────┴──────────────┴─────────────┘
```

### Phase 1: Benchmark Runner MVP & Real Model Integration (Hours 0–6) — **COMPLETED**
* **Goal:** Build the initial automated benchmark runner, connect to local Ollama (`qwen2.5:1.5b`), and establish baseline execution with deterministic scoring.
* **Deliverables:**
  * [x] Initialize Git repository structure and documentation suite.
  * [x] Implement deterministic scorers (`ExactMatchScorer`, `NumericalScorer`, `JSONValidatorScorer`).
  * [x] Implement `OllamaRunner` with latency and token telemetry.
  * [x] Author initial 10-task smoke benchmark dataset (`smoke_benchmark.json`).
  * [x] Execute live verification on `qwen2.5:1.5b` (Q4_K_M) producing timestamped JSON reports.
* **Milestone M1:** Working Automated Benchmark Runner executing live tasks on local hardware.

### Phase 2: Mock AI Provider & Controlled Mutation Harness (Hours 6–12) — **IN PROGRESS**
* **Goal:** Build the configurable local proxy capable of transparently passing traffic or deliberately introducing controlled changes across each of the four operational layers.
* **Deliverables:**
  * [ ] Implement `src/proxy/mock_provider.py` as an extensible request forwarder / simulator.
  * [ ] Implement **Layer 1 (Model) Mutation:** Toggle between Q8_0 and Q4_K_M variants.
  * [ ] Implement **Layer 2 (Serving Stack) Mutation:** Inject synthetic TTFT delay and latency variance.
  * [ ] Implement **Layer 3 (Safety Settings) Mutation:** Trigger guardrail refusals on sensitive domain vocabulary.
  * [ ] Implement **Layer 4 (Request Configuration) Mutation:** Override temperature and truncate `max_tokens`.
* **Milestone M2:** Mock AI Provider operational with toggleable four-layer perturbation injection.

### Phase 3: Synthetic Canary Probe Suite & Extended Telemetry (Hours 12–18)
* **Goal:** Expand synthetic probes and build comprehensive resource profiling.
* **Deliverables:**
  * [ ] Expand evaluation suite to 50–100 curated tasks covering all 5 capability domains:
    * Mathematics (arithmetic, percentage, algebra)
    * Logical Reasoning (syllogisms, deductive constraints)
    * Structured JSON (schema compliance, nested keys, types)
    * Instruction Following (negative constraints, casing, formatting)
    * Safety & Guardrails (benign queries with sensitive keywords)
  * [ ] Integrate NVML GPU VRAM profiling (`pynvml`) alongside host RAM tracking.
  * [ ] Add automated baseline recording (`save_baseline.py`).
* **Milestone M3:** Comprehensive canary suite capable of running low-overhead continuous or batch health probes.

### Phase 4: Statistical Drift Detection & Root-Cause Attribution (Hours 18–26)
* **Goal:** Implement the statistical decision logic and four-layer diagnostic classifier.
* **Deliverables:**
  * [ ] Implement `src/detector/statistical_test.py`:
    * Two-proportion $z$-test and Fisher's exact test per capability category.
    * Cohen's $h$ effect size calculation and degradation severity bands.
  * [ ] Implement `src/attribution/engine.py`:
    * Multi-factor signature matching against Model, Serving Stack, Safety, and Request Config.
    * Attribution confidence scoring algorithm.
    * Evidence synthesis separating observed facts from inferred causes.
* **Milestone M4:** Statistical detector producing verified drift alerts with evidence-backed attribution.

### Phase 5: Verification of 4 Controlled Experiments & Metrics (Hours 26–32)
* **Goal:** Execute the four controlled experiments and evaluate the monitor's diagnostic performance.
* **Deliverables:**
  * [ ] **Experiment 1 (Model):** Run baseline Q8 vs. Q4 -> Verify detection of reasoning drop and Model attribution.
  * [ ] **Experiment 2 (Serving Stack):** Inject latency delay -> Verify detection of TTFT shift and Serving Stack attribution.
  * [ ] **Experiment 3 (Safety Settings):** Inject refusal filters -> Verify detection of refusal surge and Safety attribution.
  * [ ] **Experiment 4 (Request Config):** Inject token truncation -> Verify detection of syntax cutoff and Request Config attribution.
  * [ ] Compute system evaluation metrics:
    * False-Alarm Rate ($\alpha$) across repeated unperturbed baseline runs.
    * Missed-Change Rate ($\beta$) across injected anomalies.
    * Attribution Accuracy (% of injections correctly diagnosed).
    * Probe execution cost and total test duration.
* **Milestone M5:** Complete validation report proving high attribution accuracy and low false-alarm rate across all 4 experiments.

### Phase 6: Reporting Dashboard, Demonstration Freeze & Rehearsal (Hours 32–36)
* **Goal:** Final polish, CLI visualization, and presentation preparation.
* **Deliverables:**
  * [ ] Terminal UI dashboard displaying real-time probe progress, category accuracy, and attribution cards.
  * [ ] Automated Markdown report generation (`generate_report.py`).
  * [ ] Rehearse live demonstration: Baseline -> Inject Mutation -> Detect Drift -> Attribute Root Cause -> Present Evidence.
* **Milestone M6:** Hackathon submission freeze and live demonstration readiness.

---

## 4. Technical Risks & Mitigations

| Risk | Impact | Mitigation Strategy |
|---|---|---|
| **GPU Out of Memory (OOM) on 4 GB VRAM** | High | Strictly enforce sequential model loading when running real model comparisons. Context window capped at 2,048 tokens. |
| **Subtle Quantization Degradation Signal** | Medium | Target small models (~1.5B parameters) which exhibit higher quantization fragility. Stress test with multi-step math and strict schema adherence. |
| **Attribution Ambiguity Across Layers** | Medium | Maintain distinct evidence signatures (e.g., serving latency changes do not alter accuracy; safety filters only hit safety probes; truncation cuts off mid-token). Explicitly report confidence percentages and separate evidence from inference. |
| **Inference Time Overrun During Probes** | Medium | Cap smoke tests at 10–30 tasks and full runs at 100 tasks. Limit `max_tokens` per task to avoid runaway generation loops. |
