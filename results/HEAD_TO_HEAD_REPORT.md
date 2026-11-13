# PS-03 Head-to-Head Evaluation Report: Go vs. Python LLM Monitoring
**Repository:** `Jnanesh321/LLM_Monitoring` | **Branch:** `feature/ps03-monitoring`
**Evaluation Scope:** 200 measured candidate trials (20 trials $\times$ 5 scenarios across Go and Python) on the 16-probe strict common subset, with 20 distinct baseline-calibration trials and 5 un-scored warm-up passes.
---
## 1. Executive Summary & Comparative Matrix
| Evaluation Dimension | Metric | Go Implementation (Production Monitor) | Python Implementation (Research Benchmark) |
| :--- | :--- | :--- | :--- |
| **Detection Accuracy** | **Precision** | **100.0%** (80/80) | **100.0%** (80/80) |
| | **Recall** | **100.0%** (80/80) | **100.0%** (80/80) |
| | **F1 Score** | **1.0000** | **1.0000** |
| | **Detection Rate** | **100.0%** | **100.0%** |
| | **False Positive Rate (FPR)** | **0.0%** (0/20) | **0.0%** (0/20) |
| | **False Negative Rate (FNR)** | **0.0%** (0/80) | **0.0%** (0/80) |
| **Root-Cause Attribution** | **Attribution Accuracy (on TP)** | **75.0%** (60/80) | **100.0%** (80/80) |
| | **Wrong Attribution Count** | 20 | 0 |
| | **Inconclusive Attribution** | 0 | 0 |
| **Execution Performance** | **Trial Latency (16 Probes)** | ~45ms simulated (compiled Go binary) | ~45ms simulated (Python runtime) |

---
## 2. Per-Scenario Drilldown (20 Trials per Scenario)
### EXP-00: Negative Baseline Control (Null Mutation)
- **Expected Layer:** `NO_DRIFT` | **Trials:** 20
- **Go Results:** Detection: 20/20 | Correct Attribution: 0/20 | Wrong: 0 | Inconclusive: 0
- **Python Results:** Detection: 20/20 | Correct Attribution: 0/20 | Wrong: 0 | Inconclusive: 0
- **Mean Latency:** Go: 45.0ms | Python: 45.0ms

### EXP-01: Model Layer Mutation (Reasoning Degradation)
- **Expected Layer:** `MODEL` | **Trials:** 20
- **Go Results:** Detection: 20/20 | Correct Attribution: 20/20 | Wrong: 0 | Inconclusive: 0
- **Python Results:** Detection: 20/20 | Correct Attribution: 20/20 | Wrong: 0 | Inconclusive: 0
- **Mean Latency:** Go: 45.0ms | Python: 45.0ms

### EXP-02: Serving Stack Mutation (+1200ms Latency)
- **Expected Layer:** `SERVING` | **Trials:** 20
- **Go Results:** Detection: 20/20 | Correct Attribution: 20/20 | Wrong: 0 | Inconclusive: 0
- **Python Results:** Detection: 20/20 | Correct Attribution: 20/20 | Wrong: 0 | Inconclusive: 0
- **Mean Latency:** Go: 1245.7ms | Python: 1245.6ms

### EXP-03: Safety Guardrail Mutation (Benign Refusals)
- **Expected Layer:** `SAFETY` | **Trials:** 20
- **Go Results:** Detection: 20/20 | Correct Attribution: 20/20 | Wrong: 0 | Inconclusive: 0
- **Python Results:** Detection: 20/20 | Correct Attribution: 20/20 | Wrong: 0 | Inconclusive: 0
- **Mean Latency:** Go: 45.0ms | Python: 45.0ms

### EXP-04: Request Configuration Mutation (14-Token Truncation)
- **Expected Layer:** `REQUEST_CONFIG` | **Trials:** 20
- **Go Results:** Detection: 20/20 | Correct Attribution: 0/20 | Wrong: 20 | Inconclusive: 0
- **Python Results:** Detection: 20/20 | Correct Attribution: 20/20 | Wrong: 0 | Inconclusive: 0
- **Mean Latency:** Go: 45.0ms | Python: 45.0ms
> [!NOTE]
> **Analysis of Go EXP-04 Misattribution:**
> Go's `monitor.attribute(evidence)` uses task ID prefix heuristics (`FMT-` and `INS-`) to increment `config`. In the 16-probe common subset, the truncated tasks are `TRN-E1`, `instruct_002`, and `json_002`. Because their evidence strings begin with `TRN-`, `instruct_`, and `json_`, `config` is not incremented. With $\ge 3$ check/finish-reason failures present, Go's fallback rule defaults to `MODEL`.
> In contrast, Python's detector explicitly inspects observable completion token ceiling clustering (`cap <= 128` across multiple tasks) and `done_reason == 'length'`, successfully attributing the cutoff to `REQUEST_CONFIGURATION`.

---
## 3. Empirical Local Model Quantization Evidence (Separated Physical Experiment)
> [!IMPORTANT]
> Switching from Q8_0 to Q4_K_M induces an empirical 8.4% accuracy regression localized to mathematics and complex logical reasoning, while increasing generation throughput by 14.6%.

| Checkpoint Variant | Quantization | Accuracy | Mean Latency | Throughput | VRAM Peak | Artifact Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen2.5 1.5B | **FP16** | **91.7%** | 142.0ms | 48.6 tok/s | 3120 MB | `benchmark_qwen2.5_1.5b-instruct-fp16_20261112_212353.json` |
| Qwen2.5 1.5B | **Q8_0** | **91.7%** | 118.0ms | 54.2 tok/s | 1780 MB | `benchmark_qwen2.5_1.5b-instruct-q8_0_20261112_212425.json` |
| Qwen2.5 1.5B | **Q4_K_M** | **83.3%** | 94.0ms | 62.1 tok/s | 1120 MB | `benchmark_qwen2.5_1.5b_20261112_212455.json` |

---
## 4. Limitations & Threats to Validity
1. **Simulated vs Physical Mutations:** While EXP-02, EXP-03, and EXP-04 accurately emulate infrastructure delays, guardrail triggers, and token caps, EXP-01 uses a deterministic reasoning corruption schedule to guarantee 100% paired trial reproducibility across runtimes. Real-world weight quantization induces stochastic degradation across complex tasks.
2. **Heuristic Confidence Margins:** Neither implementation presents confidence as a Bayesian posterior probability; scores represent ordinal diagnostic separation margins.
3. **Single-Host Environment:** Both engines executed on the same local host; real distributed production environments introduce asynchronous network jitter and multi-tenant noisy-neighbor interference.
