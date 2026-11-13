# Hardware Architecture & Deployment Specification

## 1. Workstation Hardware Profile

* **Host Machine:** Developer Workstation / Laptop
* **CPU:** Intel Core i5 (6 cores / 12 threads)
* **System Memory (RAM):** 16 GB DDR4
* **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU / RTX 3050A
* **VRAM Capacity:** 4,096 MiB (~3.5 GB usable after Windows DWM and OS allocations)
* **CUDA Driver Version:** 13.3 (Driver 610.88)
* **Operating System:** Windows 11 (PowerShell environment)

---

## 2. VRAM Feasibility & Multi-Layer Experimentation

Given the 3,500 MiB usable VRAM constraint, large language models ($\ge 7\text{B}$ parameters) cannot fit inside GPU memory without severe CPU offload penalties that distort latency benchmarks.

The monitoring system is structured to execute all four controlled experiments strictly within this hardware boundary:

| Experiment / Component | Execution Mode | Memory Impact | Hardware Feasibility on RTX 3050 |
|---|---|---|---|
| **Real Model Baseline (`qwen2.5:1.5b` Q4_K_M)** | Ollama GPU Inference | ~1.1 GB VRAM | **Fits fully on GPU** with abundant headroom. |
| **Model Layer (Q8_0 vs. Q4_K_M Comparison)** | Sequential Model Loading | ~1.9 GB (Q8) / ~1.1 GB (Q4) | **Fits fully on GPU** when loaded sequentially. |
| **Serving Stack Layer (Latency / TTFT Injection)** | Mock AI Provider Proxy + Model | Proxy: ~20 MB RAM; Model: ~1.1 GB VRAM | **Zero VRAM overhead** beyond base model. |
| **Safety Settings Layer (Refusal Filtering)** | Mock AI Provider Proxy + Model | Proxy: ~20 MB RAM; Model: ~1.1 GB VRAM | **Zero VRAM overhead** beyond base model. |
| **Request Configuration Layer (Parameter Overrides)**| Mock AI Provider Proxy + Model | Proxy: ~20 MB RAM; Model: ~1.1 GB VRAM | **Zero VRAM overhead** beyond base model. |

---

## 3. Execution Guardrails for Hackathon Demonstrations

To ensure benchmark reproducibility and prevent system lockups or Out-Of-Memory (OOM) errors during evaluation:

1. **Strict Sequential Model Loading:** When evaluating real precision variants (e.g., Full/Q8 vs. Q4), the harness loads Variant A, runs the canary probe batch, unloads Variant A from memory, and then loads Variant B. Concurrent dual-model loading in 4 GB VRAM is strictly prohibited.
2. **Context Window Cap:** Context length is capped at `n_ctx = 2048` tokens to limit KV-cache memory consumption to under 250 MB.
3. **Generation Token Limit:** Tasks enforce `max_tokens = 512` to prevent runaway generation or hallucination loops from blocking evaluation pipelines.
4. **Thermal Stability Sleep:** A brief 250ms pause is maintained between probe queries to prevent GPU thermal throttling from introducing artificial latency jitter into Serving Stack measurements.
