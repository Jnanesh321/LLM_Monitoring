"""Executable script running full 70-probe behavioural evaluation between baseline and candidate models."""

import os
import sys
import json
import time
from datetime import datetime, timezone

# Ensure python/src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.runner.ollama_runner import OllamaRunner
from src.behavioural_harness import BehaviouralEvaluationHarness


def run_full_evaluation():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/canaries/behavioural_evaluation_v1.json"))
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/evaluation"))
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("MODELWATCH: 70-QUERY BEHAVIOURAL EVALUATION WORKFLOW")
    print("=" * 80)
    print(f"Dataset path: {dataset_path}")
    print(f"Results dir:  {results_dir}")

    # Models: Baseline Q8 vs Candidate Q4
    baseline_model = "qwen2.5:1.5b-instruct-q8_0"
    candidate_model = "qwen2.5:1.5b"

    print(f"\n[1/4] Initializing Baseline Model: {baseline_model}")
    base_runner = OllamaRunner(
        model_name=baseline_model,
        default_temperature=0.0,
        default_seed=42,
        timeout_seconds=90,
    )
    if not base_runner.check_health():
        raise ConnectionError("Ollama service unreachable on http://localhost:11434")

    base_harness = BehaviouralEvaluationHarness(base_runner, results_dir=results_dir)

    print(f"\n[2/4] Executing 70 Probes on Baseline ({baseline_model})...")
    start_base = time.time()
    baseline_run = base_harness.run_suite(dataset_path, verbose=False)
    base_duration = time.time() - start_base
    print(f"  --> Baseline run complete in {base_duration:.1f}s")
    print(f"  --> Baseline Pass Rate: {baseline_run['overall_pass_rate']*100:.2f}% ({baseline_run['passed_probes']}/{baseline_run['total_probes']})")
    print(f"  --> Mean Latency: {baseline_run['mean_latency_ms']:.1f} ms")

    # Explicitly unload baseline model from VRAM before loading candidate
    print(f"\nUnloading baseline {baseline_model} from VRAM...")
    base_runner.unload()
    time.sleep(2)

    print(f"\n[3/4] Initializing Candidate Model: {candidate_model}")
    cand_runner = OllamaRunner(
        model_name=candidate_model,
        default_temperature=0.0,
        default_seed=42,
        timeout_seconds=90,
    )
    cand_harness = BehaviouralEvaluationHarness(cand_runner, results_dir=results_dir)

    print(f"\n[4/4] Executing 70 Probes on Candidate ({candidate_model})...")
    start_cand = time.time()
    candidate_run = cand_harness.run_suite(dataset_path, verbose=False)
    cand_duration = time.time() - start_cand
    print(f"  --> Candidate run complete in {cand_duration:.1f}s")
    print(f"  --> Candidate Pass Rate: {candidate_run['overall_pass_rate']*100:.2f}% ({candidate_run['passed_probes']}/{candidate_run['total_probes']})")
    print(f"  --> Mean Latency: {candidate_run['mean_latency_ms']:.1f} ms")

    # Unload candidate model
    cand_runner.unload()

    print("\nComputing head-to-head comparison...")
    comparison = BehaviouralEvaluationHarness.compare_runs(baseline_run, candidate_run)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_meta = json.load(f)

    # Save consolidated multi_query_suite_latest.json
    latest_payload = {
        "dataset": dataset_meta,
        "baseline_run": baseline_run,
        "candidate_run": candidate_run,
        "comparison": comparison,
    }
    latest_path = os.path.join(results_dir, "multi_query_suite_latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(latest_payload, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully written consolidated suite artifact to: {latest_path}")
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Baseline ({baseline_model}):  {baseline_run['overall_pass_rate']*100:.2f}% pass rate")
    print(f"Candidate ({candidate_model}): {candidate_run['overall_pass_rate']*100:.2f}% pass rate")
    print(f"Difference:                   {comparison['difference_percentage_points']:.2f} percentage points")
    print(f"Regressions:                  {comparison['regressions_count']} probes")
    print(f"Improvements:                 {comparison['improvements_count']} probes")
    print("\nCategory Breakdown:")
    for cat in comparison["category_comparison"]:
        print(f"  - {cat['category']:<36}: Baseline {(cat['baseline_pass_rate']*100):>5.1f}% | Candidate {(cat['candidate_pass_rate']*100):>5.1f}% | Diff {cat['diff_percentage_points']:>+5.1f} pp | Regressions: {cat['regressions_count']} | Status: {cat['status']}")

    if comparison["regressions"]:
        print("\nRegressed Probes (Baseline PASS -> Candidate FAIL):")
        for reg in comparison["regressions"]:
            print(f"  • {reg['probe_id']} [{reg['category']}]: {reg['failure_reason']}")

    return latest_payload


if __name__ == "__main__":
    run_full_evaluation()

