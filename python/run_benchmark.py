"""CLI entry point for running the Automated Benchmark Runner."""

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.runner.ollama_runner import OllamaRunner
from src.benchmark import BenchmarkEngine


def main():
    parser = argparse.ArgumentParser(
        description="AI Model Behaviour Drift Detection - Automated Benchmark Runner"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:1.5b",
        help="Ollama model tag to evaluate (e.g., qwen2.5:1.5b, qwen3:1.7b). Default: qwen2.5:1.5b",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(SCRIPT_DIR, "evaluation", "benchmarks", "smoke_benchmark.json"),
        help="Path to the benchmark tasks JSON file. Default: evaluation/benchmarks/smoke_benchmark.json",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server base URL. Default: http://localhost:11434",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature for deterministic testing. Default: 0.0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reproducibility. Default: 42",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "results"),
        help="Directory to save timestamped JSON results. Default: results",
    )

    args = parser.parse_args()

    # Initialize runner
    runner = OllamaRunner(
        model_name=args.model,
        base_url=args.ollama_url,
        default_temperature=args.temperature,
        default_seed=args.seed,
    )

    # Health check
    if not runner.check_health():
        print(f"[ERROR] Could not connect to Ollama server at {args.ollama_url}.")
        print("Please ensure Ollama is running (`ollama serve`).")
        sys.exit(1)

    # Initialize engine and execute
    engine = BenchmarkEngine(runner=runner, results_dir=args.results_dir)
    try:
        report = engine.run(dataset_path=args.dataset, verbose=True)
    except Exception as e:
        print(f"[FATAL] Benchmark execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

