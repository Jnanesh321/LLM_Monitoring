"""CLI entry point for running multi-precision benchmark comparisons."""

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.comparison_runner import PrecisionComparisonRunner


def main():
    parser = argparse.ArgumentParser(
        description="PS-03: Multi-Precision Benchmark Comparison Runner (FP16 vs Q8 vs Q4)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "qwen2.5:1.5b-instruct-fp16",
            "qwen2.5:1.5b-instruct-q8_0",
            "qwen2.5:1.5b",
        ],
        help="List of model variant tags to evaluate. Default: qwen2.5:1.5b-instruct-fp16 qwen2.5:1.5b-instruct-q8_0 qwen2.5:1.5b",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(SCRIPT_DIR, "evaluation", "benchmarks", "smoke_benchmark.json"),
        help="Path to benchmark tasks JSON file. Default: evaluation/benchmarks/smoke_benchmark.json",
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
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check model availability and hardware feasibility without executing inference.",
    )

    args = parser.parse_args()

    runner = PrecisionComparisonRunner(
        model_names=args.models,
        dataset_path=args.dataset,
        base_url=args.ollama_url,
        temperature=args.temperature,
        seed=args.seed,
        results_dir=args.results_dir,
    )

    if args.check_only:
        availability = runner.check_availability()
        print(f"\n{'='*95}")
        print("PRECISION VARIANT AVAILABILITY & HARDWARE FEASIBILITY")
        print(f"{'='*95}")
        for m, status in availability.items():
            mark = "[AVAILABLE]" if status["installed"] else "[MISSING]"
            print(f"{mark:<12} {m:<32} | Precision: {status['target_precision']}")
            print(f"             Download Size: {status['download_size']} | Est VRAM: {status['est_vram_mib']} MiB")
            print(f"             Hardware note: {status['notes']}")
        print(f"{'='*95}\n")
        return

    try:
        runner.run_comparison(verbose=True)
    except Exception as e:
        print(f"[FATAL] Comparison run failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

