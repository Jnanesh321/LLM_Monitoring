"""CLI entry point for running the PS-03 Controlled Layer Experiment Validation Suite."""

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.validation.experiment_suite import ControlledExperimentSuite


def main():
    parser = argparse.ArgumentParser(
        description="PS-03: Controlled Layer Fault-Injection and Attribution Validation Suite"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="qwen2.5:1.5b-instruct-q8_0",
        help="Baseline model tag in Ollama. Default: qwen2.5:1.5b-instruct-q8_0",
    )
    parser.add_argument(
        "--quant-model",
        type=str,
        default="qwen2.5:1.5b",
        help="Quantized model tag for model-layer fault injection. Default: qwen2.5:1.5b",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(SCRIPT_DIR, "evaluation", "benchmarks", "canary_suite.json"),
        help="Path to Canary probe dataset JSON file. Default: evaluation/benchmarks/canary_suite.json",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server base URL. Default: http://localhost:11434",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "results"),
        help="Directory to save timestamped JSON validation reports. Default: results",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        help="Subset of experiments to execute: 'all', 'baseline', 'model', 'serving', 'safety', 'request', or specific experiment IDs. Default: all",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run in quiet mode with minimal terminal output.",
    )

    args = parser.parse_args()

    suite = ControlledExperimentSuite(
        base_model=args.base_model,
        quant_model=args.quant_model,
        dataset_path=args.dataset,
        base_url=args.ollama_url,
        results_dir=args.results_dir,
    )

    try:
        suite.run_suite(verbose=not args.quiet, experiment_filter=args.experiments)
    except Exception as e:
        print(f"\n[FATAL] Experiment suite execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

