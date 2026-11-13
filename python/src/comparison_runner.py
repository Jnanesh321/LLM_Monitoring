"""Multi-precision comparison runner executing identical benchmarks across model variants."""

import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.runner.ollama_runner import OllamaRunner
from src.benchmark import BenchmarkEngine


class PrecisionComparisonRunner:
    """Orchestrates sequential, paired benchmark evaluations across precision variants (FP16, Q8, Q4)."""

    def __init__(
        self,
        model_names: List[str],
        dataset_path: str = "evaluation/benchmarks/smoke_benchmark.json",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        seed: int = 42,
        results_dir: str = "results",
    ):
        self.model_names = model_names

        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(dataset_path) and not os.path.exists(dataset_path):
            candidate = os.path.join(pkg_root, dataset_path)
            if os.path.exists(candidate):
                dataset_path = candidate
        self.dataset_path = dataset_path

        self.base_url = base_url
        self.temperature = temperature
        self.seed = seed

        if not os.path.isabs(results_dir) and not os.path.exists(results_dir):
            results_dir = os.path.join(pkg_root, results_dir)
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def check_availability(self) -> Dict[str, Dict[str, Any]]:
        """Verify which configured models are currently available in the local Ollama instance."""
        local_models = OllamaRunner.list_local_models(self.base_url)

        # Expected specifications for Qwen2.5 1.5B precision variants
        known_specs = {
            "qwen2.5:1.5b-instruct-fp16": {
                "target_precision": "FP16 (16-bit unquantized)",
                "download_size": "2.88 GB (3.09 GB)",
                "est_vram_mib": 3350.0,
                "est_ram_gb": 4.5,
                "notes": "Unquantized weights; fits in 4 GB VRAM under sequential loading with minimal headroom.",
            },
            "qwen2.5:1.5b-instruct-q8_0": {
                "target_precision": "Q8_0 (8-bit quantization)",
                "download_size": "1.53 GB (1.65 GB)",
                "est_vram_mib": 2100.0,
                "est_ram_gb": 3.0,
                "notes": "Near-lossless 8-bit precision; fits comfortably in RTX 3050 4 GB VRAM.",
            },
            "qwen2.5:1.5b": {
                "target_precision": "Q4_K_M (4-bit medium k-quant)",
                "download_size": "986 MB (0.92 GB)",
                "est_vram_mib": 1350.0,
                "est_ram_gb": 2.0,
                "notes": "Standard 4-bit quantization; fully fits in 4 GB VRAM with ~2.5 GB headroom.",
            },
        }

        availability = {}
        for name in self.model_names:
            is_installed = name in local_models
            spec = known_specs.get(
                name,
                {
                    "target_precision": "Custom",
                    "download_size": "Unknown",
                    "est_vram_mib": 0.0,
                    "est_ram_gb": 0.0,
                    "notes": "User-specified variant",
                },
            )
            availability[name] = {
                "installed": is_installed,
                "target_precision": spec["target_precision"],
                "download_size": spec["download_size"],
                "est_vram_mib": spec["est_vram_mib"],
                "est_ram_gb": spec["est_ram_gb"],
                "notes": spec["notes"],
            }

        return availability

    def run_comparison(self, verbose: bool = True) -> Dict[str, Any]:
        """Execute the exact same benchmark across all available precision variants."""
        availability = self.check_availability()
        installed_models = [m for m, status in availability.items() if status["installed"]]
        missing_models = [m for m, status in availability.items() if not status["installed"]]

        if verbose:
            print(f"\n{'='*100}")
            print("PRECISION VARIANT AVAILABILITY CHECK")
            print(f"{'='*100}")
            for m, status in availability.items():
                mark = "[AVAILABLE]" if status["installed"] else "[MISSING]"
                print(f"{mark:<12} {m:<32} | Precision: {status['target_precision']}")
                if not status["installed"]:
                    print(f"             Download Size: {status['download_size']} | Est VRAM: {status['est_vram_mib']} MiB")
                    print(f"             Action required: Run `ollama pull {m}` to enable local execution.")
            print(f"{'='*100}\n")

        variant_reports: Dict[str, Any] = {}

        # Sequentially evaluate each available model
        for model_name in installed_models:
            if verbose:
                print(f"\n>>> Loading and evaluating variant: {model_name} (Sequential Execution) <<<\n")

            runner = OllamaRunner(
                model_name=model_name,
                base_url=self.base_url,
                default_temperature=self.temperature,
                default_seed=self.seed,
            )

            engine = BenchmarkEngine(runner=runner, results_dir=self.results_dir)
            report = engine.run(dataset_path=self.dataset_path, verbose=verbose)
            variant_reports[model_name] = report

            # Explicitly unload the model to enforce sequential VRAM isolation
            if verbose:
                print(f"\n[CLEANUP] Unloading {model_name} from GPU VRAM to maintain isolated benchmark integrity...")
            runner.unload()

        # Build combined comparative report
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        comparison_payload: Dict[str, Any] = {
            "experiment": "Qwen2.5 1.5B Multi-Precision Comparison (FP16 vs Q8 vs Q4)",
            "timestamp_utc": timestamp_str,
            "dataset": self.dataset_path,
            "hyperparameters": {
                "temperature": self.temperature,
                "seed": self.seed,
                "execution_mode": "Sequential GPU Loading (RTX 3050 4 GB VRAM)",
            },
            "availability": availability,
            "evaluated_variants": {},
            "missing_variants": missing_models,
        }

        for model_name, rep in variant_reports.items():
            summary = rep["summary"]
            info = rep["model_info"]
            comparison_payload["evaluated_variants"][model_name] = {
                "model_name": model_name,
                "parameter_size": info.get("parameter_size", "1.5B"),
                "quantization_level": info.get("quantization_level", "Unknown"),
                "overall_accuracy_percent": summary["overall_accuracy_percent"],
                "passed_tasks": summary["passed_tasks"],
                "total_tasks": summary["total_tasks"],
                "avg_latency_ms": summary["avg_latency_ms"],
                "total_prompt_tokens": summary.get("total_prompt_tokens", 0),
                "total_completion_tokens": summary.get("total_completion_tokens", 0),
                "memory": summary.get("memory", {}),
                "category_breakdown": summary["category_breakdown"],
            }

        # Save combined report
        out_filename = f"comparison_qwen2.5_1.5b_{timestamp_str}.json"
        out_path = os.path.join(self.results_dir, out_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(comparison_payload, f, indent=2, ensure_ascii=False)

        if verbose:
            self._print_comparison_table(comparison_payload, out_path)

        return comparison_payload

    def _print_comparison_table(self, comp: Dict[str, Any], output_path: str) -> None:
        """Render side-by-side comparison table."""
        evaluated = comp["evaluated_variants"]
        missing = comp["missing_variants"]

        print(f"\n{'='*105}")
        print("MULTI-PRECISION COMPARATIVE BENCHMARK SUMMARY")
        print(f"{'='*105}")
        print(f"Dataset:          {comp['dataset']}")
        print(f"Hyperparameters:  Temperature={comp['hyperparameters']['temperature']}, Seed={comp['hyperparameters']['seed']}")
        print(f"Execution Mode:   {comp['hyperparameters']['execution_mode']}")
        print(f"{'-'*105}")

        models = list(evaluated.keys())
        if not models:
            print("[INFO] No requested variants were available for execution.")
            print(f"Missing models: {missing}")
            print(f"{'='*105}\n")
            return

        header = f"{'Metric / Capability':<32}"
        for m in models:
            header += f" | {m:<22}"
        print(header)
        print(f"{'-'*105}")

        # Row 1: Quantization
        row = f"{'Quantization Level':<32}"
        for m in models:
            q = evaluated[m]["quantization_level"]
            row += f" | {q:<22}"
        print(row)

        # Row 2: Overall Accuracy
        row = f"{'Overall Accuracy':<32}"
        for m in models:
            acc = f"{evaluated[m]['overall_accuracy_percent']}% ({evaluated[m]['passed_tasks']}/{evaluated[m]['total_tasks']})"
            row += f" | {acc:<22}"
        print(row)

        # Row 3: Avg Latency
        row = f"{'Avg Latency (ms)':<32}"
        for m in models:
            lat = f"{evaluated[m]['avg_latency_ms']:.1f} ms"
            row += f" | {lat:<22}"
        print(row)

        # Row 4: Peak GPU VRAM
        row = f"{'Peak GPU VRAM (MiB)':<32}"
        for m in models:
            vram = f"{evaluated[m]['memory'].get('peak_gpu_vram_mib', 0):.1f} MiB"
            row += f" | {vram:<22}"
        print(row)

        # Category Rows
        categories = ["mathematics", "logical_reasoning", "instruction_following", "structured_json"]
        for cat in categories:
            row = f"  - {cat.replace('_', ' ').title():<30}"
            for m in models:
                cat_data = evaluated[m]["category_breakdown"].get(cat, {})
                cat_acc = f"{cat_data.get('accuracy_percent', 0.0):.1f}% ({cat_data.get('passed', 0)}/{cat_data.get('total', 0)})"
                row += f" | {cat_acc:<22}"
            print(row)

        # JSON sub-metrics
        row_jv = f"    * JSON Syntax Validity:<30"
        row_fc = f"    * Raw Format Compliance:<30"
        has_json_sub = False
        for m in models:
            j_data = evaluated[m]["category_breakdown"].get("structured_json", {})
            if "json_validity_percent" in j_data:
                has_json_sub = True
                row_jv += f" | {j_data.get('json_validity_percent', 0.0):.1f}%{'':<16}"
                row_fc += f" | {j_data.get('format_compliance_percent', 0.0):.1f}%{'':<16}"
        if has_json_sub:
            print(row_jv)
            print(row_fc)

        if missing:
            print(f"{'-'*105}")
            print(f"Unavailable / Missing Variants (Not Evaluated): {', '.join(missing)}")

        print(f"{'='*105}")
        print(f"Combined comparison report saved to: {output_path}\n")

