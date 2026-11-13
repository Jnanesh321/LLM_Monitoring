"""Benchmark execution engine coordinating runner, dataset, and deterministic scorers."""

import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.runner.base import BaseModelRunner
from src.scorers import get_scorer, EvaluationResult
from src.telemetry.memory import MemoryTracker


class BenchmarkEngine:
    """Orchestrates loading test suites, executing tasks, and scoring results."""

    def __init__(self, runner: BaseModelRunner, results_dir: str = "results"):
        self.runner = runner
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load benchmark tasks from a JSON file."""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Benchmark dataset not found at: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(self, dataset_path: str, verbose: bool = True) -> Dict[str, Any]:
        """Execute all tasks in the dataset against the configured model runner."""
        dataset = self.load_dataset(dataset_path)
        tasks: List[Dict[str, Any]] = dataset.get("tasks", [])
        suite_name = dataset.get("benchmark_name", "Benchmark Suite")

        model_info = self.runner.get_model_info()
        model_name = getattr(self.runner, "model_name", "unknown")
        quant_level = model_info.get("quantization_level", "unknown")

        if verbose:
            print(f"\n{'='*95}")
            print(f"Running Benchmark: {suite_name}")
            print(f"Target Model:     {model_name} (Quant: {quant_level})")
            print(f"Total Tasks:      {len(tasks)}")
            print(f"{'='*95}\n")

        task_results: List[Dict[str, Any]] = []
        category_stats: Dict[str, Dict[str, Any]] = {}
        total_latency_ms = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        peak_gpu_vram_mib = 0.0
        peak_system_ram_gb = 0.0

        for idx, task in enumerate(tasks, 1):
            task_id = task.get("id", f"task_{idx}")
            category = task.get("category", "general")
            prompt = task.get("prompt", "")
            expected = task.get("expected")
            eval_type = task.get("eval_type", "exact_match")

            # Initialize category stats if new
            if category not in category_stats:
                category_stats[category] = {
                    "total": 0,
                    "passed": 0,
                    "json_valid_count": 0,
                    "format_compliant_count": 0,
                }
            category_stats[category]["total"] += 1

            if verbose:
                print(f"[{idx:>2}/{len(tasks):<2}] {task_id:<14} | {category:<22} ... ", end="", flush=True)

            # Metadata passed to scorer
            metadata = {
                "tolerance": task.get("tolerance", 0.0),
                "case_sensitive": task.get("case_sensitive", False),
                "eval_type": eval_type,
                "strict_raw": task.get("strict_raw", True),
            }

            try:
                gen_output = self.runner.generate(prompt)
                total_latency_ms += gen_output.latency_ms
                total_prompt_tokens += gen_output.prompt_tokens
                total_completion_tokens += gen_output.completion_tokens

                # Sample memory
                mem = MemoryTracker.sample()
                vram_used = mem.get("gpu_vram", {}).get("used_mib", 0.0)
                ram_used = mem.get("system_ram", {}).get("used_gb", 0.0)
                if vram_used > peak_gpu_vram_mib:
                    peak_gpu_vram_mib = vram_used
                if ram_used > peak_system_ram_gb:
                    peak_system_ram_gb = ram_used

                scorer = get_scorer(eval_type)
                eval_res: EvaluationResult = scorer.score(
                    completion=gen_output.raw_text,
                    expected=expected,
                    metadata=metadata,
                )

                if eval_res.passed:
                    category_stats[category]["passed"] += 1

                # Track JSON specific metrics
                if category == "structured_json":
                    if eval_res.metrics.get("json_valid"):
                        category_stats[category]["json_valid_count"] += 1
                    if eval_res.metrics.get("raw_json_compliant"):
                        category_stats[category]["format_compliant_count"] += 1

                status_str = "PASS" if eval_res.passed else "FAIL"
                if verbose:
                    print(f"[{status_str}] ({gen_output.latency_ms:.0f}ms)")

                record = {
                    "id": task_id,
                    "category": category,
                    "prompt": prompt,
                    "expected": expected,
                    "actual_response": gen_output.raw_text,
                    "extracted_value": eval_res.extracted_value,
                    "eval_type": eval_type,
                    "passed": eval_res.passed,
                    "score": eval_res.score,
                    "details": eval_res.details,
                    "metrics": eval_res.metrics,
                    "latency_ms": gen_output.latency_ms,
                    "prompt_tokens": gen_output.prompt_tokens,
                    "completion_tokens": gen_output.completion_tokens,
                    "tokens_per_second": gen_output.tokens_per_second,
                    "done_reason": gen_output.metadata.get("done_reason", "unknown"),
                    "effective_request_params": gen_output.metadata.get("effective_request_params", {}),
                }
                task_results.append(record)

            except Exception as e:
                if verbose:
                    print(f"[ERROR] {str(e)}")
                record = {
                    "id": task_id,
                    "category": category,
                    "prompt": prompt,
                    "expected": expected,
                    "actual_response": None,
                    "extracted_value": None,
                    "eval_type": eval_type,
                    "passed": False,
                    "score": 0.0,
                    "details": f"Execution error: {str(e)}",
                    "metrics": {},
                    "latency_ms": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "tokens_per_second": 0.0,
                    "done_reason": "error",
                    "effective_request_params": {},
                }
                task_results.append(record)

        # Aggregate metrics
        total_tasks = len(tasks)
        total_passed = sum(1 for r in task_results if r["passed"])
        overall_accuracy = (total_passed / total_tasks * 100.0) if total_tasks > 0 else 0.0
        avg_latency_ms = (total_latency_ms / total_tasks) if total_tasks > 0 else 0.0

        category_summary = {}
        for cat, stats in category_stats.items():
            tot = stats["total"]
            pas = stats["passed"]
            acc = (pas / tot * 100.0) if tot > 0 else 0.0
            cat_dict: Dict[str, Any] = {
                "total": tot,
                "passed": pas,
                "accuracy_percent": round(acc, 2),
            }
            if cat == "structured_json":
                jv = stats["json_valid_count"]
                fc = stats["format_compliant_count"]
                cat_dict["json_validity_percent"] = round((jv / tot * 100.0) if tot > 0 else 0.0, 2)
                cat_dict["format_compliance_percent"] = round((fc / tot * 100.0) if tot > 0 else 0.0, 2)
                cat_dict["json_valid_count"] = jv
                cat_dict["format_compliant_count"] = fc

            category_summary[cat] = cat_dict

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        completion_token_counts = [r["completion_tokens"] for r in task_results if r.get("completion_tokens", 0) > 0]
        length_truncation_count = sum(1 for r in task_results if r.get("done_reason") == "length")

        report = {
            "benchmark_name": suite_name,
            "timestamp_utc": timestamp_str,
            "model_info": model_info,
            "summary": {
                "total_tasks": total_tasks,
                "passed_tasks": total_passed,
                "overall_accuracy_percent": round(overall_accuracy, 2),
                "avg_latency_ms": round(avg_latency_ms, 2),
                "total_latency_ms": round(total_latency_ms, 2),
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "token_stats": {
                    "min_completion_tokens": min(completion_token_counts) if completion_token_counts else 0,
                    "max_completion_tokens": max(completion_token_counts) if completion_token_counts else 0,
                    "length_truncations_count": length_truncation_count,
                },
                "memory": {
                    "peak_gpu_vram_mib": round(peak_gpu_vram_mib, 1),
                    "peak_system_ram_gb": round(peak_system_ram_gb, 2),
                },
                "category_breakdown": category_summary,
            },
            "results": task_results,
        }

        # Save results to disk
        safe_model_name = model_name.replace(":", "_").replace("/", "_")
        filename = f"benchmark_{safe_model_name}_{timestamp_str}.json"
        output_path = os.path.join(self.results_dir, filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        if verbose:
            self._print_summary_table(report, output_path)

        return report

    def _format_cell(self, val: Any, max_len: int = 25) -> str:
        """Formats cell content cleanly for ASCII table display."""
        if val is None:
            return "None"
        if isinstance(val, dict):
            s = json.dumps(val)
        else:
            s = str(val).replace("\n", " ")
        if len(s) > max_len:
            return s[: max_len - 3] + "..."
        return s

    def _print_summary_table(self, report: Dict[str, Any], output_path: str) -> None:
        """Render a formatted ASCII summary report with individual task results."""
        summary = report["summary"]
        model_info = report["model_info"]
        results = report["results"]

        print(f"\n{'='*100}")
        print(f"INDIVIDUAL TASK RESULTS")
        print(f"{'='*100}")
        header = f"{'ID':<12} | {'Category':<20} | {'Expected':<18} | {'Extracted':<18} | {'Status':<6} | {'Latency':<9} | {'TPS':<6}"
        print(header)
        print(f"{'-'*100}")
        for r in results:
            task_id = r["id"]
            cat = r["category"]
            exp = self._format_cell(r["expected"], 18)
            ext = self._format_cell(r["extracted_value"], 18)
            status = "PASS" if r["passed"] else "FAIL"
            lat = f"{r['latency_ms']:.0f}ms"
            tps = f"{r['tokens_per_second']:.1f}"
            print(f"{task_id:<12} | {cat:<20} | {exp:<18} | {ext:<18} | {status:<6} | {lat:<9} | {tps:<6}")

        print(f"\n{'='*100}")
        print(f"BENCHMARK EXECUTION SUMMARY")
        print(f"{'='*100}")
        print(f"Model Name:       {model_info.get('model_name')}")
        print(f"Quantization:     {model_info.get('quantization_level')}")
        print(f"Overall Accuracy: {summary['overall_accuracy_percent']}% ({summary['passed_tasks']}/{summary['total_tasks']} passed)")
        print(f"Average Latency:  {summary['avg_latency_ms']} ms")
        print(f"Total Tokens:     {summary['total_prompt_tokens']} prompt, {summary['total_completion_tokens']} completion")
        if "memory" in summary:
            mem = summary["memory"]
            print(f"Peak Memory:      GPU VRAM: {mem.get('peak_gpu_vram_mib', 0)} MiB | Host RAM: {mem.get('peak_system_ram_gb', 0)} GB")
        print(f"{'-'*100}")
        print(f"{'Category':<24} | {'Total':<6} | {'Passed':<6} | {'Accuracy':<10}")
        print(f"{'-'*100}")
        for cat, data in summary["category_breakdown"].items():
            print(f"{cat:<24} | {data['total']:<6} | {data['passed']:<6} | {data['accuracy_percent']:>8.1f}%")

        if "structured_json" in summary["category_breakdown"]:
            j_data = summary["category_breakdown"]["structured_json"]
            print(f"{'-'*100}")
            print(f"Structured JSON Diagnostic Breakdown:")
            print(f"  - Syntax Parse Validity:       {j_data.get('json_validity_percent', 0.0):>5.1f}% ({j_data.get('json_valid_count', 0)}/{j_data['total']} passed)")
            print(f"  - Strict Format Compliance:   {j_data.get('format_compliance_percent', 0.0):>5.1f}% ({j_data.get('format_compliant_count', 0)}/{j_data['total']} passed without markdown fences)")

        print(f"{'='*100}")
        print(f"Detailed results saved to: {output_path}\n")
