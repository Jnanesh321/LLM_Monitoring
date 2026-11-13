"""Behavioural evaluation harness executing multi-query suites with deterministic verification."""

import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.runner.base import BaseModelRunner
from src.scorers import get_scorer, EvaluationResult
from src.telemetry.memory import MemoryTracker


class BehaviouralEvaluationHarness:
    """Harness that loads 70-query synthetic behavioural evaluation datasets,

    validates schema, executes individual probes or entire suites by category,
    verifies outputs via task-specific deterministic evaluators, records
    per-probe telemetry, and computes category & baseline comparisons.
    """

    def __init__(self, runner: BaseModelRunner, results_dir: str = "results"):
        self.runner = runner
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def load_and_validate_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Loads and validates the behavioural evaluation dataset."""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data.get("records")
        if not isinstance(records, list) or len(records) == 0:
            # Fallback for datasets organized as {"tasks": [...]}
            records = data.get("tasks", [])
            data["records"] = records

        if len(records) == 0:
            raise ValueError(f"Dataset at {dataset_path} contains 0 records.")

        # Validate schema of records
        required_fields = ["id", "category", "prompt"]
        for idx, r in enumerate(records):
            for field in required_fields:
                if field not in r:
                    raise ValueError(f"Record #{idx} is missing required field '{field}'.")
            # Normalize evaluator and evaluation_spec
            if "evaluator" not in r and "eval_type" in r:
                r["evaluator"] = r["eval_type"]
            if "evaluation_spec" not in r:
                r["evaluation_spec"] = {
                    "tolerance": r.get("tolerance", 0.0),
                    "case_sensitive": r.get("case_sensitive", False),
                    "strict_raw": r.get("strict_raw", False),
                }

        return data

    def evaluate_probe(self, probe: Dict[str, Any], completion: Optional[str] = None) -> Dict[str, Any]:
        """Evaluates a single probe. If completion is None, runs the model runner."""
        probe_id = probe["id"]
        category = probe["category"]
        prompt = probe["prompt"]
        expected = probe.get("expected")
        evaluator_name = probe.get("evaluator") or probe.get("eval_type", "exact_match")
        spec = probe.get("evaluation_spec", {})
        reference_answer = probe.get("reference_answer")

        # Telemetry metrics
        latency_ms = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        tps = 0.0
        done_reason = "cached" if completion is not None else "model_response"
        actual_response = completion

        if actual_response is None:
            try:
                gen_output = self.runner.generate(prompt)
                actual_response = gen_output.raw_text
                latency_ms = gen_output.latency_ms
                prompt_tokens = gen_output.prompt_tokens
                completion_tokens = gen_output.completion_tokens
                tps = gen_output.tokens_per_second
                done_reason = gen_output.metadata.get("done_reason", "completed")
            except Exception as run_err:
                return {
                    "probe_id": probe_id,
                    "category": category,
                    "difficulty": probe.get("difficulty", "medium"),
                    "prompt": prompt,
                    "expected": expected,
                    "reference_answer": reference_answer,
                    "actual_response": None,
                    "evaluator": evaluator_name,
                    "status": "INCONCLUSIVE",
                    "passed": False,
                    "inconclusive": True,
                    "evaluator_error": False,
                    "execution_error": str(run_err),
                    "score": 0.0,
                    "details": f"Model runner error: {run_err}",
                    "failure_reason": f"Runner failure: {run_err}",
                    "latency_ms": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "tokens_per_second": 0.0,
                    "done_reason": "error",
                }

        # Safe deterministic evaluation
        try:
            scorer = get_scorer(evaluator_name)
            meta = dict(spec)
            eval_res: EvaluationResult = scorer.score(
                completion=actual_response,
                expected=expected,
                metadata=meta,
            )
            passed = eval_res.passed
            status = "PASS" if passed else "FAIL"
            evaluator_error = False
            details = eval_res.details
            failure_reason = "" if passed else eval_res.details
        except Exception as eval_err:
            status = "EVALUATOR_ERROR"
            passed = False
            evaluator_error = True
            details = f"Evaluator internal exception: {eval_err}"
            failure_reason = details

        return {
            "probe_id": probe_id,
            "category": category,
            "difficulty": probe.get("difficulty", "medium"),
            "prompt": prompt,
            "expected": expected,
            "reference_answer": reference_answer,
            "actual_response": actual_response,
            "evaluator": evaluator_name,
            "status": status,
            "passed": passed,
            "inconclusive": status == "INCONCLUSIVE",
            "evaluator_error": evaluator_error,
            "score": 1.0 if passed else 0.0,
            "details": details,
            "failure_reason": failure_reason,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_second": tps,
            "done_reason": done_reason,
        }

    def run_suite(
        self,
        dataset_path: str,
        category: Optional[str] = None,
        smoke_count: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Runs evaluation suite with category filtering or smoke test slicing."""
        data = self.load_and_validate_dataset(dataset_path)
        dataset_name = data.get("dataset_name", "Behavioural Evaluation Canary Suite")
        dataset_version = data.get("version", "v1.0")
        all_records = data.get("records", [])

        # Filter by category if requested
        if category and category.lower() != "all":
            selected_records = [r for r in all_records if r.get("category", "").lower() == category.lower()]
        else:
            selected_records = all_records

        # Slice smoke test if requested
        if smoke_count and smoke_count > 0:
            selected_records = selected_records[:smoke_count]

        model_info = self.runner.get_model_info()
        model_name = getattr(self.runner, "model_name", None) or model_info.get("model_name", "unknown")
        quant_level = model_info.get("quantization_level", "unknown")

        start_time = datetime.now(timezone.utc)
        probe_results: List[Dict[str, Any]] = []
        category_breakdown: Dict[str, Dict[str, Any]] = {}

        if verbose:
            print(f"Executing suite: {dataset_name} ({len(selected_records)} probes) on {model_name}...")

        total_latency_ms = 0.0
        passed_probes = 0
        failed_probes = 0
        inconclusive_probes = 0
        evaluator_errors = 0

        for idx, probe in enumerate(selected_records, 1):
            res = self.evaluate_probe(probe)
            probe_results.append(res)
            cat = res["category"]

            if cat not in category_breakdown:
                category_breakdown[cat] = {
                    "category": cat,
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "inconclusive": 0,
                    "pass_rate": 0.0,
                    "mean_latency_ms": 0.0,
                    "total_latency_ms": 0.0,
                }

            cat_stats = category_breakdown[cat]
            cat_stats["total"] += 1
            total_latency_ms += res["latency_ms"]
            cat_stats["total_latency_ms"] += res["latency_ms"]

            if res["passed"]:
                passed_probes += 1
                cat_stats["passed"] += 1
            elif res["inconclusive"]:
                inconclusive_probes += 1
                cat_stats["inconclusive"] += 1
            elif res["evaluator_error"]:
                evaluator_errors += 1
            else:
                failed_probes += 1
                cat_stats["failed"] += 1

        # Finalize category breakdown
        for cat, stats in category_breakdown.items():
            if stats["total"] > 0:
                stats["pass_rate"] = stats["passed"] / stats["total"]
                stats["mean_latency_ms"] = stats["total_latency_ms"] / stats["total"]

        total_evaluated = len(selected_records)
        overall_pass_rate = passed_probes / total_evaluated if total_evaluated > 0 else 0.0
        mean_latency = total_latency_ms / total_evaluated if total_evaluated > 0 else 0.0

        run_id = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{model_name.replace(':', '_')}"
        suite_result = {
            "evaluation_id": run_id,
            "timestamp": start_time.isoformat(),
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "dataset_path": dataset_path,
            "model_name": model_name,
            "quantization_level": quant_level,
            "total_probes": total_evaluated,
            "passed_probes": passed_probes,
            "failed_probes": failed_probes,
            "inconclusive_probes": inconclusive_probes,
            "evaluator_errors": evaluator_errors,
            "overall_pass_rate": overall_pass_rate,
            "mean_latency_ms": mean_latency,
            "category_breakdown": category_breakdown,
            "probes": probe_results,
        }

        # Persist results
        out_filename = f"{run_id}.json"
        out_path = os.path.join(self.results_dir, out_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(suite_result, f, indent=2, ensure_ascii=False)

        return suite_result

    @staticmethod
    def compare_runs(baseline_run: Dict[str, Any], candidate_run: Dict[str, Any]) -> Dict[str, Any]:
        """Computes head-to-head comparison between baseline and candidate suite runs."""
        base_probes = {p["probe_id"]: p for p in baseline_run.get("probes", [])}
        cand_probes = {p["probe_id"]: p for p in candidate_run.get("probes", [])}

        common_ids = sorted(list(set(base_probes.keys()).intersection(cand_probes.keys())))
        regressions = []
        improvements = []
        consistent_passes = []
        consistent_failures = []

        for pid in common_ids:
            b = base_probes[pid]
            c = cand_probes[pid]
            if b["passed"] and not c["passed"]:
                regressions.append({
                    "probe_id": pid,
                    "category": c["category"],
                    "difficulty": c.get("difficulty", "medium"),
                    "prompt": c["prompt"],
                    "expected": c["expected"],
                    "baseline_response": b["actual_response"],
                    "candidate_response": c["actual_response"],
                    "failure_reason": c["failure_reason"],
                    "details": c["details"],
                })
            elif not b["passed"] and c["passed"]:
                improvements.append({
                    "probe_id": pid,
                    "category": c["category"],
                    "difficulty": c.get("difficulty", "medium"),
                })
            elif b["passed"] and c["passed"]:
                consistent_passes.append(pid)
            else:
                consistent_failures.append(pid)

        # Category level comparison
        categories = sorted(list(set(
            list(baseline_run.get("category_breakdown", {}).keys()) +
            list(candidate_run.get("category_breakdown", {}).keys())
        )))
        cat_comparison = []
        for cat in categories:
            b_cat = baseline_run.get("category_breakdown", {}).get(cat, {})
            c_cat = candidate_run.get("category_breakdown", {}).get(cat, {})
            b_rate = b_cat.get("pass_rate", 0.0)
            c_rate = c_cat.get("pass_rate", 0.0)
            diff = c_rate - b_rate
            cat_comparison.append({
                "category": cat,
                "total": c_cat.get("total", b_cat.get("total", 0)),
                "baseline_pass_rate": b_rate,
                "candidate_pass_rate": c_rate,
                "diff_percentage_points": round(diff * 100, 2),
                "regressions_count": sum(1 for r in regressions if r["category"] == cat),
                "status": "REGRESSED" if diff < -0.05 else ("IMPROVED" if diff > 0.05 else "STABLE"),
            })

        base_rate = baseline_run.get("overall_pass_rate", 0.0)
        cand_rate = candidate_run.get("overall_pass_rate", 0.0)
        diff_pp = round((cand_rate - base_rate) * 100, 2)

        return {
            "comparison_id": f"comp_{baseline_run.get('evaluation_id')}_vs_{candidate_run.get('evaluation_id')}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline_model": baseline_run.get("model_name"),
            "candidate_model": candidate_run.get("model_name"),
            "baseline_pass_rate": base_rate,
            "candidate_pass_rate": cand_rate,
            "difference_percentage_points": diff_pp,
            "total_evaluated": len(common_ids),
            "regressions_count": len(regressions),
            "improvements_count": len(improvements),
            "regressions": regressions,
            "improvements": improvements,
            "category_comparison": cat_comparison,
        }
