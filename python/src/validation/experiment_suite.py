"""Controlled experiment validation harness for all 4 PS-03 operational layers."""

import os
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.runner.ollama_runner import OllamaRunner
from src.proxy.mock_provider import MockAIProvider, ExperimentDefinition, LayerMutationConfig
from src.benchmark import BenchmarkEngine
from src.detector.drift_detector import DriftDetector, DriftReport
from src.attribution.engine import AttributionEngine, AttributionReport


class ControlledExperimentSuite:
    """Executes controlled fault injections across all 4 layers and validates detection & attribution accuracy."""

    def __init__(
        self,
        base_model: str = "qwen2.5:1.5b-instruct-q8_0",
        quant_model: str = "qwen2.5:1.5b",
        dataset_path: str = "evaluation/benchmarks/canary_suite.json",
        base_url: str = "http://localhost:11434",
        results_dir: str = "results",
    ):
        self.base_model = base_model
        self.quant_model = quant_model

        # Fallback to python/ relative path if invoked from root
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if not os.path.isabs(dataset_path) and not os.path.exists(dataset_path):
            candidate = os.path.join(pkg_root, dataset_path)
            if os.path.exists(candidate):
                dataset_path = candidate
        self.dataset_path = dataset_path

        if not os.path.isabs(results_dir) and not os.path.exists(results_dir):
            results_dir = os.path.join(pkg_root, results_dir)
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.detector = DriftDetector()
        self.attributor = AttributionEngine()

    def run_suite(self, verbose: bool = True, experiment_filter: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Execute baseline and 4 controlled experiments sequentially."""
        start_suite_time = time.perf_counter()

        # Step 1: Establish Baseline Run
        if verbose:
            print(f"\n{'='*105}")
            print(f"ESTABLISHING BASELINE PROFILE: {self.base_model}")
            print(f"{'='*105}\n")

        base_runner = OllamaRunner(model_name=self.base_model, base_url=self.base_url)
        alt_runner = OllamaRunner(model_name=self.quant_model, base_url=self.base_url)

        # Baseline execution
        base_engine = BenchmarkEngine(runner=base_runner, results_dir=self.results_dir)
        baseline_report = base_engine.run(dataset_path=self.dataset_path, verbose=False)
        base_runner.unload()

        # Build list of 5 experiment scenarios
        experiments: List[ExperimentDefinition] = [
            ExperimentDefinition.create_baseline_control(),
            ExperimentDefinition.create_model_experiment(reference_model=self.base_model, quantized_model=self.quant_model),
            ExperimentDefinition.create_serving_experiment(added_latency_ms=1200.0),
            ExperimentDefinition.create_safety_experiment(),
            ExperimentDefinition.create_request_experiment(max_tokens=14),
        ]

        if experiment_filter:
            filter_lower = [f.lower() for f in experiment_filter]
            if 'all' not in filter_lower:
                experiments = [
                    e for e in experiments
                    if e.experiment_id.lower() in filter_lower or e.target_layer.lower() in filter_lower
                ]

        expected_layer_mapping = {
            "baseline": "NO_DRIFT",
            "model": "MODEL_LAYER",
            "serving": "SERVING_STACK",
            "safety": "SAFETY_SETTINGS",
            "request": "REQUEST_CONFIGURATION",
        }

        experiment_results: List[Dict[str, Any]] = []

        tp = 0  # Injected drift correctly detected as drift
        fp = 0  # Null run falsely detected as drift
        tn = 0  # Null run confirmed as NO_DRIFT
        fn = 0  # Injected drift missed
        correctly_attributed = 0

        # Step 2: Sequentially execute each experiment
        for exp in experiments:
            exp_start_time = time.perf_counter()
            if verbose:
                print(f"\n{'-'*105}")
                print(f"RUNNING EXPERIMENT: [{exp.experiment_id}] -> Layer: {exp.target_layer.upper()}")
                print(f"Description: {exp.description}")
                print(f"{'-'*105}")

            # Instantiate MockAIProvider configured for this experiment
            proxy = MockAIProvider(
                base_runner=base_runner,
                experiment_id=exp.experiment_id,
                mutations=exp.modified_config,
                alternate_runners={self.quant_model: alt_runner},
            )

            # Execute canary benchmark
            exp_engine = BenchmarkEngine(runner=proxy, results_dir=self.results_dir)
            candidate_report = exp_engine.run(dataset_path=self.dataset_path, verbose=False)

            # Unload model from VRAM between runs to maintain hardware isolation
            base_runner.unload()
            alt_runner.unload()

            # Run Drift Detection
            drift: DriftReport = self.detector.detect(baseline_report, candidate_report)

            # Run Root-Cause Attribution
            attr: AttributionReport = self.attributor.attribute(drift, candidate_report.get("model_info"))

            exp_duration_sec = time.perf_counter() - exp_start_time
            expected_layer = expected_layer_mapping.get(exp.target_layer, "UNKNOWN")

            # Classification counts
            is_fault_injection = exp.target_layer != "baseline"
            if is_fault_injection:
                if drift.drift_detected:
                    tp += 1
                    if attr.likely_cause == expected_layer:
                        correctly_attributed += 1
                        attr_match = "MATCH (CORRECT)"
                    else:
                        attr_match = f"MISMATCH (Got: {attr.likely_cause})"
                else:
                    fn += 1
                    attr_match = "MISSED (NO_DRIFT)"
            else:
                # Null control run
                if not drift.drift_detected:
                    tn += 1
                    attr_match = "MATCH (CORRECT NULL)"
                else:
                    fp += 1
                    attr_match = f"FALSE POSITIVE (Got: {attr.likely_cause})"

            record = {
                "experiment_id": exp.experiment_id,
                "target_layer": exp.target_layer,
                "expected_layer": expected_layer,
                "attributed_layer": attr.likely_cause,
                "attribution_correct": (attr.likely_cause == expected_layer),
                "confidence": attr.confidence,
                "drift_detected": drift.drift_detected,
                "severity": drift.severity,
                "accuracy_delta": drift.overall_accuracy_delta,
                "latency_change_pct": drift.latency_change_percent,
                "refusal_delta": drift.refusal_rate_delta,
                "truncation_detected": drift.truncation_detected,
                "detection_latency_sec": round(exp_duration_sec, 2),
                "evidence": attr.observed_evidence,
                "inferred_diagnosis": attr.inferred_diagnosis,
            }
            experiment_results.append(record)

            if verbose:
                print(f"  Result: Drift Detected: {drift.drift_detected} (Severity: {drift.severity})")
                print(f"  Attribution: {attr.likely_cause} (Expected: {expected_layer}) -> [{attr_match}]")
                print(f"  Confidence:  {attr.confidence*100.0:.1f}%")
                print(f"  Duration:    {exp_duration_sec:.2f}s")
                print(f"  Evidence:    {'; '.join(attr.observed_evidence[:2])}")

        total_injections = tp + fn
        total_null = tn + fp
        total_experiments = len(experiments)

        detection_accuracy = ((tp + tn) / total_experiments * 100.0) if total_experiments > 0 else 0.0
        false_positive_rate = (fp / total_null * 100.0) if total_null > 0 else 0.0
        missed_change_rate = (fn / total_injections * 100.0) if total_injections > 0 else 0.0
        attribution_accuracy = (correctly_attributed / tp * 100.0) if tp > 0 else 0.0
        avg_detection_latency = sum(r["detection_latency_sec"] for r in experiment_results) / total_experiments

        suite_summary = {
            "total_experiments": total_experiments,
            "total_fault_injections": total_injections,
            "null_controls": total_null,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "detection_accuracy_percent": round(detection_accuracy, 2),
            "false_positive_rate_percent": round(false_positive_rate, 2),
            "missed_change_rate_percent": round(missed_change_rate, 2),
            "attribution_accuracy_percent": round(attribution_accuracy, 2),
            "avg_detection_latency_sec": round(avg_detection_latency, 2),
            "total_suite_runtime_sec": round(time.perf_counter() - start_suite_time, 2),
        }

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        validation_report = {
            "title": "PS-03 Controlled Layer Experiment Validation Report",
            "timestamp_utc": timestamp_str,
            "dataset": self.dataset_path,
            "base_model": self.base_model,
            "summary_metrics": suite_summary,
            "experiments": experiment_results,
        }

        # Save to disk
        out_filename = f"validation_report_{timestamp_str}.json"
        out_path = os.path.join(self.results_dir, out_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)

        if verbose:
            self._print_suite_summary(validation_report, out_path)

        return validation_report

    def _print_suite_summary(self, report: Dict[str, Any], output_path: str) -> None:
        """Render readable CLI summary of the validation experiment suite."""
        summary = report["summary_metrics"]
        experiments = report["experiments"]

        print(f"\n{'='*110}")
        print("PS-03 CONTROLLED EXPERIMENT VALIDATION RESULTS (ALL 4 LAYERS)")
        print(f"{'='*110}")
        header = f"{'Experiment ID':<26} | {'Target Layer':<15} | {'Expected':<18} | {'Attributed':<18} | {'Match':<8} | {'Latency':<8}"
        print(header)
        print(f"{'-'*110}")
        for r in experiments:
            eid = r["experiment_id"]
            layer = r["target_layer"]
            exp = r["expected_layer"]
            att = r["attributed_layer"]
            match = "PASS" if r["attribution_correct"] else "FAIL"
            lat = f"{r['detection_latency_sec']:.1f}s"
            print(f"{eid:<26} | {layer:<15} | {exp:<18} | {att:<18} | {match:<8} | {lat:<8}")

        print(f"\n{'='*110}")
        print("SYSTEM VERIFICATION METRICS SUMMARY")
        print(f"{'='*110}")
        print(f"Overall Detection Accuracy:        {summary['detection_accuracy_percent']}% ({summary['true_positives'] + summary['true_negatives']}/{summary['total_experiments']} correct)")
        print(f"False Positive Rate (FPR):         {summary['false_positive_rate_percent']}% ({summary['false_positives']}/{summary['null_controls']} false alarms on baseline)")
        print(f"Missed Change Rate (FNR):          {summary['missed_change_rate_percent']}% ({summary['false_negatives']}/{summary['total_fault_injections']} missed injections)")
        print(f"Root-Cause Attribution Accuracy:   {summary['attribution_accuracy_percent']}% ({summary['true_positives']}/{summary['total_fault_injections']} correctly diagnosed)")
        print(f"Average Detection Latency:         {summary['avg_detection_latency_sec']} seconds / probe cycle")
        print(f"Total Suite Runtime:               {summary['total_suite_runtime_sec']} seconds")
        print(f"{'='*110}")
        print(f"Detailed validation artifact saved to: {output_path}\n")

