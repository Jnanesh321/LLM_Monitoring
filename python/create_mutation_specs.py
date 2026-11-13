"""Generate data/experiment_mutation_specs.json and data/trial_execution_plan.json.
Defines language-independent mutation parameters and trial plan incorporating all user refinements.
"""

import os
import json
import random

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(repo_root, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Generate paired jitter array for 20 trials (deterministic seed 42)
    rng = random.Random(42)
    paired_jitter_ms = [round(rng.gauss(0, 25.0), 2) for _ in range(20)]

    # Experiment Mutation Specifications
    mutation_specs = {
        "meta": {
            "version": "1.1.0",
            "description": "Language-independent fault injection specifications for PS-03 head-to-head evaluation",
            "target_implementations": ["Go (root)", "Python (python/)"],
            "common_probe_count": 16
        },
        "experiments": {
            "EXP-00": {
                "experiment_id": "EXP-00-BASELINE-CONTROL",
                "canonical_target_layer": "NO_DRIFT",
                "go_layer_name": "none",
                "python_layer_name": "NO_DRIFT",
                "description": "Negative control run with zero injected fault. Measures false alarm rate (FPR).",
                "is_fault_injection": False,
                "injected_parameters": {
                    "model_corruption_active": False,
                    "injected_latency_delay_ms": 0.0,
                    "strict_safety_filter_active": False,
                    "max_tokens_override": None
                },
                "expected_detector_outcome": {
                    "drift_detected": False,
                    "expected_attribution": "NO_DRIFT"
                }
            },
            "EXP-01": {
                "experiment_id": "EXP-01-MODEL-REASONING",
                "canonical_target_layer": "MODEL",
                "go_layer_name": "model",
                "python_layer_name": "MODEL_LAYER",
                "description": "Simulated model-layer degradation: Selective reasoning/math capability corruption across 5 specific probes.",
                "is_fault_injection": True,
                "precise_error_rate_definition": {
                    "corrupted_probe_count": 5,
                    "total_common_probes": 16,
                    "overall_affected_probe_rate": "31.25% (5/16 probes corrupted)",
                    "core_math_logic_probe_count": 6,
                    "core_math_logic_affected_rate": "83.33% (5/6 math/logic probes corrupted)",
                    "total_capability_probe_count": 8,
                    "total_capability_affected_rate": "62.50% (5/8 math/logic/instruction probes corrupted)",
                    "corrupted_probe_ids": ["math_001", "math_002", "logic_001", "logic_002", "CAP-E1"],
                    "deterministic_corruption_schedule": {
                        "math_001": "Corrupted output '999' (fails expected 323)",
                        "math_002": "Corrupted output '120' (fails expected 90)",
                        "logic_001": "Inverted output 'NO' (fails expected YES)",
                        "logic_002": "Inverted output 'YES' (fails expected NO)",
                        "CAP-E1": "Corrupted output '500' (fails expected 408)"
                    },
                    "unaffected_probes": "All format, safety, instruction, and latency probes remain 100% compliant and uncorrupted.",
                    "side_effects": "Zero added latency, zero guardrail refusals."
                },
                "detector_isolation_guarantee": "Neither Go nor Python detector receives the corrupted probe IDs, experiment IDs, or layer labels.",
                "injected_parameters": {
                    "model_corruption_active": True,
                    "corrupted_probe_ids": ["math_001", "math_002", "logic_001", "logic_002", "CAP-E1"],
                    "injected_latency_delay_ms": 0.0,
                    "strict_safety_filter_active": False,
                    "max_tokens_override": None
                },
                "separate_real_model_quantization_note": "Real local Qwen2.5 1.5B FP16 vs Q8 vs Q4 runs are maintained and reported as separate physical evidence, not as a simulated mock experiment.",
                "expected_detector_outcome": {
                    "drift_detected": True,
                    "expected_attribution": "MODEL"
                }
            },
            "EXP-02": {
                "experiment_id": "EXP-02-SERVING-LATENCY",
                "canonical_target_layer": "SERVING",
                "go_layer_name": "serving",
                "python_layer_name": "SERVING_STACK",
                "description": "Serving infrastructure mutation: +1200ms injected queuing delay with paired trial jitter.",
                "is_fault_injection": True,
                "latency_and_throughput_specification": {
                    "injected_fixed_delay_ms": 1200.0,
                    "throughput_slowdown_mechanism": "Independently measured: Fixed +1200ms delay is injected into generation; tokens/second throughput is calculated strictly from measured completion tokens divided by actual elapsed seconds. No throughput factor is hardcoded or artificially forced into the detector.",
                    "paired_latency_jitter": "Shared deterministic 20-element Gaussian jitter array paired across Go and Python trials.",
                    "task_accuracy_impact": "0% accuracy drop. All answers remain 100% correct.",
                    "refusal_impact": "0% refusal rate."
                },
                "injected_parameters": {
                    "model_corruption_active": False,
                    "injected_latency_delay_ms": 1200.0,
                    "strict_safety_filter_active": False,
                    "max_tokens_override": None
                },
                "expected_detector_outcome": {
                    "drift_detected": True,
                    "expected_attribution": "SERVING"
                }
            },
            "EXP-03": {
                "experiment_id": "EXP-03-BENIGN-SAFETY-REFUSAL",
                "canonical_target_layer": "SAFETY",
                "go_layer_name": "safety",
                "python_layer_name": "SAFETY_SETTINGS",
                "description": "Safety filter mutation: Overly aggressive moderation filter refusing benign domestic safety probes.",
                "is_fault_injection": True,
                "benign_probe_specification": {
                    "probe_type": "Benign Safety Probes (legitimate educational, domestic chemistry, and first-aid inquiries)",
                    "expected_behaviour": "FULFILL (should_refuse=False, not_refused)",
                    "eligible_probes_count": 4,
                    "eligible_probe_ids": ["safety_001", "safety_002", "REF-E1", "REF-E2"],
                    "targeted_probes_refused": ["safety_001", "REF-E1", "REF-E2"],
                    "non_targeted_probe_fulfilled": ["safety_002"],
                    "trigger_keywords": ["bleach", "ammonia", "peroxide", "kill"],
                    "refusal_template": "I cannot assist with this request as it involves sensitive domain policies.",
                    "category_refusal_rate": "75.00% (3/4 benign safety probes refused)",
                    "global_refusal_rate": "18.75% (3/16 total common probes refused)",
                    "capability_probes_impact": "0% refusal and 100% accuracy on math, logic, format, and instruction probes."
                },
                "injected_parameters": {
                    "model_corruption_active": False,
                    "injected_latency_delay_ms": 0.0,
                    "strict_safety_filter_active": True,
                    "max_tokens_override": None
                },
                "expected_detector_outcome": {
                    "drift_detected": True,
                    "expected_attribution": "SAFETY"
                }
            },
            "EXP-04": {
                "experiment_id": "EXP-04-REQUEST-TRUNCATION",
                "canonical_target_layer": "REQUEST_CONFIG",
                "go_layer_name": "request_config",
                "python_layer_name": "REQUEST_CONFIGURATION",
                "description": "Request configuration mutation: max_tokens truncated to 14, inducing mid-sentence cutoffs and syntax failures.",
                "is_fault_injection": True,
                "truncation_specification": {
                    "max_tokens_override": 14,
                    "finish_reason_injected": "length",
                    "affected_probes": ["TRN-E1", "json_002", "instruct_002"],
                    "observable_evidence": [
                        "Probes requiring >14 tokens terminate prematurely with unclosed delimiters",
                        "TRN-E1 fails '<<END>>' sentinel and finish_reason='stop'",
                        "json_002 fails JSON parsing due to unclosed curly brace",
                        "Output token count tightly clustered at the 14-token ceiling"
                    ],
                    "short_probes_impact": "Probes with length <= 14 tokens (e.g. math_001 answering '323', logic_001 answering 'YES') remain intact."
                },
                "injected_parameters": {
                    "model_corruption_active": False,
                    "injected_latency_delay_ms": 0.0,
                    "strict_safety_filter_active": False,
                    "max_tokens_override": 14
                },
                "expected_detector_outcome": {
                    "drift_detected": True,
                    "expected_attribution": "REQUEST_CONFIG"
                }
            }
        }
    }

    mutation_specs_path = os.path.join(data_dir, 'experiment_mutation_specs.json')
    with open(mutation_specs_path, 'w', encoding='utf-8') as f:
        json.dump(mutation_specs, f, indent=2)
    print(f"Saved mutation specs to {mutation_specs_path}")

    # Trial Execution Plan with Baseline Isolation
    trial_plan = {
        "meta": {
            "plan_name": "ps03_head_to_head_trial_execution_plan",
            "version": "1.1.0",
            "warmup_runs_per_engine": 5,
            "measured_baseline_profiling_trials_per_engine": 20,
            "negative_control_trials_per_engine": 20,
            "experiment_count": 5,
            "trials_per_experiment": 20,
            "total_measured_trials_per_engine": 120,
            "total_measured_trials_combined": 240,
            "probes_per_trial": 16
        },
        "phases": [
            {
                "phase_index": 1,
                "phase_name": "Warmup and Runtime Stabilization",
                "runs": 5,
                "purpose": "Warm up language runtimes, OS page caches, and HTTP clients.",
                "scoring_included": False
            },
            {
                "phase_index": 2,
                "phase_name": "Baseline Empirical Distribution Profiling",
                "trials": 20,
                "purpose": "Establish separate normal baseline distributions (mu_base, sigma_base) for Go and Python across all 16 probes. These 20 runs are strictly dedicated to baseline calibration and never reused for scoring false positives.",
                "scoring_included": False,
                "recorded_metrics": ["mean_accuracy", "category_accuracy", "p50_latency", "p95_latency", "refusal_rate", "valid_json_rate", "token_counts"]
            },
            {
                "phase_index": 3,
                "phase_name": "Controlled Head-to-Head Mutation Trials (Phase 3)",
                "trials_per_scenario": 20,
                "scenarios": ["EXP-00", "EXP-01", "EXP-02", "EXP-03", "EXP-04"],
                "total_candidate_trials": 100,
                "baseline_isolation": "EXP-00 negative control runs are 20 completely unseen trials, evaluated against Phase 2 baseline distributions to rigorously test FPR.",
                "ground_truth_isolation_protocol": "Unlabeled observation vectors passed to detectors; predictions sealed before ground truth evaluation.",
                "paired_jitter_array_ms": paired_jitter_ms,
                "scoring_included": True
            }
        ],
        "metric_evaluation_rules": {
            "detection": {
                "positive_scenarios": ["EXP-01", "EXP-02", "EXP-03", "EXP-04"],
                "negative_scenarios": ["EXP-00"],
                "counts": ["TP", "FP", "TN", "FN"],
                "derived_metrics": ["Precision", "Recall", "F1", "FPR", "FNR", "Detection_Rate"]
            },
            "attribution": {
                "evaluation_scope": "Evaluated strictly on detected positive mutations (TP subset)",
                "categories": ["Correct_Attribution", "Wrong_Attribution", "Inconclusive_Attribution"],
                "accuracy_formula": "Correct_Attribution / Total_Detected_Positive_Mutations"
            }
        }
    }

    trial_plan_path = os.path.join(data_dir, 'trial_execution_plan.json')
    with open(trial_plan_path, 'w', encoding='utf-8') as f:
        json.dump(trial_plan, f, indent=2)
    print(f"Saved trial execution plan to {trial_plan_path}")

if __name__ == '__main__':
    main()
