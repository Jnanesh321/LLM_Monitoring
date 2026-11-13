"""Root-cause attribution engine mapping observed drift evidence to operational layers (PS-03)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.detector.drift_detector import DriftReport


@dataclass
class AttributionReport:
    """Diagnostic attribution isolating probable root causes across the 4 operational layers."""

    likely_cause: str  # 'MODEL_LAYER' | 'SERVING_STACK' | 'SAFETY_SETTINGS' | 'REQUEST_CONFIGURATION' | 'NO_DRIFT'
    confidence: float  # 0.0 to 1.0
    detected_changes: List[str]
    observed_evidence: List[str]  # Strictly factual measurements
    inferred_diagnosis: str  # Diagnostic deduction
    alternative_explanations: List[Dict[str, Any]]
    action_recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "likely_cause": self.likely_cause,
            "confidence": round(self.confidence, 3),
            "detected_changes": self.detected_changes,
            "observed_evidence": self.observed_evidence,
            "inferred_diagnosis": self.inferred_diagnosis,
            "alternative_explanations": self.alternative_explanations,
            "action_recommendation": self.action_recommendation,
        }


class AttributionEngine:
    """Analyzes statistical drift reports and attributes degradation to the responsible layer."""

    def attribute(self, drift_report: DriftReport, candidate_model_info: Optional[Dict[str, Any]] = None) -> AttributionReport:
        candidate_model_info = candidate_model_info or {}

        # ---------------------------------------------------------
        # Case 0: No Drift Detected
        # ---------------------------------------------------------
        if not drift_report.drift_detected:
            return AttributionReport(
                likely_cause="NO_DRIFT",
                confidence=0.95,
                detected_changes=["No statistically significant behavioural degradation"],
                observed_evidence=[
                    f"Overall accuracy delta: {drift_report.overall_accuracy_delta:+.1f}% (p={drift_report.p_value:.3f})",
                    f"Latency change: {drift_report.latency_change_percent:+.1f}%",
                    f"Refusal rate delta: {drift_report.refusal_rate_delta:+.2f}",
                ],
                inferred_diagnosis="System performance remains fully consistent with the established baseline within normal stochastic variation bounds.",
                alternative_explanations=[],
                action_recommendation="Endpoint healthy; no remediation required.",
            )

        observed_evidence = []
        detected_changes = []

        # Record factual measurements
        if drift_report.overall_accuracy_delta > 0:
            observed_evidence.append(
                f"Overall accuracy degraded by {drift_report.overall_accuracy_delta:.1f}% (p={drift_report.p_value:.3f}, Cohen's h={drift_report.cohens_h:.2f})"
            )
            detected_changes.append(f"Accuracy dropped from baseline by {drift_report.overall_accuracy_delta:.1f}%")

        if drift_report.affected_categories:
            cat_list = ", ".join([f"{c} ({drift_report.category_deltas.get(c, 0):+.1f}%)" for c in drift_report.affected_categories])
            observed_evidence.append(f"Capability regressions localized to: {cat_list}")
            detected_changes.append(f"Capabilities degraded: {', '.join(drift_report.affected_categories)}")

        if abs(drift_report.latency_change_percent) >= 20.0:
            observed_evidence.append(
                f"Generation latency shifted by {drift_report.latency_change_percent:+.1f}% ({drift_report.latency_delta_ms:+.0f}ms)"
            )
            detected_changes.append(f"Latency shifted by {drift_report.latency_change_percent:+.1f}%")

        if drift_report.refusal_rate_delta > 0:
            observed_evidence.append(
                f"Refusal rate on benign safety/domain probes increased by +{drift_report.refusal_rate_delta*100.0:.1f}%"
            )
            detected_changes.append("Benign probe refusal surge")

        if drift_report.truncation_detected:
            trunc_ev = drift_report.details.get("truncation_evidence", [])
            for ev in trunc_ev:
                if ev not in observed_evidence:
                    observed_evidence.append(ev)
            if not trunc_ev:
                observed_evidence.append("Multiple completions truncated prematurely at fixed token boundary with unclosed syntax")
            detected_changes.append("Premature token truncation")

        if drift_report.json_validity_delta >= 20.0:
            observed_evidence.append(f"JSON syntax parse validity dropped by {drift_report.json_validity_delta:.1f}%")

        # ---------------------------------------------------------
        # Layer 3: Safety Settings Attribution Signature
        # ---------------------------------------------------------
        if drift_report.refusal_rate_delta >= 0.40:
            # High refusal surge while core math/logic remain relatively intact
            math_delta = drift_report.category_deltas.get("mathematics", 0.0)
            logic_delta = drift_report.category_deltas.get("logical_reasoning", 0.0)
            confidence = 0.92 if (math_delta <= 15.0 and logic_delta <= 15.0) else 0.82

            return AttributionReport(
                likely_cause="SAFETY_SETTINGS",
                confidence=confidence,
                detected_changes=detected_changes,
                observed_evidence=observed_evidence,
                inferred_diagnosis=(
                    "Diagnostic evidence strongly indicates an upstream safety filter or moderation guardrail update. "
                    "Benign prompts containing domain vocabulary triggered unexpected refusal templates without general reasoning degradation."
                ),
                alternative_explanations=[
                    {
                        "cause": "REQUEST_CONFIGURATION",
                        "confidence": round(1.0 - confidence, 2),
                        "reason": "Client or gateway may have injected an aggressive safety system prompt override.",
                    }
                ],
                action_recommendation="Review endpoint safety moderation thresholds or system prompt guardrails for false-positive refusals on domain queries.",
            )

        # ---------------------------------------------------------
        # Layer 4: Request Configuration Attribution Signature
        # ---------------------------------------------------------
        if drift_report.truncation_detected:
            return AttributionReport(
                likely_cause="REQUEST_CONFIGURATION",
                confidence=0.90,
                detected_changes=detected_changes,
                observed_evidence=observed_evidence,
                inferred_diagnosis=(
                    "Diagnostic evidence indicates a request configuration parameter mismatch. "
                    "Completions were abruptly terminated at a low token limit, breaking JSON syntax and cutting off responses mid-sentence."
                ),
                alternative_explanations=[
                    {
                        "cause": "SERVING_STACK",
                        "confidence": 0.10,
                        "reason": "Inference server hard max_tokens cap configured in engine scheduler.",
                    }
                ],
                action_recommendation="Verify client-side request parameters (specifically `max_tokens` or `max_new_tokens`) to ensure sufficient generation budget.",
            )

        # ---------------------------------------------------------
        # Layer 2: Serving Stack Attribution Signature
        # ---------------------------------------------------------
        if drift_report.latency_change_percent >= 25.0 and drift_report.overall_accuracy_delta <= 5.0 and drift_report.refusal_rate_delta <= 0.0:
            return AttributionReport(
                likely_cause="SERVING_STACK",
                confidence=0.88,
                detected_changes=detected_changes,
                observed_evidence=observed_evidence,
                inferred_diagnosis=(
                    "Diagnostic evidence isolates this drift to the serving infrastructure layer. "
                    "Generation latency and TTFT experienced significant regression while task accuracy across all capabilities remained invariant."
                ),
                alternative_explanations=[
                    {
                        "cause": "HARDWARE_OR_NETWORK_CONGESTION",
                        "confidence": 0.12,
                        "reason": "Host machine CPU/GPU thermal throttling or transient network proxy queuing delays.",
                    }
                ],
                action_recommendation="Inspect inference engine batching configurations, GPU concurrency settings, and network proxy hop latencies.",
            )

        # ---------------------------------------------------------
        # Layer 1: Model Layer Attribution Signature
        # ---------------------------------------------------------
        if "mathematics" in drift_report.affected_categories or "logical_reasoning" in drift_report.affected_categories or drift_report.overall_accuracy_delta >= 15.0:
            return AttributionReport(
                likely_cause="MODEL_LAYER",
                confidence=0.85,
                detected_changes=detected_changes,
                observed_evidence=observed_evidence,
                inferred_diagnosis=(
                    "Diagnostic evidence indicates an underlying model or quantization checkpoint change. "
                    "Performance regressions are concentrated in precision-sensitive reasoning (mathematics and logic) without serving latency degradation."
                ),
                alternative_explanations=[
                    {
                        "cause": "REQUEST_CONFIGURATION",
                        "confidence": 0.15,
                        "reason": "Subtle changes in generation sampling temperature or system instructions.",
                    }
                ],
                action_recommendation="Audit model checkpoint hashes and quantization formats (e.g. verify if serving moved from Q8/FP16 to lower-bit quant like Q4).",
            )

        # Fallback multi-factor diagnosis
        return AttributionReport(
            likely_cause="UNCORRELATED_DRIFT",
            confidence=0.60,
            detected_changes=detected_changes,
            observed_evidence=observed_evidence,
            inferred_diagnosis="Multiple weak signals detected across latency and accuracy; unable to attribute with high confidence to a single layer.",
            alternative_explanations=[
                {"cause": "MODEL_LAYER", "confidence": 0.35, "reason": "Minor weight/quantization variance."},
                {"cause": "SERVING_STACK", "confidence": 0.35, "reason": "Minor serving latency variance."},
                {"cause": "REQUEST_CONFIGURATION", "confidence": 0.30, "reason": "Slight sampling temperature fluctuation."},
            ],
            action_recommendation="Execute extended 50-task canary benchmark to accumulate stronger statistical signal.",
        )

