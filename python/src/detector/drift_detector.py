"""Statistical drift detection comparing baseline and candidate evaluation runs (PS-03)."""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DriftReport:
    """Diagnostic report output by DriftDetector."""

    drift_detected: bool
    severity: str  # 'NONE' | 'MINOR' | 'MODERATE' | 'CRITICAL'
    p_value: float
    z_score: float
    cohens_h: float
    overall_accuracy_delta: float  # baseline_acc - candidate_acc
    affected_categories: List[str]
    category_deltas: Dict[str, float]
    latency_delta_ms: float
    latency_change_percent: float
    throughput_change_percent: float
    refusal_rate_delta: float
    truncation_detected: bool
    json_validity_delta: float
    json_compliance_delta: float
    summary_text: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "severity": self.severity,
            "p_value": round(self.p_value, 4),
            "z_score": round(self.z_score, 3),
            "cohens_h": round(self.cohens_h, 3),
            "overall_accuracy_delta": round(self.overall_accuracy_delta, 2),
            "affected_categories": self.affected_categories,
            "category_deltas": {k: round(v, 2) for k, v in self.category_deltas.items()},
            "latency_delta_ms": round(self.latency_delta_ms, 2),
            "latency_change_percent": round(self.latency_change_percent, 2),
            "throughput_change_percent": round(self.throughput_change_percent, 2),
            "refusal_rate_delta": round(self.refusal_rate_delta, 2),
            "truncation_detected": self.truncation_detected,
            "json_validity_delta": round(self.json_validity_delta, 2),
            "json_compliance_delta": round(self.json_compliance_delta, 2),
            "summary_text": self.summary_text,
            "details": self.details,
        }


class DriftDetector:
    """Compares baseline vs candidate benchmark runs and identifies statistically significant drift."""

    def __init__(
        self,
        alpha: float = 0.05,
        min_cohens_h: float = 0.20,
        latency_threshold_percent: float = 25.0,
        refusal_threshold_delta: float = 0.25,
    ):
        self.alpha = alpha
        self.min_cohens_h = min_cohens_h
        self.latency_threshold_percent = latency_threshold_percent
        self.refusal_threshold_delta = refusal_threshold_delta

    @staticmethod
    def _normal_cdf(z: float) -> float:
        """Standard normal cumulative distribution function using math.erf."""
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    @classmethod
    def two_proportion_z_test(cls, k1: int, n1: int, k2: int, n2: int):
        """Computes one-tailed z-score, p-value, and Cohen's h for p1 > p2 degradation."""
        if n1 <= 0 or n2 <= 0:
            return 0.0, 1.0, 0.0

        p1 = k1 / n1
        p2 = k2 / n2

        # Cohen's h effect size
        # Ensure values stay strictly in [0, 1] for arcsin
        val1 = max(0.0, min(1.0, p1))
        val2 = max(0.0, min(1.0, p2))
        cohens_h = 2.0 * math.asin(math.sqrt(val1)) - 2.0 * math.asin(math.sqrt(val2))

        # Pooled proportion
        p_pool = (k1 + k2) / (n1 + n2)
        if p_pool <= 0.0 or p_pool >= 1.0:
            return 0.0, 1.0, cohens_h

        se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
        if se <= 0.0:
            return 0.0, 1.0, cohens_h

        z = (p1 - p2) / se
        p_val = 1.0 - cls._normal_cdf(z)
        return z, p_val, cohens_h

    @classmethod
    def analyze_truncation_evidence(
        cls, baseline_results: List[Dict[str, Any]], candidate_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Examines observable token distributions, syntax cutoffs, and wire metadata to detect token budget truncation."""
        cand_tokens = [r.get("completion_tokens", 0) for r in candidate_results if r.get("completion_tokens", 0) > 0]
        if not cand_tokens:
            return {"truncation_detected": False, "evidence": []}

        evidence = []
        truncation_detected = False

        # 1. Wire metadata check: done_reason == "length"
        length_done_tasks = [
            r.get("id", "unknown") for r in candidate_results
            if r.get("done_reason") == "length" or r.get("metadata", {}).get("done_reason") == "length"
        ]
        if length_done_tasks:
            evidence.append(
                f"Provider telemetry reported done_reason='length' on {len(length_done_tasks)} tasks "
                f"({', '.join(length_done_tasks[:3])})"
            )
            truncation_detected = True

        # 2. Token ceiling clustering analysis
        token_freq: Dict[int, int] = {}
        for t in cand_tokens:
            token_freq[t] = token_freq.get(t, 0) + 1

        base_map = {r.get("id"): r for r in baseline_results if r.get("id")}
        max_cap = max(cand_tokens)

        # Check clustered token ceilings
        for cap, count in sorted(token_freq.items(), key=lambda x: x[0], reverse=True):
            # A truncation ceiling must be at least 10 tokens and constrain multiple tasks
            if count >= 2 and 10 <= cap <= 128:
                tasks_at_cap = [r for r in candidate_results if r.get("completion_tokens") == cap]

                # Baseline token length for these exact tasks
                base_tokens = [
                    base_map[t["id"]].get("completion_tokens", 0)
                    for t in tasks_at_cap if t.get("id") in base_map
                ]
                avg_base = (sum(base_tokens) / len(base_tokens)) if base_tokens else 0.0

                # Detect observable syntax/sentence truncation
                incomplete_endings = 0
                for t in tasks_at_cap:
                    resp = (t.get("actual_response") or "").strip()
                    # Structured JSON syntax failure at cap
                    if t.get("category") == "structured_json" and not t.get("passed"):
                        incomplete_endings += 1
                    # Prose text terminating without sentence punctuation
                    elif resp and resp[-1] not in ".!?}\"'\n`":
                        incomplete_endings += 1

                # True truncation: baseline was substantially longer (avg_base >= cap * 1.35)
                # AND completions exhibit incomplete syntax / cutoffs
                if (avg_base >= cap * 1.35 and incomplete_endings >= 1) or (
                    cap == max_cap and incomplete_endings >= 2 and avg_base >= cap * 1.2
                ):
                    truncation_detected = True
                    evidence.append(
                        f"Observable token ceiling clustering at {cap} tokens across {count} tasks "
                        f"(baseline required avg {avg_base:.1f} tokens; {incomplete_endings} completions cut off mid-syntax)"
                    )
                    break

        return {
            "truncation_detected": truncation_detected,
            "evidence": evidence,
        }

    def detect(self, baseline_report: Dict[str, Any], candidate_report: Dict[str, Any]) -> DriftReport:
        """Analyze baseline and candidate benchmark reports to detect behavioural drift."""
        base_sum = baseline_report["summary"]
        cand_sum = candidate_report["summary"]

        base_res = baseline_report.get("results", [])
        cand_res = candidate_report.get("results", [])

        # 1. Overall Accuracy Test
        k1 = base_sum.get("passed_tasks", 0)
        n1 = base_sum.get("total_tasks", len(base_res))
        k2 = cand_sum.get("passed_tasks", 0)
        n2 = cand_sum.get("total_tasks", len(cand_res))

        base_acc = (k1 / n1 * 100.0) if n1 > 0 else 0.0
        cand_acc = (k2 / n2 * 100.0) if n2 > 0 else 0.0
        acc_delta = base_acc - cand_acc

        z_score, p_val, cohens_h = self.two_proportion_z_test(k1, n1, k2, n2)

        # 2. Category Breakdown & Affected Capabilities
        base_cats = base_sum.get("category_breakdown", {})
        cand_cats = cand_sum.get("category_breakdown", {})

        affected_categories = []
        category_deltas = {}

        all_cat_names = set(base_cats.keys()).union(set(cand_cats.keys()))
        for cat in all_cat_names:
            b_info = base_cats.get(cat, {})
            c_info = cand_cats.get(cat, {})
            b_acc = b_info.get("accuracy_percent", 0.0)
            c_acc = c_info.get("accuracy_percent", 0.0)
            diff = b_acc - c_acc
            category_deltas[cat] = diff

            # If accuracy dropped meaningfully in this category
            cat_k1 = b_info.get("passed", 0)
            cat_n1 = b_info.get("total", 0)
            cat_k2 = c_info.get("passed", 0)
            cat_n2 = c_info.get("total", 0)

            _, cat_p, cat_h = self.two_proportion_z_test(cat_k1, cat_n1, cat_k2, cat_n2)
            if diff >= 25.0 or (diff > 0 and (cat_p < 0.15 or cat_h >= 0.30)):
                affected_categories.append(cat)

        # 3. JSON Validity & Format Compliance
        b_json = base_cats.get("structured_json", {})
        c_json = cand_cats.get("structured_json", {})
        json_validity_delta = b_json.get("json_validity_percent", 0.0) - c_json.get("json_validity_percent", 0.0)
        json_compliance_delta = b_json.get("format_compliance_percent", 0.0) - c_json.get("format_compliance_percent", 0.0)

        # 4. Refusal Rate Delta (Safety Probes)
        def calc_refusal_rate(results_list):
            safety_tasks = [r for r in results_list if r.get("category") == "safety"]
            if not safety_tasks:
                return 0.0
            refusals = sum(1 for r in safety_tasks if r.get("metrics", {}).get("is_refusal", False))
            return (refusals / len(safety_tasks))

        base_refusal_rate = calc_refusal_rate(base_res)
        cand_refusal_rate = calc_refusal_rate(cand_res)
        refusal_delta = cand_refusal_rate - base_refusal_rate

        # 5. Latency & Throughput Deltas
        base_lat = base_sum.get("avg_latency_ms", 1.0)
        cand_lat = cand_sum.get("avg_latency_ms", 1.0)
        lat_delta_ms = cand_lat - base_lat
        lat_change_pct = ((cand_lat - base_lat) / base_lat * 100.0) if base_lat > 0 else 0.0

        # Throughput
        def avg_tps(results_list):
            tps_vals = [r.get("tokens_per_second", 0.0) for r in results_list if r.get("tokens_per_second", 0.0) > 0]
            return sum(tps_vals) / len(tps_vals) if tps_vals else 0.0

        base_tps = avg_tps(base_res)
        cand_tps = avg_tps(cand_res)
        tps_change_pct = ((cand_tps - base_tps) / base_tps * 100.0) if base_tps > 0 else 0.0

        # 6. Truncation Detection
        trunc_diag = self.analyze_truncation_evidence(base_res, cand_res)
        truncation_detected = trunc_diag["truncation_detected"]
        truncation_evidence = trunc_diag["evidence"]

        # 7. Evaluate Multi-Factor Drift Signals
        accuracy_drift = (p_val < self.alpha and cohens_h >= self.min_cohens_h) or (acc_delta >= 15.0)
        latency_drift = lat_change_pct >= self.latency_threshold_percent
        safety_drift = refusal_delta >= self.refusal_threshold_delta
        category_drift = len(affected_categories) > 0

        drift_detected = (
            accuracy_drift
            or latency_drift
            or safety_drift
            or truncation_detected
            or (json_validity_delta >= 40.0)
            or (category_drift and acc_delta >= 8.0)
        )

        # Classify Severity
        if not drift_detected:
            severity = "NONE"
        elif acc_delta >= 30.0 or refusal_delta >= 0.50 or truncation_detected:
            severity = "CRITICAL"
        elif acc_delta >= 15.0 or lat_change_pct >= 50.0 or cohens_h >= 0.50:
            severity = "MODERATE"
        else:
            severity = "MINOR"

        # Summary text
        reasons = []
        if accuracy_drift or acc_delta > 0:
            reasons.append(f"Accuracy dropped by {acc_delta:.1f}% (p={p_val:.3f}, h={cohens_h:.2f})")
        if affected_categories:
            reasons.append(f"Degraded capabilities: {', '.join(affected_categories)}")
        if latency_drift:
            reasons.append(f"Latency increased by {lat_change_pct:.1f}% (+{lat_delta_ms:.0f}ms)")
        if safety_drift:
            reasons.append(f"Refusal rate surged by +{refusal_delta*100.0:.0f}%")
        if truncation_detected:
            if truncation_evidence:
                reasons.extend(truncation_evidence)
            else:
                reasons.append("Completion truncation detected across multiple tasks")
        if json_validity_delta >= 20.0:
            reasons.append(f"JSON parse validity dropped by {json_validity_delta:.1f}%")

        summary_text = "; ".join(reasons) if reasons else "No statistically significant drift detected (normal variation)."

        return DriftReport(
            drift_detected=drift_detected,
            severity=severity,
            p_value=p_val,
            z_score=z_score,
            cohens_h=cohens_h,
            overall_accuracy_delta=acc_delta,
            affected_categories=affected_categories,
            category_deltas=category_deltas,
            latency_delta_ms=lat_delta_ms,
            latency_change_percent=lat_change_pct,
            throughput_change_percent=tps_change_pct,
            refusal_rate_delta=refusal_delta,
            truncation_detected=truncation_detected,
            json_validity_delta=json_validity_delta,
            json_compliance_delta=json_compliance_delta,
            summary_text=summary_text,
            details={
                "baseline_accuracy": base_acc,
                "candidate_accuracy": cand_acc,
                "baseline_latency_ms": base_lat,
                "candidate_latency_ms": cand_lat,
                "baseline_tps": base_tps,
                "candidate_tps": cand_tps,
                "truncation_evidence": truncation_evidence,
            },
        )
