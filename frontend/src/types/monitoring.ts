export type NavigationTab = 'dashboard' | 'canary-testing' | 'change-detection' | 'attribution' | 'evaluation';

export interface ExperimentConfig {
  temperature: number;
  seed: number;
  max_tokens: number;
}

export interface HealthResponse {
  status: 'connected' | 'disconnected';
  alias: string;
  model_identifier: string;
  candidate_models: string[];
  provider: string;
  endpoint: string;
  mock_mode: boolean;
  has_baseline: boolean;
  baseline_file: string;
  latest_run_file: string;
  last_execution_time: string | null;
  experiment_config: ExperimentConfig;
}

export interface CanaryCheckSpec {
  type: string;
  value?: string;
  values?: string[];
  [key: string]: unknown;
}

export interface CanaryTask {
  id: string;
  division: string;
  tier: 'easy' | 'general' | 'expensive';
  prompt: string;
  system?: string;
  filler?: unknown;
  temperature?: number;
  response_format?: string;
  max_tokens: number;
  checks: CanaryCheckSpec[];
  est_in_tokens: number;
  est_out_tokens: number;
  est_total_tokens: number;
}

export interface CanaryDivision {
  id: string;
  temperature?: number;
  targets_layer?: string;
  primary_metrics?: string[];
}

export interface CanariesResponse {
  meta: Record<string, unknown>;
  divisions: CanaryDivision[];
  tasks: CanaryTask[];
  check_types: Record<string, string>;
  profiles: Record<string, unknown>;
}

export interface CheckObservation {
  type: string;
  expected: string;
  passed: boolean;
  unavailable?: boolean;
  detail?: string;
}

export interface Observation {
  canary_id: string;
  run_id: string;
  probe_execution_id: string;
  request_config_hash: string;
  requested_model?: string;
  returned_model?: string;
  system_fingerprint?: string | null;
  timestamp: string;
  response: string;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  finish_reason: string;
  http_status: number;
  retry_count: number;
  error?: string;
  checks: CheckObservation[];
  style?: number[];
}

export interface RunSummary {
  file_name: string;
  kind: 'run' | 'baseline';
  timestamp: string;
  total_tasks: number;
  passed_tasks: number;
  failed_tasks: number;
  pass_rate: number;
  observations?: Observation[];
}

export interface RunsResponse {
  runs: RunSummary[];
}

export interface ExecuteRunRequest {
  profile?: string;
  tiers?: ('easy' | 'general' | 'expensive')[];
  mode?: 'run' | 'baseline';
  temperature: number;
  seed: number;
  max_tokens: number;
}

export interface ExecuteRunResponse {
  file_name: string;
  mode: string;
  total_tasks: number;
  passed_tasks: number;
  failed_tasks: number;
  pass_rate: number;
  observations: Observation[];
}

export interface FeatureStats {
  mean: number;
  sd: number;
  n: number;
}

export interface BaselineResponse {
  available: boolean;
  message?: string;
  baseline_batches?: number;
  features?: Record<string, FeatureStats>;
  extracted_metrics?: Record<string, number>;
  observations?: Observation[];
}

export interface FeatureChange {
  feature: string;
  direction: 'increased' | 'decreased';
  effect_size: number;
  adjusted_p_value: number;
  baseline: number;
  current: number;
}

export interface DetectionResult {
  detected: boolean;
  changes: FeatureChange[];
}

export interface DriftAnalysis {
  detected: boolean;
  confidence: number;
  predicted_layer: string;
  evidence: string[];
}

export interface ThresholdsConfig {
  fdr_q: number;
  min_effect: Record<string, number>;
  [key: string]: unknown;
}

export interface DetectionResponse {
  available: boolean;
  message?: string;
  run_file?: string;
  thresholds?: ThresholdsConfig;
  detection?: DetectionResult;
  analysis?: DriftAnalysis;
  baseline_stats?: Record<string, FeatureStats>;
  baseline_metrics?: Record<string, number>;
  current_metrics?: Record<string, number>;
}

export interface LayerScore {
  layer: 'model' | 'serving' | 'safety' | 'request_config' | string;
  score: number;
  core_hit: number;
  precision: number;
}

export interface AttributionResult {
  ranked: LayerScore[];
  margin: number;
  confidence: 'high' | 'medium' | 'inconclusive' | 'n/a' | string;
  layers?: string[];
  scores?: LayerScore[];
  config_explained?: string[];
  provider_side?: string[];
  inconclusive: boolean;
  reasons?: string[];
}

export interface CheckReport {
  status: string;
  observed_evidence: DetectionResult;
  likely_cause: AttributionResult;
  suggested_next_check: string;
  calls: number;
  input_tokens: number;
  output_tokens: number;
  seconds: number;
  run_id?: string;
  profile?: string;
  baseline_id?: string;
  unchanged?: string[];
  config_hash_baseline?: string;
  config_hash_current?: string;
  unavailable_metadata?: number;
}

export interface AttributionResponse {
  available: boolean;
  message?: string;
  run_file?: string;
  detection?: DetectionResult;
  attribution?: AttributionResult;
  analysis?: DriftAnalysis;
  report?: CheckReport;
}

export type DetectionStatusLabel =
  | 'Normal variation'
  | 'Potential deviation'
  | 'Confirmed threshold breach'
  | 'Recovered behaviour';


// ==========================================
// Evaluation & Experiment Types (PS-03 Head-to-Head & Quantization)
// ==========================================

export interface HeadToHeadMetricsPerScenario {
  is_positive: boolean;
  expected_layer: string;
  trials: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  correct_attribution: number;
  wrong_attribution: number;
  inconclusive_attribution: number;
  mean_latency_ms: number;
  mean_tokens: number;
}

export interface EngineMetrics {
  engine: string;
  confusion_matrix: {
    TP: number;
    FP: number;
    TN: number;
    FN: number;
  };
  detection_metrics: {
    precision: number;
    recall: number;
    f1_score: number;
    detection_rate: number;
    false_positive_rate: number;
    false_negative_rate: number;
  };
  attribution_metrics: {
    evaluated_on_tp_subset: number;
    correct_attribution: number;
    wrong_attribution: number;
    inconclusive_attribution: number;
    attribution_accuracy: number;
  };
  per_scenario: Record<string, HeadToHeadMetricsPerScenario>;
}

export interface SmokeBenchmarkVariant {
  file: string;
  score: string;
  avg_latency_ms: number;
}

export interface CanaryBenchmarkVariant {
  file: string;
  score: string;
  avg_latency_ms: number;
  vram_mb?: number;
}

export interface QuantizationEvidence {
  model_architecture: string;
  smoke_suite_10_tasks: {
    dataset: string;
    task_count: number;
    variants: Record<string, SmokeBenchmarkVariant>;
  };
  canary_suite_12_tasks: {
    dataset: string;
    task_count: number;
    variants: Record<string, CanaryBenchmarkVariant>;
    empirical_finding: string;
  };
}

export interface OllamaComparisonItem {
  canary_id: string;
  status: string;
  reasons?: string[];
  baseline_checks_passed: number;
  eval_checks_passed: number;
  total_checks: number;
  baseline_latency_sec: number;
  eval_latency_sec: number;
}

export interface OllamaModelSummary {
  avg_latency_sec: number;
  below_baseline: number;
  check_pass_rate: number;
  comparisons: OllamaComparisonItem[];
}

export interface OllamaEvaluationData {
  baseline_file: string;
  baseline_model: string;
  experiment_id: string;
  timestamp: string;
  model_b: OllamaModelSummary;
  model_c: OllamaModelSummary;
}

export interface ScenarioEvaluationNode {
  detected: number;
  correct_attr: number;
  wrong_attr: number;
  latency_ms: number;
  tokens: number;
  status: string;
}

export interface EvaluationScenarioSummary {
  id: string;
  name: string;
  category: string;
  expected_layer: string;
  trials: number;
  go_v2: ScenarioEvaluationNode;
  python_ref: ScenarioEvaluationNode;
  observable_evidence: string[];
  notes?: string;
}

export interface MultiQueryProbeResult {
  probe_id: string;
  category: string;
  difficulty: string;
  prompt: string;
  expected: unknown;
  reference_answer?: string;
  actual_response?: string;
  evaluator: string;
  status: 'PASS' | 'FAIL' | 'INCONCLUSIVE' | 'EVALUATOR_ERROR';
  passed: boolean;
  inconclusive: boolean;
  evaluator_error: boolean;
  score: number;
  details: string;
  failure_reason: string;
  latency_ms: number;
  tokens_per_second: number;
}

export interface MultiQueryCategoryStats {
  category: string;
  total: number;
  passed: number;
  failed: number;
  inconclusive: number;
  pass_rate: number;
  mean_latency_ms: number;
}

export interface MultiQuerySuiteRun {
  evaluation_id: string;
  timestamp: string;
  dataset_name: string;
  dataset_version: string;
  model_name: string;
  quantization_level: string;
  total_probes: number;
  passed_probes: number;
  failed_probes: number;
  inconclusive_probes: number;
  overall_pass_rate: number;
  mean_latency_ms: number;
  category_breakdown: Record<string, MultiQueryCategoryStats>;
  probes: MultiQueryProbeResult[];
}

export interface MultiQueryRegression {
  probe_id: string;
  category: string;
  difficulty: string;
  prompt: string;
  expected: unknown;
  baseline_response?: string;
  candidate_response?: string;
  failure_reason: string;
  details: string;
}

export interface ScenarioSpec {
  id: string;
  name: string;
  canonical_target_layer: string;
  description: string;
  is_fault_injection: boolean;
  simulation_mode: string;
  affected_probes: string;
  injected_parameters?: Record<string, unknown>;
  expected_outcome?: Record<string, unknown>;
}

export interface ScenariosListResponse {
  scenarios: ScenarioSpec[];
  disclaimer: string;
}

export interface RunScenarioRequest {
  scenario_id: string;
}

export interface RunScenarioResponse {
  status: string;
  run_id: string;
  file_name: string;
  scenario_id: string;
  name: string;
  is_simulation: boolean;
  simulation_mode: string;
  execution_ms: number;
  total_probes: number;
  passed_probes: number;
  failed_probes: number;
  pass_rate: number;
  telemetry: {
    avg_latency_ms: number;
    effective_tokens_per_s: number;
    total_tokens_generated: number;
    intrinsic_tok_per_sec: number;
  };
  detection: {
    drift_detected: boolean;
    confidence: number;
  };
  attribution: {
    predicted_layer: string;
    canonical_layer: string;
    confidence: number;
    evidence: string[];
  };
  ground_truth: {
    canonical_layer: string;
    description: string;
    ground_truth_isolated: boolean;
    isolation_audit: string;
  };
  evaluation: {
    attribution_correct: boolean;
    false_alarm: boolean;
    false_negative: boolean;
    verdict: string;
  };
  observations?: Observation[];
}

export interface MultiQueryCategoryComparison {
  category: string;
  total: number;
  baseline_pass_rate: number;
  candidate_pass_rate: number;
  diff_percentage_points: number;
  regressions_count: number;
  status: 'STABLE' | 'REGRESSED' | 'IMPROVED';
}

export interface MultiQueryComparison {
  comparison_id: string;
  timestamp: string;
  baseline_model: string;
  candidate_model: string;
  baseline_pass_rate: number;
  candidate_pass_rate: number;
  difference_percentage_points: number;
  total_evaluated: number;
  regressions_count: number;
  improvements_count: number;
  regressions: MultiQueryRegression[];
  category_comparison: MultiQueryCategoryComparison[];
}

export interface MultiQuerySuitePayload {
  dataset: {
    dataset_name: string;
    version: string;
    record_count: number;
    category_count: number;
    categories: Array<{ id: string; probe_count: number }>;
    records: Array<Record<string, unknown>>;
  };
  baseline_run: MultiQuerySuiteRun;
  candidate_run: MultiQuerySuiteRun;
  comparison: MultiQueryComparison;
}

export interface EvaluationResponse {
  available?: boolean;
  multi_query_suite?: MultiQuerySuitePayload;
  meta: {
    evaluation_title: string;
    version: string;
    timestamp: string;
    trial_design: string;
    common_probes_count: number;
    branch: string;
  };
  go_metrics: EngineMetrics;
  python_metrics: EngineMetrics;
  real_model_quantization_evidence: QuantizationEvidence;
  ollama_evaluation?: OllamaEvaluationData;
  fault_scenarios?: EvaluationScenarioSummary[];
}
