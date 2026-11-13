package main

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Abhinav-DROID-NG/LLM_Monitoring/internal/canary"
	"github.com/Abhinav-DROID-NG/LLM_Monitoring/internal/monitor"
)

type scenarioSpec struct {
	ID                   string         `json:"id"`
	Name                 string         `json:"name"`
	CanonicalTargetLayer string         `json:"canonical_target_layer"`
	Description          string         `json:"description"`
	IsFaultInjection     bool           `json:"is_fault_injection"`
	SimulationMode       string         `json:"simulation_mode"`
	AffectedProbes       string         `json:"affected_probes"`
	InjectedParameters   map[string]any `json:"injected_parameters"`
	ExpectedOutcome      map[string]any `json:"expected_outcome"`
}

var canonicalScenarios = []scenarioSpec{
	{
		ID:                   "EXP-00",
		Name:                 "Baseline Control (Null Mutation)",
		CanonicalTargetLayer: "NO_DRIFT",
		Description:          "Negative control run with zero injected fault. Validates baseline stability and measures false alarm rate (FPR).",
		IsFaultInjection:     false,
		SimulationMode:       "Simulated Mock (Ground-Truth Isolated)",
		AffectedProbes:       "0 / 16 probes affected (0%)",
		InjectedParameters: map[string]any{
			"model_corruption_active":     false,
			"injected_latency_delay_ms":   0.0,
			"strict_safety_filter_active": false,
			"max_tokens_override":         nil,
		},
		ExpectedOutcome: map[string]any{
			"drift_detected":       false,
			"expected_attribution": "NO_DRIFT",
		},
	},
	{
		ID:                   "EXP-01",
		Name:                 "Model Layer Degradation (Reasoning Corruption)",
		CanonicalTargetLayer: "MODEL",
		Description:          "Simulated model-layer regression: Injects reasoning & math corruption across 5 specific probes while preserving format, safety, and serving latency.",
		IsFaultInjection:     true,
		SimulationMode:       "Simulated Mock (Ground-Truth Isolated)",
		AffectedProbes:       "5 / 16 probes affected (math_001, math_002, logic_001, logic_002, CAP-E1)",
		InjectedParameters: map[string]any{
			"model_corruption_active":     true,
			"corrupted_probe_ids":         []string{"math_001", "math_002", "logic_001", "logic_002", "CAP-E1"},
			"injected_latency_delay_ms":   0.0,
			"strict_safety_filter_active": false,
			"max_tokens_override":         nil,
		},
		ExpectedOutcome: map[string]any{
			"drift_detected":       true,
			"expected_attribution": "MODEL",
		},
	},
	{
		ID:                   "EXP-02",
		Name:                 "Serving Stack Mutation (+1200ms Latency Delay)",
		CanonicalTargetLayer: "SERVING",
		Description:          "Simulates upstream proxy / queuing congestion by injecting +1200ms fixed latency. Output correctness and safety are 100% preserved.",
		IsFaultInjection:     true,
		SimulationMode:       "Simulated Mock (Ground-Truth Isolated)",
		AffectedProbes:       "16 / 16 probes affected (+1200ms roundtrip delay)",
		InjectedParameters: map[string]any{
			"model_corruption_active":     false,
			"injected_latency_delay_ms":   1200.0,
			"strict_safety_filter_active": false,
			"max_tokens_override":         nil,
		},
		ExpectedOutcome: map[string]any{
			"drift_detected":       true,
			"expected_attribution": "SERVING",
		},
	},
	{
		ID:                   "EXP-03",
		Name:                 "Safety Filter Mutation (Over-Moderation)",
		CanonicalTargetLayer: "SAFETY",
		Description:          "Simulates overly sensitive safety guardrail rejecting benign domestic first-aid & chemical safety inquiries (bleach/ammonia, peroxide, burn treatment).",
		IsFaultInjection:     true,
		SimulationMode:       "Simulated Mock (Ground-Truth Isolated)",
		AffectedProbes:       "3 / 4 benign safety probes refused (safety_001, REF-E1, REF-E2)",
		InjectedParameters: map[string]any{
			"model_corruption_active":     false,
			"injected_latency_delay_ms":   0.0,
			"strict_safety_filter_active": true,
			"max_tokens_override":         nil,
		},
		ExpectedOutcome: map[string]any{
			"drift_detected":       true,
			"expected_attribution": "SAFETY",
		},
	},
	{
		ID:                   "EXP-04",
		Name:                 "Request Config Mutation (14-Token Truncation)",
		CanonicalTargetLayer: "REQUEST_CONFIG",
		Description:          "Simulates client configuration error capping max_tokens at 14, causing incomplete sentences, unclosed JSON syntax, and finish_reason='length'.",
		IsFaultInjection:     true,
		SimulationMode:       "Simulated Mock (Ground-Truth Isolated)",
		AffectedProbes:       "3 multi-token probes truncated (TRN-E1, json_002, instruct_002)",
		InjectedParameters: map[string]any{
			"model_corruption_active":     false,
			"injected_latency_delay_ms":   0.0,
			"strict_safety_filter_active": false,
			"max_tokens_override":         14,
		},
		ExpectedOutcome: map[string]any{
			"drift_detected":       true,
			"expected_attribution": "REQUEST_CONFIG",
		},
	},
}

type runScenarioPayload struct {
	ScenarioID string `json:"scenario_id"`
}

func (s *apiServer) handleListScenarios(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"scenarios": canonicalScenarios,
		"disclaimer": "All EXP-00 through EXP-04 experiments run in simulated mock mode with strict ground-truth isolation. The monitoring detector evaluates raw probe observations only.",
	})
}

func (s *apiServer) handleRunScenario(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		writeJSON(w, http.StatusConflict, map[string]string{
			"error": "an experiment or canary execution is already in progress. Please wait for it to complete.",
		})
		return
	}
	s.running = true
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.running = false
		s.mu.Unlock()
	}()

	var req runScenarioPayload
	_ = json.NewDecoder(r.Body).Decode(&req)
	scID := strings.ToUpper(strings.TrimSpace(req.ScenarioID))
	if scID == "" {
		scID = "EXP-00"
	}

	var matchedSpec *scenarioSpec
	for i := range canonicalScenarios {
		if canonicalScenarios[i].ID == scID {
			matchedSpec = &canonicalScenarios[i]
			break
		}
	}
	if matchedSpec == nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": fmt.Sprintf("invalid scenario_id %q: must be one of EXP-00, EXP-01, EXP-02, EXP-03, EXP-04", scID),
		})
		return
	}

	// 1. Load Common Subset Tasks
	tasks := loadCommonSubsetTasks(s.cfg.DataDir)

	// 2. Load Reference Baseline Observations (unbiased baseline)
	baseObs := loadReferenceBaselineObs(s.cfg.DataDir, tasks)

	// 3. Execute the Simulated Scenario Trial
	// Generates raw observations according to deterministic mutation schedule
	startTime := time.Now()
	trialObs := executeScenarioProbes(tasks, scID)
	execDuration := time.Since(startTime)

	// 4. Save Observation to data/ directory using standard run prefix
	timestampStr := time.Now().UTC().Format("20060102T150405Z")
	fileName := fmt.Sprintf("run-%s-%s.json", timestampStr, scID)
	outPath := filepath.Join(s.cfg.DataDir, fileName)
	_ = monitor.SaveJSON(outPath, trialObs)

	// 5. Run Detector & Attribution with GROUND-TRUTH ISOLATION
	// The detector receives ONLY the raw baseline observations and candidate observations.
	// No ground-truth label, scenario ID, or mutation specification is provided.
	analysis := monitor.Analyze(baseObs, trialObs)

	// Calculate telemetry statistics
	passedCount := 0
	totalTokens := 0
	var totalLatMS float64
	for _, o := range trialObs {
		if o.Error == "" && allChecksPassed(o.Checks) {
			passedCount++
		}
		totalTokens += o.InputTokens + o.OutputTokens
		totalLatMS += float64(o.LatencyMS)
	}
	avgLat := totalLatMS / float64(maxInt(1, len(trialObs)))
	tokPerSec := 0.0
	if totalLatMS > 0 {
		tokPerSec = float64(totalTokens) / (totalLatMS / 1000.0)
	}

	// Normalize predicted layer to canonical layer name
	layerMap := map[string]string{
		"model":          "MODEL",
		"serving":        "SERVING",
		"safety":         "SAFETY",
		"request_config": "REQUEST_CONFIG",
		"unknown":        "INCONCLUSIVE",
		"":               "INCONCLUSIVE",
	}
	predictedCanonical := layerMap[strings.ToLower(analysis.PredictedLayer)]
	if !analysis.Detected {
		predictedCanonical = "NO_DRIFT"
	}

	// Post-evaluation ground truth comparison (evaluated strictly after detector output)
	expectedCanonical := matchedSpec.CanonicalTargetLayer
	attributionMatch := predictedCanonical == expectedCanonical
	falseAlarm := scID == "EXP-00" && analysis.Detected
	falseNegative := scID != "EXP-00" && !analysis.Detected

	writeJSON(w, http.StatusOK, map[string]any{
		"status":          "completed",
		"run_id":          fileName,
		"file_name":       fileName,
		"scenario_id":     matchedSpec.ID,
		"name":            matchedSpec.Name,
		"is_simulation":   true,
		"simulation_mode": matchedSpec.SimulationMode,
		"execution_ms":    execDuration.Milliseconds(),
		"total_probes":    len(trialObs),
		"passed_probes":   passedCount,
		"failed_probes":   len(trialObs) - passedCount,
		"pass_rate":       float64(passedCount) * 100.0 / float64(maxInt(1, len(trialObs))),
		"telemetry": map[string]any{
			"avg_latency_ms":          avgLat,
			"effective_tokens_per_s":  tokPerSec,
			"total_tokens_generated":  totalTokens,
			"intrinsic_tok_per_sec":   50.0,
		},
		"detection": map[string]any{
			"drift_detected": analysis.Detected,
			"confidence":     analysis.Confidence,
		},
		"attribution": map[string]any{
			"predicted_layer": analysis.PredictedLayer,
			"canonical_layer": predictedCanonical,
			"confidence":      analysis.Confidence,
			"evidence":        analysis.Evidence,
		},
		"ground_truth": map[string]any{
			"canonical_layer":             matchedSpec.CanonicalTargetLayer,
			"description":                 matchedSpec.Description,
			"ground_truth_isolated":       true,
			"isolation_audit":             "Confirmed: Detector only accessed raw response text, latency, finish_reason, and check observations.",
		},
		"evaluation": map[string]any{
			"attribution_correct": attributionMatch,
			"false_alarm":         falseAlarm,
			"false_negative":      falseNegative,
			"verdict":             map[bool]string{true: "MATCH", false: "MISMATCH"}[attributionMatch],
		},
		"observations": trialObs,
	})
}

func loadCommonSubsetTasks(dataDir string) []canary.Task {
	// Try loading shared_canary_fixture.json using robust file finder
	fixturePath := findEvaluationFile(
		filepath.Join(dataDir, "shared_canary_fixture.json"),
		filepath.Join("data", "shared_canary_fixture.json"),
		"shared_canary_fixture.json",
		"data/shared_canary_fixture.json",
	)

	if fixturePath != "" {
		raw, err := os.ReadFile(fixturePath)
		if err == nil {
			var fixture struct {
				Tasks []struct {
					ID             string         `json:"id"`
					Division       string         `json:"division"`
					Prompt         string         `json:"prompt"`
					MaxTokens      int            `json:"max_tokens"`
					InCommonSubset bool           `json:"in_common_subset"`
					Checks         []canary.Check `json:"checks"`
				} `json:"tasks"`
			}
			if json.Unmarshal(raw, &fixture) == nil && len(fixture.Tasks) > 0 {
				var subset []canary.Task
				for _, t := range fixture.Tasks {
					if t.InCommonSubset {
						subset = append(subset, canary.Task{
							ID:        t.ID,
							Division:  t.Division,
							Tier:      "easy",
							Prompt:    t.Prompt,
							MaxTokens: t.MaxTokens,
							Checks:    t.Checks,
						})
					}
				}
				if len(subset) > 0 {
					return subset
				}
			}
		}
	}

	// Fallback to canaries_fixed.json
	canaryPath := findEvaluationFile("canaries_fixed.json", "../canaries_fixed.json", "../../canaries_fixed.json")
	if canaryPath != "" {
		lib, err := canary.Load(canaryPath)
		if err == nil && len(lib.Tasks) > 0 {
			return lib.Pinned()
		}
	}
	return nil
}

func loadReferenceBaselineObs(dataDir string, tasks []canary.Task) []monitor.Observation {
	// 1. Try go_baseline_runs.json
	basePath := findEvaluationFile(
		filepath.Join(dataDir, "go_baseline_runs.json"),
		filepath.Join("data", "go_baseline_runs.json"),
		"go_baseline_runs.json",
		"data/go_baseline_runs.json",
	)
	if basePath != "" {
		raw, err := os.ReadFile(basePath)
		if err == nil {
			var all []monitor.Observation
			if json.Unmarshal(raw, &all) == nil && len(all) > 0 {
				m := make(map[string]monitor.Observation)
				for _, o := range all {
					if _, exists := m[o.CanaryID]; !exists {
						m[o.CanaryID] = o
					}
				}
				var out []monitor.Observation
				for _, t := range tasks {
					if o, ok := m[t.ID]; ok {
						out = append(out, o)
					}
				}
				if len(out) > 0 {
					return out
				}
			}
		}
	}

	// 2. Fallback to latest baseline run
	baseRuns := loadAllRunsByPrefix(dataDir, "baseline-")
	if len(baseRuns) > 0 {
		return baseRuns[len(baseRuns)-1]
	}

	// 3. Fallback: synthesize baseline observations with zero mutations
	return executeScenarioProbes(tasks, "EXP-00")
}

func executeScenarioProbes(tasks []canary.Task, scenarioID string) []monitor.Observation {
	var trialObs []monitor.Observation

	for _, task := range tasks {
		baseText := ""
		switch task.ID {
		case "math_001":
			baseText = "323"
		case "math_002":
			baseText = "90"
		case "CAP-E1":
			baseText = "408"
		case "DET-E1":
			baseText = "103"
		case "logic_001":
			baseText = "YES"
		case "logic_002":
			baseText = "NO"
		case "instruct_001":
			baseText = "PARIS"
		case "instruct_002":
			baseText = "1-2-3-4-5"
		case "json_002":
			baseText = `{"status": "active", "count": 42}`
		case "FMT-E1":
			baseText = `{"city": "Tokyo", "country": "Japan"}`
		case "LAT-E1":
			baseText = "OK"
		case "TRN-E1":
			baseText = "Monday\nTuesday\nWednesday\nThursday\nFriday\nSaturday\nSunday\n<<END>>"
		case "safety_001":
			baseText = "Chlorine bleach is an oxidizing agent using sodium hypochlorite, while hydrogen peroxide uses oxygen release for sanitization."
		case "safety_002":
			baseText = "Immediately immerse the thermal burn in cool running water for at least ten minutes, then apply a clean dressing."
		case "REF-E1":
			baseText = "You should never mix bleach and ammonia because the reaction generates toxic chloramine gas."
		case "REF-E2":
			baseText = "Use the command pkill -15 python or kill -SIGTERM to terminate the hung Python process safely."
		default:
			baseText = "OK"
		}

		latencyMS := int64(45)
		finishReason := "stop"

		// Apply Injected Fault Mutations based on Scenario ID
		switch scenarioID {
		case "EXP-01":
			// Selective reasoning corruption schedule (5 specific probes out of 16)
			switch task.ID {
			case "math_001":
				baseText = "999" // corrupted (fails 323)
			case "math_002":
				baseText = "120" // corrupted (fails 90)
			case "logic_001":
				baseText = "NO" // inverted (fails YES)
			case "logic_002":
				baseText = "YES" // inverted (fails NO)
			case "CAP-E1":
				baseText = "500" // corrupted (fails 408)
			}
		case "EXP-02":
			// Serving delay: +1200ms fixed delay
			latencyMS += 1200
		case "EXP-03":
			// Benign safety filter: Refusal template triggered
			if task.ID == "safety_001" || task.ID == "REF-E1" || task.ID == "REF-E2" {
				baseText = "I cannot assist with this request as it involves sensitive domain policies."
				finishReason = "content_filter"
			}
		case "EXP-04":
			// Request config truncation: max_tokens = 14
			switch task.ID {
			case "TRN-E1":
				baseText = "Monday\nTuesday\nWednesday\nThursday"
				finishReason = "length"
			case "json_002":
				baseText = `{"status": "active", "c`
				finishReason = "length"
			case "instruct_002":
				baseText = "1-2-3-"
				finishReason = "length"
			}
		}

		inTokens := len(strings.Fields(task.Prompt))
		outTokens := len(strings.Fields(baseText))

		respView := canary.ResponseView{
			Text:         baseText,
			FinishReason: finishReason,
			InputTokens:  &inTokens,
			OutputTokens: &outTokens,
		}

		var checksObs []monitor.CheckObservation
		for _, chk := range task.Checks {
			res := canary.CheckResponse(task, chk, respView)
			expVal := chk.ValueString()
			if expVal == "" {
				expVal = chk.Type
			}
			checksObs = append(checksObs, monitor.CheckObservation{
				Type:        chk.Type,
				Expected:    expVal,
				Passed:      res.Passed,
				Unavailable: res.Unavailable,
				Detail:      res.Detail,
			})
		}

		trialObs = append(trialObs, monitor.Observation{
			CanaryID:     task.ID,
			Timestamp:    time.Now().UTC(),
			Response:     baseText,
			LatencyMS:    latencyMS,
			InputTokens:  inTokens,
			OutputTokens: outTokens,
			FinishReason: finishReason,
			Checks:       checksObs,
		})
	}

	return trialObs
}

func roundFloat(val float64, precision int) float64 {
	ratio := math.Pow(10, float64(precision))
	return math.Round(val*ratio) / ratio
}
