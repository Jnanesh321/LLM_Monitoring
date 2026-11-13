package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/Abhinav-DROID-NG/LLM_Monitoring/internal/canary"
	"github.com/Abhinav-DROID-NG/LLM_Monitoring/internal/monitor"
)

type SharedTask struct {
	ID             string         `json:"id"`
	Origin         string         `json:"origin"`
	Division       string         `json:"division"`
	Category       string         `json:"category"`
	Prompt         string         `json:"prompt"`
	MaxTokens      int            `json:"max_tokens"`
	Temperature    float64        `json:"temperature"`
	InCommonSubset bool           `json:"in_common_subset"`
	Checks         []canary.Check `json:"checks"`
}

type SharedFixture struct {
	Tasks []SharedTask `json:"tasks"`
}

type TrialResult struct {
	TrialIndex     int                   `json:"trial_index"`
	Scenario       string                `json:"scenario"`
	DurationMS     float64               `json:"duration_ms"`
	AvgLatencyMS   float64               `json:"avg_latency_ms"`
	TotalTokens    int                   `json:"total_tokens"`
	TokensPerSec   float64               `json:"tokens_per_sec"`
	Detected       bool                  `json:"drift_detected"`
	PredictedLayer string                `json:"predicted_layer"`
	Confidence     float64               `json:"confidence"`
	ConfidenceText string                `json:"confidence_text"`
	Evidence       []string              `json:"evidence"`
	Observations   []monitor.Observation `json:"observations"`
}

func main() {
	mode := flag.String("mode", "run_trials", "mode: warmup | baseline | trials")
	scenario := flag.String("scenario", "EXP-00", "scenario: EXP-00..EXP-04")
	trialCount := flag.Int("trials", 20, "number of trials")
	fixturePath := flag.String("fixture", "data/shared_canary_fixture.json", "path to shared fixture")
	baselineFile := flag.String("baseline_file", "", "path to saved baseline observations json")
	outputFile := flag.String("output", "", "path to output json")
	jitterFile := flag.String("jitter_file", "", "path to jitter array json")
	flag.Parse()

	fixtureData, err := os.ReadFile(*fixturePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading fixture: %v\n", err)
		os.Exit(1)
	}

	var fixture SharedFixture
	if err := json.Unmarshal(fixtureData, &fixture); err != nil {
		fmt.Fprintf(os.Stderr, "Error unmarshaling fixture: %v\n", err)
		os.Exit(1)
	}

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

	var jitters []float64
	if *jitterFile != "" {
		if jdata, err := os.ReadFile(*jitterFile); err == nil {
			_ = json.Unmarshal(jdata, &jitters)
		}
	}

	switch *mode {
	case "warmup":
		runWarmup(subset, 5)
		fmt.Println("WARMUP_COMPLETE")
	case "baseline":
		baselineRuns := runBaselineProfiling(subset, *trialCount)
		if *outputFile != "" {
			data, _ := json.MarshalIndent(baselineRuns, "", "  ")
			_ = os.WriteFile(*outputFile, data, 0644)
			fmt.Printf("BASELINE_SAVED: %s (%d runs)\n", *outputFile, len(baselineRuns))
		}
	case "trials":
		var baselineObs []monitor.Observation
		if *baselineFile != "" {
			if bdata, err := os.ReadFile(*baselineFile); err == nil {
				_ = json.Unmarshal(bdata, &baselineObs)
			}
		}
		results := runScenarioTrials(subset, baselineObs, *scenario, *trialCount, jitters)
		if *outputFile != "" {
			data, _ := json.MarshalIndent(results, "", "  ")
			_ = os.WriteFile(*outputFile, data, 0644)
			fmt.Printf("TRIALS_SAVED: %s (%d trials)\n", *outputFile, len(results))
		}
	default:
		fmt.Fprintf(os.Stderr, "Unknown mode: %s\n", *mode)
		os.Exit(1)
	}
}

func runWarmup(tasks []canary.Task, count int) {
	for i := 0; i < count; i++ {
		for _, task := range tasks {
			_ = executeMockProbe(task, "EXP-00", 0.0, false, 0)
		}
	}
}

func runBaselineProfiling(tasks []canary.Task, trials int) []monitor.Observation {
	var allObs []monitor.Observation
	for t := 0; t < trials; t++ {
		for _, task := range tasks {
			obs := executeMockProbe(task, "EXP-00", 0.0, false, 0)
			obs.RunID = fmt.Sprintf("baseline_run_%d", t+1)
			allObs = append(allObs, obs)
		}
	}
	return allObs
}

func runScenarioTrials(tasks []canary.Task, baselineObs []monitor.Observation, scenario string, trialCount int, jitters []float64) []TrialResult {
	var results []TrialResult

	// Compile baseline average per probe
	baseMap := make(map[string]monitor.Observation)
	for _, b := range baselineObs {
		baseMap[b.CanaryID] = b
	}

	// Pre-create baseline slice for detector
	refSlice := make([]monitor.Observation, 0, len(tasks))
	for _, task := range tasks {
		if b, ok := baseMap[task.ID]; ok {
			refSlice = append(refSlice, b)
		}
	}

	for trial := 0; trial < trialCount; trial++ {
		startTrial := time.Now()
		var trialObs []monitor.Observation
		jitter := 0.0
		if trial < len(jitters) {
			jitter = jitters[trial]
		}

		totalTokens := 0
		var totalLatMS float64

		for _, task := range tasks {
			obs := executeMockProbe(task, scenario, jitter, true, trial)
			obs.RunID = fmt.Sprintf("%s_trial_%d", scenario, trial+1)
			trialObs = append(trialObs, obs)
			totalTokens += obs.InputTokens + obs.OutputTokens
			totalLatMS += float64(obs.LatencyMS)
		}

		trialDuration := time.Since(startTrial)
		avgLat := totalLatMS / float64(len(tasks))
		tokensPerSec := 0.0
		if totalLatMS > 0 {
			tokensPerSec = float64(totalTokens) / (totalLatMS / 1000.0)
		}

		// Run Go's detector with Ground-Truth Isolation (detector receives ONLY observations)
		analysis := monitor.Analyze(refSlice, trialObs)

		// Map Go layer names to standardized canonical layers
		layerMap := map[string]string{
			"model":          "MODEL",
			"serving":        "SERVING",
			"safety":         "SAFETY",
			"request_config": "REQUEST_CONFIG",
			"unknown":        "INCONCLUSIVE",
		}
		predicted := layerMap[analysis.PredictedLayer]
		if predicted == "" {
			if !analysis.Detected {
				predicted = "NO_DRIFT"
			} else {
				predicted = "INCONCLUSIVE"
			}
		}

		results = append(results, TrialResult{
			TrialIndex:     trial + 1,
			Scenario:       scenario,
			DurationMS:     float64(trialDuration.Milliseconds()),
			AvgLatencyMS:   avgLat,
			TotalTokens:    totalTokens,
			TokensPerSec:   tokensPerSec,
			Detected:       analysis.Detected,
			PredictedLayer: predicted,
			Confidence:     analysis.Confidence,
			ConfidenceText: fmt.Sprintf("%.2f", analysis.Confidence),
			Evidence:       analysis.Evidence,
			Observations:   trialObs,
		})
	}

	return results
}

func executeMockProbe(task canary.Task, scenario string, jitterMS float64, isTrial bool, trialIdx int) monitor.Observation {
	// Base mock responses for common subset
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

	latencyMS := int64(45) // base simulated latency
	finishReason := "stop"

	// Apply Injected Fault Mutations based on Scenario
	switch scenario {
	case "EXP-01":
		// Deterministic reasoning corruption schedule (5 specific probes out of 16)
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
		// Serving delay: +1200ms fixed delay + paired trial jitter
		latencyMS += 1200 + int64(math.Round(jitterMS))
		if latencyMS < 10 {
			latencyMS = 10
		}
	case "EXP-03":
		// Benign safety filter: Refusal template triggered on keyword match
		if task.ID == "safety_001" || task.ID == "REF-E1" || task.ID == "REF-E2" {
			baseText = "I cannot assist with this request as it involves sensitive domain policies."
			finishReason = "content_filter"
		}
		// safety_002 is fulfilled normally
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

	// Token estimates
	inTokens := len(strings.Fields(task.Prompt))
	outTokens := len(strings.Fields(baseText))

	// Evaluate canary checks
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

	return monitor.Observation{
		CanaryID:     task.ID,
		Timestamp:    time.Now().UTC(),
		Response:     baseText,
		LatencyMS:    latencyMS,
		InputTokens:  inTokens,
		OutputTokens: outTokens,
		FinishReason: finishReason,
		Checks:       checksObs,
	}
}
