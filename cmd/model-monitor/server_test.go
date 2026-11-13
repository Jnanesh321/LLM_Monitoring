package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/Abhinav-DROID-NG/LLM_Monitoring/internal/canary"
	"github.com/Abhinav-DROID-NG/LLM_Monitoring/internal/monitor"
)

func setupTestServer(t *testing.T) (*httptest.Server, *apiServer) {
	t.Helper()
	canaryPath := findEvaluationFile("canaries_fixed.json", "../canaries_fixed.json", "../../canaries_fixed.json")
	if canaryPath == "" {
		canaryPath = "canaries_fixed.json"
	}
	library, err := canary.Load(canaryPath)
	if err != nil {
		t.Fatalf("failed to load canaries: %v", err)
	}

	thresholdsPath := findEvaluationFile("thresholds.json", "../thresholds.json", "../../thresholds.json")
	thresholds, err := monitor.LoadThresholds(thresholdsPath)
	if err != nil {
		thresholds = monitor.Thresholds{
			FDRQ: 0.05,
			MinEffect: map[string]float64{
				"proportion_abs":   0.1,
				"latency_relative": 0.2,
			},
		}
	}

	dataDir := findEvaluationFile("data", "../data", "../../data")
	if dataDir == "" {
		dataDir = "data"
	}

	cfg := monitor.DefaultConfig()
	cfg.Mock = true
	cfg.DataDir = dataDir

	srv := &apiServer{
		cfg:        cfg,
		library:    library,
		thresholds: thresholds,
	}

	ts := httptest.NewServer(srv.routes())
	return ts, srv
}

func TestServerCORS(t *testing.T) {
	ts, _ := setupTestServer(t)
	defer ts.Close()

	req, err := http.NewRequest(http.MethodOptions, ts.URL+"/api/health", nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}
	req.Header.Set("Origin", "http://localhost:5173")
	req.Header.Set("Access-Control-Request-Method", "GET")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("preflight request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNoContent {
		t.Errorf("expected status 204 No Content for OPTIONS, got %d", resp.StatusCode)
	}
	if origin := resp.Header.Get("Access-Control-Allow-Origin"); origin != "*" {
		t.Errorf("expected Access-Control-Allow-Origin: *, got %q", origin)
	}
	if methods := resp.Header.Get("Access-Control-Allow-Methods"); methods == "" {
		t.Errorf("expected Access-Control-Allow-Methods header to be set")
	}
}

func TestHealthEndpoint(t *testing.T) {
	ts, _ := setupTestServer(t)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/api/health")
	if err != nil {
		t.Fatalf("GET /api/health failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if origin := resp.Header.Get("Access-Control-Allow-Origin"); origin != "*" {
		t.Errorf("expected CORS header, got %q", origin)
	}

	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Fatalf("failed to decode json: %v", err)
	}
	if data["status"] != "connected" {
		t.Errorf("expected status connected, got %v", data["status"])
	}
	if data["mock_mode"] != true {
		t.Errorf("expected mock_mode true, got %v", data["mock_mode"])
	}
}

func TestCanariesEndpoint(t *testing.T) {
	ts, _ := setupTestServer(t)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/api/canaries")
	if err != nil {
		t.Fatalf("GET /api/canaries failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Fatalf("failed to decode json: %v", err)
	}
	tasks, ok := data["tasks"].([]any)
	if !ok || len(tasks) != 81 {
		t.Errorf("expected 81 pinned tasks, got %d", len(tasks))
	}
}

func TestRunsEndpoint(t *testing.T) {
	ts, _ := setupTestServer(t)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/api/runs")
	if err != nil {
		t.Fatalf("GET /api/runs failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Fatalf("failed to decode json: %v", err)
	}
	runs, ok := data["runs"].([]any)
	if !ok {
		t.Errorf("expected runs array, got %T", data["runs"])
	}
	if len(runs) == 0 {
		t.Logf("runs is empty, which is valid if no runs saved yet")
	}
}

func TestBaselineEndpoint(t *testing.T) {
	ts, _ := setupTestServer(t)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/api/baseline")
	if err != nil {
		t.Fatalf("GET /api/baseline failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Fatalf("failed to decode json: %v", err)
	}
	if avail, ok := data["available"].(bool); !ok {
		t.Errorf("expected available boolean field, got %v", data["available"])
	} else if avail {
		if _, ok := data["features"]; !ok {
			t.Errorf("expected features field in baseline response")
		}
	}
}

func TestDetectionEndpoint(t *testing.T) {
	ts, _ := setupTestServer(t)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/api/detection")
	if err != nil {
		t.Fatalf("GET /api/detection failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Fatalf("failed to decode json: %v", err)
	}
	if _, ok := data["available"]; !ok {
		t.Errorf("expected available field")
	}
}

func TestAttributionEndpoint(t *testing.T) {
	ts, _ := setupTestServer(t)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/api/attribution")
	if err != nil {
		t.Fatalf("GET /api/attribution failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Fatalf("failed to decode json: %v", err)
	}

	analysis, ok := data["analysis"].(map[string]any)
	if !ok {
		t.Fatalf("expected analysis object in attribution response, got %T", data["analysis"])
	}
	if _, exists := analysis["evidence"]; !exists {
		t.Errorf("missing analysis.evidence in attribution response")
	}
	if _, exists := analysis["predicted_layer"]; !exists {
		t.Errorf("missing analysis.predicted_layer in attribution response")
	}
}

func TestEvaluationEndpoint(t *testing.T) {
	ts, _ := setupTestServer(t)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/api/evaluation")
	if err != nil {
		t.Fatalf("GET /api/evaluation failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if origin := resp.Header.Get("Access-Control-Allow-Origin"); origin != "*" {
		t.Errorf("expected CORS header on /api/evaluation, got %q", origin)
	}

	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Fatalf("failed to decode json: %v", err)
	}

	if data["available"] != true {
		t.Errorf("expected available=true in evaluation endpoint")
	}
	if data["head_to_head"] == nil {
		t.Errorf("expected genuine head_to_head data from results/head_to_head_results_v2.json")
	}
	if data["evaluation"] == nil {
		t.Errorf("expected genuine evaluation data from data/evaluation.json")
	}
	if data["go_metrics"] == nil {
		t.Errorf("expected unpacked go_metrics in response")
	}
	if data["python_metrics"] == nil {
		t.Errorf("expected unpacked python_metrics in response")
	}
}

func TestExecuteRunEndpoint(t *testing.T) {
	ts, srv := setupTestServer(t)
	defer ts.Close()

	payload := map[string]any{
		"mode":        "run",
		"tiers":       []string{"easy"},
		"temperature": 0.0,
		"seed":        42,
		"max_tokens":  64,
	}
	body, _ := json.Marshal(payload)

	resp, err := http.Post(ts.URL+"/api/run", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /api/run failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d, body: %s", resp.StatusCode, string(respBody))
	}

	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Fatalf("failed to decode json: %v", err)
	}
	if fileName, ok := data["file_name"].(string); !ok || fileName == "" {
		t.Errorf("expected file_name in response, got %v", data["file_name"])
	} else {
		_ = os.Remove(filepath.Join(srv.cfg.DataDir, fileName))
	}
	if total, ok := data["total_tasks"].(float64); !ok || total == 0 {
		t.Errorf("expected non-zero total_tasks, got %v", data["total_tasks"])
	}
}

func TestScenariosListEndpoint(t *testing.T) {
	ts, _ := setupTestServer(t)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/api/scenarios")
	if err != nil {
		t.Fatalf("GET /api/scenarios failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Fatalf("failed to decode json: %v", err)
	}
	scenarios, ok := data["scenarios"].([]any)
	if !ok || len(scenarios) != 5 {
		t.Fatalf("expected 5 canonical scenarios (EXP-00..04), got %v", scenarios)
	}
}

func TestScenariosRunEndpoint(t *testing.T) {
	ts, srv := setupTestServer(t)
	defer ts.Close()

	// Test running EXP-04 (Request Config Truncation)
	payload := map[string]any{
		"scenario_id": "EXP-04",
	}
	body, _ := json.Marshal(payload)

	resp, err := http.Post(ts.URL+"/api/scenarios/run", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /api/scenarios/run failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d, body: %s", resp.StatusCode, string(respBody))
	}

	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Fatalf("failed to decode json: %v", err)
	}

	if status, ok := data["status"].(string); !ok || status != "completed" {
		t.Errorf("expected status 'completed', got %v", data["status"])
	}

	fileName, ok := data["file_name"].(string)
	if !ok || fileName == "" {
		t.Errorf("expected file_name in response, got %v", data["file_name"])
	} else {
		_ = os.Remove(filepath.Join(srv.cfg.DataDir, fileName))
	}

	// Verify ground truth isolation
	gt, ok := data["ground_truth"].(map[string]any)
	if !ok || gt["canonical_layer"] != "REQUEST_CONFIG" {
		t.Errorf("expected ground_truth.canonical_layer = REQUEST_CONFIG, got %v", gt)
	}

	// Verify attribution outcome
	attr, ok := data["attribution"].(map[string]any)
	if !ok {
		t.Fatalf("expected attribution in response")
	}
	if pred, _ := attr["canonical_layer"].(string); pred != "REQUEST_CONFIG" {
		t.Errorf("expected predicted canonical_layer to be REQUEST_CONFIG, got %v", pred)
	}

	// Verify evaluation verdict
	eval, ok := data["evaluation"].(map[string]any)
	if !ok || eval["verdict"] != "MATCH" {
		t.Errorf("expected evaluation.verdict = MATCH, got %v", eval)
	}
}
