package main

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

func (s *apiServer) handleEvaluation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	h2hPath := findEvaluationFile(
		filepath.Join(s.cfg.DataDir, "..", "results", "head_to_head_results_v2.json"),
		filepath.Join("results", "head_to_head_results_v2.json"),
		"head_to_head_results_v2.json",
	)
	evalPath := findEvaluationFile(
		filepath.Join(s.cfg.DataDir, "evaluation.json"),
		filepath.Join("data", "evaluation.json"),
		"evaluation.json",
	)
	multiQueryPath := findEvaluationFile(
		filepath.Join(s.cfg.DataDir, "evaluation", "multi_query_suite_latest.json"),
		filepath.Join("data", "evaluation", "multi_query_suite_latest.json"),
		"multi_query_suite_latest.json",
	)

	var h2hRaw []byte
	if h2hPath != "" {
		h2hRaw, _ = os.ReadFile(h2hPath)
	}

	var evalRaw []byte
	if evalPath != "" {
		evalRaw, _ = os.ReadFile(evalPath)
	}

	var multiQueryRaw []byte
	if multiQueryPath != "" {
		multiQueryRaw, _ = os.ReadFile(multiQueryPath)
	}

	queryFile := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("file")))
	if queryFile == "multi_query" || queryFile == "behavioural" || queryFile == "suite" {
		if len(multiQueryRaw) > 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(multiQueryRaw)
			return
		}
	} else if queryFile == "head_to_head" || queryFile == "h2h" {
		if len(h2hRaw) > 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(h2hRaw)
			return
		}
	} else if queryFile == "evaluation" || queryFile == "live" {
		if len(evalRaw) > 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(evalRaw)
			return
		}
	}

	var h2hData any
	if len(h2hRaw) > 0 {
		_ = json.Unmarshal(h2hRaw, &h2hData)
	}

	var evalData any
	if len(evalRaw) > 0 {
		_ = json.Unmarshal(evalRaw, &evalData)
	}

	var multiQueryData any
	if len(multiQueryRaw) > 0 {
		_ = json.Unmarshal(multiQueryRaw, &multiQueryData)
	}

	resp := map[string]any{
		"available":            h2hData != nil || evalData != nil || multiQueryData != nil,
		"multi_query_suite":    multiQueryData,
		"head_to_head":         h2hData,
		"head_to_head_results": h2hData,
		"evaluation":           evalData,
		"live_evaluation":      evalData,
		"source_files": map[string]string{
			"multi_query_suite": multiQueryPath,
			"head_to_head":      h2hPath,
			"evaluation":        evalPath,
		},
	}

	if h2hMap, ok := h2hData.(map[string]any); ok {
		for k, v := range h2hMap {
			if _, exists := resp[k]; !exists {
				resp[k] = v
			}
		}
	}

	if evalMap, ok := evalData.(map[string]any); ok {
		resp["live_experiment_id"] = evalMap["experiment_id"]
		resp["live_baseline_model"] = evalMap["baseline_model"]
		resp["live_model_b"] = evalMap["model_b"]
		resp["live_model_c"] = evalMap["model_c"]
	}

	writeJSON(w, http.StatusOK, resp)
}

func findEvaluationFile(candidates ...string) string {
	for _, c := range candidates {
		if c == "" {
			continue
		}
		clean := filepath.Clean(c)
		if fi, err := os.Stat(clean); err == nil && !fi.IsDir() {
			return clean
		}
	}
	cwd, err := os.Getwd()
	if err == nil {
		for _, name := range candidates {
			dir := cwd
			for depth := 0; depth < 5; depth++ {
				cand := filepath.Join(dir, name)
				if fi, err := os.Stat(cand); err == nil && !fi.IsDir() {
					return cand
				}
				parent := filepath.Dir(dir)
				if parent == dir {
					break
				}
				dir = parent
			}
		}
	}
	return ""
}
