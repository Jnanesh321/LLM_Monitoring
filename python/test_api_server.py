import subprocess
import time
import urllib.request
import json
import sys
import os

print("=== STARTING GO API SERVER FOR INTEGRATION TESTING ===")
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Start server
proc = subprocess.Popen(
    ["go", "run", "./cmd/model-monitor", "serve", "--mock", "--data-dir", "data"],
    cwd=repo_root,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

time.sleep(3) # wait for server to start

base_url = "http://localhost:8080"
endpoints = [
    ("/api/health", "GET", None),
    ("/api/canaries", "GET", None),
    ("/api/runs", "GET", None),
    ("/api/baseline", "GET", None),
    ("/api/detection", "GET", None),
    ("/api/attribution", "GET", None),
    ("/api/evaluation", "GET", None),
    ("/api/scenarios", "GET", None),
]

all_passed = True

try:
    for path, method, payload in endpoints:
        url = base_url + path
        print(f"\nTesting {method} {url}...")
        try:
            req = urllib.request.Request(url, method=method)
            if payload:
                req.data = json.dumps(payload).encode('utf-8')
                req.add_header('Content-Type', 'application/json')
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
                body = resp.read().decode('utf-8')
                data = json.loads(body)
                print(f"  Status: {status}")
                if path == "/api/health":
                    print(f"  Health status: {data.get('status')}, alias: {data.get('alias')}, mock: {data.get('mock_mode')}")
                elif path == "/api/canaries":
                    print(f"  Canary tasks count: {len(data.get('tasks', []))}")
                    if len(data.get('tasks', [])) != 81:
                        print("  WARNING: expected 81 tasks!")
                        all_passed = False
                elif path == "/api/runs":
                    print(f"  Runs count: {len(data.get('runs', []))}")
                elif path == "/api/baseline":
                    print(f"  Baseline available: {data.get('available')}, batches: {data.get('baseline_batches')}")
                elif path == "/api/detection":
                    det = data.get('detection', {})
                    print(f"  Detection available: {data.get('available')}, detected: {det.get('detected')}, changes count: {len(det.get('changes', []))}")
                elif path == "/api/attribution":
                    attr = data.get('attribution', {})
                    analysis = data.get('analysis', {})
                    print(f"  Attribution available: {data.get('available')}, confidence: {attr.get('confidence')}")
                    print(f"  Analysis predicted layer: {analysis.get('predicted_layer')}, confidence: {analysis.get('confidence')}")
                    evidence = analysis.get('evidence') or []
                    print(f"  Observable evidence count: {len(evidence)}")
                    if evidence:
                        print(f"  Sample evidence: {evidence[:2]}")
        except Exception as e:
            print(f"  FAILED: {e}")
            all_passed = False

    # Test POST /api/run
    print(f"\nTesting POST {base_url}/api/run...")
    try:
        run_payload = {
            "mode": "run",
            "tiers": ["easy"],
            "temperature": 0.0,
            "seed": 42,
            "max_tokens": 512
        }
        req = urllib.request.Request(base_url + "/api/run", data=json.dumps(run_payload).encode('utf-8'), method="POST")
        req.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print(f"  Status: {resp.status}")
            print(f"  Created file: {data.get('file_name')}, total tasks: {data.get('total_tasks')}, passed: {data.get('passed_tasks')}")
    except Exception as e:
        print(f"  FAILED POST /api/run: {e}")
        all_passed = False

    # Test POST /api/scenarios/run (Interactive Experiment Execution with Ground-Truth Isolation)
    print(f"\nTesting POST {base_url}/api/scenarios/run (EXP-04: Request Truncation)...")
    try:
        sc_payload = {"scenario_id": "EXP-04"}
        req = urllib.request.Request(base_url + "/api/scenarios/run", data=json.dumps(sc_payload).encode('utf-8'), method="POST")
        req.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print(f"  Status: {resp.status}")
            print(f"  Simulation Mode: {data.get('simulation_mode')}")
            print(f"  Run ID: {data.get('run_id')}, Total Probes: {data.get('total_probes')}")
            attr = data.get('attribution', {})
            gt = data.get('ground_truth', {})
            eval_res = data.get('evaluation', {})
            print(f"  Attribution Predicted: {attr.get('canonical_layer')}, Ground Truth: {gt.get('canonical_layer')}")
            print(f"  Verdict: {eval_res.get('verdict')} (Attribution Correct: {eval_res.get('attribution_correct')})")
            if not eval_res.get('attribution_correct') or attr.get('canonical_layer') != 'REQUEST_CONFIG':
                print(f"  ERROR: Expected attribution to REQUEST_CONFIG, got {attr.get('canonical_layer')}")
                all_passed = False
            else:
                print("  SUCCESS: Controlled experiment executed, isolated ground-truth validated!")
            # Clean up the scenario run file so subsequent test runs against latest baseline cleanly
            if data.get('file_name'):
                created_sc_path = os.path.join(repo_root, "data", data['file_name'])
                if os.path.exists(created_sc_path):
                    os.remove(created_sc_path)
    except Exception as e:
        print(f"  FAILED POST /api/scenarios/run: {e}")
        all_passed = False

    # Test Behavioral Drift Attribution with Observable Evidence (Truncation Mutation)
    print(f"\nTesting Observable Evidence Attribution with Injected Truncation Run...")
    mut_path = os.path.join(repo_root, "data", "run-20991231T235959Z.json")
    try:
        # Load the latest 81-probe baseline to create a matched truncation candidate
        base_runs = [f for f in os.listdir(os.path.join(repo_root, "data")) if f.startswith("baseline-") and f.endswith(".json")]
        base_runs.sort()
        with open(os.path.join(repo_root, "data", base_runs[-1]), encoding="utf-8") as f:
            base_obs = json.load(f)

        cand_obs = json.loads(json.dumps(base_obs))
        for o in cand_obs:
            if o["canary_id"] in ["TRN-E1", "TRN-E2", "FMT-E1"]:
                o["finish_reason"] = "length"
                o["output_tokens"] = 14
                o["response"] = o["response"][:10]
                for c in o.get("checks", []):
                    if c["type"] in ["finish_reason", "ends_with", "json_valid"]:
                        c["passed"] = False

        with open(mut_path, "w", encoding="utf-8") as f:
            json.dump(cand_obs, f, indent=2)

        with urllib.request.urlopen(base_url + "/api/attribution", timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            analysis = data.get("analysis", {})
            attr = data.get("attribution", {})
            pred = analysis.get("predicted_layer")
            evidence = analysis.get("evidence") or []
            print(f"  Attribution confidence: {attr.get('confidence')}")
            print(f"  Analysis predicted layer: {pred}")
            print(f"  Observable evidence items ({len(evidence)}):")
            for ev in evidence[:5]:
                print(f"    • {ev}")
            if pred != "request_config":
                print(f"  ERROR: expected request_config, got {pred}")
                all_passed = False
            else:
                print("  SUCCESS: Behavioral engine identified request_config with observable evidence!")
    except Exception as e:
        print(f"  FAILED Truncation Attribution Test: {e}")
        all_passed = False
    finally:
        if os.path.exists(mut_path):
            os.remove(mut_path)

finally:
    print("\nShutting down server process...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("Server stopped.")

if all_passed:
    print("\nALL 7 API ENDPOINTS VERIFIED SUCCESSFULLY!")
    sys.exit(0)
else:
    print("\nSOME API ENDPOINTS FAILED!")
    sys.exit(1)
