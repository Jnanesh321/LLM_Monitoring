import json

go = json.load(open('canaries_fixed.json', encoding='utf-8'))
py = json.load(open('python/evaluation/benchmarks/canary_suite.json', encoding='utf-8'))

print("=== PYTHON TASKS (12) ===")
for pt in py['tasks']:
    print(f"[{pt['id']}] cat={pt['category']}, eval={pt['eval_type']}")
    print(f"  Prompt: {pt['prompt']}")
    print(f"  Expected: {pt.get('expected')}")

print("\n=== SELECTED GO TASKS ===")
for gt in go['tasks']:
    if gt['id'] in ['REF-E1', 'REF-E2', 'REF-E3', 'FMT-E1', 'FMT-E2', 'DET-E1', 'CAP-E1', 'CAP-E2', 'INS-E1', 'INS-E2', 'TRU-E1', 'LAT-E1']:
        chk_strs = []
        for c in gt['checks']:
            v = c.get('value') or c.get('values') or c.get('equals') or ''
            chk_strs.append(f"{c['type']}({v})")
        print(f"[{gt['id']}] div={gt['division']}, tier={gt['tier']}")
        print(f"  Prompt: {gt['prompt']}")
        print(f"  Checks: {', '.join(chk_strs)}")

