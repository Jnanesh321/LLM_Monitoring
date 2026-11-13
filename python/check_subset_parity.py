import json

f = json.load(open('data/shared_canary_fixture.json', encoding='utf-8'))
subset = [t for t in f['tasks'] if t['in_common_subset']]

print(f"Total subset tasks: {len(subset)}")
for t in subset:
    print(f"\nID: {t['id']} | Origin: {t['origin']} | Cat: {t['category']}")
    print(f"  Prompt: {t['prompt']}")
    print(f"  Go checks: {t['checks']}")
    print(f"  Python eval: {t['python_eval']}")

