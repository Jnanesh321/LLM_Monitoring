import glob, json

for f in sorted(glob.glob('python/results/*.json')):
    with open(f) as fp:
        try: d = json.load(fp)
        except Exception: continue
    s = d.get('summary', {})
    mem = s.get('memory', {})
    vram = mem.get('peak_gpu_vram_mib', 0)
    tasks = len(d.get('results', []))
    if tasks == 12 and vram > 2000:
        print(f"12-task high vram: {f} | vram={vram} | passed={s.get('passed_tasks')}/12 | lat={s.get('avg_latency_ms')}")
