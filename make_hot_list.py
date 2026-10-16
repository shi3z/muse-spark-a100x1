#!/usr/bin/env python3
"""Pick the experts to pin on the GPU from a routing histogram (llama-moe-routing-stats CSV).
Greedy by selection count across all layers (every expert costs the same bytes), within a byte budget.
usage: make_hot_list.py routing.csv budget_gib bytes_per_expert > hot_list.csv   (lines: layer,expert)"""
import csv, sys
rows = [(int(r['layer']), int(r['expert']), int(r['count'])) for r in csv.DictReader(open(sys.argv[1]))]
budget = float(sys.argv[2]) * 2**30
per_expert = float(sys.argv[3])
n = int(budget // per_expert)
rows.sort(key=lambda r: -r[2])
chosen = [r for r in rows[:n] if r[2] > 0]
total = sum(r[2] for r in rows); covered = sum(r[2] for r in chosen)
per_layer = {}
for l, e, c in chosen: per_layer[l] = per_layer.get(l, 0) + 1
print(f"# {len(chosen)} experts, {len(chosen)*per_expert/2**30:.1f} GiB, cover {100*covered/total:.1f}% of selections; per layer min {min(per_layer.values())} max {max(per_layer.values())}", file=sys.stderr)
print("layer,expert")
for l, e, c in sorted(chosen): print(f"{l},{e}")
