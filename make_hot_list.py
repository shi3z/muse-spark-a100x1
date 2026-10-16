#!/usr/bin/env python3
"""Pick the experts to pin on the GPU from routing histograms (CSV layer,expert,count; several files are summed).
Greedy by selection count within a byte budget, optionally per layer group (one budget per device).
usage: make_hot_list.py bytes_per_expert budget_spec hist.csv [hist2.csv ...] > hot_list.csv
  budget_spec: "42" (GiB, all layers) or "0-30:60,31-61:60" (GiB per layer range)"""
import csv, sys, collections
per_expert = float(sys.argv[1]); spec = sys.argv[2]; files = sys.argv[3:]
counts = collections.Counter()
for f in files:
    for r in csv.DictReader(open(f)):
        counts[(int(r['layer']), int(r['expert']))] += int(r['count'])
groups = []
for part in spec.split(','):
    if ':' in part:
        rng, gib = part.split(':'); lo, hi = map(int, rng.split('-'))
    else:
        lo, hi, gib = 0, 10**6, part
    groups.append((lo, hi, float(gib) * 2**30))
total = sum(counts.values()); chosen = []
for lo, hi, budget in groups:
    rows = sorted(((l, e, c) for (l, e), c in counts.items() if lo <= l <= hi and c > 0), key=lambda r: -r[2])
    n = int(budget // per_expert)
    pick = rows[:n]; chosen += pick
    cov = sum(r[2] for r in pick); tot = sum(r[2] for r in rows)
    per_layer = collections.Counter(l for l, e, c in pick)
    print(f"# layers {lo}-{min(hi, max(per_layer) if per_layer else hi)}: {len(pick)} experts, {len(pick)*per_expert/2**30:.1f} GiB, cover {100*cov/max(tot,1):.1f}% of the group's selections; per layer min {min(per_layer.values()) if per_layer else 0} max {max(per_layer.values()) if per_layer else 0}", file=sys.stderr)
cov = sum(r[2] for r in chosen)
print(f"# total {len(chosen)} experts, {len(chosen)*per_expert/2**30:.1f} GiB, cover {100*cov/total:.1f}% of all selections", file=sys.stderr)
print("layer,expert")
for l, e, c in sorted(chosen): print(f"{l},{e}")
