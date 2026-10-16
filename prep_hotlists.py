#!/usr/bin/env python3
"""Compute bytes per expert for a Muse-Spark GGUF (sum of one expert's up/gate/down slabs) and write hot lists:
1 GPU (one budget) and 2 GPUs (per-layer-group budgets) from the decode histograms.
usage: prep_hotlists.py <gguf shard 1> <tag> [gib_1gpu=42] [gib_2gpu_each=60]"""
import sys, glob, os, subprocess
sys.path.insert(0, '/home/shi3z/llama.cpp-master/gguf-py')
from gguf import GGUFReader
shard, tag = sys.argv[1], sys.argv[2]
g1 = float(sys.argv[3]) if len(sys.argv) > 3 else 42
g2 = float(sys.argv[4]) if len(sys.argv) > 4 else 60
per = {}
for f in sorted(glob.glob(os.path.join(os.path.dirname(shard), '*.gguf'))):
    for t in GGUFReader(f).tensors:
        if t.name.startswith('blk.5.ffn_') and 'exps' in t.name:
            per[t.name] = (t.tensor_type.name, t.n_bytes / int(t.shape[-1]))
    if len(per) == 3: break
bpe = sum(v[1] for v in per.values())
print(f"{tag}: " + ", ".join(f"{k.split('.')[2]}={v[0]} {v[1]/1e6:.1f} MB" for k, v in per.items()) + f" -> {bpe/1e6:.1f} MB per expert, {bpe*256*62/2**30:.0f} GiB of experts", file=sys.stderr)
H = ['decode_hist_train.csv', 'decode_hist_train2.csv']
for name, spec in ((f'hot_list_{tag}_1gpu.csv', f'{g1}'), (f'hot_list_{tag}_2gpu.csv', f'0-30:{g2},31-61:{g2}')):
    with open(name, 'w') as out:
        subprocess.run(['./make_hot_list.py', str(bpe), spec] + H, stdout=out, check=True)
    print(f"wrote {name}", file=sys.stderr)
