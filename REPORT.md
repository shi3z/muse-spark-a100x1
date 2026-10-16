# Muse-Spark-Open-816B-A42B on one A100 80 GB — status report (2026-10-16)

## Goal
Run meta-models/Muse-Spark-Open-816B-A42B (816B total, 42B active, 256 experts top-8, 62 layers)
on a single A100 80 GB, using the Strata idea: dense weights on the GPU, routed experts in host RAM
computed on the CPUs. Then compare three decoding modes: plain, DFlash2 draft, and our own DSpark draft.

## What runs today
* Weights: ggml-org Q4_K_S GGUF (22 shards, 437.8 GiB) in `/mnt/ssd/models/gguf/Muse-Spark-Open-816B-A42B-Q4_K_S/`.
* Runtime: llama.cpp master (build 11675, `muse-spark` arch merged 2026-10-15) built with CUDA sm80 + AMX,
  worktree `~/llama.cpp-master` (branch `muse-dspark`).
* Split: `-ngl 99 --n-cpu-moe 62` → attention, dense FFN, embeddings, output head on the GPU
  (~24 GB with a 16K context), all 62 layers' experts (~422 GiB) in host RAM.
* Host memory layout matters: `--load-mode none` (no mmap) + `numactl --interleave=all` + `--numa distribute`
  + `kernel.numa_balancing=0` → the 433 GB are split 217/217 GB across the two sockets (checked with `numastat -p`).
* Launchers: `run_muse_spark.sh` (server), `run_muse_cmp.sh MODE` (base|dflash|dspark), `start_muse.sh MODE`
  (restart in tmux `muse`, wait until it serves), `compare_modes.sh` (sequential benchmark), `bench_client.py`.

## Measurements (GPU 2 exclusive, 24 threads, 128 generated tokens, greedy)
| mode | prompt tok | prefill tok/s | decode tok/s | draft acceptance |
|---|---|---|---|---|
| base | 158 | 12.1 | 5.67 | – |
| base | 158 | 9.6 | 6.50 | – |
| base | 3009 | 39.6 | 5.96 | – |
| dflash2 (n_max 7) | 158 | 11.7 | 5.62 | 102/166 (61%), mean accepted len 5.25 |
| dflash2 (n_max 7) | 158 (code) | 8.6 | 3.18 | 82/303 (27%) |
| dflash2 (n_max 7) | 3009 | 39.2 | 5.54 | 100/187 (53%) |
| dspark, ours (n_max 7, p_min 0.5) | 158 | 11.7 | 4.16 | 66/179 (37%), mean len 2.22 |
| dspark, ours (n_max 7, p_min 0.5) | 158 (code) | 8.5 | 5.01 | 35/69 (51%), mean len 1.73 |
| dspark, ours (n_max 7, p_min 0.5) | 3009 | 38.6 | 5.03 | 54/106 (51%), mean len 2.12 |
| dspark, ours (n_max 7, p_min 0) | 158 | 11.4 | 2.78 | 73/364 (20%), 2.46 tok/cycle |
| dspark, ours (n_max 7, p_min 0) | 158 (code) | 9.4 | 2.06 | 55/489 (11%) |
| dspark, ours (n_max 7, p_min 0) | 3009 | 38.7 | 2.30 | 61/441 (14%) |

All three modes produce identical greedy text (speculative decoding is lossless here).

## Finding 1: block drafting does not pay with CPU-resident experts
DFlash2 accepts 5.25 tokens per verification step on the short prompt, yet decode speed is unchanged
(5.6 vs 5.7 tok/s) and it is *slower* on code (3.2 tok/s at 27% acceptance). With the experts on the CPU
the cost of a verification step grows almost linearly with the number of tokens in the block: each extra
token activates its own 8 experts per layer (up to 64 distinct experts per layer for a block of 8 instead of 8),
and the CPU expert matmul is compute/latency bound rather than weight-bandwidth bound at this size
(base decode moves ~14 GB of expert weights per token = ~80 GB/s, far below the ~250 GB/s the two sockets deliver).
On a GPU-resident model the block is almost free (the weights are read once per step); here it is not.
Consequence: speculative decoding of any kind (DFlash2, DSpark) can only win on this box if the expert
path becomes cheaper per *batch* than per *token*, i.e. a Strata-style GPU hot-expert cache or a much
faster CPU expert kernel. The draft work is still useful for validating our DSpark port, but it is not the
path to higher tok/s on 1xA100 + CPU experts.

## Finding 2: our DSpark port matches the published acceptance range
Reference (model cards, 7 draft tokens, temperature 1.0 / top-p 0.95, SGLang/vLLM):
* DSpark: acceptance 28.5% (MT-Bench chat) to 42.0% (GSM8K), 2.7-3.9 tokens per cycle.
* DFlash2: 5.37 tokens per cycle on GSM8K, 3.9-4.2 on AIME/GPQA.

Ours (greedy, llama.cpp, block verification):
* DFlash2 (upstream conversion): 5.25 tokens per cycle on the CRISPR prompt, 61% acceptance -> matches the card.
* DSpark (our converter + backbone in `src/models/dflash.cpp`): 37-51% acceptance with p_min 0.5 (drafts cut short
  by the confidence cutoff, mean length 1.7-2.2). Acceptance per drafted token is inside the published range, so the
  Muse backbone port (gated attention output, weightless post norms, norm-preserving residual gates, delta-form norms,
  qk scale 5.473, normed embeddings) is functionally right. Runs with p_min 0 (full 7-token drafts) and with the
  embedding norm disabled are below.

## Finding 3: the CPU expert path is thread-bound, not bandwidth-bound (llama-bench, tg32, 2 reps)
| experts on CPU (layers) | threads | decode tok/s | prefill pp32 tok/s |
|---|---|---|---|
| 62 (all) | 16 | 4.61 | 5.23 |
| 62 (all) | 24 | 5.48 | 5.13 |
| 62 (all) | 32 | 7.19 | 5.17 |
| 62 (all) | 48 | 0.25 (thread starvation: never use every logical CPU) | 5.07 |
| 56 (6 layers' experts on the GPU, +43.5 GiB VRAM) | 16 | 5.26 | 5.94 |
| 56 | 24 | 6.26 | 5.65 |
| 56 | 32 | 5.75 +-1.54 (noisy) | 5.78 |

* Finer sweep with 56 CPU layers, 3 reps: 28 thr 5.99 +-1.42, 32 thr 6.07 +-0.80, 36 thr 7.34 +-0.43, 40 thr 6.23 +-0.75 tok/s
  (noisy: six Blender render processes of the H3 fleet plus two kcompactd threads were using ~8 cores during all these
  measurements, so every number here is a lower bound for an idle box; 32-36 threads is the usable range, 48 collapses).
* Hyperthreads help (24 -> 32 threads: +31%), so the limit is per-core throughput of the Q4_K expert kernels, not DRAM.
* Moving 6 of 62 layers' experts to the GPU buys the expected ~+14% (6/62 fewer CPU layers).
* Prefill of a 32-token batch runs at the same tok/s as decode: with experts on the CPU, a batch costs ~linearly per token
  (every expert of every layer is touched by a 32-token batch), which is the same effect that kills speculative decoding here.

## Finding 4: using the spare VRAM (server, 32 threads, 128 tokens, GPU 2; CPUs shared with the fleet's renders)
| config | short decode | code decode | long decode | GPU memory |
|---|---|---|---|---|
| all experts on CPU, 24 thr (baseline) | 5.67 | 6.50 | 5.96 | 24.5 GB |
| repack off, 32 thr | 4.18 (overlapped with another job) | 5.78 | 5.08 | 24.5 GB |
| repack off + `--moe-cache-mib 45000` (llama.cpp's GPU LRU cache of host experts), 32 thr | 4.15 | 4.45 | 3.78 | ~70 GB |
| **repack on, experts of 6 layers on the GPU (`--n-cpu-moe 56`), 32 thr** | **7.58** | **6.37** | **7.33** | ~68 GB |

* llama.cpp master already has the Strata idea built in: `--moe-cache-mib` keeps an LRU of host-resident experts on the GPU
  (PR #29887, 2026-10-07) and uploads the misses over PCIe. It does not pay here (the cache's own hit/miss statistics are printed at shutdown and were not captured): with 8 of 256
  experts per layer per token and a cache that holds ~11% of the experts, most selections must miss, and uploading one expert (29 MB, ~1.2 ms over PCIe)
  costs more than computing it on the CPU (~0.35 ms). It also needs the repacked Q4_K kernels off (the cache only reads plain
  host buffers), which by itself costs ~15%. A static placement of the hottest experts would only help if routing were
  strongly skewed; see the routing histogram below.
* Pinning whole layers' experts on the GPU keeps the repacked CPU kernels and removes 6/62 of the CPU work; with 32 threads
  this is the best configuration measured: **7.3-7.6 tok/s** decode (+30% over the 24-thread baseline), prefill 13-43 tok/s.
  It is now the default of `run_muse_spark.sh` (THREADS=32, NCMOE=56).

## Finding 5: routing is strongly skewed, so a static hot-expert cache should pay
`llama-moe-routing-stats` (our tool, patch 0003) over a 2,032-token mixed corpus (prose, code, Japanese), prefill routing:
* per layer, the 27 hottest experts (= what 45 GB holds) receive 69% of all selections on average (27% in the first layers,
  up to 100% mid-network); the single hottest expert gets 9% (max 12%); ~61 of 256 experts per layer are never selected.
* a greedy global pick within 42 GiB (`make_hot_list.py`, 1,488 experts, 14-35 per layer) covers ~68% of selections.
* unlike the LRU cache, a *static* split does not upload anything per token: hot experts are computed on the GPU, the CPU
  computes only the cold ones (a sentinel id makes the CPU kernel skip the slot). Expected: CPU work drops to ~1/3,
  so decode should land around 2x the current 7.5 tok/s if the overheads stay small.

Implemented (patch 0004, `LLAMA_MOE_HOT_LIST=hot_list.csv`): the loader gathers the listed experts into GPU tensors, the
MoE graph runs two expert chains for decode-sized batches (hot on the GPU from the copy, cold on the host; the slots of
the other side point at a dummy expert and are masked out, so all ids stay valid for the large-batch prefill path, which
is unchanged). Output is identical to the plain run.

**Result: no gain.** 42 GiB hot list, 32 threads, all 62 layers host-resident: 6.88 / 7.20 / 6.32 tok/s (short / code / long)
versus 7.58 / 6.37 / 7.33 with 6 whole layers on the GPU and ~7.2 all-CPU. Moving ~2/3 of the expert selections off the
CPU did not shorten the token time, so by this point the CPU expert matmul is no longer what the token time is made of.
Diagnostics that followed:
* top-k 8 -> 2 (`--override-kv muse-spark.expert_used_count=int:2`): 15.5-15.9 tok/s. So the expert matmul *is* still the
  cost at k=8 (about 13 ms per expert-per-token across the 62 layers, ~38 ms fixed), and the split should have paid.
* a counter in the CPU kernel (`GGML_MMID_STATS=1`): with the hot list the host still computed 7.4 distinct experts per
  matmul instead of 8 (expected ~3.6). The split works mechanically; the hot *set* was wrong.
* a kernel-side histogram of decode routing (`GGML_MMID_HIST=path`): decode routing is as skewed as prefill routing
  (top-27 experts per layer = 60% of selections), but the experts that are hot during prefill of a text corpus cover only
  20% of the selections made while *generating*. The two distributions differ a lot; a hot list must be built from
  decode routing of a representative workload (see `collect_decode_prompts.py`, `make_hot_list.py`).

Note (found while wiring the split): with `--load-mode none` and a GPU present, llama.cpp puts the CPU-overridden expert
tensors into the CUDA *pinned host* buffer (`CUDA_Host`), not into the `CPU_REPACK` buffer, so the repacked Q4_K kernels were
never in use in any run above; the `--no-repack` row differs from the default only by noise. `--no-host` would route the
experts to `CPU_REPACK` instead (to be measured).

## Finding 6: the host kernel's cost is per row, not per expert; with that fixed the split pays
The CPU expert kernel (Q4_K weights x Q8_K activations) is compute-bound: its time scales with the number of (slot, token)
rows it computes, not with how many distinct experts those rows touch. Folding hot slots into a dummy expert therefore
saved nothing (8.00 rows per matmul either way). Skipping the slot in the kernel (sentinel id -1, zero output row) does:

| config (32 threads, 128 tokens, GPU 2) | rows / host matmul | short | code | long |
|---|---|---|---|---|
| all experts on CPU (reference, llama-bench tg32) | 8.00 | ~7.2 | | |
| 6 layers' experts on the GPU (`--n-cpu-moe 56`) | 8.00 (56 layers) | 7.58 | 6.37 | 7.33 |
| hot split, prefill-trained list, dummy fold | 8.00 | 6.88 | 7.20 | 6.32 |
| hot split, decode-trained list, dummy fold | 8.00 | 6.93 | 6.93 | 6.81 |
| **hot split, decode-trained list, sentinel skip** | **4.34** | **9.18** | **8.28** | **8.41** |

* The decode-trained 42 GiB list (1,488 experts, 10-32 per layer, from 10 training prompts x 100 tokens) covers 46.7% of
  the training selections and 43.6% of the held-out benchmark prompts' selections, so it generalizes.
* Token time model (from the k=2 run): ~38 ms fixed + ~12.7 ms per row-set across the 62 layers; 4.34 rows -> ~93 ms
  predicted, 109-121 ms measured. The remaining gap is the extra graph nodes and the GPU hot chain.
* Output text is identical to the plain run on all three prompts.
* Usage: `LLAMA_MOE_HOT_LIST=hot_list.csv` (lines `layer,expert`), `--n-cpu-moe 62`; the list comes from
  `GGML_MMID_HIST=path` on a plain run driven by `collect_decode_prompts.py`, then `make_hot_list.py hist.csv 42 30300000`.
  `run_muse_cmp.sh hot` launches this configuration.

Where the next gains are: more VRAM coverage is the straight path (a second A100 would hold ~2/3 of the selections),
a cheaper host kernel (the repacked Q4_K path, never active here because the experts sit in the CUDA pinned buffer; or
AMX int8), and the 38 ms fixed part (62 GPU<->CPU hand-offs per token).

## Two A100s (GPUs 2 and 3, `-ts 31,31`, 32 threads, 128 tokens)
Layers 0-30 run on GPU 2 and 31-61 on GPU 3; each GPU holds its half of the dense weights plus 60 GiB of its layers' hottest
experts (`hot_list_2gpu120g.csv`: 4,252 experts, built from two decode histograms, 30 prompts x 100-120 tokens; it covers
73.6% of the training selections and 71.6% of the held-out benchmark prompts' selections). GPU memory: 75.6 GB / 66.4 GB.

| config | rows / host matmul | short | code | long |
|---|---|---|---|---|
| 1 GPU, hot split 42 GiB | 4.34 | 9.18 | 8.28 | 8.41 |
| **2 GPUs, hot split 2 x 60 GiB** | **3.15** | **12.19** | **11.47** | **11.05** |

| 2 GPUs, whole layers: experts of 8 layers per GPU (116 GiB), 46 layers on the CPU | 8.00 | 8.12 | 8.11 | 8.06 |
| 2 GPUs, hot split, 24 threads | | 11.69 | 11.02 | 10.17 |
| 2 GPUs, hot split, 32 threads, `--poll 100` | | 12.39 | 11.54 | 9.73 |
| 2 GPUs, hot split, 40 threads | | 8.02 | 6.58 | 7.72 |

Output identical to the plain run. The same 116-120 GiB of VRAM buys +45% as hot experts versus +10% as whole layers.
Thread count and polling no longer matter (24 vs 32 within noise, 40 collapses under the fleet's render load).
The token is now ~82-90 ms. Fitting the runs (all-CPU 62 layers, 46 CPU layers, k=2) gives ~40-48 ms that does not
depend on the number of CPU layers or rows: GPU dense work and per-kernel launch overhead (with a CPU split in every layer
llama.cpp cannot use CUDA graphs, so each of the ~2,500 small kernels per token pays its launch latency). That, not
expert coverage, is the next lever: fewer hand-offs (e.g. grouping layers) or CUDA-graph capture of the per-layer GPU segments.
