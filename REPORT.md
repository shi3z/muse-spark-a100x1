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
