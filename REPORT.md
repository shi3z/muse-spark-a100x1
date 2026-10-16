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
