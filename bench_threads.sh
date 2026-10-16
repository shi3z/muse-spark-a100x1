#!/usr/bin/env bash
# Thread-count sweep for the CPU expert path (one model load, several -t values). Needs the server stopped first.
# usage: GPU=2 THREADS_LIST=12,24,32,48 ./bench_threads.sh
set -eo pipefail
MODEL=/mnt/ssd/models/gguf/Muse-Spark-Open-816B-A42B-Q4_K_S/Muse-Spark-Open-816B-A42B-Q4_K_S-00001-of-00022.gguf
export CUDA_VISIBLE_DEVICES=${GPU:-2}
exec numactl --interleave=all /home/shi3z/llama.cpp-master/build/bin/llama-bench -m "$MODEL" -ngl 99 -ncmoe 62 -fa on \
  --load-mode none --numa distribute -t "${THREADS_LIST:-12,24,32,48}" -p "${NPROMPT:-64}" -n "${NGEN:-32}" -r "${REPS:-2}" -o md
