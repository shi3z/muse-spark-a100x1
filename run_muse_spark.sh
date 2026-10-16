#!/usr/bin/env bash
# Muse-Spark-Open-816B-A42B (Q4_K_S GGUF, 438 GiB) on ONE A100 80 GB + host RAM, llama.cpp master (CUDA + AMX build).
# Dense weights (attention, dense FFN, embeddings) on the GPU, all 62 layers' routed experts in host RAM computed on
# the CPUs (--n-cpu-moe 62); optionally pin the experts of the first N layers to the GPU with EXPERT_GPU_LAYERS.
set -eo pipefail
LLAMA=${LLAMA:-/home/shi3z/llama.cpp-master/build/bin/llama-server}
MODEL=${MODEL:-/mnt/ssd/models/gguf/Muse-Spark-Open-816B-A42B-Q4_K_S/Muse-Spark-Open-816B-A42B-Q4_K_S-00001-of-00022.gguf}
GPU=${GPU:-2}
CTX=${CTX:-16384}
THREADS=${THREADS:-24}
NCMOE=${NCMOE:-62}                 # layers whose experts stay on the CPU (62 = all)
PORT=${PORT:-8020}
EXTRA=${EXTRA:-}
export CUDA_VISIBLE_DEVICES=$GPU
# NUMACTL=1: interleave the (non-mmap) weight allocation over both sockets so the expert stream uses both memory controllers
PREFIX=""
[ "${NUMACTL:-1}" = "1" ] && command -v numactl >/dev/null && PREFIX="numactl --interleave=all"
exec $PREFIX "$LLAMA" -m "$MODEL" -ngl 99 --n-cpu-moe "$NCMOE" -c "$CTX" -t "$THREADS" -tb "$THREADS" -fa on \
  --host 0.0.0.0 --port "$PORT" --jinja --numa distribute --load-mode "${LOAD_MODE:-none}" $EXTRA
