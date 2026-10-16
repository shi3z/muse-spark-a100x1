#!/bin/bash
# Use the spare VRAM: (A) repack off baseline, (B) repack off + 45 GB MoE expert LRU cache (--moe-cache-mib),
# (C) repack on + experts of 6 layers pinned to the GPU (--n-cpu-moe 56). 32 threads everywhere. Waits for other GPU jobs first.
cd /mnt/ssdraid/project/muse-spark
while pgrep -x llama-bench >/dev/null || pgrep -x llama-moe-routing-stats >/dev/null; do sleep 15; done
export GPU=2 THREADS=32 NPRED=128
run() { # TAG EXTRA_FLAGS NCMOE
  export TAG=$1 EXTRA_FLAGS="$2" NCMOE=${3:-62}
  if ! ./start_muse.sh custom "muse_custom${TAG}_gpu2.log"; then echo "=== $TAG: failed to start" | tee -a compare_results.log; return; fi
  echo "=== custom$TAG gpu2 $(date +%T) threads=$THREADS ncmoe=$NCMOE flags='$EXTRA_FLAGS'" | tee -a compare_results.log
  python3 bench_client.py 8020 $NPRED 2>&1 | tee -a compare_results.log
  grep -E "moe_cache" "muse_custom${TAG}_gpu2.log" | tail -3 | cut -c1-200 | tee -a compare_results.log
}
run _t32_norepack "--no-repack" 62
run _t32_norepack_moecache45g "--no-repack --moe-cache-mib 45000" 62
run _t32_ncmoe56 "" 56
tmux kill-session -t muse 2>/dev/null
echo "=== vram tests done $(date +%T)" | tee -a compare_results.log
