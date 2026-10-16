#!/bin/bash
# After the running benchmark: (D) hot split with 6 complete layers (same bytes as --n-cpu-moe 56, isolates the split's overhead),
# (E) --no-host so the experts land in CPU_REPACK (repacked Q4_K kernels) with 6 layers on the GPU.
cd /mnt/ssdraid/project/muse-spark
while pgrep -f "bench_client.py 8020" >/dev/null; do sleep 10; done
export GPU=2 THREADS=32 NPRED=128
run() { # TAG EXTRA_ENV EXTRA_FLAGS NCMOE
  export TAG=$1 EXTRA_ENV="$2" EXTRA_FLAGS="$3" NCMOE=${4:-62}
  if ! ./start_muse.sh custom "muse_custom${TAG}_gpu2.log"; then echo "=== $TAG: failed to start" | tee -a compare_results.log; return; fi
  echo "=== custom$TAG gpu2 $(date +%T) threads=$THREADS ncmoe=$NCMOE env='$EXTRA_ENV' flags='$EXTRA_FLAGS'" | tee -a compare_results.log
  python3 bench_client.py 8020 $NPRED 2>&1 | tee -a compare_results.log
}
run _hot42g "LLAMA_MOE_HOT_LIST=/mnt/ssdraid/project/muse-spark/hot_list_42g.csv" "" 62
run _hotfull6 "LLAMA_MOE_HOT_LIST=/mnt/ssdraid/project/muse-spark/hot_list_full6.csv" "" 62
run _t32_ncmoe56_nohost "" "--no-host" 56
tmux kill-session -t muse 2>/dev/null
echo "=== hot tests 2 done $(date +%T)" | tee -a compare_results.log
