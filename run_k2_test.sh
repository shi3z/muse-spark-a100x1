#!/bin/bash
# Diagnostic: top-k 8 -> 2 experts (quality irrelevant). If decode barely changes, the expert matmul is not the bottleneck.
cd /mnt/ssdraid/project/muse-spark
while pgrep -f "^python3 bench_client.py" >/dev/null; do sleep 10; done
export GPU=2 THREADS=32 NPRED=64
run() { export TAG=$1 EXTRA_ENV="$2" EXTRA_FLAGS="$3" NCMOE=${4:-62}
  if ! ./start_muse.sh custom "muse_custom${TAG}_gpu2.log"; then echo "=== $TAG: failed to start" | tee -a compare_results.log; return; fi
  echo "=== custom$TAG gpu2 $(date +%T) threads=$THREADS ncmoe=$NCMOE env='$EXTRA_ENV' flags='$EXTRA_FLAGS'" | tee -a compare_results.log
  python3 bench_client.py 8020 $NPRED 2>&1 | tee -a compare_results.log; }
run _k2_t32 "" "--override-kv muse-spark.expert_used_count=int:2" 62
run _k2_t16 "" "--override-kv muse-spark.expert_used_count=int:2 -t 16 -tb 16" 62
tmux kill-session -t muse 2>/dev/null
echo "=== k2 tests done $(date +%T)" | tee -a compare_results.log
