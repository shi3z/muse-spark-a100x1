#!/bin/bash
# Sequentially benchmark base / dflash / dspark (one 438 GB server at a time); results in compare_results.log
# usage: GPU=2 MODES="dflash dspark" NPRED=160 ./compare_modes.sh
cd /mnt/ssdraid/project/muse-spark
export GPU=${GPU:-2}
for MODE in ${MODES:-base dflash dspark}; do
  LOGF=muse_${MODE}${TAG:-}_gpu${GPU}.log
  if ! ./start_muse.sh "$MODE" "$LOGF"; then
    echo "=== $MODE: server failed to start" | tee -a compare_results.log
    grep -E "error|Error|abort|Assert" "$LOGF" | tail -3 | cut -c1-200 | tee -a compare_results.log; continue
  fi
  echo "=== $MODE$TAG gpu$GPU $(date +%T) ${EXTRA_ENV:-}" | tee -a compare_results.log
  python3 bench_client.py 8020 ${NPRED:-160} 2>&1 | tee -a compare_results.log
  # acceptance stats printed by llama-server (draft accepted / drafted)
  grep -E "draft|accept|spec" "$LOGF" | tail -4 | cut -c1-200 | tee -a compare_results.log
done
echo "=== compare done $(date +%T)" | tee -a compare_results.log
