#!/bin/bash
# start_muse.sh MODE [logfile]: (re)start the Muse-Spark server in tmux session "muse" and wait until it serves
MODE=${1:-base}; LOGF=${2:-muse_$MODE.log}
cd /mnt/ssdraid/project/muse-spark
tmux kill-session -t muse 2>/dev/null; sleep 2
for i in $(seq 1 60); do pgrep -x llama-server >/dev/null || break; sleep 2; done
# tmux does not inherit the caller's environment: pass the knobs explicitly
ENVS="GPU=${GPU:-2} THREADS=${THREADS:-32} CTX=${CTX:-16384} PORT=${PORT:-8020} LOAD_MODE=${LOAD_MODE:-none} NUMACTL=${NUMACTL:-1} PMIN=${PMIN:-0.5} ${EXTRA_ENV:-}"
tmux new-session -d -s muse "cd /mnt/ssdraid/project/muse-spark && env $ENVS NCMOE=${NCMOE:-56} EXTRA_FLAGS='${EXTRA_FLAGS:-}' ./run_muse_cmp.sh $MODE 2>&1 | tee -a $LOGF"
echo "=== $(date +%T) starting $MODE (log $LOGF)"
for i in $(seq 1 240); do
  curl -s -m 5 http://127.0.0.1:8020/v1/models 2>/dev/null | grep -q '"object"' && { echo "=== ready after ~$((i*10))s"; exit 0; }
  tmux has-session -t muse 2>/dev/null || { echo "=== session died"; tail -8 $LOGF | cut -c1-160; exit 1; }
  sleep 10
done
echo "=== not ready after 40 min"; exit 1
