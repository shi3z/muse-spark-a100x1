#!/usr/bin/env bash
# run one of three configurations of Muse-Spark on GPU 5 for comparison: MODE=base | dflash | dspark
MODE=${1:-base}
export GPU=${GPU:-2} CTX=${CTX:-16384} THREADS=${THREADS:-24} PORT=${PORT:-8020}
D=/mnt/ssd/models/gguf/Muse-Spark-drafts
case "$MODE" in
  base)   EXTRA="" ;;
  dflash) EXTRA="--spec-type draft-dflash -md $D/Muse-Spark-Open-DFlash2-BF16.gguf --spec-draft-ngl 99 --spec-draft-n-max 16" ;;
  dspark) EXTRA="--spec-type draft-dspark -md $D/Muse-Spark-Open-DSpark-BF16.gguf --spec-draft-ngl 99 --spec-draft-n-max 7 --spec-draft-p-min ${PMIN:-0.5}" ;;
esac
export EXTRA
exec /mnt/ssdraid/project/muse-spark/run_muse_spark.sh
