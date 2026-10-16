#!/bin/bash
# detached download chain: IQ2_XXS to the SSD, then IQ3_XXS to the RAID (aria2, IPv4, pinned commit lists already built)
TOK=$(cat ~/.cache/huggingface/token)
for D in /mnt/ssd/models/gguf/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS /mnt/raid6/models/gguf/Muse-Spark-Open-816B-A42B-UD-IQ3_XXS; do
  cd $D || continue
  echo "=== start $(basename $D) $(date +%T)" >> download_iq.status
  aria2c --disable-ipv6 -c -i aria2.txt -j 3 -x 4 -s 4 -k 64M --timeout=30 --lowest-speed-limit=64K --max-tries=0 --file-allocation=falloc --header="Authorization: Bearer $TOK" --console-log-level=warn --summary-interval=0 > aria2.log 2>&1
  echo "=== done $(basename $D) $(date +%T) rc=$? size=$(du -sh . | cut -f1) gguf=$(ls *.gguf 2>/dev/null | wc -l) partial=$(ls *.aria2 2>/dev/null | wc -l)" >> download_iq.status
done
