#!/bin/bash
#!/usr/bin/python3.10

grep -v '^#' < thresholds.txt | {
while read -r endpoint thr
do
    printf "endpoint %s, thr %s\n" "${endpoint}" "${thr}"
    python3.10 embeddingGen.py --thr ${thr} --embedding_size 2 --endpoint ${endpoint} --filtered --epochs 0 --binarize --hidden 10
done
}