#!/bin/bash
#!/usr/bin/python3.10

grep -v '^#' < thresholds.txt | {
while read -r endpoint thr
do
    printf "endpoint %s, thr %s\n" "${endpoint}" "${thr}"
    python3.10 runCV_KNN.py --thr ${thr} --endpoint ${endpoint} --detailed_names --filtered --binarize
done
}