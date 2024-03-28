#!/bin/bash
#!/usr/bin/python3.10

grep -v '^#' < thresholds.txt | {
while read -r endpoint thr
do
    python3.10 matrixGen.py --type daylight --endpoint ${endpoint}
    python3.10 adjGen.py --endpoint ${endpoint}
    python3.10 thrAnalysis.py --endpoint ${endpoint}
done
}

grep -v '^#' < thresholds.txt | {
while read -r endpoint thr
do
    printf "endpoint %s, thr %s\n" "${endpoint}" "${thr}"
    python3.10 runCV.py --thr ${thr} --embedding_size 2 --n_cv 30 --endpoint ${endpoint} --detailed_names --filtered --epochs 1000 --binarize --hidden 10
    python3.10 embeddingGen.py --thr ${thr} --embedding_size 2 --endpoint ${endpoint} --filtered --epochs 1000 --binarize --hidden 10
done
}