#!/bin/bash
source .venv/bin/activate

for endpoint in devtox carcino estro andro skin muta hepa chrom
do
python scripts/cv_knn.py --endpoint ${endpoint}
done