#!/bin/bash
source .venv/bin/activate

for endpoint in devtox carcino estro andro skin muta hepa chrom
do
python scripts/generate_matrix.py --endpoint ${endpoint}
done
