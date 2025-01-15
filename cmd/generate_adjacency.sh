#!/bin/bash
source .venv/bin/activate

for endpoint in devtox carcino estro andro skin muta hepa chrom
do
python scripts/generate_adjacency.py --endpoint ${endpoint}
done
