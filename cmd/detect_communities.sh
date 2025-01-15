#!/bin/bash
source .venv/bin/activate

for endpoint in devtox carcino estro andro skin muta hepa chrom
do
python scripts/detect_communities.py --endpoint ${endpoint}
done
