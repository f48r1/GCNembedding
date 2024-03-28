#!/bin/bash
#!/usr/bin/python3.10

for endpoint in devtox carcino estro andro skin muta hepa chrom
do
python3.10 matrixGen.py --endpoint ${endpoint}
done
