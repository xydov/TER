#!/bin/sh

python3 /app/indexing.py --inputDataset "$inputDataset" --outputDir "$outputDir"
python3 /app/retrieve.py 
python3 /app/re-rank.py 

