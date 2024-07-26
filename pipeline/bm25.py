#!/usr/bin/env python3
import json
import argparse
import gzip
from rank_bm25 import BM25Okapi
from collections import defaultdict
import os
def parse_args():
    parser = argparse.ArgumentParser(description='Retrieve documents using BM25')
    parser.add_argument('-i', type=str, help='The input directory.', required=True)
    parser.add_argument('-o', type=str, help='The output file.', required=True)
    return parser.parse_args()

def load_index(index_file):
    index = {}
    with gzip.open(index_file, 'rt') as f:
        for line in f:
            doc = json.loads(line)
            index[doc['id']] = doc['text']
    return index

def build_bm25_model(index):
    # Tokenize the documents
    tokenized_corpus = [doc.lower().split() for doc in index.values()]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def retrieve_queries(bm25, index, queries_file, output_file):
    with open(queries_file, 'r') as f, open(output_file, 'w') as out:
        queries = [json.loads(line) for line in f]
        
        for query in queries:
            qid = query['qid']
            query_text = query['query']
            tokenized_query = query_text.lower().split()
            
            # Compute BM25 scores
            scores = bm25.get_scores(tokenized_query)
            
            # Create a ranked list of document IDs
            ranked_docs = sorted(
                [(doc_id, score) for doc_id, score in zip(index.keys(), scores)],
                key=lambda x: x[1], reverse=True
            )
            
            # Write results to the output file
            for rank, (doc_id, score) in enumerate(ranked_docs):
                out.write(f"{qid} Q0 {doc_id} {rank + 1} {score:.6f} STANDARD\n")

if __name__ == '__main__':
    args = parse_args()
    
    # Define file paths
    index_file = os.path.join(args.i, 'pseudo_index.jsonl.gz')
    queries_file = os.path.join(args.i, 'queries.jsonl')
    output_file = args.o
    
    # Load index and build BM25 model
    index = load_index(index_file)
    bm25 = build_bm25_model(index)
    
    # Retrieve documents for queries and write to output file
    retrieve_queries(bm25, index, queries_file, output_file)

