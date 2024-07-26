#!/usr/bin/env python3
import json
import gzip
import argparse
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Build an index')
    parser.add_argument('-i', type=str, help='The input directory.', required=True)
    parser.add_argument('-o', type=str, help='The index will be stored in this directory.', required=True)
    return parser.parse_args()

def transform_queries(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            # Remove 'original_query' field while keeping other fields
            filtered_data = {
                'qid': data['qid'],
                'query': data['query']
            }
            json.dump(filtered_data, outfile)
            outfile.write('\n')

def create_index(input_directory, output_directory):
    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Define file paths
    queries_file_path = os.path.join(input_directory, 'queries.jsonl')
    output_queries_file_path = os.path.join(output_directory, 'queries.jsonl')  # Changed

    # Transform the queries file
    if os.path.exists(queries_file_path):
        transform_queries(queries_file_path, output_queries_file_path)
    else:
        print(f'Warning: {queries_file_path} does not exist.')

    # Create the index file
    documents_file_path = os.path.join(input_directory, 'documents.jsonl.gz')
    index_file_path = os.path.join(output_directory, 'pseudo_index.jsonl.gz')

    with gzip.open(documents_file_path, 'rt') as documents, \
         gzip.open(index_file_path, 'wt') as f:
        for doc in documents:
            doc = json.loads(doc)
            json.dump({'id': doc['docno'], 'text': doc['text']}, f)
            f.write('\n')

    print('Done: Index stored.')

if __name__ == '__main__':
    args = parse_args()
    create_index(args.i, args.o)

