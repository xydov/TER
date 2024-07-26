# indexing.py

import pyterrier as pt
import ir_datasets
import pandas as pd
import os

def document_generator(dataset):
    for doc in dataset.docs_iter():
        yield {
            'docno': doc.doc_id,
            'text': doc.text
        }

def create_index(dataset, index_path='./index'):
    pt.init()  # Initialize PyTerrier
    
    indexer = pt.IterDictIndexer(index_path, meta=['docno', 'text'], meta_lengths=[20, 4096])
    index_ref = indexer.index(document_generator(dataset))
    return index_ref, index_path



if __name__ == "__main__":
    dataset = ir_datasets.load("vaswani")
    index_ref, index_path = create_index(dataset)
    
    # Save the index reference path to a file
    index_ref_path = './index_ref.txt'
    with open(index_ref_path, 'w') as f:
        f.write(index_path)

    
    print(f"Index created and saved at: {index_ref}")
    print(f"Index reference saved at: {index_ref_path}")


