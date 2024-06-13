#!/usr/bin/env python3
import json
#import gzip
#import argparse

from preprocess import reduce_articles_to_vocabulary,create_vocabulary,summarize_article

from cherche import data, retrieve, rank, index
import ir_datasets

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize


"""
def parse_args():
    parser = argparse.ArgumentParser(description='Build an index')
    parser.add_argument('-i', type=str, help='The input directory.', required=True)
    parser.add_argument('-o', type=str, help='The index will be stored in this directory.', required=True)
    
    return parser.parse_args()
"""
"""
def create_index(input_directory, output_directory):
    
    Load the documents from the input directory and build a pseudo index, just a list of document ids.
    
    with gzip.open(input_directory + '/documents.jsonl.gz', 'rt') as documents, gzip.open(output_directory + '/pseudo_index.jsonl.gz', 'wt') as f:
        for doc in documents:
            doc = json.loads(doc)
            f.write(doc['docno'] + '\n')
            
    print('Done: Index stored.')
"""

if __name__ == '__main__':
    #args = parse_args()
    #create_index(args.i, args.o)
    dataset = ir_datasets.load("msmarco-passage/trec-dl-2019/judged")
    #Preprocess documents#
    
    documents = [{'id':doc_id,'article':text} for (doc_id, text) in dataset.docs_iter()]#Test avec [:100]
    queries = [query.text for query in dataset.queries_iter()]
    vocabulary = create_vocabulary(queries)
    print(vocabulary)

    
    for doc in documents:
        doc['article'] = summarize_article(doc['article'])
        print('Summarize article')
        print(doc)
        doc['article']=reduce_articles_to_vocabulary(doc['article'], vocabulary)
        print('Reduce articles to vocabulary')
        print(doc)
        #doc['article']=reduce_articles_to_vocabulary(summarize_article(doc['article']), vocabulary)

    tokenized_docs = []
    for doc in documents:
        doc_id = doc['id']
        doc_text = doc['article'].lower()  # Convert text to lowercase before tokenization
        tokenized_text = word_tokenize(doc_text)
        tokenized_docs.append({'id': doc_id, 'tokens': tokenized_text})

    #INDEXATION
    bm25 = BM25Okapi([doc['tokens'] for doc in tokenized_docs])
    '''
    query = "theory science love"
    tokenized_query = word_tokenize(query.lower())
    doc_scores = bm25.get_scores(tokenized_query)
    N = 500
    top_n_doc_indices = sorted(range(len(tokenized_docs)), key=lambda i: doc_scores[i], reverse=True)[:N]

    top_n_docs = [tokenized_docs[i] for i in top_n_doc_indices]

    for i, doc in enumerate(top_n_docs):
        print(f"Document {i + 1} (ID: {doc['id']}): {' '.join(doc['tokens'])}")
    '''
    # Nombre de documents à récupérer pour chaque requête
    N =500  #N(de test)=20

    # Stocker les résultats dans un dictionnaire
    results = {} 

    for query_id,query in enumerate(queries):
        tokenized_query = word_tokenize(query.lower())
        doc_scores = bm25.get_scores(tokenized_query)
        top_n_doc_indices = sorted(range(len(tokenized_docs)), key=lambda i: doc_scores[i], reverse=True)[:N]
        #print("TOP N DOC INDICES")
        #print(top_n_doc_indices)
        top_n_docs = [tokenized_docs[i]['id'] for i in top_n_doc_indices]
        results[query_id] = top_n_docs
        print("QUERY ID, RESULTS[QUERY ID]")
        print(query_id,results[query_id])

    # Afficher les résultats pour les premières requêtes pour vérification
    '''
    for query, top_docs in list(results.items())[:3]:  # Afficher les résultats des 3 premières requêtes
        print(f"Query: {query}")
        for i, doc_id in enumerate(top_docs):
            print(f"Document {i + 1}: {doc_id}")
            print("\n") 
    '''
    # Sauvegarder les   résultats dans un fichier
    with open('bm25_results.json', 'w') as f:
        json.dump(results, f)
