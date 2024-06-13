from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(
   name='Pipeline_BM25OKAPI_KNNSearch',
   version='0.1.0',
   description='Test 1',
   url='https://github.com/xydov/TER',
   install_requires=[
       'cherche[gpu]','cherche[cpu]',
        'sentence_transformers',
        'pytrec-eval',
        'wget',
        'ir_datasets -q',
        'rank-bm25',
        'nltk',
        'transformers','faiss-cpu','faiss-gpu',
        'torch'
    ],
)
