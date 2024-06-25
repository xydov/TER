lancer load_data.py pour télecharger le data_set localement ( juste pour tester avant la submission TIRA) sinon utiliser index.py pour indexer directement un dataset de ir_dataset.
#pour l'indexation , python3 indexing.py <name of dataset> <output_dir> or python3 index.py
pour le retrieve , l'indexation genere un fichier index_ref.txt , un rep index et un fichiers queries.txt . il faut que le retrieve.py soit dans le meme repo.
et lancer python3 re-rank.py pour effectuer un re-rank.
