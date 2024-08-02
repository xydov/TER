## Comments 
- [ ] tirer les noms de fonction depuis les commentaires
- [x] faire un minimum de gestion d'erreurs (try catch)
- [ ] use if name main
- [x] add requirements.txt
- [x] finir avec l'indexation 
- [x] finir avec le retrieve
- [ ] finir avec le re-rank
- [x] débuter le rapport

## Explain
changed /etc/resolve.conf : changement du server DNS utilisé pour pouvoir resoudre des noms de domaines précedement problématique de 127.0.0.53 à 1.1.1.1
fichiers install 
apt**
mango_db modifs  : commenté repository mango_db dans les fichiers apt/sources.list/ pq y'avait un probléme de key gpg. 
elastic search - kibana - ELK (L pour LOKI)
PORT 5601 kibana / PORT 9200 elastic search 
configuration kibana pour le lier à elastic : on a changé une ligne de configuration kibana pour le connecter au service elsatic seatch présent localement dans la machine. 



kibana : visualisation indexation (bonus) et lecture de la données présente sur elasticsearch
elastic-search : indexation 
scripts d'instalation explained 

utilisation d'optuna : installation d'optuna (add to docker) et faire un re -rf __pycach__/


comments : BM25plus is highly optimized in my case that using bert to re-rank isn't changing the initialy retrieved documents


initial baseline indexation : 

DL10 -> Bm25+ : 0.4240 / 0.7042 / 0.1991 / 0.5010
DL10 -> Bm25+tf-idf : 0.4241 / 0.6663 / 0.2294 / 0.5258
DL10 -> Jelinec_mercel : 0.4122 / 0.6591 / 0.2263 / 0.5155
dsitilbert : "ndcg_cut_10: 0.0226
recip_rank: 0.0823
recall_10: 0.0103
P_10: 0.0309
"
indexation spacy et inverted index: 

SPACY : BM25-tf : Loading qrels...
Loading results...
Evaluating results...
Printing metrics...
ndcg_cut_10: 0.4569
recip_rank: 0.7439
recall_10: 0.2337
P_10: 0.5330

spacy : jelinek-mercer : "Printing metrics...
ndcg_cut_10: 0.4537
recip_rank: 0.7356
recall_10: 0.2371
P_10: 0.5330"

spacy : Query likelihood + JM : 
Printing metrics...
ndcg_cut_10: 0.4543
recip_rank: 0.7382
recall_10: 0.2367
P_10: 0.5320


initial baseline indexation : 

DL1000 -> Bm25+ : 0.2141 / 0.4797 / 0.0582 / 0.2629
DL1000 -> Bm25+tf-idf : 0.2157 / 0.4771 / 0.0587 / 0.2629
DL1000 -> Jelinec_mercel : 0.3823 / 0.7029 / 0.1194 / 0.4412
distil bert : "ndcg_cut_10: 0.0205
recip_rank: 0.0907
recall_10: 0.0042
P_10: 0.0412"


indexation spacy et inverted index : 

SPACY : JLNK-MERCER "Evaluating results...
Printing metrics...
ndcg_cut_10: 0.4061
recip_rank: 0.7274
recall_10: 0.1414
P_10: 0.4804
"
SPACY : BM25-TF
Printing metrics...
ndcg_cut_10: 0.4299
recip_rank: 0.7354
recall_10: 0.1459
P_10: 0.5134
"

SPACY : QLM - JELINEK MERCER : 
Printing metrics...
ndcg_cut_10: 0.4098
recip_rank: 0.7289
recall_10: 0.1402
P_10: 0.4804


