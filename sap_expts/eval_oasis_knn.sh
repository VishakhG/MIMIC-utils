#!/bin/bash
#Script to run oasis evaluations, it will test with knn and precision at top k
N=-1
fname="oasis_eval_last.pk"
python oasis_knn.py\
       -s "/data/ml2/vishakh/sap/oasis/raw_feat/aspirin/"\
       -n $N\
       -raw True\
       -d "mimic-aspirin"\
       -fn $fname\
       -c True &

python oasis_knn.py\
       -s "/data/ml2/vishakh/sap/oasis/raw_feat/cancer/"\
       -raw True\
       -n $N\
       -d "mimic-cancer"\
       -fn $fname\
       -c True&

python oasis_knn.py\
       -s "/data/ml2/vishakh/sap/oasis/latent_feat/aspirin/"\
       -raw True\
       -n $N\
       -d "mimic-aspirin"\
       -fn $fname\
       -c True&

python oasis_knn.py\
       -s "/data/ml2/vishakh/sap/oasis/latent_feat/cancer/"\
       -raw True\
       -n $N\
       -d "mimic-cancer"\
       -fn $fname\
       -c True&

wait 
echo Finished
