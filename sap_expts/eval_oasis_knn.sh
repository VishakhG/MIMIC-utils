#!/bin/bash

N=-1
python oasis_knn.py\
       -s "/data/ml2/vishakh/sap/oasis/raw_feat/aspirin/"\
       -n $N\
       -raw True\
       -d "mimic-aspirin"\
       -c True 

# python oasis_knn.py\
#        -s "/data/ml2/vishakh/sap/oasis/raw_feat/cancer/"\
#        -raw True\
#        -n $N\
#        -d "mimic-cancer"\
#        -c True&

# python oasis_knn.py\
#        -s "/data/ml2/vishakh/sap/oasis/latent_feat/aspirin/"\
#        -raw True\
#        -n $N\
#        -d "mimic-aspirin"\
#        -c True&

# python oasis_knn.py\
#        -s "/data/ml2/vishakh/sap/oasis/latent_feat/cancer/"\
#        -raw True\
#        -n $N\
#        -d "mimic-cancer"\
#        -c True

wait 
echo Finished
