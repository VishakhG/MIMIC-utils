#!/bin/bash

N=20000

python oasis_run.py\
       -s "/data/ml2/vishakh/sap/oasis/raw_feat/aspirin/"\
       -n $N\
       -raw True\
       -d "mimic-aspirin"\
       -c False\
       -run True\
       -e1 False  &

python oasis_run.py\
       -s "/data/ml2/vishakh/sap/oasis/raw_feat/cancer/"\
       -raw True\
       -n $N\
       -d "mimic-cancer"\
       -c False\
       -run True\
       -e1 False &



python oasis_run.py\
       -s "/data/ml2/vishakh/sap/oasis/latent_feat/aspirin/"\
       -raw True\
       -n $N\
       -d "mimic-aspirin"\
       -c False\
       -run True\
       -e1 False &

python oasis_run.py\
       -s "/data/ml2/vishakh/sap/oasis/latent_feat/cancer/"\
       -raw True\
       -n $N\
       -d "mimic-cancer"\
       -c False\
       -run True\
       -e1 False &

wait 
echo Finished
