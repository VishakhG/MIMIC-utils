#!/bin/bash

#LATENT-MORTALITY
N=0
python eval_baselines.py\
       -raw False\
       -c False\
       -d "mimic-cancer"\
       -s "latent_feat/mortality/"\
       -n $N\
       -f "latent-mortality" &
       

#LATENT-CANCER
python eval_baselines.py\
       -raw False\
       -c True\
       -d "mimic-cancer"\
       -s "latent_feat/cancer/"\
       -n $N\
       -f "latent-cancer" &


#LATENT-ASPIRIN
python eval_baselines.py\
       -raw False\
       -c True\
       -d "mimic-aspirin"\
       -s "latent_feat/aspirin/"\
       -n $N\
       -f "latent-mortality" &


#RAW-MORTALITY
python eval_baselines.py\
       -raw True\
       -c False\
       -d "mimic-cancer"\
       -s "raw_feat/mortality/" \
       -n $N\
       -f "raw-mortality" &


#RAW CANCER
python eval_baselines.py\
       -raw True\
       -c True\
       -d "mimic-cancer"\
       -s "raw_feat/cancer/"\
       -n $N\
       -f "raw-cancer" &



#RAW ASPIRIN
python eval_baselines.py\
       -raw True\
       -c True\
       -d "mimic-aspirin"\
       -s "raw_feat/aspirin/"\
       -n $N\
       -f "latent-aspirin" 


wait
echo "All Baselines complete"
