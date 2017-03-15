#!/bin/bash

python /home/vrg251/MIMIC-utils/tools/extract_features.py\
       -test True\
       -o "/data/ml2/vishakh/patient-similarity/temp/"\
       -fn "provenance_testFeat"\
       -tn "provenance__testTarg"\
       -n 10\
       -w 48
       
