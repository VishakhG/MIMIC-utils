#!/bin/bash

python /home/vrg251/MIMIC-utils/tools/extract_features.py\
       -o "/data/ml2/vishakh/patient-similarity/temp/mimic-features-split/"\
       -fn "features"\
       -tn "targets"\
       -suff "1"\
       -start 0 \
       -end 10000 &

cat /home/vrg251/MIMIC-utils/tools/extract_features.py|\
    ssh crunchy3 python -\
	-o "/data/ml2/vishakh/patient-similarity/temp/mimic-features-split/"\
       -fn "features"\
       -tn "targets"\
       -start 10000\
       -end 20000\
       -suff "2" &

cat /home/vrg251/MIMIC-utils/tools/extract_features.py|\
    ssh crunchy4 python -\
	-o "/data/ml2/vishakh/patient-similarity/temp/mimic-features-split/"\
       -fn "features"\
       -tn "targets"\
       -suff "3"\
       -start 20000\
       -end 30000 &

cat /home/vrg251/MIMIC-utils/tools/extract_features.py|\
    ssh crunchy6 python -\
	-o "/data/ml2/vishakh/patient-similarity/temp/mimic-features-split/"\
       -fn "feature"\
       -tn "targets"\
       -suff "4"\
       -start 30000 & 
wait
