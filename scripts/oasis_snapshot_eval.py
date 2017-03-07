import gzip
import numpy as np
import pandas as pd
import cPickle as pickle
from sklearn.model_selection import train_test_split
from oasis import Oasis
import os 
from os.path import join as pjoin
import cPickle as pickle
from feature_tester import FeatureTester
#Examine snapshots over time
#look at how the training test curves behave and look at oasis loss as well

nrows = 10000 
ncols = 10000
out = []
baseDir = "/data/ml2/vishakh/patient-similarity/metric-learning"

snap_shot_dir = pjoin(baseDir, "oasis", "snapshots-truncated")
saveDir = pjoin(baseDir, "oasis")
eval_fname = "oasisEval.pk"

ft = FeatureTester()
ft.load_from_csv(nrows = nrows, ncols = ncols)
ft.prepare_for_testing(sparse=False)

for fname in sorted(os.listdir(snap_shot_dir)):
    if not fname.startswith('.'):
        cPath = pjoin(snap_shot_dir, fname)
        print("Loading from snapshot {0}".format(cPath))
        info = ft.oasis(snap_shot_path = cPath)
        out.append(info)
    else:
        print("ignoring file {0}").format(fname)
    
f = gzip.open(pjoin(saveDir, eval_fname), 'wb')
pickle.dump(out, f)
f.close()


