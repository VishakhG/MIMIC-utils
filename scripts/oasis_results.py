import gzip
import numpy as np
import pandas as pd
import cPickle as pickle
from sklearn.model_selection import train_test_split
from oasis import Oasis
import os 
from os.path import join as pjoin
import cPickle as pickle

nrows = None

features = np.array(pd.read_csv("/data/ml2/vishakh/mimic-out-umls/feature_mat_first48.csv", nrows=nrows))
targets = np.array(pd.read_csv("/data/ml2/vishakh/mimic-out-umls/target_mat_first48.csv", nrows=nrows)).ravel()

split = train_test_split(features, targets, train_size = .7 , stratify=targets)
x_train, x_test , y_train, y_test = split

out = []
baseDir = "/data/ml2/vishakh/temp/oasis/snapshots"
saveDir = "/data/ml2/vishakh/temp/oasis/"
for fName in sorted(os.listdir(baseDir)):
    print fName
    print pjoin(baseDir, fName)

    model = Oasis()

    model.read_snapshot(pjoin(baseDir,fName))

    info = model.predict(x_test, x_train, y_test, y_train, maxk=1000)
    
    out.append(info)
    
f = gzip.open(pjoin(saveDir, "evaluation.pk"), 'wb')
pickle.dump(out, f)
f.close()


