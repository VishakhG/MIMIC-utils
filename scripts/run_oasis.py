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

#Just run oasis and save the snapshots somewhere

baseDir = "/data/ml2/vishakh/patient-similarity/metric-learning/"
save_dir = pjoin(baseDir, "oasis/snapshots-pickle-test")
nrows = 10000
ncols = 10000

predict = False

ft = FeatureTester()

ft.load_from_csv(nrows = nrows, ncols = ncols)
ft.prepare_for_testing(sparse=False)
out = ft.oasis(save_path = save_dir, predict=predict, niter = 100, save_every = 5)
print out
