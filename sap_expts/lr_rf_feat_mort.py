import pandas as pd
import gzip
import numpy as np
from os.path import join as pjoin
from feature_tester import FeatureTester
from experiment import Experiment

#BASELINES
#MORTALITY
#RAW



outDir = "/data/ml2/vishakh/sap/baselines/"
ft = FeatureTester(outDir = outDir, fname='mortality-raw')
ft.load_from_hdf5_raw(dname='mimic-cancer', cohort=False)


e = Experiment()
e.begin_experiment(fname="mortality-raw")
ft.logistic_regression(save_lc=True)
e.log(ft.get_metadata(), ft.get_result())
ft.random_forest()
e.log(ft.get_metadata(), ft.get_result())
ft.nearest_neighbors()
e.log(ft.get_metadata(), ft.get_result())
e.view_experiment()
e.end_experiment()
