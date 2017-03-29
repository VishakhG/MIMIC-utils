import pandas as pd
import gzip
import numpy as np
from os.path import join as pjoin
from feature_tester import FeatureTester
from experiment import Experiment

#BASELINES
#LATENT
#COHORT
#SSI


#########
#CANCER
########

outDir = "/data/ml2/vishakh/sap/baselines/"

ft = FeatureTester(outDir = outDir, fname='cancer-latent-mu-ssi')
ft.load_from_hdf5_latent(dname='mimic-cancer', feat_name ='mu', ssi=True, cohort = True)


e = Experiment()
e.begin_experiment(fname="mortality-latent-mu")
ft.logistic_regression(save_lc=True)
e.log(ft.get_metadata(), ft.get_result())
ft.random_forest()
e.log(ft.get_metadata(), ft.get_result())
ft.nearest_neighbors()
e.log(ft.get_metadata(), ft.get_result())
e.end_experiment()

############
#ASPIRIN
###########


ft = FeatureTester(outDir = outDir, fname='aspirin-latent-mu-ssi')
ft.load_from_hdf5_latent(dname='mimic-aspirin', feat_name ='mu', ssi=True, cohort = True)


e = Experiment()
e.begin_experiment(fname="aspirin-latent-mu-ssi")
ft.logistic_regression(save_lc=True)
e.log(ft.get_metadata(), ft.get_result())

ft.random_forest()
e.log(ft.get_metadata(), ft.get_result())

ft.nearest_neighbors()
e.log(ft.get_metadata(), ft.get_result())
e.end_experiment()

