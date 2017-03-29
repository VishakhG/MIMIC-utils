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
import matplotlib.pyplot as plt
import argparse
from experiment import Experiment

parser = argparse.ArgumentParser()

parser.add_argument(
     "-s", "--savedir",
     help = "save directory",
     default = "/data/ml2/vishakh/sap/oasis/"
 )

parser.add_argument(
    "-f", "--fname",
    help ="filename to use",
    default = ""
)

parser.add_argument(
     "-raw", "--raw",
     help = "whether raw or latent",
     default = True
 )

parser.add_argument(
     "-d", "--dname",
     help = "datasetname",
     default = "mimic-aspirin"
 )

parser.add_argument(
     "-c", "--cohort",
     help = "use cohorts",
     default = True
 )

parser.add_argument(
     "-n", "--nrows",
     help = "use n rows",
     default = 0,
    type = int
 )

args = parser.parse_args()

if args.nrows == 0:
    nrows = None
else:
    nrows = args.nrows


savedir = pjoin("/data/ml2/vishakh/sap/baselines", args.savedir)

ft = FeatureTester(outDir = savedir)

if args.raw:
    ft.load_from_hdf5_raw(dname=args.dname, cohort=args.cohort, nrows=nrows)

else:
    ft.load_from_hdf5_latent(dname=args.dname, cohort=args.cohort, nrows=nrows)


e = Experiment()
e.begin_experiment(fname=args.fname, baseDir=savedir)
ft.logistic_regression(save_lc=True)
e.log(ft.get_metadata(), ft.get_result())

ft.random_forest()
e.log(ft.get_metadata(), ft.get_result())

ft.nearest_neighbors()
e.log(ft.get_metadata(), ft.get_result())

e.end_experiment()
