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

parser = argparse.ArgumentParser()
parser.add_argument(
     "-s", "--savedir",
     help = "save directory",
     default = "/data/ml2/vishakh/sap/oasis/"
 )

parser.add_argument(
     "-raw", "--raw",
     help = "whether raw or latent",
     default = True
 )


parser.add_argument(
     "-n", "--nsnapshots",
     help = "number of iterations",
     default = 10,
    type = int 
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
    "-fn", "--fname" ,
    help="The name of the final pickle file ",
    default = "oasis_eval.pk"

)


args = parser.parse_args()

if args.nsnapshots == 0:
    nsnapshots = None
else:
    nsnapshots = args.nsnapshots
    
eval_fname = args.fname
ft = FeatureTester()
save_dir = args.savedir

if args.raw:
    ft.load_from_hdf5_raw(dname=args.dname, cohort=args.cohort)
else:
    ft.load_from_hdf5_latent(dname=args.dname, cohort=args.cohort)
        
ft.eval_oasis(save_dir,save_dir, eval_fname,  nsnapshots)
