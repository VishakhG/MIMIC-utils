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

#OASIS


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
     "-n", "--niters",
     help = "number of iterations",
     default = 10000
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

args = parser.parse_args()

     
save_dir = args.savedir
predict = False
niter = int(args.niters)
    
ft = FeatureTester()
    
if args.raw:
    ft.load_from_hdf5_raw(dname=args.dname, cohort=args.cohort)

else:
    ft.load_from_hdf5_latent(dname=args.dname, cohort=args.cohort)

out, loss = ft.oasis(save_path = save_dir, predict=predict, niter = niter,  save_every = 1000)
pickle.dump(loss, open(pjoin(save_dir, "oasis.pk"), 'wb'))



     
     

