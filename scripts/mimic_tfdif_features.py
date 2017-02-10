import pandas as pd
import numpy as np
import os
import pickle
import fnmatch
import math

from sklearn.feature_extraction.text import TfidfTransformer
import theano
from models.vae import VAE
import theanomodels

import time

from collections import OrderedDict
from datasets.load import loadDataset


#Load the sparse array
sparse  = pd.read_csv("/data/ml2/vishakh/sparse_mat.csv")

print("Loaded dataframe")

#turn into np array and remove index
sparse_array = np.array(sparse)
sparse_array = sparse_array[:,1:sparse_array.shape[1]]

#remove outliers
sums = sparse_array.sum(axis=0)
mean = np.mean(sums)
std = np.std(sums)
n_bound = 2  * std

how_odd = lambda x: np.abs(x - mean) > n_bound
sparse_array = typical_sparses_array = sparse_array[map(how_odd, sums)]

print("removed outliers")

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(sparse_array)
tfidf_array = tfidf.toarray()

print("pickling the tfidf array")
np.save("/data/ml2/vishakh/mimic_output/tfidf_features.npy", tfidf_array)
