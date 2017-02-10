import pandas as pd
import numpy as np
import os
import pickle
import fnmatch
import math

from sklearn.feature_extraction.text import TfidfTransformer
import theano

import theanomodels

from models.vae import VAE

import time

from collections import OrderedDict
from datasets.load import loadDataset

#Easy Toggle
test = False

if not test:
    #Load the sparse array pandas is faster but we want an npy at the end
    print("Loading whole dataset")
    sparse  = pd.read_csv("/data/ml2/vishakh/mimic_mortalitypred_data/sparse_outlier_free.csv")
    sparse = np.array(sparse)
    print("Loaded whole dataset")

else:
    print("Loading testing dataset")
    #Load the sparse array pandas is faster but we want an npy at the end
    sparse  = pd.read_csv("/data/ml2/vishakh/mimic_mortalitypred_data/sparse_outlier_free.csv", nrows=2)
    sparse = np.array(sparse)
    print("Loaded testing  dataset")

#turn into np array and remove index
sparse_array = np.array(sparse)

#parameters for VAE
hmap = OrderedDict() 
hmap['lr']= 8e-4
hmap['q_dim_hidden']= 400
hmap['p_dim_hidden']= 200
hmap['p_layers']=2
hmap['q_layers']=2
hmap['nonlinearity']='softplus'
hmap['optimizer']='adam'
hmap['batch_size']=200
hmap['epochs']=100
hmap['inference_model']='single'
hmap['input_dropout']=0.1
hmap['reg_type']    = 'l2'
hmap['reg_value']   = 0.01
hmap['reg_spec']    = '_'
combined   = ''
hmap['seed'] = 1
hmap['dim_observations'] = sparse_array.shape[1]
hmap['init_scheme'] = 'uniform'
hmap['init_weight'] = .1
hmap['data_type'] = 'binary'

#latent dim
hmap['dim_stochastic']= 100

#train
pfile= '/data/ml2/vishakh/vae_out/pfile.pkl'
print 'Training model from scratch. Parameters in: ', pfile


hmap['dim_stochastic']= 100

vae  = VAE(hmap, paramFile = pfile)

vae.learn(          sparse_array,
                    epoch_start=0 , 
                    epoch_end  = hmap['epochs'], 
                    batch_size = hmap['batch_size'],
                    savefreq   = 50,
                    savefile   = '/data/ml2/vishakh/vae_out',
                    dataset_eval= sparse_array,
                    replicate_K= 5
                    )

#Save the latent space info
print "getting latent space"
latent_space = vae.infer(sparse_array)

latent_space = np.array(latent_space)

fname = "vae_latent_space_dim"  + ".npy"
fname = os.path.join("/data/ml2/vishakh", "vae_out", fname)

print("saving latent space info")
np.save("\data\ml2\vishakh\latent.npy", latent_space)

print("done")
