from theanomodels.models.vae import VAE
import numpy as np
import pandas as pd
from collections import OrderedDict
import os
from os.path import join as pjoin
from sklearn.model_selection import train_test_split
from pprint import pprint
import sys 
import cPickle as pickle
import gzip 
class VaeDimensionReducer(object):
    def __init__(self, saveDir = None, save_prefix = ""):
        self.vae = None
        self.dataset = None
        self.batch_size = None
        self.epochs = None
        self.train = None
        self.validation = None
        self.test = None
        self.params = None
        self.latent_samples = None
        self.savefreq = None
        self.save_prefix = save_prefix
        if saveDir == None:
            saveDir =  "/data/ml2/vishakh/vae/out/"
        self.saveDir =saveDir

    def load_config(ftype="pk"):
        print "Loading VAE params from pickle file"


    def write_config(ourDir = None):
        print "Writing a configuration to file"


    def load_from_csv(self, path=None, nrows=None):
        if path == None:
            path = "/data/ml2/vishakh/mimic-out-umls/feature_mat_first48.csv"
        data = np.array(pd.read_csv(path, nrows=nrows))
        self.dataset = data
        self.train, self.validation = train_test_split(data, train_size=.7)
        

    def learn_reducer(self, use_default_config=True, batch_size = 200, epochs = 200, savefreq = 100,  latentD = 2, reloadFile = None, pfile = None):

        self.batch_size = batch_size
        self.epochs = epochs
        self.savefreq = savefreq
        
        if use_default_config == True:
            print "Using default params"
            self._use_default_config()
        
        params = self.params

        if latentD != 2:
            params['dim_stochastic'] = latentD
        
        print "Training VAE"
        print "Latent Dimension is {0}".format(latentD)

        uuid = self.make_uuid(params)

        saveFile = pjoin(params['savedir'], self.save_prefix, uuid)
        
        
        if pfile is None:
            pfile =  params['savedir']+'/'+ uuid +'-config.pkl'
            print 'Training model from scratch. Parameters in: ',pfile


        else:
            print "loading params from file"
            pfile = pfile

        if reloadFile is None:
            vae  = VAE(params, paramFile = pfile)
        else:
            print "Warning, using vae loaded from weights"
            vae = VAE(params, paramFile = pfile, reloadFile = reloadFile)


        xvae.learn(self.train, epoch_start=0 , epoch_end = self.epochs,
                  batch_size = self.batch_size, savefreq  = self.savefreq,
                  savefile   = saveFile, dataset_eval= self.validation,
                  replicate_K = 5)
        
        self.vae = vae

        self.latent_samples = self.vae.infer(self.dataset)

        print "Done learning VAE"


            
    def visualize(self, colors=None):
        if self.latent_samples != None:
            assert len(self.latent_samples.shape == 2)

            x = z[:,0].ravel()
            y = z[:,1].ravel()
            plt.scatter(x, y, c=colors)
            plt.show()

    def _use_default_config(self):
        print "Using a default VAE configuration"

        if sys.argv == ['']:
            from theanomodels.utils.parse_args_vae import params
        else:
            f = open(pjoin(self.saveDir, "default_params.pk"))
            params = pickle.load(f)
            f.close()

        params['batch_size'] = self.batch_size
        params['epochs']= self.epochs
        params['dim_observations'] = self.train.shape[1]
        params['dim_stochastic']= 2
        params['savedir'] = self.saveDir
        params['paramFile'] = "pfile.pk"
        params['reloadFile'] = './NOSUCHFILE'
        params['savefreq'] = self.savefreq
        params['data_type'] = 'real'
        params['unique_id'] = 'uuid'
        
        self.params = params

    def make_uuid(self, params = None):
        if params is None :
            if self.params is not None:
                params = self.params
            else:
                print "No parameters from which to make uuid"
                return


        #Define a map between the keys and the shortcut
        hmap       = OrderedDict() 
        hmap['lr']='lr'
        hmap['q_dim_hidden']='qh'
        hmap['p_dim_hidden']='ph'
        hmap['dim_stochastic']='ds'
        hmap['p_layers']='pl'
        hmap['q_layers']='ql'
        hmap['nonlinearity']='nl'
        hmap['optimizer']='opt'
        hmap['batch_size']='bs'
        hmap['epochs']='ep'
        hmap['inference_model']='infm'
        hmap['input_dropout']='idrop'
        hmap['reg_type']    = 'reg'
        hmap['reg_value']   = 'rv'
        hmap['reg_spec']    = 'rspec'

        combined   = ''
        for k in hmap:
            if k in params:
                if type(params[k]) is float:
                    combined+=hmap[k]+'-'+('%.4e')%(params[k])+'-'
                else:
                    combined+=hmap[k]+'-'+str(params[k])+'-'
        uuid = combined[:-1]+'-'+ params['unique_id']
        uuid = 'VAE_'+uuid.replace('.','_')

        return uuid        

        
    def dimension_reduce(self, data):
        print "Reducing the dimension of data"

    def save_latent(self, saveDir = None):
        if saveDir == None:
            saveDir = self.saveDir 

        latent_path = pjoin(saveDir, self.save_prefix + "latent_samples.pk")

        f = gzip.open(latent_path, 'wb')
        pickle.dump(self.latent_samples, f)

    def to_string():
        if self.config != None:
            pprint(self.config)
