from __future__ import division


import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from metric_learn  import LMNN
from metric_learn import NCA
from oasis import Oasis

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import cPickle as pickle
from sklearn.model_selection import learning_curve
from pprint import pprint 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import make_classification

from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler

from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split
import gzip

import datetime

from pprint import pprint

import scipy 
import scipy.sparse

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise  import euclidean_distances

from os.path import join as pjoin 

class FeatureTester(object) :
    def __init__(self, fname = "", verbose=False):
        self.features = None
        self.targets = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.current_result = None
        self.current_metadata = None
        self.fname = fname
        self.cv_score_func = None
        self.verbose = verbose
        
    def load_from_array(self, feature_set, target_set):
        self.features = feature_set
        self.targets = target_set.ravel()
        print "loaded data"
        
    def load_from_csv(self, feature_path=None, target_path=None, nrows=None):
        if feature_path is None:
            feature_path = "/data/ml2/vishakh/mimic-out-umls/feature_mat_first48.csv"
        if target_path is  None:
            target_path = "/data/ml2/vishakh/mimic-out-umls/target_mat_first48.csv"


        features = np.array(pd.read_csv(feature_path, nrows = nrows)).astype('float')
        targets = np.array(pd.read_csv(target_path, nrows=nrows)).ravel()

        self.features = features
        self.targets = targets

        print "feature dimension " + str(features.shape)
        print "target dimension " + str(targets.shape)
        
    def create_intermediate_datasets(self, featurePathBase, targetsPathBase):
        self.load_from_csv(featurePathBase + ".csv", targetsPathBase + ".csv")

        features = self.features
        targets = targets

        #avoid bugs
        self.features = None
        targets = None

        fB_path = featurePathBase + "B"
        tB_path = targetsPathBase + "B"

        fBT_path = featurePathBase + "BT"
        tBT_path = targetsPathBase + "BT"

        print fB_path
        print tB_path
        print fBT_path
        print tBT_path
        
        ros = RandomOverSampler()
        features, targets = ros.fit_sample(features, targets) 

        pd.DataFrame(features).to_csv(fB_path + ".csv")
        pd.DataFrame(targets).to_csv(tB_path + ".csv")

        transformer = TfidfTransformer(smooth_idf=False)
        tfidf_data = transformer.fit_transform(features)
        features = tfidf_data.toarray()

        pd.DataFrame(features).to_csv(fBT_path + ".csv")
        pd.DataFrame(targets).to_csv(tBT_path + ".csv")
        
    def load_from_pk(self, feature_path, target_path):
        features = pickle.load(open(feature_path))
        targets = pickle.load(open(target_path))

        self.features = features
        self.targets = targets
        
        print "feature dimension " + features.shape
        print "target dimension" + target.shape

    def prepare_for_testing(self, tfidf=True, balance=True, sparse=True, cv_score_func=None, validation=False): 
        if cv_score_func is None:
            self.cv_score_func = 'roc_auc'
        else :
            self.cv_score_func= 'accuracy'

       
        features, targets = (self.features, self.targets)
        
        split = train_test_split(features, targets, train_size = .7 , stratify=targets)
        x_train, x_test , y_train, y_test = split

        if balance:
            ros = RandomOverSampler()
            x_train, y_train = ros.fit_sample(x_train, y_train)
            print "Oversampled minority class with replacement"

        
        if tfidf:
            transformer = TfidfTransformer(smooth_idf=False)
            x_train = transformer.fit_transform(x_train).toarray()
            x_test = transformer.fit_transform(x_test).toarray()

        if sparse:
            x_train = csr_matrix(x_train)
            print("Representing train set as sparse matrix")

        metadata = '''
        TFIDF:{9}
        Balanced Train : {10}
        Feature set dimensions: {0}
        Target Set dimensions:{1} 
        Target set composition:Died:{2}, Survived:{3}
        Training set composition:Died:{4}, Survived{5},
        Testing  set composition Died:{6}, Survived: {7} 
        Feature Set sparsity: {8} 
        Special Note:{11}'''.format(
            features.shape, targets.shape,
            len(targets[targets == 0]), len(targets[targets == 1]),
            len(y_train[y_train == 1]), len(y_train[y_train == 0]),
            len(y_test[y_test == 1]), len(y_test[y_test == 0]),
            np.sum(features) / np.product(features.shape),
            tfidf,
            balance, self.fname)

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        
        print("Data prepared for testing")


        self.current_metadata = metadata
            
        print  metadata

    def prepare_valid_set(self):
        x_train, x_valid, y_train, y_valid  = train_test_split(
            self.x_train, self.y_train, train_size = .7, stratify=self.y_train)

        return {
            "x_train":x_train,
            "y_train":y_train,
            "x_valid":x_valid,
            "y_valid":y_valid
        }
        
    def physionet_score(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        Se = tp/(tp + fn)
        P = tp/(tp + fp)
        score = min(Se, P)
        return score
        
    def test_classifier(self, classifier_i, params_i = None, name=None):

        x_test = self.x_test
        y_test = self.y_test
        x_train = self.x_train
        y_train = self.y_train

        
        clf = GridSearchCV(classifier_i, params_i, scoring=self.cv_score_func, cv=5)
        clf.fit(x_train, y_train)

        top_score = clf.best_score_
        train_pred = clf.predict(x_train)
        test_pred = clf.predict(x_test)

        phys_score =[self.physionet_score(y_train, train_pred),
                     self.physionet_score(y_test, test_pred)]
        
        accuracy = [accuracy_score(i[0], i[1]) for i in
                    [(y_train, train_pred), (y_test, test_pred)]]

        if (not (0 in train_pred and 1 in train_pred)):
            print "Predicted only one class for train_pred"

        if (not (0 in test_pred and 1 in test_pred)):
            print "Predicted only one class for train_pred"

        try:
            roc = [roc_auc_score(i[0], i[1]) for i in
                   [(y_train, train_pred ), (y_test, test_pred)]]

        except:
            roc = None
            print "Can't calc ROC"

        test_Mmask = y_test == 0
        train_Mmask = y_train == 0

        train_Mpred = train_pred[train_Mmask]
        train_Mtrue = y_train[train_Mmask]

        test_Mpred = test_pred[test_Mmask]
        test_Mtrue = y_test[test_Mmask]

        Macc_test = (test_Mpred, test_Mtrue)
        Macc_train = (train_Mpred, train_Mtrue)

        majority_accuracy  = [accuracy_score(i[0], i[1]) for i in
                              [Macc_train, Macc_test]]
        


        result_out =  {'name' : name,
                'classifier': str(clf.best_params_),
                'cv_score':top_score,
                'accuracy:': {
                    "train":accuracy[0],
                    "test":accuracy[1]
                },
                'auc_roc:': {
                    "train":roc[0],
                    "test":roc[1]
                },

                'majority_accuracy:':{
                    "train": majority_accuracy[0],
                    "test":majority_accuracy[1]
                },
                'physionet score':{
                    "train" : phys_score[0],
                    "test" : phys_score[1]
                }
        }

        self.check_record(result_out)
        return result_out

    def check_record(self, stats):
        stats = pd.DataFrame.from_dict(stats)
        self.current_result = stats


    def logistic_regression(self, elided_search = False):
        lr = LogisticRegression()

        if elided_search:
            lr_params = {'penalty':['l1','l2']}

        else:
            lr_params = {'penalty':['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

        log_stats = self.test_classifier(lr, lr_params, name = str(lr)[0:18])
        
    def random_forest(self):
        rf = RandomForestClassifier()
        rf_params = {'criterion':('gini', 'entropy'), 'n_estimators':[10^i for i in range(3)]}    
        rf_stats = self.test_classifier(rf, rf_params, name="rf")

        self.check_record(rf_stats)

        
    def nearest_neighbors(self, name=None):
        knn = KNeighborsClassifier()
        knn_params = {
            "n_neighbors" : [5, 6]
        }
        
        if name is None:
            name =  "KNN"
        knn_stats  = self.test_classifier(knn, knn_params, name)
        self.check_record(knn_stats)    


    def oasis(self, aggress=.1, niter=10000, maxk=10, verbose=True):
        model = Oasis(n_iter=niter, aggress = aggress, do_psd=False, psd_every=3,
                      save_path = pjoin("/data/ml2/vishakh/temp/oasis", self.fname))
        
        model.fit(self.x_train, self.y_train, verbose = verbose)
        
        out = model.predict(self.x_test, self.x_train,  self.y_test, self.y_train, maxk=maxk)

        return out                                                                              

    def LMNN(self):
        print "Warning, the features will be transformed"
        lmnn = LMNN(k=5, learn_rate = 1e-6)
        lmnn.fit(self.features, targets)

        self.features = lmnn.transform(self.features)
        self.prepare_for_testing()
        self.nearest_neighbors("LMNN + KNN")



    def NCA(self):
        print "Warning the features will be transformed"
        lmnn = NCA()
        NCA.fit(self.features, targets)
        
        self.features = NCA.transform(self.features)
        self.prepare_for_testing()

        #Evaluate with nn
        self.nearest_neighbors("NCA + KNN")             
        
    def get_metadata(self):
        return self.current_metadata

    def get_result(self):
        return self.current_result

    def knn_custom(self, maxk=200, cosineSim=False):
        split = self.prepare_valid_set()
        
        X_train = split['x_train']
        X_valid = split['x_valid']
        X_test = split['x_test']
        
        y_train = split['y_train']
        y_valid = split['y_valid']
        y_test = split['y_test']

        
        maxk = min(maxk, X_train.shape[0])  # K can't be > numcases in X_train
        
        numqueries = X_test.shape[0]
        numqueries_train = X_train.shape[0]

        numqueriesMaj = X_test[y_test == 0].shape[0]

        numqueriesMaj_train = X_train[y_train == 0].shape[0]

        if not cosineSim:
            s =  euclidean_distances(X_test, X_train)
            s_train =  euclidean_distances(X_train, X_train)
            
        else:
            s = cosine_similarity(X_test, precomp.T)
            s_train = cosine_similarity(X_train, X_train)
        
        ind = np.argsort(s, axis=1)
        ind_train = np.argsort(s_train, axis=1)

        # Voting based on nearest neighbours
        # make sure it is int

        # Newer version of ndarray.astype takes a copy keyword argument
        # With this, we won't have to check

        if y_train.dtype.kind != 'int':
            queryvotes = y_train[ind[:, :maxk]].astype('int')
            queryvotes_train = y_train[ind_train[:, :maxk]].astype('int')

        else:
            queryvotes = y_train[ind[:, :maxk]]
            queryvotes_train = y_train[ind_train[:, :maxk]]

        errsum = np.empty((maxk,))
        errsum_train = np.empty((maxk, ))

        
        errsumMaj = np.empty((maxk, ))
        errsumMaj_train = np.empty((maxk, ))

        errsumMin = np.empty((maxk, ))
        errsumMin_train = np.empty((maxk, ))

        aucs = []
        aucs_train = []

        phys_scores = []
        phys_scores_train = []

        for kk in xrange(maxk):
            labels = np.empty((numqueries,), dtype='int')
            labels_train = np.empty((numqueries_train, ), dtype='int')
            
            for i in xrange(numqueries):
                b = np.bincount(queryvotes[i, :kk + 1])
                labels[i] = np.argmax(b)  # get winning class

            for i in xrange(numqueries_train):
                b = np.bincount(queryvotes_train[i, :kk + 1])
                labels_train[i] = np.argmax(b)
                
            errors = labels != y_test
            errors_train = labels_train != y_train
            
            errorsMaj = errors[y_test == 0]
            errorsMaj_train = errors_train[y_train == 0]

            phys_scores.append(self.physionet_score(y_test, labels))
            phys_scores_train.append(self.physionet_score(y_train, labels_train))

            try:
                auc = roc_auc_score(y_test, labels)
                auc_train = roc_auc_score(y_train, labels_train)
                aucs.append(auc)
                aucs_train.append(auc_train)

            except:
                aucs.append(None)

            errsum[kk] = sum(errors)
            errsum_train[kk] = sum(errors_train)
            
            errsumMaj[kk] = sum(errorsMaj)
            errsumMaj_train[kk] = sum(errorsMaj_train)

            errrate = errsum / numqueries
            errrate_train = errsum_train / numqueries_train
            
            errrateMaj = errsumMaj / numqueriesMaj
            errrateMaj_train = errsumMaj_train / numqueriesMaj_train

            
        result_out =  {
            'name' : "knn_custom",
            'accuracy:':{
                "train": [1 - min(errrate_train), np.argmin(errrate_train)+1],
                "test": [ 1- min(errrate), np.argmin(errrate)+1]
            },
            
            'auc_roc:': {
                "train":[max(aucs_train), np.argmax(aucs_train)+1],
                "test":[max(aucs), np.argmax(aucs)+1]
            },
            
            'majority_accuracy:':{
                "train": [1 - min(errrateMaj_train), np.argmin(errrateMaj_train)+1],
                "test": [ 1 - min(errrateMaj), np.argmin(errrateMaj)+1]
            },
            
            'physionet score':{
                "train" : [max(phys_scores_train), np.argmax(phys_scores_train)+1],
                "test" : [max(phys_scores), np.argmax(phys_scores)+1]
            }
        }       

        return result_out
                      
      
    













