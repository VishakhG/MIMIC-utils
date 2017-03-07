from __future__ import division

import cPickle as pickle
from pprint import pprint
import gzip
import datetime
from os.path import join as pjoin 

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from metric_learn  import LMNN
from metric_learn import NCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise  import euclidean_distances

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer

from oasis import Oasis

from imblearn.over_sampling import RandomOverSampler


import scipy 
import scipy.sparse
from scipy.sparse import csr_matrix

class FeatureTester(object) :
    """A class that allows the testing of features with basic classifiers """
    
    def __init__(self, fname = "", outDir=None,  verbose=False):
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
        if outDir is None:
            outDir = "/data/ml2/vishakh/patient-similarity/"
        self.outDir = outDir
        
    def load_from_array(self, feature_set, target_set):
        """Set features and targets from numpy arrays"""
        
        self.features = feature_set
        self.targets = target_set.ravel()
        print "loaded data"
        
    def load_from_csv(self, feature_path=None, target_path=None, nrows=None, ncols=None):
        """Load features and targests from csv files """

        if nrows is not None:
            print("Using {0} rows").format(nrows)
            
        if feature_path is None:
            feature_path = pjoin("/data/ml2/vishakh/patient-similarity",
                                 "mimic-derived", "features48.csv")
            print("Using default features {0}").format(feature_path)
            
        if target_path is  None:
            target_path = pjoin("/data/ml2/vishakh/patient-similarity",
                                "mimic-derived", "targets48.csv")
            
            print("Using default targets {0}").format(target_path)


        features = np.array(pd.read_csv(feature_path, nrows = nrows)).astype('float')
        targets = np.array(pd.read_csv(target_path, nrows=nrows)).ravel()

        if ncols is not None:
            print("Using {0} columns").format(ncols)
            features = features[:, 1:ncols]
                        
        self.features = features
        self.targets = targets

        print "feature dimension " + str(features.shape)
        print "target dimension " + str(targets.shape)
        
    def create_intermediate_datasets(self, featurePathBase, targetsPathBase):
        """Create intermediate datasets that make things faster """
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
        """Load data from pickle files """
        features = pickle.load(open(feature_path))
        targets = pickle.load(open(target_path))

        self.features = features
        self.targets = targets
        
        print "feature dimension " + features.shape
        print "target dimension" + target.shape

    def prepare_for_testing(self, tfidf=True, balance=True, sparse=True, cv_score_func=None, validation=False):
        """ Get everything ready for testing, balance training set, create test train split and use  
            TFIDF if needed. """
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
        """Sometimes it may be easier to use a validation set instead of cross validation for 
          Effeciency reasons, in these cases use this function to prepare that validation set.
        """
        x_train, x_valid, y_train, y_valid  = train_test_split(
            self.x_train, self.y_train, train_size = .7, stratify=self.y_train)

        return {
            "x_train":x_train,
            "y_train":y_train,
            "x_valid":x_valid,
            "y_valid":y_valid
        }
        
    def physionet_score(self, y_true, y_pred):
        """The score that was used to score physionet 2012 challenge A """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        Se = tp/(tp + fn)
        P = tp/(tp + fp)
        score = min(Se, P)
        return score
        
    def test_classifier(self, classifier_i, params_i = None, name=None, save=False):
        """The general setup to test a classifier using grid search and k-fold CV for a sklearn classifier """

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

        if save:
            fpath = pjoin(self.outDir, str(name) + str(clf.best_params_) + str(roc[1]))
            print "Saving learning data to {0}".format(fpath)
            f = gzip.open(fpath, 'wb')
            #Some model information that could be useful.
            pickle.dump([clf.cv_results_, clf], f)
            f.close()
        return result_out

    def check_record(self, stats):
        stats = pd.DataFrame.from_dict(stats)
        self.current_result = stats


    def logistic_regression(self, elided_search = False, save_lc=False):
        lr = LogisticRegression()

        if elided_search:
            lr_params = {'penalty':['l1','l2']}

        else:
            lr_params = {'penalty':['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

        log_stats = self.test_classifier(lr, lr_params, name = str(lr)[0:18], save=save_lc)
        
    def random_forest(self, save_lc = False):
        rf = RandomForestClassifier()
        rf_params = {
            'criterion':['gini', 'entropy'],
            'n_estimators':[10**i for i in range(4)] + [2000, 3000, 4000],
            'max_features':['sqrt','log2', .1, .2, .3]
        }
        
        rf_stats = self.test_classifier(rf, rf_params, name="rf", save = save_lc)

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


    def oasis(self, predict=True, snap_shot_path = None, save_path = None, aggress=.4, niter=10000, maxk=100, verbose=True, save_every = None):

        if snap_shot_path is None:
            print("Learning metric with oasis using agressivenes of {0} and {1} iterations").format(
            aggress, niter)
        
        split = self.prepare_valid_set()
        
        X_train = split['x_train']
        X_valid = split['x_valid']
        X_test = self.x_test
        
        y_train = split['y_train']
        y_valid = split['y_valid']
        y_test = self.y_test

        if save_path is None:
            save_path = pjoin(self.outDir, "metric-learning", "snapshots", self.fname)
        
        if snap_shot_path is None:
            model = Oasis(n_iter=niter, aggress = aggress, do_psd=False, psd_every=3,
                           save_path = save_path, save_every=save_every)

            model.fit(X_train, y_train, verbose = verbose)

        else:
            model = Oasis()
            model.read_snapshot(snap_shot_path)
            
        n_features = X_train.shape[1]
        W = model._weights
        W.shape = (np.int(np.sqrt(W.shape[0])), np.int(np.sqrt(W.shape[0])))
        
        if predict:
            out = self.knn_precomp(sim_weights=W, useSim=True)
        else:
            out = None
        
        return out, model.batch_loss                                                                              

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

    def knn_precomp(self, maxk=200, cosineSim=False, sim_weights = None, useSim = False):
    
        #We want to use a validation set 
        split = self.prepare_valid_set()
        
        X_train = split['x_train']
        X_valid = split['x_valid']
        X_test = self.x_test
        
        y_train = split['y_train']
        y_valid = split['y_valid']
        y_test = self.y_test

        
        maxk = min(maxk, X_train.shape[0])  # K can't be > numcases in X_train
        
        numqueries_valid = X_valid.shape[0]
        numqueries_test = X_test.shape[0]
        numqueries_train = X_train.shape[0]

        numqueriesMaj_test = X_test[y_test == 0].shape[0]
        numqueriesMaj_valid = X_valid[y_test == 0].shape[0]
        numqueriesMaj_train = X_train[y_train == 0].shape[0]

        #Euclidean distance if not cosine
        if not cosineSim and not useSim:
            print("Using Euclidean distance")
            s_test =  euclidean_distances(X_test, X_train)
            s_valid = euclidean_distances(X_valid, X_train)
            s_train =  euclidean_distances(X_train, X_train)

        #cosine distance if not euclidean
        elif cosineSim:
            print("Using Cosine useSim measure")
            s_test = cosine_similarity(X_test, X_train)
            s_valid = cosine_similarity(X_valid, X_train)
            s_train = cosine_similarity(X_train, X_train)

        elif useSim:
            print("Using a custom similarity score maybe for OASIS")
            precomp = np.dot(sim_weights, X_train.T)
            print "precomp done"
            s_valid = np.dot(X_valid, precomp)
            print "s_valid done"
            s_test = np.dot(X_test, precomp)
            print "s_test done"
            s_train = np.dot(X_train, precomp)
            print "s_train done"
            
            
        if cosineSim or useSim:
            print "Sorting backwards because its a similarity not distance"
            #We are using a useSim score so argsort backwards
            ind_test = np.argsort(s_test, axis=1)[:, ::-1]
            ind_train = np.argsort(s_train, axis = 1)[:, :: -1]
            ind_valid = np.argsort(s_valid, axis = 1)[:, :: -1]

        else:
            "Sorting regularly because its a distance not similarity"
            #We are using a distance so sort regularly
            ind_test = np.argsort(s_test, axis=1)
            ind_valid = np.argsort(s_valid, axis=1)
            ind_train = np.argsort(s_train, axis=1)

        # Voting based on nearest neighbours
        # make sure it is int

        # Newer version of ndarray.astype takes a copy keyword argument
        # With this, we won't have to check



        if y_train.dtype.kind != 'int':
            queryvotes_test = y_train[ind_test[:, :maxk]].astype('int')
            queryvotes_valid = y_train[ind_valid[:, :maxk]].astype('int')
            queryvotes_train = y_train[ind_train[:, :maxk]].astype('int')

        else:
            queryvotes_test = y_train[ind_test[:, :maxk]]
            queryvotes_valid = y_train[ind_valid[:, :maxk]]
            queryvotes_train = y_train[ind_train[:, :maxk]]


        errsum_valid = np.empty((maxk,))
        errsum_train = np.empty((maxk, ))

        errsumMaj_train = np.empty((maxk, ))
        errsumMaj_valid = np.empty((maxk, ))

        aucs_valid = []
        aucs_train = []

        phys_scores_valid = []
        phys_scores_train = []


        for kk in xrange(maxk):
            print "Begining the voting"
            print kk
            labels = np.empty((numqueries_valid,), dtype='int')
            labels_train = np.empty((numqueries_train, ), dtype='int')
            
            for i in xrange(numqueries_valid):
                b = np.bincount(queryvotes_valid[i, :kk + 1])
                labels[i] = np.argmax(b)  # get winning class

            for i in xrange(numqueries_train):
                b = np.bincount(queryvotes_train[i, :kk + 1])
                labels_train[i] = np.argmax(b)
                
            errors = labels != y_valid
            errors_train = labels_train != y_train
            
            errorsMaj = errors[y_valid == 0]
            errorsMaj_train = errors_train[y_train == 0]

            phys_scores_valid.append(self.physionet_score(y_valid, labels))
            phys_scores_train.append(self.physionet_score(y_train, labels_train))

            try:
                auc = roc_auc_score(y_valid, labels)
                auc_train = roc_auc_score(y_train, labels_train)
                aucs_valid.append(auc)
                aucs_valid.append(auc_train)

            except:
                aucs.append(None)

            errsum_valid[kk] = sum(errors)
            errsum_train[kk] = sum(errors_train)
            
            errsumMaj_valid[kk] = sum(errorsMaj)
            errsumMaj_train[kk] = sum(errorsMaj_train)

            errrate_valid = errsum_valid / numqueries_valid
            errrate_train = errsum_train / numqueries_train
            
            errrateMaj_valid  = errsumMaj_valid / numqueriesMaj_valid
            errrateMaj_train = errsumMaj_train / numqueriesMaj_train

        #try on test set
        opt_k = np.argmax(aucs_valid)

        labels = np.empty((numqueries_test,), dtype='int')
        labels_train = np.empty((numqueries_train))

        for i in xrange(numqueries_test):
            b = np.bincount(queryvotes_test[i, :opt_k + 1])
            labels[i] = np.argmax(b)  # get winning class

        for i in xrange(numqueries_train):
            b = np.bincount(queryvotes_train[i, :opt_k + 1])
            labels_train[i] = np.argmax(b)

        errors = labels != y_test
        errorsMaj = errors[y_test == 0]
        phys_score_test = self.physionet_score(y_test, labels)

        try:
            auc_test = roc_auc_score(y_test, labels)
        except:
            auc_test = None

        errsum_test = sum(errors)
        errsumMaj_test = sum(errorsMaj)
        

        errrate_test = errsum_test / numqueries_test
        errrateMaj_test  = errsumMaj_test / numqueriesMaj_test

        train_out = {
            "Misclassification error": errrate_train,
            "Majority missclassification error": errrateMaj_train,
            "AUC":auc_train, "physionet_score":phys_scores_train
        }
        
        valid_out = {
            "Misclassification error": errrate_valid,
            "Majority missclassification error": errrateMaj_valid,
            "AUC":aucs_valid, "physionet_score":phys_scores_valid
        } 


        results = {
            "train" : train_out,
            "validation":valid_out,
            "physionet_score_test":phys_score_test,
            "Optimal K" : opt_k,
            "Missclassification error in test": 1-errrate_test,
            "Majority missclassification error in test": 1-errrateMaj_test,
            "Physionet_score in test": phys_score_test,
            "AUC_test": auc_test
        }
        
        return results
                      
      
