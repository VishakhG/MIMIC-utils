""" Python implementation of the OASIS algorithm.
Graham Taylor

Based on Matlab implementation of:
Chechik, Gal, et al.
"Large scale online learning of image similarity through ranking."
The Journal of Machine Learning Research 11 (2010): 1109-1135.
"""

from __future__ import division
import numpy as np
from sklearn.base import BaseEstimator
from datetime import datetime
import matplotlib.pyplot as plt
import cPickle as pickle
import os
import gzip
from sys import stdout
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def snorm(x):
    """Dot product based squared Euclidean norm implementation

    See: http://fseoane.net/blog/2011/computing-the-vector-norm/
    Doesn't matter if row or column vectors are passed in
    Since everything is flattened
    """
    return np.dot(x.flatten().T, x.flatten())

def physionet_score( y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        Se = tp/(tp + fn)
        P = tp/(tp + fp)
        score = min(Se, P)
        return score

def make_psd(W):
    """ Make matrix positive semi-definite. """
    w, v = np.linalg.eig(0.5 * (W + W.T))  # eigvec in columns
    D = np.diagflat(np.maximum(w, 0))
    W[:] = np.dot(np.dot(v, D), v.T)


def symmetrize(W):
    """ Symmetrize matrix."""
    W[:] = 0.5 * (W + W.T)


class Oasis(BaseEstimator):
    """ OASIS algorithm. """

    def __init__(self, aggress=0.1, random_seed=None, do_sym=False,
                 do_psd=False, n_iter=10 ** 6, save_every=None, sym_every=1,
                 psd_every=1, save_path=None):

        self.aggress = aggress
        self.random_seed = random_seed
        self.n_iter = n_iter
        self.do_sym = do_sym
        self.do_psd = do_psd
        self.sym_every = sym_every
        self.psd_every = psd_every
        self.save_path = save_path
        self.batch_loss = []
        if save_every is None:
            self.save_every = int(np.ceil(self.n_iter / 10))
        else:
            self.save_every = save_every

        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

    def _getstate(self):
        return self._weights

    def _setstate(self, state):
        weights = state
        self._weights = weights

    def _save(self, n=None):
        """ Pickle the model."""
        fname = self.save_path + "/model%04d.pklz" % n

        try:
            f = gzip.open(fname, 'wb')
            state = self._getstate()
            pickle.dump(state, f)
            f.close()
        except:
            f.close()
            print("Can't compress state with gzip, using alternative method")
            state = self._getstate()
            np.save(fname, state)

    def read_snapshot(self, fname):
        """ Read model state snapshot from gzipped pickle. """
        try:
            f = gzip.open(fname, 'rb')
            state = pickle.load(f)
            self._setstate(state)
            
        except:
            f.close()
            state = np.load(fname)
            self._setstate(state)

    def _fit_batch(self, W, X, y, class_start, class_sizes, n_iter,
                   verbose=False):
        """ Train batch inner loop. """

        batch_loss = []
        loss_steps_batch = np.empty((n_iter,), dtype='bool')
        n_samples, n_features = X.shape
        
        # assert(W.shape[0] == n_features)
        # assert(W.shape[1] == n_features)

        for ii in xrange(n_iter):
            if verbose:
                if np.mod(ii + 1, 100) == 0:
                    print ".",
                if np.mod(ii + 1, 1000) == 0:
                    print "%d" % (ii + 1),
                if np.mod(ii + 1, 10000) == 0:
                    print "[%s]" % str(datetime.now())
                # http://stackoverflow.com/questions/5101151/print-without-newline-under-function-doesnt-work-as-it-should-in-python
                stdout.flush()

            # Sample a query image
            p_ind = self.init.randint(n_samples)
            label = y[p_ind]

            # Draw random positive sample
            pos_ind = class_start[label] + \
                self.init.randint(class_sizes[label])

            # Draw random negative sample
            neg_ind = self.init.randint(n_samples)
            while y[neg_ind] == label:
                neg_ind = self.init.randint(n_samples)

            p = X[p_ind]

            samples_delta = X[pos_ind] - X[neg_ind]

            loss = 1 - np.dot(np.dot(p, W), samples_delta)


            if loss > 0:
                batch_loss.append(loss)
                # Update W
                grad_W = np.outer(p, samples_delta)

                loss_steps_batch[ii] = True

                norm_grad_W = np.dot(p, p) * np.dot(samples_delta,
                                                    samples_delta)

                # constraint on the maximal update step size
                tau_val = loss / norm_grad_W  # loss / (V*V.T)
                tau = np.minimum(self.aggress, tau_val)

                W += tau * grad_W
                # plt.figure(10)
                # plt.imshow(W,interpolation='nearest')
                # plt.draw()

            #print "loss = %f" % loss
        
        return W, loss_steps_batch, batch_loss

    def fit(self, X, y, overwrite_X=False, overwrite_y=False, verbose=False):
        """ Fit an OASIS model. """
        
        if not overwrite_X:
            X = X.copy()
        if not overwrite_y:
            y = y.copy()

        n_samples, n_features = X.shape

        self.init = np.random.RandomState(self.random_seed)

        # Parameter initialization
        self._weights = np.eye(n_features).flatten()
        # self._weights = np.random.randn(n_features,n_features).flatten()
        W = self._weights.view()
        W.shape = (n_features, n_features)

        ind = np.argsort(y)

        y = y[ind]
        X = X[ind, :]

        classes = np.unique(y)
        classes.sort()

        n_classes = len(classes)

        # Translate class labels to serial integers 0, 1, ...
        y_new = np.empty((n_samples,), dtype='int')

        for ii in xrange(n_classes):
            y_new[y == classes[ii]] = ii

        y = y_new
        class_sizes = [None] * n_classes
        class_start = [None] * n_classes

        for ii in xrange(n_classes):
            class_sizes[ii] = np.sum(y == ii)
            # This finds the first occurrence of that class
            class_start[ii] = np.flatnonzero(y == ii)[0]
        
        loss_steps = np.empty((self.n_iter,), dtype='bool')
        n_batches = int(np.ceil(self.n_iter / self.save_every))
        steps_vec = np.ones((n_batches,), dtype='int') * self.save_every
        steps_vec[-1] = self.n_iter - (n_batches - 1) * self.save_every

        if verbose:
            print 'n_batches = %d, total n_iter = %d' % (n_batches,
                                                         self.n_iter)

        for bb in xrange(n_batches):
            if verbose:
                print 'run batch %d/%d, for %d steps ("." = 100 steps)\n' \
                      % (bb + 1, n_batches, self.save_every)

            W, loss_steps_batch, batch_loss = self._fit_batch(W, X, y,
                                                  class_start,
                                                  class_sizes,
                                                  steps_vec[bb],
                                                  verbose=verbose)

            #print "loss_steps_batch = %d" % sum(loss_steps_batch)
            print("batch loss{0}").format(sum(batch_loss))
            self.batch_loss.append(sum(batch_loss))
            
            loss_steps[bb * self.save_every:min(
                (bb + 1) * self.save_every, self.n_iter)] = loss_steps_batch

            if self.do_sym:
                if np.mod(bb + 1, self.sym_every) == 0 or bb == n_batches - 1:
                    if verbose:
                        print "Symmetrizing"
                    symmetrize(W)

            if self.do_psd:
                if np.mod(bb + 1, self.psd_every) == 0 or bb == n_batches - 1:
                    if verbose:
                        print "PSD"
                    make_psd(W)

            if self.save_path is not None:
                self._save(bb + 1)  # back up model state

        return self

    def get_weights(self):
        return self._weights.view()
        
    
    def predict(self, X_test, X_train, y_test, y_train, maxk=200):
        '''
        Evaluate an OASIS model by KNN classification
        Examples are in rows
        '''

        W = self._weights.view()
        W.shape = (np.int(np.sqrt(W.shape[0])), np.int(np.sqrt(W.shape[0]))) # 1 

        maxk = min(maxk, X_train.shape[0])  # K can't be > numcases in X_train
        
        numqueries = X_test.shape[0]
        numqueries_train = X_train.shape[0]

        numqueriesMaj = X_test[y_test == 0].shape[0]
        numqueriesMaj_train = X_train[y_train == 0].shape[0]

        precomp = np.dot(W, X_train.T)

        # compute similarity scores
        s = np.dot(X_test, precomp)
        s_train = np.dot(X_train, precomp)

        # argsort sorts in ascending order
        # so we need to reverse the second dimension
        
        ind = np.argsort(s, axis=1)[:, ::-1]
        ind_train = np.argsort(s_train, axis = 1)[:, :: -1]
        
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

            phys_scores.append(physionet_score(y_test, labels))
            phys_scores_train.append(physionet_score(y_train, labels_train))

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
                "train": [errrate_train],
                "test": [errrate]
            },
            'auc_roc:': {
                "train":[aucs_train],
                "test":[aucs]
            },

            'majority_accuracy:':{
                "train": [errrateMaj_train],
                "test": [errrateMaj]
            },
                    
            'physionet score':{
                "train" : [phys_scores_train],
                "test" : [phys_scores]
            }
        }       

        return result_out
        
if __name__ == "__main__":
    from sklearn import datasets
    digits = datasets.load_digits()

    X_train = digits.data[500:] / 16
    X_test = digits.data[:500] / 16
    y_train = digits.target[500:]
    y_test = digits.target[:500]

    model = Oasis(n_iter=1000, do_psd=True, psd_every=3,
                  save_path="/tmp/gwtaylor/oasis_test").fit(X_train, y_train,
                                                            verbose=True)

    errrate = model.predict(X_test, X_train, y_test, y_train, maxk=1000)
    print "Min error rate: %6.4f at k=%d" \
          % (min(errrate), np.argmin(errrate) + 1)
    plt.figure()
    plt.plot(errrate)

    n_features = X_train.shape[1]
    W = model._weights.view()
    W.shape = (n_features, n_features)

    W[0:5, 0:5]
