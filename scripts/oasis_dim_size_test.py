

import numpy as np 
import pandas as pd
from feature_tester import FeatureTester
#Check the output for various dimensions and sizes

#read all them in 
ft = FeatureTester()
features = np.array(pd.read_csv("/data/ml2/vishakh/temp/feature_mat_first48.csv"))
targets = np.array(pd.read_csv("/data/ml2/vishakh/temp/target_mat_first48.csv"))

#1000 rows, 100 dimension
feat = features[1:1000]
feat = feat[:,1:1000]
targ = targets[1:1000]

ft.load_from_array(feature_set=feat, target_set=targ)
ft.prepare_for_testing(sparse=False)
out = ft.oasis()

print "shape {0}".format(feat.shape)
print " AUC in test {0}".format(out['Auc in test'])
print "Physionet_score in test{0}".format(out['Physionet_score in test'])
print "optimal k {0}".format( out['Optimal K'])



#10000 rows 100 dimensions
feat = features[1:10000]
targ = targets[1:10000]
feat = feat[:,1:100]
ft.load_from_array(feature_set=feat, target_set=targ)
ft.prepare_for_testing(sparse=False)
out = ft.oasis()
print "shape {0}".format(feat.shape)
print " AUC in test {0}".format(out['Auc in test'])
print "Physionet_score in test{0}".format(out['Physionet_score in test'])
print "optimal k {0}".format( out['Optimal K'])


#10000 dimensions 100 rows
feat = features[1:100]
targ = targets[1:100]
feat = feat[:,1:10000]
ft.load_from_array(feature_set=feat, target_set=targ)
ft.prepare_for_testing(sparse=False)
out = ft.oasis()
print "shape {0}".format(feat.shape)
print " AUC in test {0}".format(out['Auc in test'])
print "Physionet_score in test{0}".format(out['Physionet_score in test'])
print "optimal k {0}".format( out['Optimal K'])


#10000 dimensions 1000 rows
feat = features[1:1000]
targ = targets[1:1000]
feat = features[1:10000]
ft.load_from_array(feature_set=feat, target_set=targ)
ft.prepare_for_testing(sparse=False)
out = ft.oasis()


print "shape {0}".format(feat.shape)
print " AUC in test {0}".format(out['Auc in test'])
print "Physionet_score in test{0}".format(out['Physionet_score in test'])
print "optimal k {0}".format( out['Optimal K'])



