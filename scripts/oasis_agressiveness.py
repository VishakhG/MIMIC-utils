import numpy as np
import pandas as pd
import cPickle as pickle
from os.path import join as pjoin
import gzip

from feature_tester import FeatureTester

ft = FeatureTester(fname = "agress")

baseDir = "/data/ml2/vishakh/mimic-out-umls/"
featureDir = pjoin(baseDir, "feature_mat_first48.csv")
targetDir = pjoin(baseDir, "target_mat_first48.csv")

nrows = 1000

ft.load_from_csv(feature_path = featureDir, target_path = targetDir, nrows = nrows)

ft.prepare_for_testing(sparse = False)

test_range = [0.1, 0.4, .7]
results = []
for idx, i in enumerate(test_range):
    print "using {0} as the agressiveness".format(i)
    print "{0} of {1}".format(idx, len(test_range))
    out = ft.oasis(aggress=i, maxk = 100)
    results.append(out)
    
f = gzip.open("/data/ml2/vishakh/temp/oasis/oasisAgressTests", 'wb')
pickle.dump(results, f)
f.close()
    



