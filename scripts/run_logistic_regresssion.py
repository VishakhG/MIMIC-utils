import pandas as pd
import gzip
import numpy as np
from feature_tester import FeatureTester

outDir = "/data/ml2/vishakh/patient-similarity/mortality-pred/feature-testing/"

feat_dir = "/data/ml2/vishakh/SHARED/mimic-features-notes/features_concat.csv"
targ_path = "data/ml2/vishakh/SHARED/mimic-features-notes/targets_concat.csv"

ft = FeatureTester(outDir = outDir)
ft.load_from_csv()
ft.prepare_for_testing()
ft.logistic_regression(save_lc=True)
print ft.current_result
