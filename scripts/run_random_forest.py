import pandas
import numpy
from feature_tester import FeatureTester

outDir = "/data/ml2/vishakh/patient-similarity/mortality-pred/feature-testing"

nrows = 1000
ncols = None

ft = FeatureTester(outDir = outDir)
ft.load_from_csv(nrows = nrows, ncols = ncols)
ft.prepare_for_testing(sparse=False)
ft.random_forest(save_lc=True)
ft.current_result()


