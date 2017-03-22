from experiment import Experiment
from feature_tester import FeatureTester

e = Experiment()
ft = FeatureTester()
e.begin_experiment(fName = "benchmarks-hp", note="More hyperparameters in random_forest")


ft.load_from_csv()
ft.prepare_for_testing()
ft.logistic_regression()
e.log(ft.get_metadata(), ft.get_result())
ft.random_forest()
e.log(ft.get_metadata(), ft.get_result())
ft.nearest_neighbors()
e.log(ft.get_metadata(), ft.get_result())
e.view_experiment()
e.end_experiment()
