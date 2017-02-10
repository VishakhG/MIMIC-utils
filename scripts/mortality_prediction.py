import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from metric_learn  import LMNN
from metric_learn import NCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.model_selection import learning_curve

print("Loading Data")

data_path = "/data/ml2/vishakh/mimic_out"
sparse_path = os.path.join(data_path, "sparse_mat-umls.csv")
tfidf_path = os.path.join(data_path, "tfidf_features.npy")
mortality_path = os.path.join(data_path, "mortality_targets.csv")

print("Using real data")

sparse, targets = [np.array(pd.read_csv(i)) for i in [sparse_path, mortality_path]]

sparse = sparse[:,1:len(sparse)] 
sums = np.sum(sparse, axis=0)

low = np.percentile(sums, 30)
high = np.percentile(sums, 80)

sparse = sparse[: ,(sums > low) & (sums < high)]

split = int(np.floor(.8 * len(sparse)))

x_train = sparse[: split]
x_test = sparse[split :]

y_train = targets[:split].ravel()
y_test = targets[split:].ravel()

dataset_collection = [sparse]   
file_out_names = ["predacc_raw-umls.csv"]

print("Begin the loop")


def test_classifier(classifier, params, dat = None):
    global x_train
    global x_test
    global y_train
    global y_testW

    if dat != None:
        x_train, x_test  = dat
        `
    clf = GridSearchCV(classifier, params, scoring='accuracy')
    clf.fit(x_train, y_train)

    top_score = clf.best_score_

    train_pred = clf.predict(x_train)
    test_pred = clf.predict(x_test)
    accuracy = [accuracy_score(i[0], i[1]) for i in [(train_pred, y_train), (test_pred, y_test)]]

    try:
        roc = [roc_auc_score(i[0], i[1]) for i in [(train_pred, y_train), (test_pred, y_test)]]

    except:
        roc = None 

    test_Mmask = y_test == 0
    train_Mmask = y_train == 0

    test_Mpred = test_pred[test_Mmask]
    test_Mtrue = test_pred[test_Mmask]

    train_Mpred = train_pred[train_Mmask]
    train_Mtrue = train_pred[train_Mmask]

    
    Macc_test = (test_Mpred, test_Mtrue)
    Macc_train = (train_Mpred, train_Mtrue)

    majority_accuracy  = [accuracy_score(i[0], i[1]) for i in  [Macc_train, Macc_test]]

    try:
        majority_roc = [roc_auc_score(i[0], i[1]) for i in [Macc_train, Macc_test]]

    except:
        majority_roc = None

    return {'cv_score':top_score,
            'accuracy': accuracy,
            'roc': roc,
            'majority_accuracy':majority_accuracy,
            'majority_roc':majority_roc}


for d in range(len(dataset_collection)):

   print("Starting loop for" + " " + file_out_names[d][1:-3])

    data = dataset_collection[d]
    classifier_stats = dict.fromkeys([
        "logistic_regression", "nearest_neighbor",
        "random_foreset", "metric_learning"])
    
    print("Logistic Regression")

    lr = LogisticRegression()
    classifier_stats["logistic_regression"] = test_classifier(lr)

    pickle.dump(classifier_stats, open( "/data/ml2/vishakh/mimic_out/pred_statscheckpoint.pk", 'wb'))
    
    print("Random Forest")
    rf = RandomForestClassifier()
    rf_params = {'criterion':('gini', 'entropy')}    
    classifier_stats["random_forest"] = test_classifier(lr, lr_params)

    pickle.dump(classifier_stats, open( "/data/ml2/vishakh/mimic_out/pred_statscheckpoint.pk", 'wb'))

    print("Nearest Neighbors")
    nn = KNeighborsClassifier()
    nn_params = {"n_neighbors" : range(5, max(6, len(data)/10)), 'leaf_size':range(30,100)}
    
    classifier_stats["nearest_neighbors"] = test_classifier(nn, nn_params)
    pickle.dump(classifier_stats, open( "/data/ml2/vishakh/mimic_out/pred_statscheckpoint.pk", 'wb'))

    print("Metric learning")

    nca1 = NCA(max_iter=1000, learning_rate=0.01)
    nca1.fit(x_train], y_train) 
    t_x_train = nca1.transform()
    nca2 = NCA(max_iter=1000, learning_rate=0.01)
    nca2.fit(x_test, np.array(y_test))
    t_x_test = nca2.transform()
    dat = [t_x_train, t_x_test]
    nn_metric = KNeighborsClassifier()
    nn_metric_params = {"n_neighbors" : range(5, max(6, len(data)/10)),
                        'leaf_size':range(30,100)}

    classifier_stats['nn_metric'] = test_classifier(nn_metric, nn_metric_params, dat)
    


    out = pickle.dump(classifier_stats, open("/data/ml2/vishakh/mimic_out/predacc_raw_umls.csv", 'wb'))
