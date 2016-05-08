from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from evaluationstrategy import compute_metrics_classifier
import pandas as pd
import sys
import os
import json
import yaml


#

if __name__ == "__main__":
    n_runs = 18

    param_grids = {
        "SVC": [{'kernel': ['linear'], 'C': [0.0001,0.001,0.01,0.1,1, 10, 100]},{'kernel':['rbf'], 'gamma':['auto', 1,0.1,0.01, 0.001],'C':[0.0001,0.001,0.01,0.1,1,10,100]},{'kernel':['poly'], 'C':[0.0001,0.001,0.01,0.1,1,10,100], 'degree':[2,3,4,5]}],
        "LogisticRegression": {'C':[0.0001,0.001,0.01,0.1,1,10,100,1000]},
        "RandomForestClassifier":{'n_estimators':[5,10,15,20,25,30,40,50], 'criterion':['gini', 'entropy'], 'max_features':['auto', 'sqrt', 'log2', None], 'max_depth':[10,20,30,40,50,100,None]}
    }


    to_do = [
            {
             "feature":"Features/Lower_res/features_all.csv",
             "use_tree":False,
             "classifier":LogisticRegression(),
             "classifier_name":"LogisticRegression"
            },
            {
             "feature":"Features/Lower_res/features_lowlevel.csv",
             "use_tree":False,
             "classifier":LogisticRegression(),
             "classifier_name":"LogisticRegression"
            },
            {
             "feature":"Features/Lower_res/features_selected_all.csv",
             "use_tree":False,
             "classifier":LogisticRegression(),
             "classifier_name":"LogisticRegression"
            },
            {
             "feature":"Features/Default/features_lowlevel.csv",
             "use_tree":False,
             "classifier":LogisticRegression(),
             "classifier_name":"LogisticRegression"
            },
            {
             "feature":"Features/Higher_res/features_lowlevel.csv",
             "use_tree":False,
             "classifier":RandomForestClassifier(),
             "classifier_name":"RandomForestClassifier"
            },
            {
             "feature":"Features/Lower_res/features_all.csv",
             "use_tree":True,
             "classifier_name":"SVC",
             "classifier":SVC()
            },
           ]



    for id_experiment in range(len(to_do)):
        print "========== running experiment " +str(id_experiment) +" of "+str(len(to_do))
        df = pd.read_csv(to_do[id_experiment]["feature"], sep=',', header=0)
        to_do[id_experiment]["metric_tests"] = []
        X = df.iloc[:,:-1].as_matrix()
        y = df.iloc[:,-1].as_matrix().astype('U')
        for id_run in range(n_runs):
            print("id run: "+ str(id_run))
            report_map = compute_metrics_classifier(classifier=to_do[id_experiment]["classifier"], param_grid = param_grids[to_do[id_experiment]["classifier_name"]], 
            X = X, y = y, cv_parameter_estimation = 3, add_features = to_do[id_experiment]["use_tree"])
            to_do[id_experiment]["metric_tests"].append(report_map["metric_test"])
        del to_do[id_experiment]["classifier"]

    f = open("summary_plots_18_runs.json", "w")
    json.dump(to_do, f)
    f.close()

