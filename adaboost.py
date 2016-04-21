########################################################################3
# Computing best parameters for Decision Trees
########################################################################3

from evaluationstrategy import compute_metrics_classifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import sys
import pandas as pd
import json

if __name__ == "__main__":
    #python adaboost.py [input_file] [output.json] [--addTreeFeatures]
    if len(sys.argv) >= 3:
        add_features = False;
        if len(sys.argv) == 4:
            if sys.argv[3] == "--addTreeFeatures":
                add_features = True;
            
        df = pd.read_csv(sys.argv[1], sep=',', header=0)

        X = df.iloc[:,:-1].as_matrix()
        y = df.iloc[:,-1].as_matrix().astype('U')

        #param_grid = {'base_estimator':[DecisionTreeClassifier(max_depth=3),DecisionTreeClassifier(max_depth=5),DecisionTreeClassifier(max_depth=8),DecisionTreeClassifier(max_depth=13),DecisionTreeClassifier(max_depth=21)],'n_estimators':[5,10,15,20,25,30,40,50,100,200], 'learning_rate':[0.001,0.01,0.1,0.2,0.3]}

        param_grid = {'n_estimators':[5,10,15,20,25,30,40,50,100,200], 'learning_rate':[0.001,0.01,0.1,0.2,0.3]}

        classifier = AdaBoostClassifier()

        report_map = compute_metrics_classifier(classifier=classifier, param_grid = param_grid, 
            X = X, y = y, cv_parameter_estimation = 3, add_features = add_features)

        f = open(sys.argv[2], "w")
        json.dump(report_map, f)
        f.close()
