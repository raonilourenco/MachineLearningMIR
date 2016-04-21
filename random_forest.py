########################################################################3
# Computing best parameters for Decision Trees
########################################################################3

from evaluationstrategy import compute_metrics_classifier
from sklearn.ensemble import RandomForestClassifier
import sys
import pandas as pd
import json

if __name__ == "__main__":

    #python random_forest.py [input_file] [output.json] [--addTreeFeatures]
    if len(sys.argv) >= 3:
        add_features = False;
        if len(sys.argv) == 4:
            if sys.argv[3] == "--addTreeFeatures":
                add_features = True;
 

        df = pd.read_csv(sys.argv[1], sep=',', header=0)

        X = df.iloc[:,:-1].as_matrix()
        y = df.iloc[:,-1].as_matrix().astype('U')

        param_grid = {'n_estimators':[5,10,15,20,25,30,40,50], 'criterion':['gini', 'entropy'], 'max_features':['auto', 'sqrt', 'log2', None], 'max_depth':[10,20,30,40,50,100,None]}

        classifier = RandomForestClassifier()

        report_map = compute_metrics_classifier(classifier=classifier, param_grid = param_grid, 
            X = X, y = y, cv_parameter_estimation = 3, add_features = add_features)

        f = open(sys.argv[2], "w")
        json.dump(report_map, f)
        f.close()
