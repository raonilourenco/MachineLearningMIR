########################################################################3
# Computing best parameters for Decision Trees
########################################################################3

from evaluationstrategy import compute_metrics_classifier
from sklearn.tree import DecisionTreeClassifier
import sys
import pandas as pd
import json

if __name__ == "__main__":
    #python decisiontree.py [input_file] [output.json]
    if len(sys.argv) == 3:
        df = pd.read_csv(sys.argv[1], sep=',', header=0)

        X = df.iloc[:,:-1].as_matrix()
        y = df.iloc[:,-1].as_matrix().astype('U')

        param_grid = {'criterion':['gini', 'entropy'], 'splitter': ['best', 'random'],
                    'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 'min_samples_split': [2,3,4,5],
                    'min_samples_leaf': [1,2,3,4,5,6]}

        classifier = DecisionTreeClassifier()

        report_map = compute_metrics_classifier(classifier=classifier, param_grid = param_grid, 
            X = X, y = y, cv_parameter_estimation = 2)

        f = open(sys.argv[2], "w")
        json.dump(report_map, f)
        f.close()
