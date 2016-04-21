########################################################################3
# Computing best parameters for Decision Trees
########################################################################3

from evaluationstrategy import compute_metrics_classifier
from sklearn.ensemble import GradientBoostingClassifier
import sys
import pandas as pd
import json

if __name__ == "__main__":
    #python decisiontree.py [input_file] [output.json]
    if len(sys.argv) == 3:
        df = pd.read_csv(sys.argv[1], sep=',', header=0)

        X = df.iloc[:,:-1].as_matrix()
        y = df.iloc[:,-1].as_matrix().astype('U')

        param_grid = {'learning_rate':[0.1]}

        classifier = GradientBoostingClassifier()

        report_map = compute_metrics_classifier(classifier=classifier, param_grid = param_grid, X = X, y = y, cv_parameter_estimation = 3)

        f = open(sys.argv[2], "w")
        json.dump(report_map, f)
        f.close()
