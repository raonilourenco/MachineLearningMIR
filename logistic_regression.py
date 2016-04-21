########################################################################3
# Computing best parameters for Linear SVMs
########################################################################3

from evaluationstrategy import compute_metrics_classifier
from sklearn.linear_model import LogisticRegression
import sys
import pandas as pd
import json

if __name__ == "__main__":


    #python logistic_regression.py [input_file] [output.json] [--addTreeFeatures]
    if len(sys.argv) >= 3:
        add_features = False;
        if len(sys.argv) == 4:
            if sys.argv[3] == "--addTreeFeatures":
                add_features = True;
 

        df = pd.read_csv(sys.argv[1], sep=',', header=0)

        X = df.iloc[:,:-1].as_matrix()
        y = df.iloc[:,-1].as_matrix().astype('U')

        param_grid = {'C':[0.0001,0.001,0.01,0.1,1,10,100,1000]}

        classifier = LogisticRegression()

        report_map = compute_metrics_classifier(classifier=classifier, param_grid = param_grid, X = X, y = y, cv_parameter_estimation = 3, add_features = add_features)

        f = open(sys.argv[2], "w")
        json.dump(report_map, f)
        f.close()
