########################################################################3
# Computing best parameters for Linear SVMs
########################################################################3

from evaluationstrategy import compute_metrics_classifier
from sklearn.svm import SVC
import sys
import pandas as pd
import json

if __name__ == "__main__":

    #python svm.py [input_file] [output.json] [--addTreeFeatures]
    if len(sys.argv) >= 3:
        add_features = False;
        if len(sys.argv) == 4:
            if sys.argv[3] == "--addTreeFeatures":
                add_features = True;
 

        df = pd.read_csv(sys.argv[1], sep=',', header=0)
        X = df.iloc[:,:-1].as_matrix()
        y = df.iloc[:,-1].as_matrix().astype('U')
        param_grid = [{'kernel': ['linear'], 'C': [0.0001,0.001,0.01,0.1,1, 10, 100]},{'kernel':['rbf'], 'gamma':['auto', 1,0.1,0.01, 0.001],'C':[0.0001,0.001,0.01,0.1,1,10,100]},{'kernel':['poly'], 'C':[0.0001,0.001,0.01,0.1,1,10,100], 'degree':[2,3,4,5]}]
        classifier = SVC()
        report_map = compute_metrics_classifier(classifier=classifier, param_grid = param_grid, X = X, y = y, cv_parameter_estimation = 3, add_features = add_features)
        f = open(sys.argv[2], "w")
        json.dump(report_map, f)
        f.close()
