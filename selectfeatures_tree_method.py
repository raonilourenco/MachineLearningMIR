from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import sys
import pandas as pd
import json
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

if __name__ == "__main__":
    #python selectfeatures.py [input_file] [output_file - new_data.csv]
    if len(sys.argv) == 3:
        df = pd.read_csv(sys.argv[1], sep=',', header=0)
        

        X = df.iloc[:,:-1].as_matrix()
        y = df.iloc[:,-1].as_matrix().astype('U')

        lsvc = ExtraTreesClassifier().fit(X, y)
        model = SelectFromModel(lsvc, prefit=True, threshold=0.01)
        idx_important_features = model.get_support()

        features = df.iloc[:,:-1]

        print(np.where(idx_important_features)[0])

        X_2 = features.iloc[:,idx_important_features]

        df2 = pd.DataFrame(X_2)
        df2['class'] = y

        df2.to_csv(sys.argv[2], sep=',', index=False)
