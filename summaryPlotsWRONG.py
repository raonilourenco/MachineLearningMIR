from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from evaluationstrategy import evaluate_classifier
import pandas as pd
import sys
import os
import json
import yaml


feature_resolutions = ["Default", "Higher_res", "Lower_res"]
datasets = ["Features_all", "Features_lowlevel", "Features_rhythm", "Features_selected_all", "Features_tonal"]
classifier_names = set()

if len(sys.argv) != 4:
    print("usage: summaryPlots.py [feature_folder] [output_classifier_folder] [output_file.json] ")
else:
    results = {}
    feature_folder = sys.argv[1]
    output_classifier_folder = sys.argv[2]
    output_file = sys.argv[3]
    nruns = 10


    for resolution in feature_resolutions:
        for dataset in datasets:
            output_dataset_folder = os.path.join(output_classifier_folder, resolution, dataset)
            for algorithm_name in os.listdir(output_dataset_folder):
                f = open(os.path.join(output_dataset_folder, algorithm_name), 'rb')
                json_data = yaml.safe_load(f)
                f.close()
                classifier_names.add(json_data["classifier"])
                classifier = eval(json_data["classifier"])()
                parameters = json_data["best_params"]

                feature_file = os.path.join(feature_folder, resolution, dataset.lower() + ".csv")
                df = pd.read_csv(feature_file, sep=',', header=0)
                X = df.iloc[:,:-1].as_matrix()
                y = df.iloc[:,-1].as_matrix().astype('U')

                add_features = algorithm_name.find("_with_tree_features") > 0                
                results[(resolution, dataset, json_data["classifier"], add_features)] = []

                for iteration in range(nruns):
                    accuracy = evaluate_classifier(classifier=classifier, parameters = parameters, X = X, y = y, add_features = add_features)
                    results[(resolution, dataset, json_data["classifier"], add_features)].append(accuracy)


    f = open(output_file, "w")
    json.dump(results, f)
    f.close()


    