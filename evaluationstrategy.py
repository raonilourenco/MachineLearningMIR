from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import numpy as np


def compute_metrics_classifier (classifier, param_grid, X, y, metric = accuracy_score, percentage_train = 0.7, cv_parameter_estimation = 3):
    nsamples = X.shape[0]
    nsamples_train = np.floor(percentage_train * nsamples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(nsamples-nsamples_train))

    grid_search = GridSearchCV(classifier, param_grid = param_grid, cv = cv_parameter_estimation, n_jobs=8)
    grid_search.fit(X_train, y_train) #finding best parameters

    y_hat_train = grid_search.predict(X_train)
    y_hat_test = grid_search.predict(X_test)

    metric_train = metric(y_train, y_hat_train)
    metric_test = metric(y_test, y_hat_test)

    best_params = grid_search.best_params_


    report_map = {}
    report_map['classifier'] = str(classifier).split("(",1)[0]
    report_map['best_params'] = best_params
    report_map['metric'] = accuracy_score.func_name
    report_map['metric_train'] = metric_train
    report_map['metric_test'] = metric_test

    return(report_map)

