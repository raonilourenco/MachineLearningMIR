from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    
    with np.errstate(divide='ignore', invalid='ignore'):
        train_normalized = (train - train.min(axis=0))/(train.max(axis=0)-train.min(axis=0))
        test_normalized = (test - train.min(axis=0))/(train.max(axis=0)-train.min(axis=0))
        
        train_normalized[train_normalized == np.inf] = 0
        train_normalized = np.nan_to_num(train_normalized)
        
        test_normalized[test_normalized == np.inf] = 0
        test_normalized = np.nan_to_num(test_normalized)
        
    return train_normalized,test_normalized


def evaluate_classifier (classifier, parameters, X, y, metric = accuracy_score, percentage_train = 0.7, add_features=False):
    nsamples = X.shape[0]
    nsamples_train = np.floor(percentage_train * nsamples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(nsamples-nsamples_train))
    X_train, X_test = feature_normalization(X_train, X_test)

    if(add_features):
        [X_train, X_test] = compute_new_features(X_train,y_train,X_test,y_test)

    classifier.set_params(**parameters)

    classifier.fit(X_train, y_train) #finding best parameters

    y_hat_test = classifier.predict(X_test)

    metric_test = metric(y_test, y_hat_test)

    return(metric_test)



def compute_metrics_classifier (classifier, param_grid, X, y, metric = accuracy_score, percentage_train = 0.7, cv_parameter_estimation = 3, add_features=False):
    nsamples = X.shape[0]
    nsamples_train = np.floor(percentage_train * nsamples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(nsamples-nsamples_train))
    X_train, X_test = feature_normalization(X_train, X_test)

    if(add_features):
        [X_train, X_test] = compute_new_features(X_train,y_train,X_test,y_test)

    grid_search = GridSearchCV(classifier, param_grid = param_grid, cv = cv_parameter_estimation, n_jobs=-1, verbose = 100)
    grid_search.fit(X_train, y_train) #finding best parameters

    y_hat_train = grid_search.predict(X_train)
    y_hat_test = grid_search.predict(X_test)

    metric_train = metric(y_train, y_hat_train)
    metric_test = metric(y_test, y_hat_test)

    best_params = grid_search.best_params_


    report_map = {}
    report_map['classifier'] = str(classifier).split("(",1)[0]
    report_map['best_params'] = best_params
    report_map['metric'] = str(accuracy_score)
    report_map['metric_train'] = metric_train
    report_map['metric_test'] = metric_test

    return(report_map)



def compute_new_features(X_train,y_train,X_test,y_test):
    classifier = DecisionTreeClassifier(max_leaf_nodes=50)
    classifier.fit(X_train,y_train)
    idx_train = classifier.apply(X_train)
    idx_train = idx_train.reshape([-1,1])
    enc = OneHotEncoder()
    enc.fit(idx_train)
    new_features_train = enc.transform(idx_train).toarray()
    
    idx_test = classifier.apply(X_test)
    idx_test = idx_test.reshape([-1,1])
    new_features_test = enc.transform(idx_test).toarray()

    return ([np.hstack([X_train,new_features_train]), np.hstack([X_test,new_features_test])])

