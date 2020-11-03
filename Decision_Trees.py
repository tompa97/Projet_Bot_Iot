import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score


def Trees(X_train, X_test, y_train, y_test):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=20, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=25, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    print("Randomized Tree is fitting fitting")
    rf_random.fit(X_train, y_train)

    print("Models paramaters : ")
    print(rf_random.best_params_)
    predictions = rf_random.predict(X_test)

    print(classification_report(predictions, y_test))
    print(confusion_matrix(predictions, y_test))
