import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC


def Linear_SVC_Search(X_train, X_test, y_train, y_test):
    param = {'penalty': ['l1', 'l2'], 'loss': ['hinge', 'squared_hinge']}
    linearSVCSearch = RandomizedSearchCV(LinearSVC(), param, cv=10, n_jobs=-1)
    linearSVCSearch.fit(X_train, y_train)

    print('Best penalty:', linearSVCSearch.best_estimator_.get_params()['penalty'])
    print('Best loss:', linearSVCSearch.best_estimator_.get_params()['loss'])
    print('best params : ', linearSVCSearch.best_params_)
    print('Best score : ', linearSVCSearch.best_score_)

    print(linearSVCSearch.best_params_)
    Linearcsv_best_model = linearSVCSearch.best_estimator_

    Linearcsv_best_model.fit(X_train, y_train)
    predicted = Linearcsv_best_model.predict(X_test)

    print(confusion_matrix(predicted, y_test))
    print(classification_report(predicted, y_test))
