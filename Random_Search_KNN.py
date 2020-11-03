from Data_Manip import retrieve_data
from sklearn.metrics import make_scorer, roc_auc_score, \
    recall_score, matthews_corrcoef, balanced_accuracy_score, accuracy_score, \
    precision_score, f1_score, \
    confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier


def ResearchBestKNN(X_train, X_test, y_train, y_test):
    print("####################################################################")
    print("best_KNN_Params")
    leaf_size = list(range(1, 50))
    n_neighbors = list(range(1, 30))
    p = [1, 2]
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']

    Recall_score = make_scorer(recall_score)
    Accuracy_score = make_scorer(accuracy_score)
    F1_score = make_scorer(f1_score)
    Precision_score = make_scorer(precision_score)

    scoring = {"recall": Recall_score, "Accuracy": Accuracy_score, "f1-score": F1_score, "Precision": Precision_score}

    parameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p, algorithm=algorithm)
    knn = KNeighborsClassifier()
    rsCV = RandomizedSearchCV(knn, parameters, n_jobs=-1, scoring=scoring, refit="Precision")

    rsCV_best_model = rsCV.fit(X_train, y_train)

    print('Best leaf_size:', rsCV_best_model.best_estimator_.get_params()['leaf_size'])
    print('Best p:', rsCV_best_model.best_estimator_.get_params()['p'])
    print('Best n_neighbors:', rsCV_best_model.best_estimator_.get_params()['n_neighbors'])
    print('best params : ', rsCV_best_model.best_params_)
    print('Best score : ', rsCV_best_model.best_score_)

    KNN = rsCV_best_model.best_estimator_

    KNN.fit(X_train, y_train)
    predicted = KNN.predict(X_test)

    print(confusion_matrix(predicted, y_test))
    print(classification_report(predicted, y_test))
