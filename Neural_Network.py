import time

import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from kerastuner.tuners import RandomSearch
from sklearn.metrics import confusion_matrix, classification_report


def Search_Neural_Network(X_train_, X_test_, y_train_, y_test_, feature_selection, method):
    # titre du dossier qui contiendra les parametres du modele que nous souhaitons
    if feature_selection == 0:
        title = "verification/feature selection : True/method : " + str(method) + " "
        input_data = 10
    else:
        title = "verification/feature selection : False/method : " + str(method)
        input_data = 24

    """
    fonction qui contient les layers ainsi que tout les parametres/hyperparametres du model que nous recherchons
    """

    def model_to_tune(hp):
        model = Sequential()
        model.add(
            Dense(
                units=hp.Int('units_layer1', min_value=16, max_value=512, step=16, default=16),
                activation=hp.Choice('dense_activation_1', values=['relu', 'tanh', 'sigmoid'], default='relu'),
                input_dim=input_data,
                name='layer_1'
            )
        )
        model.add(
            Dropout(
                rate=hp.Float(
                    'droupout1',
                    min_value=0.0,
                    max_value=0.5,
                    default=0.25,
                    step=0.05
                )
            )
        )
        for i in range(hp.Int("n_layers", 1, 5)):
            model.add(
                Dense(
                    units=hp.Int('Suppliment_units_' + str(i + 1), min_value=16, max_value=512, step=16, default=16),
                    activation=hp.Choice('Suppliment_dense_activation_' + str(i + 1),
                                         values=['relu', 'tanh', 'sigmoid'], default='relu'),
                    name='Suppliment_Layer_' + str(i + 1)
                )
            )
            model.add(
                Dropout(
                    rate=hp.Float(
                        'Suppliment_dropout_' + str(i + 1),
                        min_value=0.0,
                        max_value=0.5,
                        default=0.25,
                        step=0.05
                    )
                )
            )

        model.add(
            Dense(
                1,
                activation='sigmoid',
                name="final_layer"
            )
        )

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    """
    Fonctions qui va entrainer le model tel un RandomSearchCV
    Elle retournera le model contenant les meilleurs (hyper)paramètres.
    """

    def tuner_evaluation(tuner, x_test, x_train, y_test, y_train):

        # Overview of the task
        tuner.search_space_summary()

        # Performs the hyperparameter tuning
        print("Start hyperparameter tuning")
        search_start = time.time()
        tuner.search(x=x_train, y=y_train, epochs=10, validation_split=0.1)

        search_end = time.time()
        elapsed_time = search_end - search_start
        print("Elapsed time : ", elapsed_time)
        # Show a summary of the search
        tuner.results_summary()

        # Retrieve the best model
        best_model = tuner.get_best_models(num_models=1)[0]
        print(best_model.summary())

        # Evaluate the best model.
        loss, accuracy = best_model.evaluate(x_test, y_test)

        return elapsed_time, loss, accuracy, best_model

    tuner = RandomSearch(
        model_to_tune,
        objective='val_accuracy',
        seed=1,
        max_trials=2,
        executions_per_trial=10,

        project_name=title,

    )

    # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=42)
    elapsed_time, loss, accuracy, best_model = tuner_evaluation(tuner, X_test_, X_train_, y_test_, y_train_)

    print("elapsed time : ", elapsed_time)
    print("loss", loss)
    print("accuracy", accuracy)

    prediction = best_model.predict_classes(X_test_)
    print(confusion_matrix(prediction, y_test_))
    print(classification_report(prediction, y_test_))

    from keras.utils.vis_utils import plot_model

    plot_model(best_model, to_file='model.png', show_shapes=True, show_layer_names=True)
