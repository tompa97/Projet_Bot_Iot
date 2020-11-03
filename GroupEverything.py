from Data_Manip import retrieve_data, Undersamplingpart, Feature_Selection
from Decision_Trees import Trees
from Neural_Network import Search_Neural_Network
from Random_Search_KNN import ResearchBestKNN
from imblearn.under_sampling import TomekLinks, NearMiss, CondensedNearestNeighbour
from sklearn.model_selection import train_test_split
from svc import Linear_SVC_Search

UnderSamplingMethods = {
    "NearMiss1": NearMiss(version=1),
    "CondensedNEarestNeighbour": CondensedNearestNeighbour(random_state=42),
    # "Tomek_links" : TomekLinks(),       #test at school
    # "NearMiss2" : NearMiss(version=2), #test at school
    # "NearMiss3" : NearMiss(version=3), #test at school
}

# retirer/traiter les données que nous manipulerons ensuite
dataframe = retrieve_data()
print("data has been retrieved")
# une boucle pour chaque méthode d'UnderSampling
for (method, function) in UnderSamplingMethods.items():
    # une boucle pour chqaue état du dataframe 10/all features
    for Feature_Selection_condition in range(1):
        X_1, y_1 = Feature_Selection(dataframe, Feature_Selection_condition)  # transformation ou non du dataFrane
        X, y = Undersamplingpart(X_1, y_1, function)  # application de la méthode d'UnderSampling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=42)  # separation des données

        print("##################################################################################################")
        print("\t\t\tKNN while using ", method, "\n\n")
        ResearchBestKNN(X, y, X_train, X_test, y_train, y_test)

        print("##################################################################################################")
        print("\t\t\tLinear SVC while using ", method, "\n\n")
        Linear_SVC_Search(X, y, X_train, X_test, y_train, y_test)

        print("##################################################################################################")
        print("\t\t\tRandomForest Classifier while using ", method, "\n\n")
        Trees(X, y, X_train, X_test, y_train, y_test)

        print("##################################################################################################")
        print("\t\t\tNeural Network while using ", method, "\n\n")
        Search_Neural_Network(X_train, X_test, y_train, y_test, Feature_Selection_condition, method)

        print("##################################################################################################")
        print("##################################################################################################")
        print("##################################################################################################")
