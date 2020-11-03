import os
import random
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("####################################################################")
print("retrieve data")
"""
    fonction permettant de : 
         - retirer deux dataset du dossier
         - retirer tout signaux qui ne sont pas des attacks afin d'augmenter le nombre de non attack ( 99% d'attacks)
         - encoder les variables qui ont des dtype ≠ int64/float64
         - standardiser & normaliser les données pour que l'utilisation du model KNN soit plus simple (affect la distanciation)
         - undersample les données car 99% d'attaques, réduire le nombre d'attaque nous évitera l'oversampling
"""


def retrieve_data():
    dataframe = pd.DataFrame()
    index = []
    i = 0
    while i < 2:
        # i+=1
        file_index = random.randint(1, 74)
        if file_index in index:
            i -= i
        file_name = '/Users/tomwilliams/Desktop/Entire_Dataset/UNSW_2018_IoT_Botnet_Dataset_' + str(
            file_index) + '.csv'
        print(file_name)
        if path.exists(
                '/Users/tomwilliams/Desktop/Entire_Dataset/UNSW_2018_IoT_Botnet_Dataset_' + str(
                    file_index) + '.csv') == True:
            print('hello')
            i += 1
            dataframe = pd.concat([dataframe, pd.read_csv(file_name, sep=',')], axis=0)

    i = 0
    while i < 74:
        i += 1
        file_name = '/Users/tomwilliams/Desktop/Entire_Dataset/UNSW_2018_IoT_Botnet_Dataset_' + str(
            i) + '.csv'
        if path.exists(
                '/Users/tomwilliams/Desktop/Entire_Dataset/UNSW_2018_IoT_Botnet_Dataset_' + str(
                    i) + '.csv') == True:
            dataframe = pd.concat(
                [dataframe, pd.read_csv(file_name, sep=',')[pd.read_csv(file_name, sep=',').attack == 0]], axis=0
            )

    """maintenant nous allons nous amuser"""

    print("####################################################################")
    print("label encode data")
    # boucle permettant d'encoder les valeurs sous forme string en int
    for column in dataframe:
        index = dataframe[dataframe[column] == '<'].index
        dataframe.drop(index, inplace=True)
        if type(dataframe[column].values[0]) == str:
            print(column)
            encoder = LabelEncoder()
            encoder.fit(dataframe[column])
            dataframe[column] = encoder.transform(dataframe[column])
            print(column)

    print("drop empy / null data")

    dataframe = dataframe.dropna(axis=1)
    dataframe.drop_duplicates(dataframe.columns, keep='last')
    print(dataframe.describe())

    # Display new class counts
    print("attack")
    print(dataframe.attack.value_counts())
    print("category")
    print(dataframe.category.value_counts())
    print("subcategory")
    print(dataframe.subcategory.value_counts())

    return dataframe


"""
Fonction permettant soit 
    - de réduir le dataframe sur 10 features
    - de garder le dataframe dans son integralité
"""


def Feature_Selection(dataframe_, condition):
    if condition == 0:
        print("##########################################")
        print("with feature selection")
        print("##########################################")

        dataframe = dataframe_[
            ["bytes", "sbytes", "pkSeqID", "spkts", "pkts", "state", "proto", "stime", "daddr", "drate", "attack"]]
        X = dataframe.drop(['attack'], axis=1).values

    else:
        print("##########################################")
        print("without feature selection")
        print("##########################################")
        dataframe = dataframe_
        X = dataframe.drop(['attack', 'category', 'subcategory'], axis=1).values

    # if(only_attack):
    y = dataframe["attack"].values  # contient que le resultat que nous cherchons
    # else:
    #     y = dataframe[["attack","category", "subcategory"]].values

    print("Select features from the most important")
    feature_selector = VarianceThreshold()
    VarianceFiltered_X = feature_selector.fit_transform(X)

    print("standardise and scale features")
    standard = StandardScaler().fit(X)
    Standardised_VarianceFiltered_X = standard.transform(VarianceFiltered_X)

    print(len(Standardised_VarianceFiltered_X[0, :]))

    return Standardised_VarianceFiltered_X, y


"""
Fonction permettant d'appliquer les methodes d'UnderSampling sur le dataframe
"""


def Undersamplingpart(X, y, UnderSamplingMethod):
    print("make dataset more realistic as there are supposed to less attacks than normal signals")
    UnderSampleMethod = UnderSamplingMethod
    Standardised_VarianceFiltered_undeSampled_X, y = UnderSampleMethod.fit_resample(X, y)
    # Standardised_VarianceFiltered_undeSampled_X = Standardised_VarianceFiltered_X

    (unique, counts) = np.unique(y, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print("frequency of each class")
    print(frequencies)

    print("histograme of y (attacks & non attacks")
    plt.figure()
    plt.hist(y)
    plt.show()

    return Standardised_VarianceFiltered_undeSampled_X, y
