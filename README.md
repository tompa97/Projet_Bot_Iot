# Projet_Bot_Iot
## Introduction
Voici le projet Bot-IOT que nous avons du accomplit le semestre précédent étant notre premier projet de la Majeure IA. Nous avions eu pour but de manipuler une base de donnée provenant d'Australie se basant sur l'IOT qui regroupait les données de signaux entrant. ces signaux sont soit des attaques soit des signaux normaux. Notre but était d'écrire un programme permettant d'analyser, manipuler les données ainsi que trouver le(s) modèles de ML le(s) plus efficace(s) pour prédire si un signal est une attaque ou non.

## description des fichiers
* GroupEverything : ce fichier regroupe toute les fonctions des autres fichier python
* DataManip : retir, manipule et traite le dataFrame que nous allons utiliser pour les models ML

afin de trouver les meilleurs paramètres pour nos modèles ML, nous avons utilisé RandomSearchCV ( et non GridSearchCV du à la consommation demandé et aux conditions dans lequel nous étions )
* Fichiers ML 
  * Decision_Trees.py : utilisation du RandomForestClassifier
  * Random_Search_KNN.py : utilisation du KNearestNeighbour
  * svc.py : utilisation de Suport Vector Machine 
  * Neural_Network.py : utilisation des réseaux de neurones

## Vous trouverez dans le répertoir le sujet du projet ainsi que notre rapport final qui vous aidera à comprendre notre démarche 
