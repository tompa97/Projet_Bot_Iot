# Projet_Bot_Iot
## Introduction
Voici le projet Bot-IOT que nous avons accomplit le semestre précédent étant notre premier projet de la Majeure IA. Nous avions eu pour but de manipuler une base de donnée provenant d'Australie se basant sur l'IOT qui regroupait les données de signaux entrant. ces signaux sont soit des attaques soit des signaux normaux. Notre but était d'écrire un programme permettant d'analyser, manipuler les données ainsi que trouver le(s) modèles de ML le(s) plus efficace(s) pour prédire si un signal est une attaque ou non.

## description des fichiers
* **GroupEverything.py** : ce fichier regroupe toute les fonctions des autres fichier python et lance le projet dans son ensemble, il suffit de run ce fichier et attendre le resultat final
* **DataManip.py** : retirer, manipuler et traiter la dataFrame que nous allons utiliser pour les models ML

Afin de trouver les meilleurs paramètres pour nos modèles ML, nous avons utilisé RandomSearchCV ( et non GridSearchCV du à la consommation demandé et aux conditions dans lequel nous étions )
* Fichiers ML 
  * **Decision_Trees.py** : utilisation du RandomForestClassifier
  * **Random_Search_KNN.py** : utilisation du KNearestNeighbour
  * **svc.py** : utilisation de Suport Vector Machine 
  * **Neural_Network.py** : utilisation des réseaux de neurones

Pour plus d'information sur notre dérmache, veuillez visiter notre rapport de projet **Project_Report.pdf**.

Les recherches sur le projet à partir duquel nous nous sommes appuyés se trouvent dans **dataset_summary.pdf**.

Le sujet original de notre projet se trouve dans **WB_analyse_logs.pdf**.


# *À noter*

* nous avons utilisé le logiciel PyCharm pour coder

* vous pouvez trouver le lien de la base de donnée à la toute fin de notre rapport

* si vous souhaitez lancer le programmes, il faut modifier les lignes suivantes pour éviter les erreurs:
  * **DataManip.py** : lignes 32, 36, 45, 48 (localisation de la base de donnée)
