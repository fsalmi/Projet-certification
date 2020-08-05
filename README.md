# Projet-certification
Projet de nettoyage des boîtes mails de manière automatique. 

Dataset: ENRON dataset public 

Méthode: 2 types de recommandation de suppression
1. Détection des mails threads (ceux incluent dans des fils de discussion), des mails de newsletter, des mails de pub, etc... 
2. Mails ayant une forte probabilité de suppression grâce à une classification qui a été entraînée sur des mails labélisés au préalable 

Utilisation de bases de données neo4j et des algorithmes de détection de communauté et de centralité 

Application flask mettant en exemple le travail réalisé avec un dash board final indiquant les différents statistiques de nettoyage réalisé. 
Pour lancer l'application, il faut créer un environnement virtuel. Suivre les étapes suivantes: 
1. Allez dans le répertoire app_flask
2. Installez virtualenv si ce n'est déjà fait : pip install virtualenv
3. Créez un environnement virtuel appelé venv : virtualenv -p /usr/bin/python3.6 venv
4. Activez l'environnement virtuel : source venv/bin/activate
5. Installez les libraires nécessaires pour lancer l'application : pip3 install -r requirements.txt
6. Lancez l'application avec python3 : python3 Modele_flask12.py
7. Lorsque vous avez fini, désactivez l'environnement virtuel : deactivate
