# Projet-certification
Projet de nettoyage des boîtes mails de manière automatique. 

Dataset: ENRON dataset public 

Méthode: 2 types de recommandation de suppression
1. Détection des mails threads (ceux incluent dans des fils de discussion), des mails de newsletter, des mails de pub, etc... 
2. Mails ayant une forte probabilité de suppression grâce à une classification qui a été entraînée sur des mails labélisés au préalable 

Utilisation de bases de données neo4j et des algorithmes de détection de communauté et de centralité 

Application flask mettant en exemple le travail réalisé avec un dash board final indiquant les différents statistiques de nettoyage réalisé. 
