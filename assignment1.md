Description du sujet

Le projet consiste à analyser la consommation d’énergie des foyers afin d’identifier des comportements anormaux pouvant correspondre à des fraudes.

Le dataset ne contient pas de variable indiquant directement si un client est frauduleux ou non. Il n’est donc pas possible d’utiliser directement un modèle de classification supervisée.

Le projet utilise une approche de machine learning non supervisé pour la détection d’anomalies, combinée à un modèle de régression supervisée pour estimer la consommation normale.

Problématique

Comment modéliser la consommation énergétique normale des clients afin d’identifier des comportements anormaux pouvant correspondre à des fraudes, tout en limitant les erreurs de détection ?

Méthodologie

Le projet suit les étapes suivantes :

1. Analyse exploratoire des données pour comprendre les relations entre les variables

2. Construction d’un modèle de régression supervisée pour prédire la consommation énergétique à partir des variables disponibles :

* Household_Size
* Avg_Temperature_C
* Has_AC
* Peak_Hours_Usage_kWh

Le modèle utilisé est un Random Forest Regressor

3. Calcul de l’écart entre la consommation réelle et la consommation prédite

4. Détection des anomalies à partir de ces écarts

Un modèle de **Isolation Forest (non supervisé) est utilisé pour identifier les observations atypiques

5. Analyse des profils détectés comme anormaux

Dataset utilisé

Le dataset principal contient les variables suivantes :

* Energy_Consumption_kWh
* Household_Size
* Avg_Temperature_C
* Has_AC
* Peak_Hours_Usage_kWh
* Date

Un second dataset contient des données météo par région :

* température moyenne, minimale et maximale
* précipitations
* vitesse du vent
* nombre de jours froids et chauds

Ces données permettent d’enrichir l’analyse.

Limites

Le dataset ne contient pas de variable cible indiquant la fraude.

Les anomalies détectées ne correspondent pas forcément à des fraudes, mais à des comportements inhabituels.

Objectif

L’objectif est de construire un modèle capable de détecter automatiquement des anomalies dans la consommation énergétique, afin d’identifier des cas potentiels de fraude.