# Description du sujet

Le projet consiste à construire un modèle de machine learning supervisé de régression permettant de prédire la consommation énergétique d’un foyer.

Le modèle s’appuie sur plusieurs variables décrivant les caractéristiques du foyer et son contexte de consommation, comme la taille du foyer, la température moyenne, la présence de climatisation, la surface du logement, le type de chauffage, le temps passé à domicile et la consommation pendant les heures de pointe.

L’objectif est de comprendre les facteurs qui influencent la consommation énergétique et de prédire la consommation en kWh.

## Problématique

Comment prédire la consommation énergétique d’un foyer à partir de ses caractéristiques afin d’identifier les principaux facteurs qui influencent la demande d’énergie ?

## Méthodologie

Le projet suit les étapes suivantes :

1. Analyse exploratoire des données pour comprendre la structure du dataset et les relations entre les variables.

2. Étude des principales variables explicatives :
   - `Household_Size`
   - `Avg_Temperature_C`
   - `Has_AC`
   - `surface_m2`
   - `heating_type`
   - `hours_at_home`
   - `Peak_Hours_Usage_kWh`

3. Feature engineering :
   - transformation de `Has_AC` en variable numérique avec `Has_AC_Binary`
   - encodage de `heating_type` en variables indicatrices
   - création de variables d’interaction comme `temperature_x_ac`
   - création de variables d’interaction comme `household_size_x_ac`

4. Construction d’un modèle de régression supervisée pour prédire la variable cible :

   `Energy_Consumption_kWh`

5. Évaluation du modèle avec des métriques adaptées à la régression :
   - MAE
   - MSE
   - R²

## Modèle utilisé

Le premier modèle utilisé est une régression linéaire avec standardisation des variables.

Ce modèle permet d’obtenir une première prédiction simple et interprétable de la consommation énergétique.

Un modèle plus avancé, comme un Random Forest Regressor, peut ensuite être testé pour comparer les performances.

## Dataset utilisé

Le dataset principal contient les variables suivantes :

- `Household_ID`
- `Date`
- `Energy_Consumption_kWh`
- `Household_Size`
- `Avg_Temperature_C`
- `Has_AC`
- `Peak_Hours_Usage_kWh`
- `surface_m2`
- `heating_type`
- `hours_at_home`

La variable cible du projet est :

`Energy_Consumption_kWh`

Il s’agit donc d’un problème de régression supervisée, car on cherche à prédire une valeur numérique continue.

## Limites

Le dataset couvre une période courte, du 1er avril 2025 au 8 avril 2025. Il permet donc d’analyser les relations entre les variables sur cette période, mais ne permet pas de conclure sur des tendances longues ou saisonnières.

Le dataset ne contient pas de variable indiquant directement des cas de fraude. Le projet ne porte donc pas sur une détection de fraude confirmée, mais sur la prédiction de la consommation énergétique.

## Objectif

L’objectif est de construire un modèle capable de prédire la consommation énergétique d’un foyer à partir de ses caractéristiques, puis d’évaluer la qualité des prédictions obtenues.
