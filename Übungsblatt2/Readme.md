
# Übungsblatt 2, Smart Data Analytics, Gruppe 1: Predictive Maintainance (Windrad)
Dieses Verzeichnis enthällt alle Lösungen für das 2.Übungsblatt der Gruppe 1 in Form von Jupyter Notebooks. Teilweise sind Python files als Hilfsdateien enthalten.
Python requirements sind in requitements.txt angegeben. 


Im folgenden ist eine Übersicht über die Inhalte gegeben:

## Datenexploration
1. Data_Exploration
2. Data_Exploration_Concept_Drift
3. Data_Exploration_Big_Correlation_Analysis
4. Data_Exploration_Importance_of_Features
5. Data_Exploration_Outlier_Dopplungen
6. Data_Exploration_Outlier_Analysis

## Data Pre-Processing:
Für die Datenvorverarbeitung existiert eine Hilfsdatei die alle notwendigen Methoden enthält. 
Die folgenden Methoden werden bereitgestellt:
- Ausreißer entfernen
- Scaling mit Min-Max-Scaler
- Methoden zur Entfernung von nicht benötigten Variablen
- Methoden zur Entfernung von stark korrelierten Variablen
- Methoden zur Entfernung von driftenden Variablen
- Methoden zur Auswahl von "guten Variablen", gemäß Data_Exploration_Importance_of_Features

## Feature engineering / Feature extraction:
Wir haben folgende Schritte bezüglich feature engineering / feature extraction unternommen:
- Generierung der features mean, median, min, max, Standartabweichung, Varianz aus jeweils einer CSV Datei. Anschließend wurden der Code des Gebiets, sowie das Labels entsprechend hinzugefügt. Die generierten features wurden als separate CSV Dateien abgespeichert.
- Generierung von gekürzten Zeitserien mittels mean und median mit rolling window über 1 Stunde, 3 Stunden, 6 Stunden, 12 Stunden.

## Klassifizierung mit klassischen Methoden
- SVM (linear)
- SVM (polynomial)
- KNN 
- Lineare Regression
- Ridge Classification
- Logistische Regression

## Klassifizierung mit deep learning:
- Deep Learning: Autoencoder
- Deep Learning: Neuronales Netz

