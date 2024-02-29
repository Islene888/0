import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score

# Load the dataset
diabetes_data = pd.read_csv('diabetes.csv')  # Replace 'path_to_your_file/' with the actual file path

# Split the data into features and target label
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

# Define the models
rf_model_3 = RandomForestClassifier(n_estimators=3)
rf_model_50 = RandomForestClassifier(n_estimators=50)
ada_model_3 = AdaBoostClassifier(n_estimators=3)
ada_model_50 = AdaBoostClassifier(n_estimators=50)

# Perform cross-validation and compute scores
rf_scores_3 = cross_val_score(rf_model_3, X, y, cv=5)
rf_scores_50 = cross_val_score(rf_model_50, X, y, cv=5)
ada_scores_3 = cross_val_score(ada_model_3, X, y, cv=5)
ada_scores_50 = cross_val_score(ada_model_50, X, y, cv=5)

# Print mean scores for comparison
print("Mean scores:")
print("RF with 3 estimators:", rf_scores_3.mean())
print("AdaBoost with 3 estimators:", ada_scores_3.mean())
print("RF with 50 estimators:", rf_scores_50.mean())
print("AdaBoost with 50 estimators:", ada_scores_50.mean())
