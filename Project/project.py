import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
data_path = 'diabetes_project.csv'
data = pd.read_csv(data_path)

# Step 1: Pre-processing
data = data.astype(float)  # Ensure all data is float

# Removing outliers using IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Imputing missing values
imputer = SimpleImputer(strategy='median')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Normalizing data
scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Step 2: Generate labels via K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0)
data['Cluster'] = kmeans.fit_predict(data[['Glucose', 'BMI', 'Age']])
cluster_glucose = data.groupby('Cluster')['Glucose'].mean()
data['Outcome'] = (data['Cluster'] == cluster_glucose.idxmax()).astype(int)
data.drop(columns=['Cluster'], inplace=True)

# Step 3: Feature extraction
X_train, X_test, y_train, y_test = train_test_split(data.drop('Outcome', axis=1), data['Outcome'], test_size=0.2, random_state=42)
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Step 4: Classification using a super learner
base_classifiers = [
    ('nb', GaussianNB()),
    ('nn', MLPClassifier(max_iter=1000)),
    ('knn', KNeighborsClassifier())
]
meta_learner = DecisionTreeClassifier()

# Using GridSearchCV for hyperparameter tuning
param_grid = {
    'nn__hidden_layer_sizes': [(50,), (100,)],
    'nn__activation': ['tanh', 'relu'],
    'knn__n_neighbors': [5, 10],
    'final_estimator__max_depth': [None, 10, 20],
    'final_estimator__min_samples_split': [2, 10]
}
super_learner = StackingClassifier(estimators=base_classifiers, final_estimator=meta_learner, cv=5)
grid_search = GridSearchCV(estimator=super_learner, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_pca, y_train)

# Evaluation
print(f"Best Cross-Validation Score: {grid_search.best_score_}")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_pca)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")


# C:\Users\40825\AppData\Local\Programs\Python\Python39\python.exe "D:\Notets\python\Data Science\Project\project.py"
# Best Cross-Validation Score: 0.906176911544228
# Test Accuracy: 85.52%
# Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.85      0.92      0.89        88
#            1       0.86      0.75      0.80        57
#
#     accuracy                           0.86       145
#    macro avg       0.86      0.84      0.84       145
# weighted avg       0.86      0.86      0.85       145
#
# Confusion Matrix:
# [[81  7]
#  [14 43]]
#
# 进程已结束，退出代码为 0

