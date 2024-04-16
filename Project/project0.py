import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to preprocess data
def preprocess_data(data):
    # Convert categorical data if any
    if data.select_dtypes(include=['object']).empty is False:
        encoder = LabelEncoder()
        for column in data.select_dtypes(include=['object']).columns:
            data[column] = encoder.fit_transform(data[column])

    # Imputing missing values
    imputer = SimpleImputer(strategy='median')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Normalizing data
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data, scaler

# Load Diabetes data
diabetes_path = 'diabetes_project.csv'
diabetes_data = pd.read_csv(diabetes_path)
diabetes_data, diabetes_scaler = preprocess_data(diabetes_data)

# Step 2: Generate labels via K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(diabetes_data[['Glucose', 'BMI', 'Age']])
diabetes_data['Cluster'] = clusters
cluster_glucose = diabetes_data.groupby('Cluster')['Glucose'].mean()
diabetes_data['Outcome'] = (diabetes_data['Cluster'] == cluster_glucose.idxmax()).astype(int)
diabetes_data.drop(columns=['Cluster'], inplace=True)

# Step 3: Feature extraction for Diabetes data
X_diabetes = diabetes_data.drop('Outcome', axis=1)
y_diabetes = diabetes_data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)
diabetes_pca = PCA(n_components=3)
X_train_pca = diabetes_pca.fit_transform(X_train)
X_test_pca = diabetes_pca.transform(X_test)

# Step 4: Classification using a super learner
base_classifiers = [
    ('nb', GaussianNB()),
    ('nn', MLPClassifier(max_iter=1000)),
    ('knn', KNeighborsClassifier())
]
meta_learner = DecisionTreeClassifier()
super_learner = StackingClassifier(estimators=base_classifiers, final_estimator=meta_learner, cv=5)
grid_search = GridSearchCV(estimator=super_learner, param_grid={'nn__hidden_layer_sizes': [(50,), (100,)], 'nn__activation': ['tanh', 'relu'], 'knn__n_neighbors': [5, 10], 'final_estimator__max_depth': [None, 10, 20], 'final_estimator__min_samples_split': [2, 10]}, cv=5, scoring='accuracy')
grid_search.fit(X_train_pca, y_train)

# Evaluation
print(f"Best Cross-Validation Score: {grid_search.best_score_}")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_pca)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Step 5: Independent handling for Employee Leave dataset
# Load Employee Leave data
employee_data_path = 'Employee_Leave.csv'
employee_data = pd.read_csv(employee_data_path)
employee_data, employee_scaler = preprocess_data(employee_data)

# Use PCA on Employee data independently
X_employee = employee_data.drop('EmployeeLeave', axis=1)  # Assuming 'EmployeeLeave' is the outcome column
y_employee = employee_data['EmployeeLeave'].astype(int)
employee_pca = PCA(n_components=3)
X_employee_pca = employee_pca.fit_transform(X_employee)

# Re-train super learner model on Employee dataset (can use same hyperparameters)
grid_search.fit(X_employee_pca, y_employee)
best_model_employee = grid_search.best_estimator_
y_pred_employee = best_model_employee.predict(X_employee_pca)

# Evaluation on Employee dataset
print(f"Employee Test Accuracy: {accuracy_score(y_employee, y_pred_employee) * 100:.2f}%")
print(f"Employee Classification Report:\n{classification_report(y_employee, y_pred_employee)}")
print(f"Employee Confusion Matrix:\n{confusion_matrix(y_employee, y_pred_employee)}")


# C:\Users\40825\AppData\Local\Programs\Python\Python39\python.exe "D:\Notets\python\Data Science\Project\project0.py"
# Best Cross-Validation Score: 0.9386064030131827
# Test Accuracy: 92.52%
# Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.94      0.95      0.94        92
#            1       0.91      0.89      0.90        55
#
#     accuracy                           0.93       147
#    macro avg       0.92      0.92      0.92       147
# weighted avg       0.92      0.93      0.93       147
#
# Confusion Matrix:
# [[87  5]
#  [ 6 49]]
# Employee Test Accuracy: 79.99%
# Employee Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.81      0.91      0.86      3053
#            1       0.77      0.59      0.67      1600
#
#     accuracy                           0.80      4653
#    macro avg       0.79      0.75      0.76      4653
# weighted avg       0.80      0.80      0.79      4653
#
# Employee Confusion Matrix:
# [[2772  281]
#  [ 650  950]]
#
# 进程已结束，退出代码为 0
