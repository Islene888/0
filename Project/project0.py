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

    # Removing outliers using IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    return data

# Load Diabetes data
diabetes_path = 'diabetes_project.csv'
diabetes_data = pd.read_csv(diabetes_path)
diabetes_data = preprocess_data(diabetes_data)

# Step 2: Generate labels via K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(diabetes_data[['Glucose', 'BMI', 'Age']])
diabetes_data['Cluster'] = clusters
cluster_glucose = diabetes_data.groupby('Cluster')['Glucose'].mean()
diabetes_data['Outcome'] = (diabetes_data['Cluster'] == cluster_glucose.idxmax()).astype(int)
diabetes_data.drop(columns=['Cluster'], inplace=True)

# Step 3: Feature extraction
X_train, X_test, y_train, y_test = train_test_split(diabetes_data.drop('Outcome', axis=1), diabetes_data['Outcome'], test_size=0.2, random_state=42)
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

# Step 5: Employing the model on other datasets
# Load Employee Leave data
employee_data_path = '/mnt/data/Employee_Leave.csv'
employee_data = pd.read_csv(employee_data_path)
employee_data = preprocess_data(employee_data)

# Reuse model for Employee Leave dataset
X_emp = employee_data.drop('EmployeeLeave', axis=1)
y_emp = employee_data['EmployeeLeave']
X_emp_pca = pca.transform(X_emp)  # using the same PCA transformation
y_pred_emp = best_model.predict(X_emp_pca)

# Evaluation on new dataset
print(f"Employee Test Accuracy: {accuracy_score(y_emp, y_pred_emp) * 100:.2f}%")
print(f"Employee Classification Report:\n{classification_report(y_emp, y_pred_emp)}")
print(f"Employee Confusion Matrix:\n{confusion_matrix(y_emp, y_pred_emp)}")
