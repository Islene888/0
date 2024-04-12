import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load data
data_path = 'diabetes_project.csv'
data = pd.read_csv(data_path)

# Step 1: Pre-processing
data = data.astype(float)  # Ensure all data is float
imputer = SimpleImputer(strategy='median')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
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
super_learner = StackingClassifier(estimators=base_classifiers, final_estimator=meta_learner, cv=5)
super_learner.fit(X_train_pca, y_train)

# Evaluation
print(f"Cross-Validation Scores: {cross_val_score(super_learner, X_train_pca, y_train, cv=5)}")
y_pred = super_learner.predict(X_test_pca)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
