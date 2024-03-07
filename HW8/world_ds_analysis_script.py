
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'D:\\Notets\\python\\Data Science\\HW8\\world_ds.csv'  # Change this to your file's actual path
world_data = pd.read_csv(file_path)

# Separate features and target variable
X = world_data.drop(['Country', 'development_status'], axis=1)
y = world_data['development_status']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA transformation
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# LDA transformation
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# Split the datasets for original, PCA, and LDA data
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.25, random_state=42)
X_train_lda, X_test_lda, y_train_lda, y_test_lda = train_test_split(X_lda, y, test_size=0.25, random_state=42)

# Initialize a KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train and test the classifier on the original data
knn_classifier.fit(X_train_orig, y_train_orig)
y_pred_orig = knn_classifier.predict(X_test_orig)
accuracy_orig = accuracy_score(y_test_orig, y_pred_orig)

# Train and test the classifier on the PCA data
knn_classifier.fit(X_train_pca, y_train_pca)
y_pred_pca = knn_classifier.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)

# Train and test the classifier on the LDA data
knn_classifier.fit(X_train_lda, y_train_lda)
y_pred_lda = knn_classifier.predict(X_test_lda)
accuracy_lda = accuracy_score(y_test_lda, y_pred_lda)

print(f'Accuracy with original features: {accuracy_orig}')
print(f'Accuracy with PCA features: {accuracy_pca}')
print(f'Accuracy with LDA features: {accuracy_lda}')
