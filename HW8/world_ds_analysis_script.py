import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt

# 加载数据集
file_path = 'D:\\Notets\\python\\Data Science\\HW8\\world_ds.csv'  # 更改为您文件的实际路径
world_data = pd.read_csv(file_path)

# 分离特征和目标变量
X = world_data.drop(['Country', 'development_status'], axis=1)
y = world_data['development_status']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 前向选择法选择最佳的三个特征
knn_classifier = KNeighborsClassifier(n_neighbors=5)
sfs = SFS(knn_classifier,
          k_features=3,
          forward=True,
          scoring='accuracy',
          cv=5)
sfs.fit(X_scaled, y)

# PCA变换
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# LDA变换
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# 分割数据集
X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_scaled[:, list(sfs.k_feature_idx_)], y, test_size=0.25, random_state=42)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.25, random_state=42)
X_train_lda, X_test_lda, y_train_lda, y_test_lda = train_test_split(X_lda, y, test_size=0.25, random_state=42)

# 在前向选择的特征、PCA和LDA数据上训练和测试KNN分类器
def test_model(X_train, X_test, y_train, y_test):
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    return accuracy_score(y_test, y_pred)

accuracy_fs = test_model(X_train_fs, X_test_fs, y_train_fs, y_test_fs)
accuracy_pca = test_model(X_train_pca, X_test_pca, y_train_pca, y_test_pca)
accuracy_lda = test_model(X_train_lda, X_test_lda, y_train_lda, y_test_lda)

print(f'Accuracy with Forward Selection features: {accuracy_fs}')
print(f'Accuracy with PCA features: {accuracy_pca}')
print(f'Accuracy with LDA features: {accuracy_lda}')

# 解释每个PC
print("PCA component correlations:")
features = world_data.columns.drop(['Country', 'development_status'])
for i, component in enumerate(pca.components_):
    print(f"PC{i+1}:")
    for j, feature in enumerate(features):
        print(f"{feature}: {component[j]:.4f}")
    print()

# 可视化PCA成分
plt.figure(figsize=(12, 8))
for i, component in enumerate(pca.components_):
    plt.subplot(2, 2, i + 1)
    plt.barh(range(len(features)), component)
    plt.yticks(range(len(features)), features)
    plt.title(f"PCA Component {i+1}")
plt.tight_layout()
plt.show()
