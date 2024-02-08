import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

def normalize_data(org_df, col):
    col_array = np.array(org_df[col]).reshape(-1, 1)
    scaler = preprocessing.StandardScaler()
    scaler.fit(col_array)
    org_df[col] = scaler.transform(col_array)
    return org_df

def remove_outliers_and_impute(org_df, cols):
    for col in cols:
        # 移除异常值
        values = org_df[col].dropna()  # 排除NaN值用于计算分位数
        q3, q1 = np.percentile(values, [75, 25])
        fence = 1.5 * (q3 - q1)
        upper_band = q3 + fence
        lower_band = q1 - fence
        org_df.loc[(org_df[col] < lower_band) | (org_df[col] > upper_band), col] = np.nan

    # 使用迭代填充器填充缺失值
    imputer = IterativeImputer(max_iter=10, random_state=0)
    org_df[cols] = imputer.fit_transform(org_df[cols])
    return org_df

def prepare_data(org_df):
    org_df = org_df.loc[:, org_df.columns.isin(['Gender', 'Age', 'Income', 'Spending'])]
    org_df = pd.get_dummies(org_df, columns=['Gender'], dtype='int')
    return org_df

# 加载数据集
org_df = pd.read_csv("market_ds.csv")

# 移除异常值并填充缺失值
org_df = remove_outliers_and_impute(org_df, ['Age', 'Income', 'Spending'])

# 标准化数据
org_df = normalize_data(org_df, 'Income')
org_df = normalize_data(org_df, 'Spending')
org_df = normalize_data(org_df, 'Age')

# 准备数据
train_feat = prepare_data(org_df)

# 使用KMeans聚类算法，聚类数为4
model = KMeans(n_clusters=4)
model.fit(train_feat)

# 分别为每个聚类准备数据
clusters = [train_feat.loc[model.labels_ == i, :] for i in range(4)]

# 为每个聚类指定颜色
colors = ['red', 'black', 'blue', 'green']

# 绘制散点图
plt.figure(figsize=(10, 5))
for i, cluster in enumerate(clusters):
    plt.scatter(cluster['Income'], cluster['Spending'], color=colors[i], label=f'Cluster {i+1}')
plt.title('Income vs Spending for 4 Clusters')
plt.xlabel('Income')
plt.ylabel('Spending')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
for i, cluster in enumerate(clusters):
    plt.scatter(cluster['Income'], cluster['Age'], color=colors[i], label=f'Cluster {i+1}')
plt.title('Income vs Age for 4 Clusters')
plt.xlabel('Income')
plt.ylabel('Age')
plt.legend()
plt.show()


# 使用肘部方法确定最佳聚类数
inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(train_feat)
    inertias.append(kmeans.inertia_)  # 记录每次聚类的总内平方和

# 绘制肘部图
plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.show()

# 使用层次聚类方法绘制树状图
linkage_data = linkage(train_feat, method='single', metric='euclidean')
dendrogram(linkage_data, truncate_mode='level', p=5)
plt.show()

