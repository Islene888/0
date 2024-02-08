# 导入必要的库
import numpy as np  # 用于高效的数值计算
from sklearn.experimental import enable_iterative_imputer  # 允许使用实验性质的迭代填充器
from sklearn.impute import IterativeImputer  # 导入迭代填充器，用于处理数据集中的缺失值
import pandas as pd  # 用于数据处理
from sklearn.model_selection import train_test_split  # 分割数据集，虽然在这段代码中没有使用
import matplotlib.pyplot as plt  # 用于绘图
from sklearn.cluster import KMeans  # 导入KMeans聚类算法
from sklearn_extra.cluster import KMedoids  # 导入KMedoids聚类算法
from kmodes.kmodes import KModes  # 导入KModes聚类算法，适用于处理分类数据
import pyclustering.cluster.kmedians  # 导入KMedians聚类算法
from scipy.cluster.hierarchy import linkage, dendrogram  # 导入层次聚类算法和绘制树状图的函数
from sklearn.cluster import AgglomerativeClustering  # 导入凝聚层次聚类算法
from sklearn import preprocessing  # 导入预处理模块

# 定义数据标准化函数
def normalize_data(org_df,col):
    # 将指定列的数据进行标准化处理（均值为0，标准差为1）
    col_array = np.array(org_df[col]).reshape(-1, 1)  # 将列数据转换为numpy数组
    scaler = preprocessing.StandardScaler()  # 创建标准化对象
    scaler.fit(col_array)  # 计算标准化所需的均值和标准差
    org_df[col] = scaler.transform(col_array)  # 应用标准化
    return org_df
'''
这段代码定义了一个名为 `normalize_data` 的函数，它的目的是对数据集中指定的列进行标准化处理。标准化是数据预处理中的一种常见方法，旨在使得不同特征之间具有可比性，同时帮助一些基于距离的算法（如K-均值聚类）更好地工作。下面逐行解释这段代码的功能：
1. **函数定义**: `def normalize_data(org_df, col):` 定义了一个函数 `normalize_data`，它接收两个参数：`org_df` 和 `col`。`org_df` 是一个Pandas的DataFrame对象，代表原始数据集；`col` 是一个字符串，代表需要被标准化处理的列名。
2. **将列数据转换为numpy数组**: `col_array = np.array(org_df[col]).reshape(-1, 1)` 这一行代码首先使用 `np.array(org_df[col])` 将DataFrame中指定列的数据转换为numpy数组。然后，使用 `.reshape(-1, 1)` 方法将这个数组转换为二维数组（即列向量），以满足 `StandardScaler` 的输入要求。`-1` 在reshape方法中表示不指定该维度的大小，让numpy自动计算它。
3. **创建标准化对象**: `scaler = preprocessing.StandardScaler()` 这里使用了 `preprocessing.StandardScaler()` 来创建一个标准化器（`scaler`）。`StandardScaler` 是来自scikit-learn库的预处理模块中的一个类，用于将数据标准化到具有零均值和单位方差的分布。
4. **计算标准化所需的均值和标准差**: `scaler.fit(col_array)` 这一步调用 `scaler` 的 `fit` 方法，将列向量传入，计算并存储该列数据的均值和标准差。这些统计量将用于后续的转换步骤。
5. **应用标准化**: `org_df[col] = scaler.transform(col_array)` 这里使用 `scaler` 的 `transform` 方法对原始列向量进行标准化转换。标准化的计算公式为：`(x - mean) / std`，其中 `x` 是原始数据值，`mean` 是均值，`std` 是标准差。转换后的数据被重新赋值给原始DataFrame的相应列，实现了原地更新。
6. **返回更新后的DataFrame**: `return org_df` 最后，函数返回更新后的DataFrame，其中指定的列已经被标准化处理。
总之，这个函数的作用是将DataFrame中某一列的数据进行标准化处理，使得该列的数据均值为0，标准差为1，这样处理后的数据对于许多机器学习算法来说更加适用。'''
# 定义数据预处理函数
def prepare_data(org_df):
    # 选择目标属性进行分析
    org_df = org_df.loc[:,org_df.columns.isin(['MIC','MIC_Interpretation', 'Antimicrobial', 'Patient_Age', 'Bacteria'])]

    # 将分类数据转换为数值数据（one-hot编码）
    org_df = pd.get_dummies(org_df, columns=['MIC_Interpretation', 'Antimicrobial', 'Bacteria'], dtype='int')

    # 移除年龄中的异常值
    values = [vl for vl in org_df['Patient_Age'] if not np.isnan(vl)]
    q3, q1 = np.percentile(values, [75, 25])  # 计算第一和第三四分位数
    fence = 1.5 * (q3 - q1)  # 计算四分位间距
    upper_band = q3 + fence  # 计算上界
    lower_band = q1 - fence  # 计算下界
    # 标记超出上下界的年龄值为缺失值
    org_df.loc[(org_df['Patient_Age'] < lower_band) | (org_df['Patient_Age'] > upper_band), 'Patient_Age'] = None

    # 使用迭代填充器填充缺失值
    imputer = IterativeImputer(max_iter=10, random_state=0)
    imputed_dataset = imputer.fit_transform(org_df)  # 填充操作
    imputed_dataframe = pd.DataFrame(imputed_dataset, columns=org_df.columns)  # 转换回DataFrame格式
    return imputed_dataframe

'''
这段代码定义了一个名为 `prepare_data` 的函数，用于对给定的原始数据集（`org_df`）进行预处理，包括选择特定的列、处理分类变量、移除异常值以及填充缺失值。以下是详细的逐行解释：
1. **选择目标属性进行分析**:
   - `org_df = org_df.loc[:,org_df.columns.isin(['MIC','MIC_Interpretation', 'Antimicrobial', 'Patient_Age', 'Bacteria'])]`: 这行代码使用 `.loc` 方法和 `.isin` 函数筛选出DataFrame中的特定列。它将 `org_df` 更新为仅包含列名在给定列表中的列。这里选择的列是 `'MIC'`, `'MIC_Interpretation'`, `'Antimicrobial'`, `'Patient_Age'`, 和 `'Bacteria'`，因为这些属性对于后续的数据分析来说是重要的。
2. **将分类数据转换为数值数据（one-hot编码）**:
   - `org_df = pd.get_dummies(org_df, columns=['MIC_Interpretation', 'Antimicrobial', 'Bacteria'], dtype='int')`: 这行代码使用 `pd.get_dummies` 函数将指定的分类列（`'MIC_Interpretation'`, `'Antimicrobial'`, `'Bacteria'`）转换为数值数据，方法是为每个唯一的类别值创建一个新的虚拟（dummy）列。这种转换是必要的，因为大多数机器学习算法在处理数据时需要数值输入。
3. **移除年龄中的异常值**:
   - 这部分代码首先通过列表推导和 `np.isnan` 函数过滤出 `'Patient_Age'` 列中的非NaN值，然后计算这些值的第一和第三四分位数（`q1` 和 `q3`），并基于这些值计算IQR（四分位间距）。接着，它计算异常值的上下界限，并将超出这些界限的年龄值标记为缺失值（`None`）。这样做是为了减少极端值对数据分析的影响。
4. **使用迭代填充器填充缺失值**:
   - `imputer = IterativeImputer(max_iter=10, random_state=0)`: 这行代码创建了一个迭代填充器（`IterativeImputer`）的实例，用于处理缺失值。迭代填充器是一种模型，它以迭代方式模拟每个特征与其他特征之间的关系，用以估算缺失值。
   - `imputed_dataset = imputer.fit_transform(org_df)`: 这行代码应用迭代填充器于原始数据集，填充缺失值。`fit_transform` 方法首先训练填充器以学习数据的模式，然后对数据进行转换，填充所有缺失值。
   - `imputed_dataframe = pd.DataFrame(imputed_dataset, columns=org_df.columns)`: 将填充后的数据（现在是一个NumPy数组）转换回Pandas DataFrame格式，确保列名保持不变。
5. **返回预处理后的数据集**:
   - `return imputed_dataframe`: 最后，函数返回预处理和填充后的数据集。

总的来说，这个函数通过一系列预处理步骤准备数据，使其适合进行后续的数据分析或机器学习模型训练。通过选择重要的特征、将分类变量转换为数值型、移除异常值和填充缺失值，确保了数据的质量和一致性。'''

# 读取数据集
org_df = pd.read_csv("DS_Dataset.csv")  # 假设有一个数据集文件
train_feat = prepare_data(org_df)  # 对数据进行预处理

# 应用KMeans聚类算法
model = KMeans(n_clusters=2)  # 指定聚类数为2
model.fit(train_feat)  # 训练模型

# 根据聚类结果筛选数据
first_cluster = train_feat.loc[model.labels_ == 1,:]  # 第一类的数据
second_cluster = train_feat.loc[model.labels_ == 0,:]  # 第二类的数据
'''这段代码展示了如何使用`KMeans`聚类算法对数据集`train_feat`进行聚类，并基于聚类结果将数据分为两个群体。这里是逐步解释：

### 1. 应用KMeans聚类算法
- `model = KMeans(n_clusters=2)`: 这行代码创建了一个`KMeans`聚类模型的实例。`n_clusters=2`指明你想要将数据集分成的聚类数目为2。KMeans算法的目标是在特征空间中找到聚类的中心点，并将每个数据点分配给最近的中心点，从而形成聚类。

### 2. 训练模型
- `model.fit(train_feat)`: 这行代码使用`train_feat`数据集对`KMeans`模型进行训练。`train_feat`应该是一个包含了用于聚类的特征的数据集。在训练过程中，算法会迭代地调整聚类中心，直到找到最优的聚类中心位置。

### 3. 根据聚类结果筛选数据
- `model.labels_`属性包含了训练数据集中每个样本所属的聚类标签。在这个例子中，聚类标签为0或1，因为我们指定`n_clusters=2`。
- `first_cluster = train_feat.loc[model.labels_ == 1,:]`: 这行代码选取了所有被模型分配到第一类（标签为1）的数据点。`train_feat.loc[model.labels_ == 1,:]`利用Pandas的索引功能，根据`model.labels_`数组中的值为1的索引位置，从`train_feat`数据集中选取相应的行。
- `second_cluster = train_feat.loc[model.labels_ == 0,:]`: 类似地，这行代码选取了所有被分配到第二类（标签为0）的数据点。

总结来说，这段代码通过`KMeans`算法将数据集`train_feat`按特征空间中的相似度分成了两个聚类，并利用聚类结果对数据进行了分类。这种方法在各种应用中都非常有用，比如客户细分、图像分割、数据压缩等。'''

# 绘制散点图，展示两个聚类的结果
plt.scatter(first_cluster.loc[:, 'Patient_Age'], first_cluster.loc[:, 'MIC'], color='red')
plt.scatter(second_cluster.loc[:, 'Patient_Age'], second_cluster.loc[:, 'MIC'], color='black')
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
