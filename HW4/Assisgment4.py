import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 加载数据集
train_df = pd.read_csv('hw4_train.csv')
test_df = pd.read_csv('hw4_test.csv')

# 训练多元线性回归模型，'BloodPressure'作为因变量
X_train = train_df.drop(columns=['BloodPressure', 'Outcome'])  # 移除非特征列
y_train_bp = train_df['BloodPressure']
linear_model = LinearRegression()
linear_model.fit(X_train, y_train_bp)

# 使用回归模型预测测试集中的'BloodPressure'
# 确保测试集只包含训练时用过的特征
X_test = test_df.drop(columns=['BloodPressure', 'Outcome'])
predicted_bp = linear_model.predict(X_test)
test_df['BloodPressure'] = predicted_bp  # 更新测试集

# 准备KNN模型训练的特征和标签
X_train_knn = X_train  # KNN模型使用相同的训练特征
y_train_knn = train_df['Outcome']

# 初始化存储性能指标的列表
accuracy_list = []
sensitivity_list = []
specificity_list = []

# 训练KNN模型，k从1到19
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_knn, y_train_knn)

    # 预测测试集的Outcome
    y_pred_knn = knn.predict(X_test)  # 使用已更新的测试集特征进行预测

    # 计算并存储性能指标
    accuracy = accuracy_score(test_df['Outcome'], y_pred_knn)
    tn, fp, fn, tp = confusion_matrix(test_df['Outcome'], y_pred_knn).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    accuracy_list.append(accuracy)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)

# 确定最佳的k值
best_k = accuracy_list.index(max(accuracy_list)) + 1  # +1因为列表索引从0开始
best_accuracy = accuracy_list[best_k - 1]  # 使用最佳k对应的准确率
best_sensitivity = sensitivity_list[best_k - 1]  # 使用最佳k对应的敏感性
best_specificity = specificity_list[best_k - 1]  # 使用最佳k对应的特异性

print(f"Best K: {best_k}")
print(f"Accuracy: {best_accuracy}")
print(f"Sensitivity: {best_sensitivity}")
print(f"Specificity: {best_specificity}")

