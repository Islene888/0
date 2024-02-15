import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 加载数据集
train_df = pd.read_csv('hw4_train.csv')
test_df = pd.read_csv('hw4_test.csv')

# 训练多元线性回归模型，'BloodPressure'作为因变量
X_train = train_df.drop(columns=['BloodPressure', 'Outcome'])
y_train_bp = train_df['BloodPressure']
linear_model = LinearRegression()
linear_model.fit(X_train, y_train_bp)

# 使用回归模型预测测试集中的'BloodPressure'
X_test = test_df.drop(columns=['BloodPressure', 'Outcome'])
predicted_bp = linear_model.predict(X_test)
test_df['BloodPressure'] = predicted_bp

# 准备KNN模型训练的特征和标签
X_train_knn = X_train
y_train_knn = train_df['Outcome']

# 初始化存储性能指标的列表
accuracy_list = []
sensitivity_list = []
specificity_list = []

# 训练KNN模型，k从1到19
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_knn, y_train_knn)
    y_pred_knn = knn.predict(X_test)
    accuracy = accuracy_score(test_df['Outcome'], y_pred_knn)
    tn, fp, fn, tp = confusion_matrix(test_df['Outcome'], y_pred_knn).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy_list.append(accuracy)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)

# 确定最佳的k值
best_k = accuracy_list.index(max(accuracy_list)) + 1
best_accuracy = accuracy_list[best_k - 1]
best_sensitivity = sensitivity_list[best_k - 1]
best_specificity = specificity_list[best_k - 1]


print(f"Accuracy: {best_accuracy}")
print(f"Sensitivity: {best_sensitivity}")
print(f"Specificity: {best_specificity}")
print(f"Best K: {best_k}")

# 解释为什么选择了这个K值
explanation0 = f"I chose K={best_k} because it provides the highest accuracy ({best_accuracy:.2f}) among the models tested. "
explanation1 = f"While sensitivity ({best_sensitivity:.2f}) and specificity ({best_specificity:.2f}) are also important, "
explanation2 = "the primary goal in this context was to maximize the overall correctness of the predictions, "
explanation3 = "which is captured by the accuracy metric. "
explanation4 = "This balance ensures that the model is not overly biased towards predicting just one class correctly."
print(explanation0)
print(explanation1)
print(explanation2)
print(explanation3)
