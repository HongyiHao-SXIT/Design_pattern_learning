# 导入必要的库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np

# 1. 数据获取
iris = load_iris()
X = iris.data
y = iris.target

# 2. 数据预处理
# 2.1 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2.2 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 模型选择与训练
# 3.1 决策树模型
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# 3.2 支持向量机模型
svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train)

# 4. 模型评估
# 4.1 决策树模型评估
dt_y_pred = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, dt_y_pred)
dt_recall = recall_score(y_test, dt_y_pred, average='macro')
dt_f1 = f1_score(y_test, dt_y_pred, average='macro')

print("决策树模型评估结果：")
print(f"准确率: {dt_accuracy}")
print(f"召回率: {dt_recall}")
print(f"F1值: {dt_f1}")

# 4.2 支持向量机模型评估
svm_y_pred = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_y_pred)
svm_recall = recall_score(y_test, svm_y_pred, average='macro')
svm_f1 = f1_score(y_test, svm_y_pred, average='macro')

print("\n支持向量机模型评估结果：")
print(f"准确率: {svm_accuracy}")
print(f"召回率: {svm_recall}")
print(f"F1值: {svm_f1}")

# 5. 交叉验证
# 决策树模型交叉验证
dt_cv_scores = cross_val_score(dt_model, X_train_scaled, y_train, cv=5)
print("\n决策树模型交叉验证平均准确率: ", np.mean(dt_cv_scores))

# 支持向量机模型交叉验证
svm_cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)
print("支持向量机模型交叉验证平均准确率: ", np.mean(svm_cv_scores))

# 6. 模型优化（以支持向量机为例，使用网格搜索）
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

print("\n支持向量机模型优化结果：")
print("最佳参数: ", grid_search.best_params_)
print("最佳得分: ", grid_search.best_score_)

# 使用优化后的模型进行预测
best_model = grid_search.best_estimator_
best_y_pred = best_model.predict(X_test_scaled)
best_accuracy = accuracy_score(y_test, best_y_pred)
print(f"优化后模型准确率: {best_accuracy}")
    