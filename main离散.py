from 离散贝叶斯幂模型2 import BayesianSINDy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告
# ================================
# 1. 读取数据
# ================================
df = pd.read_excel('data.xlsx')
x = np.array(df.iloc[:, 3])  # 假设第2列是目标序列
t = np.arange(1, len(x)+1)

# ================================
# 2. 划分训练集和测试集索引
# ================================
split_idx = int(len(x) * 0.75)
train_idx = slice(0, split_idx)
test_idx = slice(split_idx, len(x))

# ================================
# 3. 一阶累加（训练集）
# ================================
x_train = x[train_idx]
x1_train = np.cumsum(x_train)

# ================================
# 4. 拟合 Bayesian SINDy 模型
# ================================
model = BayesianSINDy(include_trig=True, prune_threshold=0.1, max_terms=8)
model.fit(x1_train)



# ================================
# 5. 一次性预测整个数据集累加序列
# ================================
x1_pred = model.predict(x0=x1_train[0], steps=len(x)+4+12*5)  # 训练+测试
# 转回原始序列
x_pred = np.concatenate([[x1_train[0]], np.diff(x1_pred)])

# ================================
# 6. 区分训练集和测试集预测
# ================================
x_train_pred = x_pred[train_idx]
x_test_pred = x_pred[test_idx]


# ================================
# 7. 定义 MAPE 函数
# ================================
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 防止除以零
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ================================
# 8. 误差计算
# ================================
# 训练误差
train_mse = mean_squared_error(x_train, x_train_pred)
train_mae = mean_absolute_error(x_train, x_train_pred)
train_r2 = r2_score(x_train, x_train_pred)
train_mape = mean_absolute_percentage_error(x_train, x_train_pred)

# 测试误差
x_test = x[test_idx]
test_mse = mean_squared_error(x_test, x_test_pred)
test_mae = mean_absolute_error(x_test, x_test_pred)
test_r2 = r2_score(x_test, x_test_pred)
test_mape = mean_absolute_percentage_error(x_test, x_test_pred)

print("\n===== Identified Model Formula =====")
print(model.get_formula())

print("\n===== 训练集误差 =====")
print(f"MSE={train_mse:.4f}, MAE={train_mae:.4f}, R2={train_r2:.4f}, MAPE={train_mape:.2f}%")

print("\n===== 测试集误差 =====")
print(f"MSE={test_mse:.4f}, MAE={test_mae:.4f}, R2={test_r2:.4f}, MAPE={test_mape:.2f}%")

# ================================
# 9. 绘图
# ================================


plt.figure(figsize=(12,5))
plt.plot(t[train_idx], x[train_idx], label='训练集真实', linewidth=2)
plt.plot(t[train_idx], x_train_pred, '--', label='训练集预测', linewidth=2)
plt.plot(t[test_idx], x[test_idx], label='测试集真实', linewidth=2)
plt.plot(t[test_idx], x_test_pred, '--', label='测试集预测', linewidth=2)
plt.xlabel('时间')
plt.ylabel('x')
plt.title('训练集与测试集预测对比')
plt.legend()
plt.grid()
plt.show()
