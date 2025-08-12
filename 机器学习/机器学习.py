import numpy as np
from sklearn.linear_model import LinearRegression

# 示例数据（需正确定义 x 和 y）
x = np.array([[1], [2], [3]])  # 特征值（必须为2D数组）
y = np.array([2, 4, 6])        # 目标值

# 建立线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(x,y)

# 输出模型参数
print('斜率：',model.coef_)
print('截距：',model.intercept_)

# 进行预测
x_new = np.array([[6],[7]])  # 新的自变量
y_pred = model.predict(x_new)  # 预测因变量
print('预测结果：',y_pred)