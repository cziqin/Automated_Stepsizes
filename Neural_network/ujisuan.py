import numpy as np

# 定义矩阵 A
A = np.array([
    [0.4,  0,  0, 0.6,  0],
    [0.5, 0.5,  0,  0,  0],
    [0.4,  0.3, 0.1,  0, 0.2],
    [0,  0,  0, 0.4,  0.6],
    [0,  0.6,  0.2,  0, 0.2]
])

# 计算 A 的特征值和左特征向量
eigenvalues, eigenvectors = np.linalg.eig(A.T)

# 找到对应于特征值1的左特征向量
index = np.where(np.isclose(eigenvalues, 1))[0][0]
u = np.real(eigenvectors[:, index])

# 归一化 u 使其和为 5
u = u * 5 / np.sum(u)

print("找到的 u 向量为：", u)

import numpy as np

# 定义向量 u 和矩阵 A
u = np.array([1.24087591, 1.31386861, 0.2189781, 1.24087591, 0.98540146])
A = np.array([
    [0.4,  0,  0, 0.6,  0],
    [0.5, 0.5,  0,  0,  0],
    [0.4,  0.3, 0.1,  0, 0.2],
    [0,  0,  0, 0.4,  0.6],
    [0,  0.6,  0.2,  0, 0.2]
])

# 计算 u^T A
result = np.dot(u, A)

# 检查结果是否接近原始的 u 向量
if np.allclose(result, u):
    print("是的，这个向量是矩阵 A 对应于特征值 1 的左特征向量。")
else:
    print("不是，这个向量不是矩阵 A 对应于特征值 1 的左特征向量。")
