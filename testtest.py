import numpy as np

x = np.full((100, 8*8), 1)
print(x.shape)

# データ数
n = x.shape[0]
# 一次元配列の要素数
d = x.shape[1]
# axis
axis_list = np.array([i for i in range(n)])
# th
th_list = np.array([i for i in range(n-1)])
# # sign
# sign_list = np.array([-1, 1])

x_mat = np.tile(x, (n-1, 1, 1))
x_mat = x_mat.transpose(2, 1, 0)
print(x_mat.shape)

m0 = x.T[:, :n-1]
m1 = x.T[:, 1:]
m_mat = (m0 + m1) / 2
m_mat = np.tile(m_mat, (n, 1, 1))
m_mat = m_mat.transpose(1, 0, 2)
print(m_mat.shape)

values = x_mat - m_mat
preds = np.where(values >= 0, 1, -1)

