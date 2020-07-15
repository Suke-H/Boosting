import numpy as np
import cvxopt
import itertools
from tqdm import tqdm

class SVM:
    def __init__(self, dim):

        self.W = np.zeros(dim)
        self.bias = 0

    def train(self, x, y):

        N = len(y)

        # Q_ij = y_i*y_j*dot(x[i],x[j])
        _Q = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                _Q[i, j] = y[i]*y[j]*np.dot(x[i], x[j])
        Q = cvxopt.matrix(_Q)

        # p = [-1, -1, ...]
        p = cvxopt.matrix(-np.ones(N))
        # G = -I
        G = cvxopt.matrix(-np.eye(N))
        # h = [0, 0, ...]T
        h = cvxopt.matrix(np.zeros(N))
        # A = [y_0, y_1, ..., y_N]
        A = cvxopt.matrix(y[np.newaxis], (1, N), 'd')
        # b = 0
        b = cvxopt.matrix(0.0)
        # 凸２次計画問題を解く
        solution = cvxopt.solvers.qp(Q, p, G, h, A, b)
        alpha = np.array(solution['x']).flatten()

        # 正負のSVを同時に出す
        top2_sv_indices = alpha.argsort()[-2:]

        # w_d = Σ alpha * y[i] * x[i][d] = dot(alpha*y, x[d])
        self.W = np.dot(alpha * y, x)
        # b = (dot(w, x+) + dot(w, x-)) * (-1/2) 
        self.bias = - np.dot(x[top2_sv_indices], self.W).mean()

    def eval(self, x):

        N = len(x)

        # y = wx + b
        eval_y = np.dot(self.W, x.T) + self.bias

        # y >= 0なら1、y < 0なら-1を出力
        output = np.full(N, -1)
        p_indices = np.where(eval_y >= 0)
        output[p_indices] = 1

        return output

