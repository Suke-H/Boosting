import numpy as np
import cvxopt
import itertools
from tqdm import tqdm

class SVM:
    """
    ハードマージン線形SVM

    Attribute
    self.W : 重み
    self.bias : バイアス

    """
    def __init__(self, dim):
        """
        Attribute
        dim : 入力次元(今回は28*28)

        """

        self.W = np.zeros(dim)
        self.bias = 0

    def train(self, x, y):
        """
        SVMを学習

        https://satopirka.com/2018/12/theory-and-implementation-of-linear-support-vector-machine/
        を参考に実装

        Q[i][j] = y[i]*y[j]*dot(x[i],x[j])
        p = [-1, -1, ...]
        G = -I
        h = [0, 0, ...]T
        A = [y[0], y[1], ..., y[N-1]]
        b = 0

        として凸2次計画問題をcvxoptにより解き、alphaを算出した

        その後、

        W[d] = Σ alpha * y[i] * x[i][d] = dot(alpha*y, x[d])
        b = (dot(W, x+) + dot(W, x-)) * (-1/2) 

        によりWとdを算出(X+, X-はサポートベクター)

        Attribute
        x : 入力データ
        y : 正解ラベル

        """

        N = len(y)

        # Q
        _Q = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                _Q[i, j] = y[i]*y[j]*np.dot(x[i], x[j])
        Q = cvxopt.matrix(_Q)

        # p, G, h, A, b
        p = cvxopt.matrix(-np.ones(N))
        G = cvxopt.matrix(-np.eye(N))
        h = cvxopt.matrix(np.zeros(N))
        A = cvxopt.matrix(y[np.newaxis], (1, N), 'd')
        b = cvxopt.matrix(0.0)

        # 凸２次計画問題を解く
        solution = cvxopt.solvers.qp(Q, p, G, h, A, b)
        alpha = np.array(solution['x']).flatten()

        # 正負のSVを同時に出す
        top2_sv_indices = alpha.argsort()[-2:]

        # W, bを算出
        self.W = np.dot(alpha * y, x)
        self.bias = - np.dot(x[top2_sv_indices], self.W).mean()

    def eval(self, x):
        """
        推論ラベルを返す

        Attribute
        x : 入力データ
        
        """

        N = len(x)

        # y = wx + b
        eval_y = np.dot(self.W, x.T) + self.bias

        # y >= 0なら1、y < 0なら-1を出力
        output = np.full(N, -1)
        p_indices = np.where(eval_y >= 0)
        output[p_indices] = 1

        return output

