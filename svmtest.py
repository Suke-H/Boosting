import numpy as np
import cvxopt
import matplotlib.pyplot as plt

def Random(a, b):
    """ aからbまでの一様乱数を返す """
    return (b - a) * np.random.rand() + a

def make_artificial_data(n, phase):
    """
    -1 <= x, y <= 1での一様乱数により作られた2次元のデータを
    label 1: 右
    label -1: 左
    の2クラスにしたデータ

    train: 各ラベルの数を同じにしたデータ生成
    val, test: ランダムにデータ生成
    """

    # trainデータ生成
    if phase == "train":
        # 正解ラベル
        labels = np.array([1 if i < int(n/2) else -1 for i in range(n)])
        # データ
        dataset = np.zeros((n, 2))

        for i, label in enumerate(labels):

            # label 1
            if label == 1:
                dataset[i] = [Random(0, 1), Random(-1, 1)]

            # label -1
            else:
                dataset[i] = [Random(-1, 0), Random(-1, 1)]

        # シャッフル
        perm = np.random.permutation(n)
        dataset, labels = dataset[perm], labels[perm]

    # val, testデータ生成
    else:
        # データ
        dataset = np.array([[Random(-1, 1), Random(-1, 1)] for i in range(n)])

        # 正解ラベル
        labels = np.zeros(n)

        for i, data in enumerate(dataset):
            x, y = data

            # label 1
            if x >= 0:
                labels[i] = 1

            # label -1
            else:
                labels[i] = -1

    return np.array(dataset, dtype="float32"), np.array(labels, dtype="int")

if __name__ == '__main__':

    # 訓練データ生成
    N = 1000
    X, y = make_artificial_data(N, phase="train")

    # Q_ij = y_i*y_j*dot(x[i],x[j])
    _Q = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            _Q[i, j] = y[i]*y[j]*np.dot(X[i], X[j])
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
    W = np.dot(alpha * y, X)
    # b = (dot(w, x+) + dot(w, x-)) * (-1/2) 
    bias = - np.dot(X[top2_sv_indices], W).mean()

    # testデータを作成
    test_X, test_y = make_artificial_data(100, phase="test")

    # y = wx + b >= 0を+1に、y < 0を-1に
    eval_y = np.dot(W, test_X.T) + bias
    p_indices = np.where(eval_y >= 0)
    p_datas = test_X[p_indices]
    n_datas = np.delete(test_X, p_indices, axis=0)

    plt.scatter(p_datas[:, 0], p_datas[:, 1], label="positive")
    plt.scatter(n_datas[:, 0], n_datas[:, 1], label="negative")

    plt.legend()
    plt.grid()
    plt.show()