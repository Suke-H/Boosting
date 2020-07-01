import numpy as np
import cvxopt
import itertools

class binary_SVM:
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
        self.bias = - np.dot(x[top2_sv_indices], W).mean()

    def eval(self, x):

        N = len(x)

        # y = wx + b
        eval_y = np.dot(self.W, x.T) + self.bias

        # y >= 0なら1、y < 0なら0を出力
        output = np.zeros(N)
        p_indices = np.where(eval_y >= 0)
        output[p_indices] = 1

        return output

class one_vs_one_SVM:
    def __init__(self, class_num, dim):

        self.class_num = class_num
        self.dim = dim

        # クラスの組み合わせ
        self.combinations = np.array(list(itertools.combinations([i for i in range(class_num)], 2)))

        # one_vs_one_SVMのリスト(combinationsの順に格納)
        self.svm_list = []

    def train(self, x):
        """
        xは0, 1, ..., 9の順で同じ数並んでいるとする
        """

        # 1クラスの数
        num = len(y) // self.class_num

        for combi in self.combinations:

            # combi[0] vs combi[1] svmを生成
            svm = binary_SVM(self.dim)

            # combi[0]とcombi[1]のデータを結合
            vs_x = np.concatenate([x[num*combi[0]:num*(combi[0]+1)], x[num*combi[1]:num*(combi[1]+1)]], axis=0)
            # combi[0]が+1, combi[1]が-1
            vs_y = np.array([1 if i < num else -1 for i in range(num*2)])

            # 学習
            svm.train(vs_x, vs_y)

            # svm_listに保存
            self.svm_list.append(svm)

    def eval(self, x):

        N = len(x)

        # 全ての画像分の勝敗表
        standings = np.zeros((self.class_num, self.class_num, N))

        for i, combi in enumerate(self.combinations):
            # 各svmに全画像を入力し、出力
            output = self.svm_list[i].eval(x)
            # 勝敗表に記入(上半分のみ)
            standings[combi[0], combi[1]] = output

        # 各画像での各svmの勝ち数
        standings = standings.transpose(2, 0, 1)
        win_nums = np.sum(standings, axis=1)

        # 勝ち数が一番多いラベルを出力
        eval_y = np.argmax(win_nums, axis=1)
        
        return eval_y



