import numpy as np
import cvxopt
import itertools
from tqdm import tqdm

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
        self.bias = - np.dot(x[top2_sv_indices], self.W).mean()

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
        print(self.combinations)

        # one_vs_one_SVMのリスト(combinationsの順に格納)
        self.svm_list = []

    def train(self, x):
        """
        xは0, 1, ..., 9の順で同じ数並んでいるとする
        """

        # 1クラスの数
        num = len(x) // self.class_num

        for combi in tqdm(self.combinations):

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

        for i, combi in enumerate(tqdm(self.combinations)):
            # 各svmに全画像を入力し、出力
            outputs = self.svm_list[i].eval(x)

            # 勝敗表に記入
            for i, out in enumerate(outputs):
                # 出力が1なら上半分に1を記入
                if out == 1:
                    standings[combi[0], combi[1], i] = 1
                # 出力が0なら下半分に1を記入
                else:
                    standings[combi[1], combi[0], i] = 1

        # 各画像での各svmの勝ち数
        standings = standings.transpose(2, 0, 1)
        win_nums = np.sum(standings, axis=2)

        print(standings)
        print(win_nums)

        # 勝ち数が一番多いラベルを出力
        eval_y = np.argmax(win_nums, axis=1)
        
        return eval_y

class one_vs_other_SVM:

    def __init__(self, class_num, dim):

        self.class_num = class_num
        self.dim = dim

        # one_vs_other_SVMのリスト
        # (0_vs_other, 1_vs_other, ...の順に格納)
        self.svm_list = []

    def train(self, x):
        """
        xは0, 1, ..., 9の順で同じ数並んでいるとする
        """

        # 1クラスの数
        num = len(x) // self.class_num

        for label in tqdm(range(self.class_num)):

            # label vs other svmを生成
            svm = binary_SVM(self.dim)

            # labelとotherを結合
            # otherは1クラスの数分をランダムで取ってくる
            label_x = x[label*num:(label+1)*num]
            other_x = np.delete(x, [i for i in range(label*num, (label+1)*num)], axis=0)
            other_x = other_x[np.random.choice(len(x)-num, num), :]
            vs_x = np.concatenate([label_x, other_x])

            # ラベルがlabelのとき1, それ以外を-1にする
            vs_y = np.array([1 if i < num else -1 for i in range(num*2)])

            # label以外のxを全てotherに回す場合(遅いうえによくない)
            # vs_x = x[:, :]
            # vs_y = np.array([1 if label*num <= i <= (label+1)*num else -1 for i in range(len(x))])

            print(vs_x.shape, vs_y.shape)

            # 学習
            svm.train(vs_x, vs_y)

            # svm_listに保存
            self.svm_list.append(svm)

    def eval(self, x):

        N = len(x)

        # 勝った回数
        win_nums = np.zeros((N, self.class_num))

        for i, label in enumerate(tqdm(range(self.class_num))):
            # 各svmに全画像を入力し、出力
            outputs = self.svm_list[i].eval(x)

            # 勝ち数を追加していく
            for j, out in enumerate(outputs):
                # 出力が1ならそのラベルにのみ+1
                if out == 1:
                    win_nums[j, label] += 1

                # 出力が0ならそのラベル以外+1
                else:
                    other_index = np.delete(np.array([i for i in range(self.class_num)]), label)
                    win_nums[j, other_index] += 1

        print(win_nums)

        # 勝ち数が一番多いラベルを出力(同数なら若い方を選択)
        eval_y = np.argmax(win_nums, axis=1)
        
        return eval_y


