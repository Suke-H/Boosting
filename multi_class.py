import numpy as np
import cvxopt
import itertools
from tqdm import tqdm
from copy import copy

from svm import SVM
from boosting import AdaBoost

class one_vs_one:
    def __init__(self, model, class_num, dim):

        self.class_num = class_num
        self.dim = dim

        # クラスの組み合わせ
        self.combinations = np.array(list(itertools.combinations([i for i in range(class_num)], 2)))

        # one_vs_one_modelのリスト(combinationsの順に格納)
        self.model_list = [copy(model) for i in range(class_num*(class_num-1) // 2)]

    def train(self, x):
        """
        xは0, 1, ..., 9の順で同じ数並んでいるとする

        """

        # 1クラスの数
        num = len(x) // self.class_num

        for j, combi in enumerate(tqdm(self.combinations)):

            # combi[0]とcombi[1]のデータを結合
            vs_x = np.concatenate([x[num*combi[0]:num*(combi[0]+1)], x[num*combi[1]:num*(combi[1]+1)]], axis=0)
            # combi[0]が+1, combi[1]が-1
            vs_y = np.array([1 if i < num else -1 for i in range(num*2)])

            # 学習
            # self.model_list[j].train(vs_x, vs_y)

            ImageSize = 28 
            binary_SVM = SVM(ImageSize**2)
            adaboost = AdaBoost(binary_SVM, 10)
            adaboost.train(vs_x, vs_y)
            self.model_list.append(adaboost)

    def eval(self, x):

        N = len(x)

        # 全ての画像分の勝敗表
        standings = np.zeros((self.class_num, self.class_num, N))

        for i, combi in enumerate(tqdm(self.combinations)):
            # 各modelに全画像を入力し、出力
            outputs = self.model_list[i].eval(x)

            # 勝敗表に記入
            for j, out in enumerate(outputs):
                # 出力が1なら上半分に1を記入
                if out == 1:
                    standings[combi[0], combi[1], j] = 1
                # 出力が-1なら下半分に1を記入
                else:
                    standings[combi[1], combi[0], j] = 1

        # 各画像での各modelの勝ち数
        standings = standings.transpose(2, 0, 1)
        win_nums = np.sum(standings, axis=2)

        print(standings)
        print(win_nums)

        # 勝ち数が一番多いラベルを出力
        eval_y = np.argmax(win_nums, axis=1)
        
        return eval_y

class one_vs_other:

    def __init__(self, model, class_num, dim):

        self.class_num = class_num
        self.dim = dim

        # one_vs_other_modelのリスト
        # (0_vs_other, 1_vs_other, ...の順に格納)
        # self.model_list = [copy(model) for i in range(class_num)]
        # self.model = model
        self.model_list = []

    def train(self, x):
        """
        xは0, 1, ..., 9の順で同じ数並んでいるとする
        """

        # 1クラスの数
        num = len(x) // self.class_num

        for j, label in enumerate(tqdm(range(self.class_num))):

            # labelとotherを結合
            # otherは1クラスの数分をランダムで取ってくる
            label_x = x[label*num:(label+1)*num]
            other_x = np.delete(x, [i for i in range(label*num, (label+1)*num)], axis=0)
            other_x = other_x[np.random.choice(len(x)-num, num), :]
            vs_x = np.concatenate([label_x, other_x])

            # ラベルがlabelのとき1, それ以外を-1にする
            vs_y = np.array([1 if i < num else -1 for i in range(num*2)])

            # 学習
            # self.model_list[j].train(vs_x, vs_y)
            # copy_model = copy(self.model).train(vs_x, vs_y)
            # self.model_list.append(copy_model)

            ImageSize = 28 
            binary_SVM = SVM(ImageSize**2)
            adaboost = AdaBoost(binary_SVM, 10)
            adaboost.train(vs_x, vs_y)
            self.model_list.append(adaboost)
            

    def eval(self, x):

        N = len(x)

        # for m in self.model_list:
        #     print("="*50)
        #     print(m.alpha)
        # a = input()

        # 勝った回数
        win_nums = np.zeros((N, self.class_num))

        for i, label in enumerate(tqdm(range(self.class_num))):
            # 各modelに全画像を入力し、出力
            outputs = self.model_list[i].eval(x)

            # 勝ち数を追加していく
            for j, out in enumerate(outputs):
                # 出力が1ならそのラベルにのみ+1
                if out == 1:
                    win_nums[j, label] += 1

                # 出力が-1ならそのラベル以外+1
                else:
                    other_index = np.delete(np.array([i for i in range(self.class_num)]), label)
                    win_nums[j, other_index] += 1

        print(win_nums)

        # 勝ち数が一番多いラベルを出力(同数なら若い方を選択)
        eval_y = np.argmax(win_nums, axis=1)
        
        return eval_y