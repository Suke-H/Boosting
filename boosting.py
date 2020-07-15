import numpy as np
from tqdm import tqdm
from copy import copy

class AdaBoost:
    def __init__(self, model, epoch):
        self.epoch = epoch
        self.weak_learners = [copy(model) for i in range(epoch)]

        self.alpha = np.zeros(epoch)

    def train(self, x, y):
        """
        各弱学習器の重みalpha_m(m=1...epoch)を算出すれば終わり

        """

        print(x.shape, y.shape)

        N = len(x)
        eta = N // 2

        # 重み初期化
        w = np.ones(N)/N

        # for m in tqdm(range(self.epoch)):
        for m in range(self.epoch):

            # 重みに応じて訓練データをサンプリング
            sample = np.random.choice(N, eta, p=w)
            x_sample, y_sample = x[sample], y[sample]

            # 学習
            self.weak_learners[m].train(x_sample, y_sample)

            # 推論
            mistakes = (self.weak_learners[m].eval(x) != y).astype(np.int)

            # acc 100% のときどっか1つ間違えとく
            if np.count_nonzero(mistakes) == 0: 
                print("no mistake")
                mistakes[np.random.choice(N, 1)] = 1

            # acc < 50% のとき警告
            elif np.count_nonzero(mistakes) > eta:
                print("acc < 50%")

            # Eとalpha算出
            # (各データの重みwによりEを計算)
            e = np.sum(w * mistakes)
                
            self.alpha[m] = np.log(1.0/e - 1) / 2

            # 重み更新
            w = w * np.exp(self.alpha[m] * mistakes)
            w = w / np.sum(w)

            # print(mistakes)
            # print(w)
            # print(self.alpha[m])
            # a = input()


        # for m in self.weak_learners:
        #     print("="*50)
        #     print(m.W)
        #     print(m.bias)
        # a = input()

    def eval(self, x):
        '''
        弱学習器f_mで各画像をpredictした後、alphaにより重みつけして出力
        f = sign(Σ alpha_m * f_m(x))

        '''

        # 弱学習器f_mで各画像をpredict
        predicts = np.zeros((self.epoch, len(x)))
        for m in range(self.epoch):
            predicts[m] = self.weak_learners[m].eval(x)

        #  f = sign(Σ alpha_m * f_m(x))
        return np.sign(np.dot(predicts.T, self.alpha))