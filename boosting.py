import numpy as np
from tqdm import tqdm

class AdaBoost:
    def __init__(self, model, epoch):
        self.epoch = epoch
        self.model_list = [model for i in range(epoch)]

        self.alpha = np.zeros(epoch)

    def fit(self, x, y):
        """
        各弱学習器の重みalpha_m(m=1...epoch)を算出すれば終わり

        """

        N = len(x)
        # 重み初期化
        w = np.ones(N)/N

        for m in tqdm(range(self.epoch)):
            # 学習
            self.model_list[m].train(x, y)
            mistakes = (self.model_list[m].eval(x) != y)

            # Eとalpha算出
            # (各データの重みwによりEを計算)
            e = np.sum(w * mistakes)
            self.alpha[m] = np.log((1.0-e)/e) / 2

            # 重み更新
            w = w * np.exp(self.alpha[m] * mistakes)
            w = w / np.sum(w)

    def predict(self, x):
        '''
        弱学習器f_mで各画像をpredictした後、alphaにより重みつけして出力
        f = sign(Σ alpha_m * f_m(x))

        '''

        # 弱学習器f_mで各画像をpredict
        predicts = np.zeros((self.epoch, len(x)))
        for m in range(self.epoch):
            predicts[m] = self.model_list[m].eval(x)

        #  f = sign(Σ alpha_m * f_m(x))
        return np.sign(np.dot(predicts.T, self.alpha))