import numpy as np
from tqdm import tqdm
from PIL import Image
import itertools

def HadamardMatrix(d, n):
    """ 2^d次のアダマール行列から0~n-1行を抽出 """

    # アダマール行列の初期値(d=0)
    H = np.array([[1]])
    
    # H' = [[H, H], [H, -H]]
    for i in range(d):
        H0 = np.concatenate([H, H], axis=1)
        H1 = np.concatenate([H, -H], axis=1)
        H = np.concatenate([H0, H1], axis=0)

    return H[:n]

class ErrorDetectAndCorrect:
    """
    誤り訂正符号により2^d個の2値分類器から1つの結果を出す

    Attitude:
    models: 2^d個の2値分類器リスト
    d: 分類器の数2^d個(=アダマール行列の次元)
    class_num: 分類するクラス数

    """

    def __init__(self, model, d, class_num, input_dim):
        self.model = model
        self.d
        self.class_num = class_num
        self.input_dim = input_dim

        self.class_code = HadamardMatrix(d, class_num)
        self.model_list = [model for i in range(2**d)]

    def train(self, x):

        data_num = len(x) // self.class_num

        for i in range(2**self.d):
            # class_codeの各列が+1のラベルをplus_label、-1のラベルをminus_labelとする
            plus_label = np.where(self.class_code[:, i] == 1)[0]
            minus_label = np.where(self.class_code[:, i] == 1)[1]

            # クラスが全て+1/-1ならmodelはNoneにする
            # (多分最初の行のみ)
            if len(plus_label) == self.class_num or len(plus_label) == 0:
                self.model_list[i] = None
                break

            # +1と-1のラベルを同じ数にする
            if len(plus_label) > len(minus_label):
                plus_label = np.random.choice(plus_label, len(minus_label))
                np.sort(plus_label)

            elif len(plus_label) < len(minus_label):
                minus_label = np.random.choice(minus_label, len(plus_label))
                np.sort(minus_label)

            label_num = len(plus_label)

            bit_x = np.empty((0, self.input_dim))

            # xを結合
            for plus in plus_label:
                bit_x = np.concatenate([bit_x, x[plus*num:(plus+1)*num]], axis=0)

            for minus in minus_label:
                bit_x = np.concatenate([bit_x, x[minus*num:(minus+1)*num]], axis=0)

            print("plus: {}, minus: {}".format(plus_label, minus_label))
            print("x: {}".format(len(bit_x)))

            # ラベル
            bit_y = np.array([1 if j < label_num*num else -1 for j in range(2*label*num)])

            # モデルiを学習
            self.model_list[i].train(bit_x, bit_y)
            
    def eval(self, x):

        N = len(x)
        out_code = np.zeros((N, 2**self.d))

        for i, model in enumerate(self.model_list):

            # モデルが定義されてないときはドントケア
            if model is None:
                code = np.full(N, self.class_code[0, i])
                out_code[:, i] = code
                continue
        
            out_code[:, i] = self.model_list[i].eval(x)
            
        



