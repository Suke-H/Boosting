import numpy as np
from tqdm import tqdm

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
        self.d = d
        self.class_num = class_num
        self.input_dim = input_dim

        self.class_code = HadamardMatrix(d, class_num)
        self.model_list = [model for i in range(2**d)]

    def train(self, x):

        data_num = len(x) // self.class_num

        for i in tqdm(range(2**self.d)):
            # class_codeの各列が+1のラベルをplus_label、-1のラベルをminus_labelとする
            plus_label = np.where(self.class_code[:, i] == 1)[0]
            minus_label = np.where(self.class_code[:, i] == -1)[0]

            print(plus_label)
            print(minus_label)

            # クラスが全て+1/-1ならmodelはNoneにする
            # (多分最初の行のみ)
            if len(plus_label) == self.class_num or len(plus_label) == 0:
                self.model_list[i] = None
                print("SKIP")
                continue

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
                print("plus: {}:{}".format(plus*data_num, (plus+1)*data_num))
                bit_x = np.concatenate([bit_x, x[plus*data_num:(plus+1)*data_num]], axis=0)
            print(bit_x.shape)

            for minus in minus_label:
                print("plus: {}:{}".format(minus*data_num, (minus+1)*data_num))
                bit_x = np.concatenate([bit_x, x[minus*data_num:(minus+1)*data_num]], axis=0)
            print(bit_x.shape)

            print("plus: {}, minus: {}".format(plus_label, minus_label))
            print("x: {}".format(len(bit_x)))

            # ラベル
            bit_y = np.array([1 if j < label_num*data_num else -1 for j in range(2*label_num*data_num)])

            # シャッフル
            perm = np.random.permutation(2*label_num*data_num)
            bit_x = bit_x[perm]
            bit_y = bit_y[perm]

            # モデルiを学習
            self.model_list[i].train(bit_x, bit_y)
            
    def eval(self, x):

        N = len(x)
        out_codes = np.zeros((N, 2**self.d))

        print(self.model_list)

        for i, model in enumerate(tqdm(self.model_list)):

            # モデルが定義されてないときはドントケア
            if model is None:
                code = np.full(N, self.class_code[0, i])
                out_codes[:, i] = code
                print("SKIP")
                continue
        
            # 画像ごとにコードを出力
            out_codes[:, i] = model.eval(x)

        print(out_codes)

        # 画像分クラスコードを用意する
        class_codes = np.tile(self.class_code, (N, 1, 1))

        # 出力コードをクラスコードの形に合わせる
        out_codes = out_codes[:, np.newaxis, :]
        out_codes = np.tile(out_codes, (1, self.class_num, 1))
        print(out_codes)

        # 各画像で、各クラスコードとのハミング距離を算出
        hammings = np.count_nonzero((out_codes - class_codes), axis=2)

        print(hammings)

        # ハミング距離最小のラベルを出力(同じなら若い方を選択)
        eval_y = np.argmin(hammings, axis=1)

        print(eval_y)

        return eval_y

        

        
            
        



