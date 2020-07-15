import numpy as np
from tqdm import tqdm
from skimage import io

from svm import SVM
from multi_class import one_vs_one, one_vs_other

def LoadDataset(DataFile, SampleNum, ClassNum, ImageSize):
    """
    データセットのロード

    Attribute
    DataFile: 画像のパスのリスト
    SampleNum: 画像数
    ClassNum: クラス数
    ImageSize: 画像の一辺

    """

    datas = np.zeros((SampleNum,ImageSize,ImageSize), dtype=np.int16)
    labels = np.zeros(SampleNum, dtype=np.uint8)

    # 入力データ(datas)と正解ラベル(labels)を用意
    i = 0
    for label in tqdm(range(0, ClassNum)):
        for sample in range(0, SampleNum // ClassNum):
            filename = DataFile.format(label, sample)
            datas[i,:,:] = io.imread(filename).astype(np.int16)
            labels[i]=label
            i += 1

    # (N, width, height) -> (N, width*height)にリサイズ
    datas = datas.reshape(SampleNum, ImageSize**2).astype(np.float32)

    # 0~1に変換
    datas /= 255.0

    return datas, labels

def TestResult(ys, ts, ClassNum):
    """
    検証
    混合行列(Confusion matrix)と識別率(Total Recognition accuracy)を出力

    Attribute
    ys: 推論ラベル
    ts: 正解ラベル
    ClassNum: クラス数

    """
    # 混合行列
    results = np.zeros((ClassNum, ClassNum))

    Num = len(ys)
    each_class_num = Num // ClassNum

    # 混合行列に格納していく
    for i, (y, t) in enumerate(tqdm(zip(ys, ts))):
        results[t, y] += 1

        # クラスごとの分類結果をプリント
        if (i != 0) and ((i+1) % each_class_num == 0):
            tested_class = i // each_class_num
            print(tested_class, each_class_num)
            print("{0:1d}: {1:.4f}".format(y, results[tested_class, tested_class] / each_class_num))

    # 混合行列
    print("= Confusion matrix ===========")
    for t_label in range(0, ClassNum):
        for m_label in range(0, ClassNum):
            print("{:04g}, ".format(results[t_label, m_label]), end="")
        print("")

    # 識別率
    print("= Total Recognition accuracy ===========")
    total_correct_num = 0
    for t_label in range(0, ClassNum):
        total_correct_num += results[t_label, t_label]
    print ("TOTAL: {0:.4f}".format(total_correct_num / Num))

if __name__ == '__main__':

    TrainingSampleNum = 2000 # 学習サンプル総数
    TestSampleNum = 100 # テストサンプル総数
    ClassNum = 10 # クラス数（今回は10）
    ImageSize = 28 # 画像サイズ(一辺の長さ)
    TrainingDataFile = './Images/TrainingSamples/{0:1d}-{1:04d}.png' # 学習データのパス
    TestDataFile = './Images/TestSamples/{0:1d}-{1:04d}.png' # テストデータのパス

    # 学習、テストデータをロード
    train_x, train_t = LoadDataset(TrainingDataFile, TrainingSampleNum, ClassNum, ImageSize)
    test_x, test_t = LoadDataset(TestDataFile, TestSampleNum, ClassNum, ImageSize)
    
    # SVM(2値分類器)を作成
    binary_SVM = SVM(ImageSize**2)

    # 多クラス分類器にする
    # 1 vs 1 分類器
    # multi = one_vs_one(binary_SVM, ClassNum, ImageSize**2)
    # 1 vs other 分類器
    multi = one_vs_other(binary_SVM, ClassNum, ImageSize**2)

    # 学習
    multi.train(train_x)
    # 推論
    y = multi.eval(test_x)
    # 検証
    TestResult(y, test_t, ClassNum)

