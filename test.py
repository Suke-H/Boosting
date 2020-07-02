from skimage import io
import numpy as np
from tqdm import tqdm
from PIL import Image
import itertools

from Step05_kNearestNeighbor import ReturnMatchLabel

def ImageCompression(DataFile, OutFile, SampleNum, ClassNum, ImageSize):
    datas = np.zeros((SampleNum,ImageSize,ImageSize), dtype=np.int16)

    i = 0
    for label in tqdm(range(0, ClassNum)):
        for sample in range(0, SampleNum // ClassNum):
            filename = DataFile.format(label, sample)
            img = Image.open(filename)
            # img = io.imread(filename)

            # 8*8にリサイズ
            img = img.resize((8, 8))

            filename = OutFile.format(label, sample)
            img.save(filename)
            # io.imsave(filename, img)

# LoadDataset
def LoadDataset(DataFile, SampleNum, ClassNum, ImageSize):
    labels = np.zeros(SampleNum, dtype=np.uint8)
    datas = np.zeros((SampleNum,ImageSize,ImageSize), dtype=np.int16)

    i = 0
    for label in tqdm(range(0, ClassNum)):
        for sample in range(0, SampleNum // ClassNum):
            filename = DataFile.format(label, sample)
            datas[i,:,:] = io.imread(filename).astype(np.int16)
            labels[i]=label
            i += 1

    return datas, labels

def DecisionStump(x, phase="DS")
    # データ数
    n = x.shape[0]
    # 一次元配列の要素数
    d = x.shape[1]
    # axis
    axis_list = np.array([i for i in range(n)])
    # th
    th_list = np.array([i for i in range(n-1)])
    # # sign
    # sign_list = np.array([-1, 1])

    x_mat = np.tile(x, (n-1, 1, 1))
    x_mat = x_mat.transpose(2, 1, 0)
    print(x_mat.shape)

    m0 = x.T[:, :n-1]
    m1 = x.T[:, 1:]
    m_mat = (m0 + m1) / 2
    m_mat = np.tile(m_mat, (n, 1, 1))
    print(m_mat.shape)

def DecisionStump(target, args):
    # 見る次元の場所, しきい値、符号(+1, -1)
    axis, th, sign = args
    # 1次元に変換
    flat_target = target.ravel()

    print(flat_target)

    if flat_target[axis] >= th:
        return sign

    else:
        return -sign

def BinaryModelError(models, x, ClassCode, ClassNum, args_set):
    # クラスコード
    for i, y in enumerate(ClassCode.T):
        print("=== {} bit ===========".format(16-i))
        BitModelError(models[i], x, y, ClassNum, args_set[i])


def BitModelError(bitmodel, x, y, ClassNum, args):
    results = np.zeros((ClassNum, ClassNum))
    DataNum = x.shape[0]
    # y = ClassCode[:, BitIndex].T

    for i, (t_img, label) in enumerate(tqdm(zip(x, y))):
        match_label = bitmodel(t_img, args)
        results[label, match_label] += 1

    total_correct_num = 0
    for t_label in range(0, ClassNum):
        total_correct_num += results[t_label, t_label]
    total_error = (DataNum - total_correct_num) / DataNum
    print ("TOTAL_ERROR: {0:.4f}".format(total_error))


def TestModel(model, test_x, test_y, ClassNum, args):
    results = np.zeros((ClassNum, ClassNum))
    TestSampleNum = len(test_y)
    each_class_num = TestSampleNum // ClassNum

    for i, (t_img, label) in enumerate(tqdm(zip(test_x, test_y))):
        match_label = model(t_img, args)
        results[label, match_label] += 1
        if (i != 0) and ((i+1) % each_class_num == 0):
            tested_class = i // each_class_num
            print(tested_class, each_class_num)
            print("{0:1d}: {1:.4f}".format(label, results[tested_class, tested_class] / each_class_num))

    print("= Confusion matrix ===========")
    for t_label in range(0, ClassNum):
        for m_label in range(0, ClassNum):
            print("{:04g}, ".format(results[t_label, m_label]), end="")
        print("")

    print("= Total Recognition accuracy ===========")
    total_correct_num = 0
    for t_label in range(0, ClassNum):
        total_correct_num += results[t_label, t_label]
    print ("TOTAL: {0:.4f}".format(total_correct_num / TestSampleNum))

def HadamardMatrix(d):
    # 2^d次のアダマール行列生成

    # アダマール行列の初期値(d=0)
    H = np.array([[1]])
    
    # H' = [[H, H], [H, -H]]
    for i in range(d):
        H0 = np.concatenate([H, H], axis=1)
        H1 = np.concatenate([H, -H], axis=1)
        H = np.concatenate([H0, H1], axis=0)

    return H

if __name__ == '__main__':
    TrainingSampleNum = 2000 # 学習サンプル総数
    TestSampleNum = 100 # テストサンプル総数
    ClassNum = 10 # クラス数（今回は10）
    ImageSize = 28 # 画像サイズ（今回は縦横ともに28）
    TrainingDataFile = './Images/TrainingSamples/{0:1d}-{1:04d}.png'
    TestDataFile = './Images/TestSamples/{0:1d}-{1:04d}.png'
    OutFile = './Images/TrainingCompressionSamples/{0:1d}-{1:04d}.png'
    OutNum = 100
    OutImageSize = 8

    x, y = LoadDataset(OutFile, OutNum, ClassNum, OutImageSize)

    d = 4
    ClassCode = HadamardMatrix(d)
    print(ClassCode)
    BitNum = 2**d
    models = [DecisionStump for i in range(BitNum)]
    # args = 見る次元の場所, しきい値、符号(+1, -1)
    args_set = np.array([[11, 10, 1] for i in range(BitNum)])

    BinaryModelError(models, x, ClassCode, ClassNum, args_set)

    # model = ReturnMatchLabel
    # ClassNum = 10
    # k = 3
    # args = [train_x, train_y, k]

    # TestModel(model, test_x, test_y, ClassNum, args)
