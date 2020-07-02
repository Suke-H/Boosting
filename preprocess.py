from skimage import io
import numpy as np
from tqdm import tqdm
from PIL import Image
import itertools

def ImageCompression(DataFile, OutFile, SampleNum, ClassNum, BeforeImageSize, AfterImageSize):

    """
    Attribute

    DataFile: 圧縮する画像のパスのリスト
    OutFile: 圧縮後の画像のパスのリスト。あらかじめ作っておく
    SampleNum: 画像数
    ClassNum: クラス数
    BeforeImageSize: 圧縮前の画像の一辺
    AfterImageSize: 圧縮後の画像の一辺

    """
    datas = np.zeros((SampleNum,BeforeImageSize,BeforeImageSize), dtype=np.int16)

    for label in tqdm(range(0, ClassNum)):
        for sample in range(0, SampleNum // ClassNum):

            # PIL形式で画像を開く
            filename = DataFile.format(label, sample)
            img = Image.open(filename)
            # 8*8にリサイズ
            img = img.resize((AfterImageSize, AfterImageSize))
            # 保存
            filename = OutFile.format(label, sample)
            img.save(filename)

if __name__ == '__main__':
    TrainingSampleNum = 2000 # 学習サンプル総数
    TestSampleNum = 10000 # テストサンプル総数
    ClassNum = 10 # クラス数（今回は10）
    ImageSize = 28 # 画像サイズ（今回は縦横ともに28）
    TrainingDataFile = './Images/TrainingSamples/{0:1d}-{1:04d}.png'
    TestDataFile = './Images/TestSamples/{0:1d}-{1:04d}.png'
    OutTrainingFile = './Images/TrainingCompressionSamples/{0:1d}-{1:04d}.png'
    OutTestFile = './Images/TestCompressionSamples/{0:1d}-{1:04d}.png'
    OutImageSize = 8 # 圧縮後のサイズ

    # ImageCompression(TrainingDataFile, OutTrainingFile, TrainingSampleNum, ClassNum, ImageSize, OutImageSize)
    ImageCompression(TestDataFile, OutTestFile, TestSampleNum, ClassNum, ImageSize, OutImageSize)
    
