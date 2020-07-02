import numpy as np

if __name__ == '__main__':
    TrainingSampleNum = 2000 # 学習サンプル総数
    TestSampleNum = 100 # テストサンプル総数
    ClassNum = 10 # クラス数（今回は10）
    ImageSize = 8 # 画像サイズ（今回は縦横ともに8）
    TrainingDataFile = './Images/TrainingCompressionSamples/{0:1d}-{1:04d}.png'
    TestDataFile = './Images/TestCompressionSamples/{0:1d}-{1:04d}.png'

    train_x, train_y = LoadDataset(TrainingDataFile, TrainingSampleNum, ClassNum, ImageSize)