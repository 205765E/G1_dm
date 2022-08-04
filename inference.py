#モデルの精度を出す
from unittest import result
from unittest.mock import patch
import numpy as np
from PIL import Image
from keras.models import load_model
import learn
import doctest

def model_test(data, keras_param):
    """与えられたデータを判定する，全て同じ季節としてその割合を出す．
    Argments:
        data (numpy.ndarray): テストするデータ
        keras_param (str): 機械学習のモデルの保存先
    Returns:
        なし．判定結果がprintされる
    """
    model = load_model(keras_param)
    sum_len = len(data)
    sum_num = 0
    win_num = 0
    for img in data:
        prd = model.predict(np.array([img]))
        print(prd) # 精度の表示
        prelabel = np.argmax(prd, axis=1)
        if prelabel == 0:
            print(">>> 夏")
            sum_num += 1
        elif prelabel == 1:
            print(">>> 冬")
            win_num += 1
    print("夏：" + str(sum_num/sum_len), "冬：" + str(win_num/sum_len))


def evaluate(keras_param, X_test, y_test):
    """夏と冬のテストデータを使ってモデルの評価を行う．
        Argments:
            keras_param (str): 機械学習のモデルの保存先
            X_test (numpy.ndarray): テスト用のデータ
            y_test (numpy.ndarray): X_testのデータが夏か冬かを0, 1で表現したリスト
        Returns:
            なし．判定結果がprintされる
    """
    sum_data = []
    win_data = []
    for i, img in enumerate(X_test):    
        if y_test[i][0] == 1.0:
            sum_data.append(img)
        else:
            win_data.append(img)
    print("夏の評価")
    model_test(sum_data, keras_param)
    print("----------------------------------------------------")
    print("冬の評価")
    model_test(win_data, keras_param)


def use(path, keras_param):
    """実際に画像を夏か冬かを判断する．
        Argments:
            path (str): 判定する画像のパスを入力
            keras_param (str): 機械学習のモデルのパス
        Returns:
            ans (str): 判定結果
        Example:
            >>> use("./30.png", "./cnn_1.h5")
            冬
    """
    img = Image.open(path)
    img = img.convert('RGB')
    # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
    imsize = (128, 128)
    img = img.resize(imsize)
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img / 255.0

    model = load_model(keras_param)
    prd = model.predict(np.array([img]))
    print(prd) # 精度の表示
    prelabel = np.argmax(prd, axis=1)
    
    if prelabel == 0:
        ans = "夏" 
    elif prelabel == 1:
        ans = "冬"
    return ans


if __name__ == "__main__":
    keras_param = "./cnn_1.h5"
    data = "./summer_winter_1.npy"
    #X_train, y_train, X_test, y_test = learn.load_data(data)
    #evaluate(keras_param, X_test, y_test)

    #path = "./30.png"
    #print(use(path ,keras_param))

    doctest.testmod()

