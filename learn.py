# 作成した学習データを学習させる
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop # TensorFlow1系
# from keras.optimizers import RMSprop # エラー（ImportError: cannot import name 'RMSprop' from 'keras.optimizers' (/usr/local/lib/python3.7/dist-packages/keras/optimizers.py)）が発生
# from tensorflow.keras.optimizers import RMSprop # TensorFlow2系

from keras.utils import np_utils
import numpy as np

def load_data(data):
    """データを読み込む関数.
    Argments:
        data (str): make_data()で作成したデータが保存されているファイル名．拡張子はnpy
    Returns:
        X_train (numpy.ndarray): 前処理後の教師データが行列の形で出力
        y_train (numpy.ndarray): X_trainのデータが夏か冬かを0, 1で表現したリスト
        X_test (numpy.ndarray): 前処理後のテスト用のデータが行列の形で出力
        y_test (numpy.ndarray): X_testのデータが夏か冬かを0, 1で表現したリスト
    """
    # indexを教師ラベルとして割り当てるため、0にはdogを指定し、1には猫を指定
    classes = ["summer", "winter"]
    num_classes = len(classes)
    # ファイル名変更
    X_train, X_test, y_train, y_test = np.load(data, allow_pickle=True)
    # 入力データの各画素値を0-1の範囲で正規化(学習コストを下げるため)
    X_train = X_train.astype("float") / 255
    X_test  = X_test.astype("float") / 255
    # to_categorical()にてラベルをone hot vector化
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test  = np_utils.to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test


    
def train(X, y, modelname):
    """モデルを学習する関数
    Argments:
        X (numpy.ndarray): 前処理後の教師データで行列の形で入力
        y (numpy.ndarray): Xのデータが夏か冬かを0, 1で表現したリスト
        modelneame (str): 学習したmodelの名前
    Returns:
        model (keras.engine.sequential.Sequential): 学習したモデル，modelneameの名前で保存される
    """
    model = Sequential()

    # Xは(1200, 64, 64, 3)
    # X.shape[1:]とすることで、(64, 64, 3)となり、入力にすることが可能です。
    
    print(X.shape)
    model.add(Conv2D(32,(3,3), padding='same',input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(96,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.30))

    model.add(Conv2D(128,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.35))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.45))
    model.add(Dense(2)) # 夏と冬を識別するため、２クラス分類のため2を指定
    model.add(Activation('softmax'))

    # https://keras.io/ja/optimizers/
    # 今回は、最適化アルゴリズムにRMSpropを利用
    opt = RMSprop(lr=0.00005, decay=1e-6)
    # https://keras.io/ja/models/sequential/
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    model.fit(X, y, batch_size=28, epochs=40)
    # HDF5ファイルにKerasのモデルを保存
    # 学習データ変更ごとにファイル名変更
    model.save(modelname)
    return model


if __name__ == "__main__":
    data = "./summer_winter_1.npy"
    modelname = "./cnn_1.h6"
    # データの読み込み
    X_train, y_train, X_test, y_test = load_data(data)
    # モデルの学習
    #model = train(X_train, y_train, modelname)

    #print(type(model))


