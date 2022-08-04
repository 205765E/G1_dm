# 教師データ作成
from PIL import Image
import os, glob
import numpy as np
from PIL import ImageFile
# IOError: image file is truncated (0 bytes not processed)回避のため
ImageFile.LOAD_TRUNCATED_IMAGES = True

def make_data(classes, save_name, num_testdata=10):
    """教師データとテストデータを作成する．
    Argments:
        classes (list): カテゴリデータ．今回は「夏服」と「冬服」．
        save_name (str): データを保存するときの名前.拡張子はnpy.
        num_testdata (int): テストデータの数,初期値は10でありこの値が画像データの数を超えるとエラーが起きる．
    Returns:
        X_train (numpy.ndarray): 画像データのRGBが教師データとして行列の形で出力
        y_train (numpy.ndarray): X_trainのデータが夏か冬かを0, 1で表現したリスト
        X_test (numpy.ndarray): 画像データのRGBがテスト用のデータとして行列の形で出力
        y_test (numpy.ndarray): X_testのデータが夏か冬かを0, 1で表現したリスト
    """
    #画像のサイズを128で固定する．
    image_size = 128
    X_train = []
    X_test  = []
    y_train = []
    y_test  = []

    # indexを教師ラベルとして割り当てるため、0にはsummerを指定し、1にはwinterを指定
    for index, classlabel in enumerate(classes):
        photos_dir = "./IMAGE_2/" + classlabel
        files = glob.glob(photos_dir + "/*.png")  #ここでパスを指定して画像を読み込む
        
        for i, file in enumerate(files):
            image = Image.open(file)
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
            #image.save("./tmp_img.jpg")
            #exit()

            data = np.asarray(image)
            if i < num_testdata:
                X_test.append(data)
                y_test.append(index)
            else:
                # angleに代入される値
                # -20
                # -15
                # -10
                #  -5
                # 0
                # 5
                # 10
                # 15
                # 画像を5度ずつ回転
                for angle in range(-20, 20, 5):

                    img_r = image.rotate(angle)
                    data = np.asarray(img_r)
                    X_train.append(data)
                    y_train.append(index)
                    # FLIP_LEFT_RIGHT　は 左右反転
                    img_trains = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                    data = np.asarray(img_trains)
                    X_train.append(data)
                    y_train.append(index) # indexを教師ラベルとして割り当てるため、0にはdogを指定し、1には猫を指定

    X_train = np.array(X_train)
    X_test  = np.array(X_test)
    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    xy = (X_train, X_test, y_train, y_test)
    #print(xy)
    print(X_train.shape)
    np.save(save_name, xy)
    return xy

if __name__ == "__main__":
    classes = ["re_men_summer", "re_men_winter"]
    save_name = "./summer_winter_1.npy"
    make_data(classes, save_name)
    

