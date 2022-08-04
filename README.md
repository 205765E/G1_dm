#Check_fashion
"Check_fashion"は画像からファッションの季節を判別するプロジェクトです．


#DEMO
精度の結果出力の例
夏の評価
夏：0.9 冬：0.1
冬の評価
夏：0.3 冬：0.7


#Features
精度判定の出力は画像一つ一つの判定割合を確認することができます．


#Requirement
Python 3.7.6
PIL
glob
numpy
keras
TensorFlow


#Installation
---Pythonのダウンロード方法---
以下でpyenvをダウンロードする
brew install pyenv
以下でpyenvでPythonをダウンロードする
pyenv install 3.7.6

---PILのインストール方法---
pip install Pillow

---globのインストール方法---
標準ライブラリ

---numpy---
pip install numpy

---keras---
pip3 install keras

---TensorFlow---
pip3 install tensorflow


#Usage
gitからtest4.py，learn.py，inference.pyをダウンロードする．
教師データとテストデータとしてネットから夏服と冬服の画像を複数枚用意する．
test4.pyは教師データとテストデータに前処理を加えて作成します．
learn.pyは教師データをもとに学習を行います．
inference.pyはテストデータを使用して判別の精度を出力します．
ターミナルから以下の順に実行する．
pyhton test4.py
pyhton learn.py
python inference.py


#Note
learn.pyは学習フェイズであり，実行に時間がかかります．


#Author
琉球大学工学部工学科知能情報コース生4名の共同開発です．
