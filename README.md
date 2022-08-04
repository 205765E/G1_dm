## Check_fashion
"Check_fashion"は画像からファッションの季節を判別するプロジェクトです．


## DEMO
精度の結果出力の例<br>
夏の評価<br>
夏：0.9 冬：0.1<br>
冬の評価<br>
夏：0.3 冬：0.7


## Features
精度判定の出力は画像一つ一つの判定割合を確認することができます．


## Requirement
Python 3.7.6<br>
PIL<br>
glob<br>
numpy<br>
keras<br>
TensorFlow


## Installation
- Pythonのダウンロード方法
以下でpyenvをダウンロードする
```
brew install pyenv
```
以下でpyenvでPythonをダウンロードする
```
pyenv install 3.7.6
```
- PILのインストール方法
```python
pip install Pillow
```
- globのインストール方法
標準ライブラリ

- numpy
```python
pip install numpy
```

- keras
```python
pip3 install keras
```
- TensorFlow
```python
pip3 install tensorflow
```

## Usage
gitからtest4.py，learn.py，inference.pyをダウンロードする．<br>
教師データとテストデータとしてネットから夏服と冬服の画像を複数枚用意する．<br>
test4.pyは教師データとテストデータに前処理を加えて作成します．<br>
learn.pyは教師データをもとに学習を行います．<br>
inference.pyはテストデータを使用して判別の精度を出力します．<br>
ターミナルから以下の順に実行する．
```python
pyhton test4.py
pyhton learn.py
python inference.py
```

## Note
learn.pyは学習フェイズであり，実行に時間がかかります．<br>
教師データは著作権の都合により添付していません.


## Author
琉球大学工学部工学科知能情報コース生4名の共同開発です．
