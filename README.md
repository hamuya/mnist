# MNIST 手書き数字認識

このプロジェクトは、MNISTデータセットを使用してモデルをトレーニングし、手書き数字を認識する方法を示します。

## プロジェクト構成

mnist_digit_recognition/

│


├── data/

│ ├── upload_sample_image_here.txt

│

├── model/

│ └── train_and_save_model.py

│

├── src/

│ ├── predict_digit.py

│

├── .gitignore

├── README.md

└── requirements.txt

## 使用方法

### 1. モデルのトレーニングと保存

まず、モデルをトレーニングし、保存します。以下のコマンドを実行してください：

python model/train_and_save_model.py

### 2. 画像から数字を予測
次に、アップロードした画像から手書き数字を予測します。以下のコマンドを実行してください：

python src/predict_digit.py --image_path <画像ファイルのパス>

例：python src/predict_digit.py --image_path data/your_image.png

## 依存関係
必要なパッケージをインストールするには、以下のコマンドを実行してください：
pip install -r requirements.txt

## ディレクトリ詳細

data/: サンプル画像をアップロードするためのディレクトリ。
model/: モデルのトレーニングと保存を行うスクリプトが含まれます。
src/: 画像を予測するためのスクリプトが含まれます。
.gitignore: Gitに含めないファイルを指定します。
README.md: このファイルです。
requirements.txt: プロジェクトで必要なPythonパッケージがリストされています。

## 注意事項
画像は、背景が白で数字が黒のものを推奨します。
画像ファイルはJPEG、PNG、BMPなどの一般的なフォーマットを使用できます。
ライセンス
このプロジェクトはMITライセンスのもとで公開されています。詳細はLICENSEファイルを参照してください。



このREADME.mdファイルは、プロジェクトの概要、ディレクトリ構成、使用方法、依存関係のインストール方法、注意事項、ライセンス情報を含んでいます。GitHubのリポジトリにこのファイルを追加することで、他の開発者がプロジェクトを理解しやすくなります。
