# mnist
プロジェクト構成
css
コードをコピーする
mnist_digit_recognition/ 

│

├── data/

│   ├── upload_sample_image_here.txt

│

├── model/

│   └── train_and_save_model.py

│

├── src/

│   ├── predict_digit.py

│

├── .gitignore

├── README.md

└── requirements.txt

各ファイルの内容

.gitignore

モデルファイルや仮想環境など、Gitに含めないファイルを指定します。

markdown
コードをコピーする
__pycache__/
*.h5
*.png
*.jpg
*.jpeg
*.bmp
README.md
プロジェクトの説明や使用方法を記載します。

markdown
コードをコピーする
# MNIST Digit Recognition

This project demonstrates how to train a model on the MNIST dataset and use it to recognize handwritten digits.

## Project Structure

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

perl
コードをコピーする

## Usage

1. Train the model and save it:
    ```bash
    python model/train_and_save_model.py
    ```

2. Predict a digit from an uploaded image:
    ```bash
    python src/predict_digit.py --image_path <path_to_image>
    ```

## Dependencies

Install the required packages using:
```bash
pip install -r requirements.txt
shell
コードをコピーする

#### requirements.txt

必要なPythonパッケージをリストします。

tensorflow
pillow
matplotlib
numpy

csharp
コードをコピーする

#### train_and_save_model.py

モデルをトレーニングし、保存するスクリプトです。

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# データセットをロード
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# データを前処理
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# モデルを構築
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# モデルをコンパイル
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# モデルを学習
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

# モデルを評価
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# モデルを保存
model.save('model/mnist_model.h5')
predict_digit.py
アップロードされた画像を使って予測するスクリプトです。

python
コードをコピーする
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# 手書き数字の画像を前処理
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img = np.array(img).astype('float32') / 255
    img = img.reshape(1, 28, 28, 1)
    return img

# 手書き数字を判別
def predict_digit(image_path, model_path='model/mnist_model.h5'):
    model = tf.keras.models.load_model(model_path)
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)
    return predicted_digit

# メイン関数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict digit from an image.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
    args = parser.parse_args()

    digit = predict_digit(args.image_path)
    print(f'Predicted digit: {digit}')

    # 画像を表示
    img = Image.open(args.image_path)
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted digit: {digit}')
    plt.axis('off')
    plt.show()
プロジェクトの準備と実行
リポジトリを作成:
GitHubで新しいリポジトリを作成し、ローカルにクローンします。

ファイルをアップロード:
上記のディレクトリ構成とファイルをローカルのリポジトリに配置します。

リポジトリにプッシュ:
必要なファイルを追加し、GitHubにプッシュします。

bash
コードをコピーする
git add .
git commit -m "Initial commit"
git push origin main
モデルのトレーニング:
train_and_save_model.pyを実行してモデルをトレーニングし、保存します。
bash
コードをコピーする
python model/train_and_save_model.py
画像の予測:
predict_digit.pyを実行して、画像をアップロードして予測します。
bash
コードをコピーする
python src/predict_digit.py --image_path data/your_image.png
