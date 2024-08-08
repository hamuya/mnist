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
