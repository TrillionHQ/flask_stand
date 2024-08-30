from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
import os



app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask_cors import CORS
CORS(app)


# Загрузка модели TensorFlow
model = tf.keras.models.load_model('edge-detection-tfteed-60k-mish.keras')


def preprocess_image(image):
    if image.shape[2] == 4:
        image = image[:, :, :3]  # Убираем альфа-канал

    image = cv2.resize(image, (224, 224))  # Размер зависит от модели
    image = image.astype(np.float32)
    image -= [103.939, 116.779, 123.68]  # Вычитание средних значений
    return np.expand_dims(image, axis=0)  # Добавляем размерность для батча, если это необходимо

def postprocess_output(outputs):
    # Суммирование всех четырех тензоров поэлементно
    summed_output = outputs[0] + outputs[1] + outputs[2] + outputs[3]
    
    summed_output = np.squeeze(summed_output)
    summed_output = cv2.resize(summed_output, (640, 480))  # Возвращаем к размеру исходного изображения
    summed_output = (summed_output - summed_output.min()) / (summed_output.max() - summed_output.min())  # Нормализация
    summed_output = (summed_output * 255).astype(np.uint8)  # Преобразование в 8-битное изображение

    # Преобразование в 3 канала для совместимости с cv2.imencode
    summed_output = cv2.cvtColor(summed_output, cv2.COLOR_GRAY2RGB)
    
    return summed_output

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No image file", 400

    file = request.files['image'].read()
    image = np.array(Image.open(BytesIO(file)))

    preprocessed_image = preprocess_image(image)
    outputs = model.predict(preprocessed_image)

    processed_output = postprocess_output(outputs)  # Суммируем поэлементно все выходные тензоры

    # Преобразование результата в изображение для отправки клиенту
    try:
        _, img_encoded = cv2.imencode('.png', processed_output)
        return send_file(BytesIO(img_encoded), mimetype='image/png')
    except cv2.error as e:
        print(f"Failed to encode image: {e}")
        return "Failed to encode image", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)