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


model = tf.keras.models.load_model('edge-detection-tfteed-60k-mish.keras')


def preprocess_image(image):
    if image.shape[2] == 4:
        image = image[:, :, :3] 

    image = cv2.resize(image, (512, 512))
    image = image.astype(np.float32)
    image -= [103.939, 116.779, 123.68] 
    return np.expand_dims(image, axis=0)  

def postprocess_output(outputs):

    output = np.squeeze(outputs[-1])
    output = cv2.resize(output, (640, 480))  
    output = (output - output.min()) / (output.max() - output.min())  
    output = (output * 255).astype(np.uint8)  
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
   
    return output

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No image file", 400

    file = request.files['image'].read()
    image = np.array(Image.open(BytesIO(file)))

    preprocessed_image = preprocess_image(image)
    outputs = model.predict(preprocessed_image)

    processed_output = postprocess_output(outputs)  

    try:
        _, img_encoded = cv2.imencode('.png', processed_output)
        return send_file(BytesIO(img_encoded), mimetype='image/png')
    except cv2.error as e:
        print(f"Failed to encode image: {e}")
        return "Failed to encode image", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)