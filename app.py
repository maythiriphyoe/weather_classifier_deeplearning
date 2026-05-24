import io
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from tensorflow.keras import layers, models
from PIL import Image

app = Flask(__name__)

CLASSES = ['cloudy', 'rainy', 'shiny', 'sunrise']

def build_local_model():
    model = models.Sequential([
        # We explicitly provide the input shape here to instantiate the layer shapes immediately
        layers.Input(shape=(150, 150, 3)),
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5), 
        layers.Dense(len(CLASSES), activation='softmax')
    ])
    return model

# 2. Initialize the empty layout and load the raw weights matrix safely
model = build_local_model()
model.load_weights('weather_cnn.weights.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part found in the request'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'})

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        img = img.resize((150, 150))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASSES[predicted_index]
        confidence = float(predictions[0][predicted_index]) * 100

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': f"Failed to process image: {str(e)}"})

if __name__ == '__main__':
    app.run(port=5000, debug=True)