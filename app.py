from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = r"C:\Users\manya\OneDrive\Desktop\lung-cancer\best_model.weights.h5"
model = load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = [
    "Adenocarcinoma (Left Lower Lobe)",
    "Large Cell Carcinoma (Left Hilum)",
    "Squamous Cell Carcinoma (Left Hilum)",
    "Normal"
]

# Ensure the upload folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})
    
    # Check if the file is a valid image (PNG, JPG, JPEG)
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"error": "Invalid file type. Only PNG, JPG, and JPEG are allowed."})
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Preprocess the image and make prediction
    try:
        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions)
        predicted_class = CLASS_LABELS[class_idx]
        confidence = float(np.max(predictions))
        
        # Clean up the uploaded file after prediction
        os.remove(file_path)

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence,
            "file_path": file_path
        })
    except Exception as e:
        # Handle any prediction errors
        os.remove(file_path)  # Clean up uploaded file in case of error
        return jsonify({"error": f"Prediction error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
