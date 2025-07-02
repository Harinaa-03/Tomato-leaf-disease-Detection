from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model (Make sure CNN.model is in your project folder)
model = load_model('CNN.model')
data_dir = "data"
class_names = os.listdir(data_dir)  # Load class names from the dataset

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Control or treatment suggestions for leaf diseases
control_measures = {
    "bacterial spot": "Use copper sprays at regular intervals and ensure proper crop rotation.",
    "early blight": "Apply fungicides such as chlorothalonil or copper sprays, and remove affected foliage.",
    "late blight": "Ensure good drainage and use fungicides like mancozeb or copper sprays.",
    "leaf mold": "Improve air circulation, avoid overhead watering, and use fungicide treatments.",
    "mosaic virus": "Remove infected plants immediately and control insect vectors like aphids.",
    "septoria leaf spot": "Remove infected leaves, water at the base of plants, and use a copper fungicide.",
    "spider mites two-spotted spider mite": "Spray with miticides or insecticidal soap, and increase humidity to discourage mites.",
    "target spot": "Remove infected leaves and apply a fungicide such as mancozeb.",
    "yellow leaf curl virus": "Control whitefly populations, and remove infected plants. There's no cure for viral infections.",
    "healthy": "No disease detected. Keep monitoring and maintain optimal plant care."
}

# Function to check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image and calculate the affected area percentage
def preprocess_image(file_path):
    # Load the image
    img = cv2.imread(file_path)
    img = cv2.medianBlur(img, 1)
    
    # Convert to grayscale for thresholding
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create a binary image for affected areas
    _, thresh = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours for the affected regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate the total area of the image
    total_area = img.shape[0] * img.shape[1]
    
    # Calculate the affected area using contours
    affected_area = 0
    for contour in contours:
        affected_area += cv2.contourArea(contour)
    
    # Calculate the percentage of affected area
    affected_percentage = (affected_area / total_area) * 100
    
    # Resize the image for model input
    img_resized = cv2.resize(img, (50, 50))  # Resize the image to match the model input size
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = img_resized / 255.0  # Normalize the image

    return img_resized, affected_percentage

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and classification
@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the image and calculate the affected area percentage
        img, affected_percentage = preprocess_image(file_path)
        
        # Make predictions
        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        class_label = class_names[class_index]
        
        # Get the control measure for the predicted disease
        control_measure = control_measures.get(class_label, "No control measure found.")
        
        # Render the result page with the uploaded image, classification result, control measure, and affected percentage
        return render_template('index.html', label=class_label, control_measure=control_measure, image_url=file_path, affected_percentage=affected_percentage)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
