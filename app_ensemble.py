from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2  # Import OpenCV
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from predict_ensemble import generate_heatmap

# Load the saved ensemble models
model1 = tf.keras.models.load_model('pretrain_model.keras')
model2 = tf.keras.models.load_model('pretrain2_model.keras')
model3 = tf.keras.models.load_model('pretrain_efficientnet_model.keras')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Preprocess a single image for prediction
def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict using an ensemble voting mechanism
def predict_with_voting(image_path, target_size):
    # Get predictions from each model
    preds1 = model1.predict(preprocess_image(image_path, target_size))
    preds2 = model2.predict(preprocess_image(image_path, target_size))
    preds3 = model3.predict(preprocess_image(image_path, target_size))

    # Convert predictions to class labels
    class1 = np.argmax(preds1)
    class2 = np.argmax(preds2)
    class3 = np.argmax(preds3)

    # Voting mechanism
    votes = [class1, class2, class3]
    final_prediction = np.bincount(votes).argmax()

    # Map the final prediction to class labels
    class_labels = ['Not Weed', 'Weed']
    predicted_class_label = class_labels[final_prediction]
    return predicted_class_label

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Create the upload directory if it doesn't exist
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            file.save(filepath)

            # Perform prediction
            target_size = (139, 139)  # Match model input size
            predicted_class = predict_with_voting(filepath, target_size)

            # Generate heatmap and overlay
            heatmap, overlay = generate_heatmap(model1, filepath, target_size)  # Using model1 for Grad-CAM

            # Save heatmap and overlay
            heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'heatmap_' + filename)
            overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], 'overlay_' + filename)

            cv2.imwrite(heatmap_path, heatmap)
            cv2.imwrite(overlay_path, overlay)

            return render_template(
                'index.html',
                filename=filename,
                predicted_class=predicted_class,
                heatmap_path=heatmap_path,
                overlay_path=overlay_path,
            )
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploaded_images/{filename}'))

if __name__ == '__main__':
    app.run(debug=True)