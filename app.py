from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2  # Ensure OpenCV is imported
from predict import model, predict_image, generate_heatmap

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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
            
            # Predict and generate heatmap
            class_labels = ['Not Weed', 'Weed']  # Replace with your class labels
            target_size = (139, 139)  # Same target size as used during training
            prediction = predict_image(model, filepath, target_size)
            predictions, heatmap, overlay, predicted_class = generate_heatmap(model, filepath, target_size, class_labels)
            
            heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'heatmap_' + filename)
            overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], 'overlay_' + filename)
            
            cv2.imwrite(heatmap_path, heatmap)
            cv2.imwrite(overlay_path, overlay)
            
            return render_template('index.html', filename=filename, prediction=predictions, predicted_class=predicted_class, heatmap_path=heatmap_path, overlay_path=overlay_path)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploaded_images/{filename}'))

if __name__ == '__main__':
    app.run(debug=True)