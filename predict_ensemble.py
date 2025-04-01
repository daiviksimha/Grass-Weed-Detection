
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2  # Import OpenCV
import matplotlib.pyplot as plt

# Load the saved models
model1 = tf.keras.models.load_model('pretrain_model.keras')
model2 = tf.keras.models.load_model('pretrain2_model.keras')
model3 = tf.keras.models.load_model('pretrain_efficientnet_model.keras')

# Function to preprocess a single image for prediction
def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict on a single image
def predict_image(model, img_path, target_size):
    img_preprocessed = preprocess_image(img_path, target_size)
    prediction = model.predict(img_preprocessed)
    return prediction

# Function to perform voting among the three models
def predict_with_voting(image_path, target_size):
    # Get predictions from each model
    preds1 = predict_image(model1, image_path, target_size)
    preds2 = predict_image(model2, image_path, target_size)
    preds3 = predict_image(model3, image_path, target_size)

    # Convert predictions to class labels (0 for Not Weed, 1 for Weed)
    class1 = np.argmax(preds1)
    class2 = np.argmax(preds2)
    class3 = np.argmax(preds3)

    # Voting mechanism
    votes = [class1, class2, class3]
    final_prediction = np.bincount(votes).argmax()  # Get the class with the maximum votes

    # Map the final prediction to class labels
    predicted_class_label = 'Weed' if final_prediction == 1 else 'Not Weed'
    print(f'The final prediction is: {predicted_class_label}')

    return predicted_class_label

import matplotlib.pyplot as plt

def generate_heatmap(model, img_path, target_size):
    img_array = preprocess_image(img_path, target_size)
    last_conv_layer_name = "conv2d"  # Replace with the name of the last convolutional layer in your model

    # Get the gradients of the predicted class
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply pooled gradients with convolution outputs and average
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Overlay the heatmap on the image
    img = cv2.imread(img_path)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

    return heatmap_colored, overlay


# Example usage
if __name__ == "__main__":
    # Define the target size
    target_size = (139, 139)  # InceptionV3 input size

    # Path to the image you want to predict
    image_path = 'sample2.jpg'  # Replace with your image path

    # Get the final prediction
    predict_with_voting(image_path, target_size)