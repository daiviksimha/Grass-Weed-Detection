import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import cv2  # Import OpenCV

# Load the saved model
model = tf.keras.models.load_model('pretrain_model.keras')

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

# Function to generate heatmap
def generate_heatmap(model, img_path, target_size, class_labels):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = tf.image.resize(img_array, (139, 139))
    img_array = tf.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    last_conv_layer = model.get_layer('mixed10')
    heatmap_model = tf.keras.models.Model(model.inputs, [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = heatmap_model(img_array)
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))

    img_array_uint8 = (img_array[0].numpy() * 255).astype(np.uint8)
    heatmap_resized_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_resized_uint8 = cv2.applyColorMap(heatmap_resized_uint8, cv2.COLORMAP_RAINBOW)
    superimposed_img = cv2.addWeighted(img_array_uint8, 0.6, heatmap_resized_uint8, 0.4, 0)

    predicted_class_label = class_labels[predicted_class]
    return predictions, heatmap_resized_uint8, superimposed_img, predicted_class_label