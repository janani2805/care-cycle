import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image                                                                                                                                                                            model = load_model('your_model_path.h5')  # Replace with your model's path
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Adjust size if needed
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img                                                                                                                                                                                                                                                      def predict_disease(image_path):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)
    disease_labels = ['disease1', 'disease2', ...]  # Replace with your labels
    predicted_disease = disease_labels[predicted_class]
    return predicted_disease                                                                                                                                                                                                                   image_path = input("Enter the path to the image file: ")
predicted_disease = predict_disease(image_path)
print("Predicted disease:", predicted_disease)