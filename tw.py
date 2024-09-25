import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('/Users/praveenajk/farm-app/backend/.backend/models/wound_model.h5')

# Define class indices based on the dataset
class_indices = {
    0: "Abrasions",
    1: "Bruises",
    2: "Burns",
    3: "Cut",
    4: "Ingrown_nails",
    5: "Laceration",
    6: "Stab_wound"
}

# Function to preprocess the image and make predictions
def predict_wound_type(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  # Modify based on model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    # Predict the class probabilities
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])  # Get the index of the highest score
    predicted_label = class_indices[predicted_class_index]

    # Example: Return first aid instructions based on the predicted label
    first_aid_instructions = {
        "Abrasions": "Clean the wound and apply an antibiotic ointment.",
        "Bruises": "Apply a cold compress to reduce swelling.",
        "Burns": "Cool the burn with cool running water.",
        "Cut": "Clean the wound, stop the bleeding, and cover it.",
        "Ingrown_nails": "Soak in warm water and gently lift the nail.",
        "Laceration": "Apply pressure to stop bleeding and clean the wound.",
        "Stab_wound": "Seek immediate medical help and apply pressure to stop the bleeding."
    }

    # Get first aid instructions for the predicted class
    return predicted_label, first_aid_instructions[predicted_label]

# Test the prediction function
test_image_path = '/Users/praveenajk/Downloads/ab1.jpeg'
predicted_label, first_aid_instructions = predict_wound_type(test_image_path)

# Print results
print(f'Predicted class: {predicted_label}')
print(f'First aid instructions: {first_aid_instructions}')