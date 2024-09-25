import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set the path to your dataset
data_dir = '/Users/praveenajk/Downloads/Wound_dataset'  # Replace with the actual path to your dataset

# Preprocessing and augmenting data
data_gen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,  # Use 20% of the data for validation
    rotation_range=20,      # Augmentation: rotate images by 20 degrees
    width_shift_range=0.2,  # Augmentation: shift width
    height_shift_range=0.2, # Augmentation: shift height
    shear_range=0.2,        # Augmentation: shear the images
    zoom_range=0.2,         # Augmentation: zoom into the image
    horizontal_flip=True    # Augmentation: flip images horizontally
)

# Create training and validation data generators
train_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  # Use 224x224 for pre-trained models
    batch_size=32,
    class_mode='categorical',  # Use categorical for multi-class classification
    subset='training',  # Set as training data
)

validation_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  # Use 224x224 for pre-trained models
    batch_size=32,
    class_mode='categorical',  # Use categorical for multi-class classification
    subset='validation',  # Set as validation data
)

# Determine number of classes
num_classes = len(train_generator.class_indices)

# Load the pre-trained VGG16 model (excluding top dense layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model (we won't train these layers)
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the pre-trained base
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),  # Dense layer with 256 units
    layers.Dropout(0.5),  # Dropout layer to reduce overfitting
    layers.Dense(128, activation='relu'),  # Dense layer with 128 units
    layers.Dropout(0.5),  # Another dropout layer
    layers.Dense(num_classes, activation='softmax')  # Final output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and reducing learning rate if accuracy plateaus
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,  # Adjust the number of epochs as needed
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Save the model
model.save('./backend/models/wound_model.h5')  # Save the trained model
print('Model trained and saved successfully.')