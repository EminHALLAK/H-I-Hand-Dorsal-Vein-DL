import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import cv2  # OpenCV for image processing
import matplotlib.pyplot as plt

# Global variable for number of classes
NUM_CLASSES = 60

def preprocess_image(image):
  """
  Preprocesses the input image by applying contrast enhancement,
  CLAHE, and Gaussian blur.

  Args:
      image: Input image in numpy array format.

  Returns:
      preprocessed_image: The preprocessed image.
  """
  # Convert to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

  # Contrast enhancement
  alpha = 1.5  # Simple contrast control
  beta = 0     # Simple brightness control
  adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

  # Apply CLAHE
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  clahe_image = clahe.apply(adjusted)

  # Apply Gaussian Blur
  blurred = cv2.GaussianBlur(clahe_image, (5, 5), 0)

  # Convert back to RGB
  preprocessed_image = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)

  return preprocessed_image

def create_data_generators(dataset_dir, img_size=(300, 300), batch_size=12, validation_split=0.2):
  """
  Creates training and validation data generators with preprocessing and augmentation.

  Args:
      dataset_dir (str): Path to the dataset directory.
      img_size (tuple): Target image size.
      batch_size (int): Batch size.
      validation_split (float): Fraction of data to use for validation.

  Returns:
      train_generator, validation_generator: The data generators.
  """

  # Custom preprocessing function for ImageDataGenerator
  def preprocessing_function(image):
      image = img_to_array(image)
      image = preprocess_image(image)
      # Convert to float32
      image = image.astype('float32')
      return image

  # Data augmentation parameters
  train_datagen = ImageDataGenerator(
      preprocessing_function=preprocessing_function,
      rescale=1.0 / 255.0,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      brightness_range=[0.8, 1.2],
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest',
      validation_split=validation_split,
  )

  # Validation data generator should not have data augmentation
  validation_datagen = ImageDataGenerator(
      preprocessing_function=preprocessing_function,
      rescale=1.0 / 255.0,
      validation_split=validation_split,
  )

  train_generator = train_datagen.flow_from_directory(
      dataset_dir,
      target_size=img_size,
      batch_size=batch_size,
      class_mode='categorical',
      subset='training',
      shuffle=True,
  )

  validation_generator = validation_datagen.flow_from_directory(
      dataset_dir,
      target_size=img_size,
      batch_size=batch_size,
      class_mode='categorical',
      subset='validation',
      shuffle=False,
  )

  return train_generator, validation_generator

def create_model(input_shape=(300, 300, 3), num_classes=NUM_CLASSES):
  """
  Creates and compiles the model using EfficientNetB3.

  Args:
      input_shape (tuple): Input image shape.
      num_classes (int): Number of classes.

  Returns:
      model: The compiled model.
  """
  # Load EfficientNetB3 with pre-trained weights
  base_model = EfficientNetB3(
      weights='imagenet',
      include_top=False,
      input_shape=input_shape
  )

  # Freeze the base model layers
  base_model.trainable = False

  # Create the model
  inputs = Input(shape=input_shape)
  x = base_model(inputs, training=False)
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.5)(x)
  outputs = Dense(num_classes, activation='softmax')(x)

  model = Model(inputs, outputs)

  # Compile the model
  model.compile(
      optimizer=Adam(learning_rate=1e-4),
      loss='categorical_crossentropy',
      metrics=['accuracy'],
  )

  return model

def train_model(model, train_generator, validation_generator, epochs=100):
  """
  Trains the model with the given data generators.

  Args:
      model: The compiled model.
      train_generator: Training data generator.
      validation_generator: Validation data generator.
      epochs (int): Number of epochs.

  Returns:
      history: The training history.
  """
  # Callbacks
  early_stopping = EarlyStopping(
      monitor='val_loss',
      patience=10,
      restore_best_weights=True,
  )

  reduce_lr = ReduceLROnPlateau(
      monitor='val_loss',
      factor=0.2,
      patience=5,
      min_lr=1e-6,
  )

  checkpoint = ModelCheckpoint(
      'best_model.h5',
      monitor='val_loss',
      save_best_only=True,
      mode='min',
  )

  # Train the model
  history = model.fit(
      train_generator,
      epochs=epochs,
      validation_data=validation_generator,
      callbacks=[early_stopping, reduce_lr, checkpoint],
  )

  return history

def plot_training_history(history):
  # Plot training & validation accuracy values
  plt.figure(figsize=(12, 4))

  plt.subplot(1, 2, 1)
  plt.plot(history.history['accuracy'], label='Train')
  plt.plot(history.history['val_accuracy'], label='Validation')
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(loc='lower right')

  # Plot training & validation loss values
  plt.subplot(1, 2, 2)
  plt.plot(history.history['loss'], label='Train')
  plt.plot(history.history['val_loss'], label='Validation')
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(loc='upper right')

  plt.show()

def main():
  # Define parameters
  dataset_dir = 'Veins_Dataset'  # Replace with your dataset path
  img_size = (300, 300)    # EfficientNetB3 default size is 300x300
  batch_size = 12
  epochs = 100

  # Determine number of classes
  try:
      num_classes = len([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
  except FileNotFoundError:
      print(f"Error: The directory {dataset_dir} does not exist.")
      return

  global NUM_CLASSES
  NUM_CLASSES = num_classes

  # Create data generators
  train_generator, validation_generator = create_data_generators(
      dataset_dir,
      img_size=img_size,
      batch_size=batch_size,
  )

  # Display a sample preprocessed image
  images, labels = next(train_generator)
  plt.figure(figsize=(6, 6))
  plt.imshow(images[0])
  plt.title("Sample Preprocessed Image")
  plt.axis('off')
  plt.show()

  # Create and train the model
  model = create_model(input_shape=img_size + (3,), num_classes=num_classes)
  history = train_model(model, train_generator, validation_generator, epochs=epochs)

  # Plot training history
  plot_training_history(history)

  # Evaluate the model
  val_loss, val_accuracy = model.evaluate(validation_generator)
  print(f'Validation Loss: {val_loss:.4f}')
  print(f'Validation Accuracy: {val_accuracy:.4f}')

if __name__ == '__main__':
  main()