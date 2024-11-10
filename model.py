import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import cv2
import matplotlib.pyplot as plt

# Global variable for number of classes
NUM_CLASSES = 60

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    alpha = 1.5  # Contrast control
    beta = 0     # Brightness control
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(adjusted)
    blurred = cv2.GaussianBlur(clahe_image, (5, 5), 0)
    preprocessed_image = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
    return preprocessed_image

def create_data_generators(dataset_dir, img_size=(300, 300), batch_size=12, validation_split=0.2):
    def preprocessing_function(image):
        image = img_to_array(image)
        image = preprocess_image(image)
        image = image.astype('float32')
        return image

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rotation_range=50,
        width_shift_range=0.3,
        height_shift_range=0.3,
        brightness_range=[0.7, 1.3],
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=validation_split,
    )

    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
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
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model

def fine_tune_model(model, base_model, num_unfreeze=20):
    for layer in base_model.layers[-num_unfreeze:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

def train_model(model, train_generator, validation_generator, epochs=20):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr, checkpoint],
    )

    return history

def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.show()

def main():
    dataset_dir = 'Veins_Dataset'
    img_size = (300, 300)
    batch_size = 12
    initial_epochs = 50
    fine_tune_epochs = 50

    num_classes = len([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    global NUM_CLASSES
    NUM_CLASSES = num_classes

    train_generator, validation_generator = create_data_generators(
        dataset_dir,
        img_size=img_size,
        batch_size=batch_size,
    )

    model = create_model(input_shape=img_size + (3,), num_classes=num_classes)
    history = train_model(model, train_generator, validation_generator, epochs=initial_epochs)

    base_model = model.layers[1]  # EfficientNetB3 base model
    model = fine_tune_model(model, base_model, num_unfreeze=20)
    fine_tune_history = train_model(model, train_generator, validation_generator, epochs=fine_tune_epochs)

    plot_training_history(history)
    plot_training_history(fine_tune_history)

    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}')

if __name__ == '__main__':
    main()
