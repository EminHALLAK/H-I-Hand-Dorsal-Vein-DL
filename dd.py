import os
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    Concatenate,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Number of classes (update with your actual number)
NUM_CLASSES = 60  # Replace with your actual number of classes

# Preprocessing function
def prepare_data(dataset_dir, img_size=(256, 256), batch_size =10, validation_split=0.2):
    """
    Prepares the data generators for training and validation.

    Args:
        dataset_dir (str): Path to the dataset directory.
        img_size (tuple): Target image size.
        batch_size (int): Batch size for data generators.
        validation_split (float): Fraction of data to use for validation.

    Returns:
        train_data: Training data generator.
        val_data: Validation data generator.
    """
    data_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split,
    )

    train_data = data_gen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        color_mode="rgb",
        class_mode="categorical",
        subset="training",
        batch_size=batch_size,
        shuffle=True,
    )

    val_data = data_gen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        color_mode="rgb",
        class_mode="categorical",
        subset="validation",
        batch_size=batch_size,
        shuffle=True,
    )

    return train_data, val_data

# Modified U-Net model for classification
def unet_with_pretrained_encoder(input_size=(256, 256, 3), num_classes=NUM_CLASSES):
    # Load pre-trained VGG16 model + higher level layers
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_size)

    # Freeze the layers except the last block
    for layer in vgg.layers[:-4]:
        layer.trainable = False

    inputs = vgg.input

    # Use VGG16 layers as the encoder
    c1 = vgg.get_layer('block1_conv2').output  # 256x256
    c2 = vgg.get_layer('block2_conv2').output  # 128x128
    c3 = vgg.get_layer('block3_conv3').output  # 64x64
    c4 = vgg.get_layer('block4_conv3').output  # 32x32

    # Bottleneck
    c5 = vgg.get_layer('block5_conv3').output  # 16x16

    # Decoding path
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(512, (3, 3), activation="relu", padding="same")(u6)
    c6 = Conv2D(512, (3, 3), activation="relu", padding="same")(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(256, (3, 3), activation="relu", padding="same")(u7)
    c7 = Conv2D(256, (3, 3), activation="relu", padding="same")(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(128, (3, 3), activation="relu", padding="same")(u8)
    c8 = Conv2D(128, (3, 3), activation="relu", padding="same")(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(64, (3, 3), activation="relu", padding="same")(u9)
    c9 = Conv2D(64, (3, 3), activation="relu", padding="same")(c9)

    # Output layer for classification
    gap = GlobalAveragePooling2D()(c9)
    outputs = Dense(num_classes, activation="softmax")(gap)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# ResNet model for classification
def resnet_classification_model(input_shape=(256, 256, 3), num_classes=NUM_CLASSES):
    base_model = ResNet50(
        weights='imagenet', include_top=False, input_shape=input_shape
    )
    base_model.trainable = False  # Freeze base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Main function to run the training pipeline
def main():
    # Prepare data
    dataset_dir = "Veins_Dataset"  # Replace with your dataset directory
    img_size = (256, 256)
    batch_size =10

    train_data, val_data = prepare_data(
        dataset_dir, img_size=img_size, batch_size=batch_size, validation_split=0.2
    )

    # Display a sample preprocessed image
    images, labels = next(train_data)
    plt.figure(figsize=(6, 6))
    plt.imshow(images[0])
    plt.title("Sample Preprocessed Image")
    plt.axis('off')
    plt.show()

    # Prepare U-Net classification model
    unet = unet_with_pretrained_encoder(input_size=(256, 256, 3))
    unet.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Prepare ResNet model
    resnet = resnet_classification_model(input_shape=(256, 256, 3))
    resnet.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Implement Early Stopping and Checkpoint
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=False,
    )

    checkpoint_unet = ModelCheckpoint(
        'best_unet_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
    )

    checkpoint_resnet = ModelCheckpoint(
        'best_resnet_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
    )

    # Train U-Net classification model
    print("Training U-Net for classification...")
    history_unet = unet.fit(
        train_data,
        epochs=100,
        validation_data=val_data,
        callbacks=[early_stopping, checkpoint_unet],
    )

    # Train ResNet model
    print("Training ResNet for classification...")
    history_resnet = resnet.fit(
        train_data,
        epochs=100,
        validation_data=val_data,
        callbacks=[early_stopping, checkpoint_resnet],
    )

    # Plot accuracy over epochs for U-Net
    plt.figure()
    plt.plot(
        history_unet.history["accuracy"], label="U-Net Train Accuracy"
    )
    plt.plot(
        history_unet.history["val_accuracy"],
        label="U-Net Validation Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("U-Net Classification Accuracy")
    plt.show()

    # Plot accuracy over epochs for ResNet
    plt.figure()
    plt.plot(
        history_resnet.history["accuracy"], label="ResNet Train Accuracy"
    )
    plt.plot(
        history_resnet.history["val_accuracy"],
        label="ResNet Validation Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("ResNet Classification Accuracy")
    plt.show()

    # Display final accuracy table
    final_accuracy = {
        "U-Net Training Accuracy": history_unet.history["accuracy"][-1],
        "U-Net Validation Accuracy": history_unet.history["val_accuracy"][-1],
        "ResNet Training Accuracy": history_resnet.history["accuracy"][-1],
        "ResNet Validation Accuracy": history_resnet.history["val_accuracy"][-1],
    }
    print("Final Model Accuracies:")
    for key, value in final_accuracy.items():
        print(f"{key}: {value:.4f}")

# Run the main function
if __name__ == "__main__":
    main()
