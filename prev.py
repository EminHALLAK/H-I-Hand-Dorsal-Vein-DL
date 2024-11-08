import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    Dropout,
    GlobalAveragePooling2D,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam


# Modified U-Net model for classification
def unet_classification_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Encoding path
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c5 = Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    c5 = Conv2D(256, (3, 3), activation="relu", padding="same")(c5)

    # Decoding path
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c2])
    c6 = Conv2D(128, (3, 3), activation="relu", padding="same")(u6)
    c6 = Conv2D(128, (3, 3), activation="relu", padding="same")(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c1])
    c7 = Conv2D(64, (3, 3), activation="relu", padding="same")(u7)
    c7 = Conv2D(64, (3, 3), activation="relu", padding="same")(c7)

    # Global Average Pooling and Dense layer for classification
    gap = GlobalAveragePooling2D()(c7)
    outputs = Dense(60, activation="softmax")(gap)  # For 60 classes

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# ResNet model for classification
def resnet_classification_model(input_shape=(256, 256, 1)):
    base_model = ResNet50(
        weights=None, include_top=False, input_shape=input_shape
    )
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    output = Dense(60, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model


# Main function to run the training pipeline
def main():
    # Prepare U-Net classification model
    unet = unet_classification_model()
    unet.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Prepare ResNet model
    resnet = resnet_classification_model()
    resnet.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Data generators with augmentation
    data_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_data = data_gen.flow_from_directory(
        "Dataset",
        target_size=(256, 256),
        color_mode="grayscale",
        class_mode="categorical",
        subset="training",
        batch_size=16,
    )
    val_data = data_gen.flow_from_directory(
        "Dataset",
        target_size=(256, 256),
        color_mode="grayscale",
        class_mode="categorical",
        subset="validation",
        batch_size=16,
    )

    # Train U-Net classification model
    print("Training U-Net for classification...")
    history_unet = unet.fit(
        train_data, epochs=100, validation_data=val_data
    )

    # Train ResNet on the same data
    print("Training ResNet for classification...")
    history_resnet = resnet.fit(
        train_data, epochs=100, validation_data=val_data
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
        "U-Net Validation Accuracy": history_unet.history["val_accuracy"][
            -1
        ],
        "ResNet Training Accuracy": history_resnet.history["accuracy"][-1],
        "ResNet Validation Accuracy": history_resnet.history[
            "val_accuracy"
        ][-1],
    }
    print("Final Model Accuracies:")
    for key, value in final_accuracy.items():
        print(f"{key}: {value:.4f}")


# Run the main function
if __name__ == "__main__":
    main()
