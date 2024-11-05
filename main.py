# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.optimizers import Adam
#
#
# # Load and preprocess dataset
# def load_and_preprocess_images(image_path, target_size=(256, 256)):
#     img = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=target_size)
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = img_array / 255.0  # Normalize
#     return img_array
#
#
# # Define U-Net model
# def unet_model(input_size=(256, 256, 1)):
#     inputs = Input(input_size)
#
#     # Encoding path
#     c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
#     c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
#     p1 = MaxPooling2D((2, 2))(c1)
#
#     c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
#     c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
#     p2 = MaxPooling2D((2, 2))(c2)
#
#     # Bottleneck
#     c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
#     c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
#
#     # Decoding path
#     u6 = UpSampling2D((2, 2))(c5)
#     u6 = Concatenate()([u6, c2])
#     c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
#     c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
#
#     u7 = UpSampling2D((2, 2))(c6)
#     u7 = Concatenate()([u7, c1])
#     c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
#     c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
#
#     outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
#     model = Model(inputs=[inputs], outputs=[outputs])
#
#     return model
#
#
# # Define ResNet-based classification model
# def resnet_classification_model(input_shape=(256, 256, 1)):
#     base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
#     x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
#     x = Dropout(0.5)(x)
#     output = tf.keras.layers.Dense(60, activation='softmax')(x)  # Assume 60 classes for 60 hands
#     model = Model(inputs=base_model.input, outputs=output)
#     return model
#
#
# # Compile and train U-Net
# unet = unet_model()
# unet.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
#
# # Compile and train ResNet
# resnet = resnet_classification_model()
# resnet.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Load, augment and split data for training
# # Assuming images are stored in folders by class, e.g., "hand_1", "hand_2", etc.
# data_gen = ImageDataGenerator(rescale=1.0 / 255.0, rotation_range=15, width_shift_range=0.1,
#                               height_shift_range=0.1, horizontal_flip=True, validation_split=0.2)
#
# train_data = data_gen.flow_from_directory('path_to_data', target_size=(256, 256), color_mode='grayscale',
#                                           class_mode='categorical', subset='training', batch_size=8)
# val_data = data_gen.flow_from_directory('path_to_data', target_size=(256, 256), color_mode='grayscale',
#                                         class_mode='categorical', subset='validation', batch_size=8)
#
# # Train U-Net
# history_unet = unet.fit(train_data, epochs=10, validation_data=val_data)
#
#
# # Use U-Net to generate segmented images
# def generate_segmented_images(unet_model, data_gen):
#     segmented_images = []
#     labels = []
#     for img_batch, label_batch in data_gen:
#         segmented_batch = unet_model.predict(img_batch)
#         segmented_images.extend(segmented_batch)
#         labels.extend(label_batch)
#     return np.array(segmented_images), np.array(labels)
#
#
# segmented_train_images, train_labels = generate_segmented_images(unet, train_data)
# segmented_val_images, val_labels = generate_segmented_images(unet, val_data)
#
# # Train ResNet on segmented images
# history_resnet = resnet.fit(segmented_train_images, train_labels, epochs=10,
#                             validation_data=(segmented_val_images, val_labels))
#
# # Plot accuracy over epochs for ResNet
# plt.plot(history_resnet.history['accuracy'], label='Train Accuracy')
# plt.plot(history_resnet.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
#
# # Display final accuracy table
# final_accuracy = {
#     'Training Accuracy': history_resnet.history['accuracy'][-1],
#     'Validation Accuracy': history_resnet.history['val_accuracy'][-1]
# }
# print(final_accuracy)
