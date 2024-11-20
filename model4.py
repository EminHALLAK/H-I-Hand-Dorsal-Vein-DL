import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import json  # Added import for json module

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

IMG_HEIGHT, IMG_WIDTH = 250, 250  # Görüntü boyutları
BATCH_SIZE = 32  # Parti boyutu
EPOCHS = 100  # Eğitim döngüsü sayısı
INITIAL_LEARNING_RATE = 0.005


def load_images(data_dir):
    images = []
    labels = []
    label_map = {}
    label_counter = 0

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            if class_name not in label_map:
                label_map[class_name] = label_counter
                label_counter += 1
            label = label_map[class_name]

            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img)
                labels.append(label)
    images = np.array(images) / 255.0
    labels = np.array(labels)
    return images, labels, label_map  # Modified to return label_map


def create_pairs(images, labels):
    pair_images = []
    pair_labels = []
    num_classes = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(num_classes)]

    for idx_a in range(len(images)):
        current_image = images[idx_a]
        label = labels[idx_a]
        idx_b = idx_a
        while idx_b == idx_a:
            idx_b = random.choice(idx[label])
        pos_image = images[idx_b]

        pair_images.append([current_image, pos_image])
        pair_labels.append(1)

        neg_label = random.choice([i for i in range(num_classes) if i != label])
        idx_c = random.choice(idx[neg_label])
        neg_image = images[idx_c]

        pair_images.append([current_image, neg_image])
        pair_labels.append(0)

    return np.array(pair_images), np.array(pair_labels)


def siamese_network(input_shape):
    inputs = Input(shape=input_shape)

    # İlk konvolüsyon bloğu
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # İkinci konvolüsyon bloğu
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Üçüncü konvolüsyon bloğu
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Dördüncü konvolüsyon bloğu
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Beşinci konvolüsyon bloğu
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # Tam bağlantılı katman
    outputs = layers.Dense(256, activation='relu')(x)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Dropout(0.5)(outputs)
    outputs = layers.Dense(128, activation='relu')(outputs)

    return Model(inputs, outputs)


def pair_generator(pairs, labels, batch_size, augment=False):
    """Eğitim için çift partileri oluştur."""
    num_samples = len(pairs)
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    ) if augment else None

    while True:
        # Verileri karıştır
        idx = np.random.permutation(num_samples)
        pairs_shuffled = pairs[idx]
        labels_shuffled = labels[idx]

        # Parti parti ver
        for i in range(0, num_samples, batch_size):
            batch_pairs = pairs_shuffled[i:i + batch_size]
            batch_labels = labels_shuffled[i:i + batch_size]
            x1 = np.array([img[0] for img in batch_pairs])
            x2 = np.array([img[1] for img in batch_pairs])

            if augment:
                x1 = next(datagen.flow(x1, batch_size=len(x1), shuffle=False))  # Ensure batch size matches x1 length
                x2 = next(datagen.flow(x2, batch_size=len(x2), shuffle=False))

            yield [x1, x2], batch_labels


data_dir = 'Veins_Dataset'
images, labels, label_map = load_images(data_dir)

# Veriyi böl
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)

# Çiftleri oluştur
train_pairs, train_pair_labels = create_pairs(train_images, train_labels)
val_pairs, val_pair_labels = create_pairs(val_images, val_labels)

# Modeli oluştur
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
base_model = siamese_network(input_shape)

# Siamese modeli oluştur
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_model(input_a)
processed_b = base_model(input_b)

# Uzaklık hesapla
distance = layers.Lambda(
    lambda tensors: tf.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
outputs = layers.Dense(1, activation='sigmoid')(distance)
model = Model([input_a, input_b], outputs)

# Öğrenme oranı planlaması ve model derleme
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    INITIAL_LEARNING_RATE, decay_steps=1000, decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Eğitim parametrelerini ayarla
steps_per_epoch = len(train_pairs) // BATCH_SIZE
validation_steps = len(val_pairs) // BATCH_SIZE

# Geri çağırmalar
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True
)


# Modeli eğit
history = model.fit(
    pair_generator(train_pairs, train_pair_labels, BATCH_SIZE, augment=True),
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=pair_generator(val_pairs, val_pair_labels, BATCH_SIZE),
    validation_steps=validation_steps,
    callbacks=[checkpoint]  # Include early_stopping if desired
)

# Modeli değerlendir
loss, acc = model.evaluate(
    pair_generator(val_pairs, val_pair_labels, BATCH_SIZE),
    steps=validation_steps
)
print(f"Validation Accuracy: {acc * 100:.2f}%")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc="upper left")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()

with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

# Etiket eşleştirmelerini kaydet
with open('label_map.json', 'w') as f:
    json.dump(label_map, f)
