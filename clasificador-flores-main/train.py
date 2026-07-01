import os
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from kagglehub import dataset_download

# 1. Descargar dataset desde Kaggle
dataset_path = dataset_download("alxmamaev/flowers-recognition")
print("Ruta del dataset:", dataset_path)

# El dataset descarga una carpeta "flowers" con las 5 categorías dentro
for folder in os.listdir(dataset_path):
    print(folder)

DATA_DIR = os.path.join(dataset_path, "flowers")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# 2. Dividir en entrenamiento (80%) y validación/prueba (20%)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_name = train_ds.class_names
print("Clases detectadas:", class_name)

# Vista rápida de ejemplos
plt.figure(figsize=(10, 8))
for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_name[labels[i]])
        plt.axis("off")
plt.tight_layout()
plt.savefig("ejemplos_dataset.png")

# 3. Preprocesamiento para MobileNetV2
AUTOTUNE = tf.data.AUTOTUNE
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

train_ds = train_ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)).prefetch(AUTOTUNE)

# 4. Modelo base MobileNetV2 (transfer learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(class_name), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# 5. Entrenamiento
EPOCHS = 10

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# 6. Evaluación
loss, acc = model.evaluate(val_ds, verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {acc:.4f}")

# 7. Guardar modelo y clases
OUTPUT_DIR = Path("modelo_flores_mobilenet")
OUTPUT_DIR.mkdir(exist_ok=True)

model.save(OUTPUT_DIR / "flowers_mobilenet.h5")
model.save(OUTPUT_DIR / "flowers_mobilenet.keras")

with open(OUTPUT_DIR / "class_name.json", "w", encoding="utf-8") as f:
    json.dump(class_name, f, ensure_ascii=False, indent=2)

print("Modelo guardado en:", OUTPUT_DIR / "flowers_mobilenet.h5")
print("Clases guardadas en:", OUTPUT_DIR / "class_name.json")