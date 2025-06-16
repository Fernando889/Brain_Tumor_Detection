import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image, image_dataset_from_directory
from keras.layers import (Rescaling, RandomFlip, RandomZoom,
                          RandomRotation, Conv2D, MaxPooling2D, Flatten, Dense)
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay

train_dataset = image_dataset_from_directory(
    'dataset/brain_tumor_train_dataset',
    shuffle=True,
    image_size=(64, 64),
    batch_size=32
)

test_dataset = image_dataset_from_directory(
    'dataset/brain_tumor_test_dataset',
    shuffle=True,
    image_size=(64, 64),
    batch_size=32
)

class_names = train_dataset.class_names

normalize_layer = Rescaling(1./255)
data_augmentation = tf.keras.Sequential([
    RandomFlip('horizontal'),
    RandomZoom(0.2),
    RandomRotation(0.1),
])

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.map(
    lambda x, y: (normalize_layer(x), y)
).map(
    lambda x, y: (data_augmentation(x), y)
).prefetch(AUTOTUNE)

test_dataset = test_dataset.map(
    lambda x, y: (normalize_layer(x), y)
).prefetch(AUTOTUNE)

cnn = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=2, strides=2),
    Conv2D(32, 2, activation='relu'),
    MaxPooling2D(pool_size=2, strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(train_dataset, validation_data=test_dataset, epochs=100)

prediction_folder = 'dataset/prediction'

image_files = sorted([
    os.path.join(prediction_folder, fname)
    for fname in os.listdir(prediction_folder)
    if fname.lower().endswith(('.png', '.jpeg', '.jpg'))
])

images = []
for img_paths in image_files:
    img = image.load_img(img_paths, target_size=(64, 64))
    img_array = image.img_to_array(img)
    images.append(img_array)

images_array = np.array(images)/255
results = cnn.predict(images_array)

for img_path, results in zip(image_files, results):
    if results[0] > 0.5:
        prediction = class_names[1]
    else:
        prediction = class_names[0]
    print(f"{os.path.basename(img_path)} --> {prediction}")


y_test = []
y_pred = []

for images, labels in test_dataset:
    preds = cnn.predict(images)
    preds = (preds > 0.5).astype('int32')
    y_test.extend(labels.numpy())
    y_pred.extend(preds.flatten())

cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
reportDataFrame = pd.DataFrame(report).transpose()
print("Confusion Matrix : \n", cm)
print(f"Accuracy Score = {round(score * 100)} %")
print("Classification Report : \n", reportDataFrame)

cmDisplay = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=class_names)
cmDisplay.plot(cmap=plt.cm.Blues)
plt.title('Brain Tumor Detection - Confusion Matrix')
plt.show()
