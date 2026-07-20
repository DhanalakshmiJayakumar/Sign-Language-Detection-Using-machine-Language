import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

data_dir = "dataset"
categories = os.listdir(data_dir)

data = []
output = []
IMG_SIZE = 64

for idx, category in enumerate(categories):
    folder = os.path.join(data_dir, category)
    for img_name in os.listdir(folder)[:500]:  # limit to 500 images per letter to reduce training time
        img_path = os.path.join(folder, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            output.append(idx)
        except Exception as e:
            continue

X = np.array(data) / 255.0
y = to_categorical(output)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
model.save("model/sign_model.h5")

with open("labels.txt", "w") as f:
    for cat in categories:
        f.write(f"{cat}\n")

print("Training complete. Model saved.")
