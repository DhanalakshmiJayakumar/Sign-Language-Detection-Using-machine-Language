import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/sign_model.h5")

with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

IMG_SIZE = 64
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[100:300, 100:300]
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0) / 255.0

    prediction = model.predict(img, verbose=0)
    class_index = np.argmax(prediction)
    class_label = labels[class_index]

    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.putText(frame, f"Prediction: {class_label}", (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
