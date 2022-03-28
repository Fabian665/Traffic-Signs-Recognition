import os
import cv2
import numpy as np
from tensorflow import keras

try:
    model = keras.models.load_model(os.path.join('models', 'sign_recognition.h5'))
except OSError:
    raise OSError('Please run train_signs.py or upload sign_recognition.h5 to models folder')

filename = input('filename:')

image = cv2.imread(os.path.join('data', 'Test', filename))
original_image = image
original_image = cv2.resize(original_image, (75, 75))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (30, 30))

X_test = np.array([image])
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print(y_pred)
cv2.imshow('original', original_image)
cv2.waitKey()
