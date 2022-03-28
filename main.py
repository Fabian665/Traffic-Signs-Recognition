import os
import cv2
import numpy as np
from tensorflow import keras


def get_model():
    try:
        model = keras.models.load_model(os.path.join('models', 'sign_recognition.h5'))
    except OSError:
        raise OSError('Please run train_signs.py or upload sign_recognition.h5 to models folder')
    return model


def get_input():
    filename = input('filename:')
    if '\\' in filename:
        tup = tuple(filename.split('\\'))
    elif '/' in filename:
        tup = tuple(filename.split('/'))
    else:
        return filename
    return os.path.join(*tup)


def get_image(im_path: str):
    """
    Get image from path
    :param im_path: str - Relative or Absolute path
    :return: tuple of image and image resized
    """
    image = cv2.imread(im_path)
    original_image = image
    original_image = cv2.resize(original_image, (75, 75))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (30, 30))
    return original_image, image


def predict(image: np.ndarray, model):
    image = np.array([image])
    prediction = model.predict(image)
    prediction = np.argmax(prediction, axis=1)
    return prediction


model = get_model()
path = get_input()
original_image, image = get_image(path)
pred = predict(image, model)
print(pred)
cv2.imshow('original', original_image)
cv2.waitKey()
