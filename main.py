import os
import sys
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras


def get_model():
    """
    gets the model
    :return: keras sequential model
    """
    try:
        return_model = keras.models.load_model(os.path.join('models', 'sign_recognition.h5'))
    except OSError:
        raise OSError('Please run train_signs.py or upload sign_recognition.h5 to models folder')
    return return_model


def get_input():
    """
    gets input from user and turn in into valid path
    :return:
    """
    while True:
        filename = input('filename:')
        res, string = validate_path(filename)
        if res:
            return string
        else:
            continue


def validate_path(input_text: str) -> tuple:
    if '\\' in input_text:
        file_path = tuple(input_text.split('\\'))
    elif '/' in input_text:
        file_path = tuple(input_text.split('/'))
    else:
        file_path = input_text

    if not isinstance(file_path, str):
        im_path = os.path.join(*file_path)
    else:
        im_path = file_path

    if not os.path.isfile(im_path):
        print(f'{im_path} is not a file or the path is wrong, please try again')
        return False, None
    else:
        return True, im_path


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


if __name__ == '__main__':
    if len(sys.argv) == 0:
        path = get_input()
    else:
        res, string = validate_path(sys.argv[1])
        if res:
            path = string
        else:
            path = get_input()

    model = get_model()
    original_im, im = get_image(path)
    pred = predict(im, model)
    print(pred)
    cv2.imshow('original', original_im)
    cv2.waitKey()
