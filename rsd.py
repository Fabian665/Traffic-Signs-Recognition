import os
import sys
import cv2
import imghdr
import utils
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras


class RoadSignsDetection:
    def __init__(self):
        self.model = self._get_model()
        self.classes_dict = {
            1: 'Speed limit (20km/h)',
            2: 'Speed limit (30km/h)',
            3: 'Speed limit (50km/h)',
            4: 'Speed limit (60km/h)',
            5: 'Speed limit (70km/h)',
            6: 'Speed limit (80km/h)',
            7: 'End of speed limit (80km/h)',
            8: 'Speed limit (100km/h)',
            9: 'Speed limit (120km/h)',
            10: 'No passing',
            11: 'No passing veh over 3.5 tons',
            12: 'Right-of-way at intersection',
            13: 'Priority road',
            14: 'Yield',
            15: 'Stop',
            16: 'No vehicles',
            17: 'Veh > 3.5 tons prohibited',
            18: 'No entry',
            19: 'General caution',
            20: 'Dangerous curve left',
            21: 'Dangerous curve right',
            22: 'Double curve',
            23: 'Bumpy road',
            24: 'Slippery road',
            25: 'Road narrows on the right',
            26: 'Road work',
            27: 'Traffic signals',
            28: 'Pedestrians',
            29: 'Children crossing',
            30: 'Bicycles crossing',
            31: 'Beware of ice/snow',
            32: 'Wild animals crossing',
            33: 'End speed + passing limits',
            34: 'Turn right ahead',
            35: 'Turn left ahead',
            36: 'Ahead only',
            37: 'Go straight or right',
            38: 'Go straight or left',
            39: 'Keep right',
            40: 'Keep left',
            41: 'Roundabout mandatory',
            42: 'End of no passing',
            43: 'End no passing veh > 3.5 tons'
        }
        self.interpret_classes = lambda x: self.classes_dict[x + 1]

    @staticmethod
    def _get_model():
        """
        gets the model
        :return: keras sequential model
        """
        try:
            return_model = keras.models.load_model(os.path.join('models', 'sign_recognition.h5'))
        except OSError:
            raise OSError('Please run train.py or upload sign_recognition.h5 to models folder')
        return return_model

    def get_input(self):
        """
        gets input from user and turn in into valid path
        :return:
        """
        while True:
            filename = input('filename:')
            res, string = self.validate_path(filename)
            if res:
                return string
            else:
                continue

    def predict_one(self, image: np.ndarray):
        image = np.array([image])
        prediction = self.model.predict(image)
        prediction = np.argmax(prediction, axis=1)
        return self.classes_dict[prediction[0] + 1]

    def predict_list(self, images_list: list):
        images = np.array(images_list)
        predictions = self.model.predict(images)
        predictions = np.argmax(predictions, axis=1)
        predictions = list(map(self.interpret_classes, predictions))
        return predictions


if __name__ == '__main__':
    app = RoadSignsDetection()
    if len(sys.argv) == 1:
        path = app.get_input()
    else:
        res, string = utils.validate_path(sys.argv[1])
        if res:
            path = string
        else:
            path = app.get_input()

    original_im, im = utils.get_image(path)
    pred = app.predict_one(im)
    print(pred)
    cv2.imshow('original', original_im)
    cv2.waitKey()
