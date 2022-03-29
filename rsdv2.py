import os
import sys
import cv2
import imghdr
import numpy as np
from imageai.Detection import ObjectDetection

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RoadSignsDetection:
    def __new__(cls):
        if os.path.isfile(os.path.join("models", "resnet50_coco_best_v2.1.0.h5")):
            return super().__new__(cls)
        else:
            raise OSError('Please upload "resnet50_coco_best_v2.1.0.h5" to models directory')

    def __init__(self):
        self.model = self._get_model()
        self.custom = self.model.CustomObjects(stop_sign=True)
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
        :return: ObjectDetection
        """
        model_path = os.path.join("models", "resnet50_coco_best_v2.1.0.h5")
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(model_path)
        detector.loadModel(detection_speed='fastest')

        return detector

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

    @staticmethod
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

    @staticmethod
    def get_image(im_path: str):
        """
        Get image from path
        :param im_path: str - Relative or Absolute path
        :return: tuple of image and image resized
        """
        image = cv2.imread(im_path)
        original_image = image
        original_image = cv2.resize(original_image, (150, 150))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (30, 30))
        return original_image, image

    def predict(self, im_path):
        self.validate_path(im_path)
        im_type = imghdr.what(im_path)
        if im_type is None:
            raise ValueError(f'{im_path} is not a valid image')
        detections = self.model.detectObjectsFromImage(
            custom_objects=self.custom,
            input_image=f'{im_path}.{im_type}',
            output_image_path=f'{im_path}.resnet.{im_type}',
            minimum_percentage_probability=50
        )
        if len(detections) == 0:
            return False, None
        else:
            image = cv2.imread(f'{im_path}.resnet.{im_type}')
            xmin, ymin, xmax, ymax = detections[0]['box_points']
            image = image[ymin:ymax, xmin:xmax]
            cv2.imshow('original', image)
            return True, (image, (ymin, ymax, xmin, xmax))




if __name__ == '__main__':
    app = RoadSignsDetection()
    if len(sys.argv) == 0:
        path = app.get_input()
    else:
        res, string = app.validate_path(sys.argv[1])
        if res:
            path = string
        else:
            path = app.get_input()

    original_im, im = app.get_image(path)
    pred = app.predict(im)
    print(pred)
    cv2.imshow('original', original_im)
    cv2.waitKey()
