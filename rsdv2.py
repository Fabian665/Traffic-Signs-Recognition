import os
import cv2
import imghdr
from utils import validate_path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from imageai.Detection import ObjectDetection


class RoadSignsDetection:
    def __new__(cls):
        if os.path.isfile(os.path.join("models", "resnet50_coco_best_v2.1.0.h5")):
            return super().__new__(cls)
        else:
            raise OSError('Please upload "resnet50_coco_best_v2.1.0.h5" to models directory')

    def __init__(self):
        self.model = self._get_model()
        self.custom = self.model.CustomObjects(stop_sign=True)

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

    def predict(self, im_path):
        validate_path(im_path)
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


    def detect(self, im_path):
        raise NotImplementedError


if __name__ == '__main__':
    from utils import get_path, get_image
    app = RoadSignsDetection()
    path = get_path()
    original_im, im = get_image(path)
    pred = app.predict(im)
    print(pred)
    cv2.imshow('original', original_im)
    cv2.waitKey()
