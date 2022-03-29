import imghdr
import os
import cv2


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


def validate_path(input_text: str):
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
        return None
    else:
        return im_path


def get_images_from_dir(im_path):
    if not os.path.isdir(im_path):
        raise OSError(f'{im_path} is not a valid path')
    data = []
    for image in os.listdir(im_path):
        try:
            if imghdr.what(os.path.join(im_path, image)) is None:
                continue
            image = cv2.imread(os.path.join(im_path, image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (30, 30))
            data.append(image)
        except PermissionError:
            continue
    return data


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
