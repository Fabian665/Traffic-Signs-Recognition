import imghdr
import os
import cv2
import sys


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


def validate_path(path: str, file=True):
    if file:
        if not os.path.isfile(path):
            raise OSError(f'{path} is not a file, please try again')
    else:
        if not os.path.isdir(path):
            raise OSError(f'{path} is not a directory, please try again')


def get_images_from_dir(directory):
    validate_path(directory, False)
    data = []
    for image in os.listdir(directory):
        try:
            if imghdr.what(os.path.join(directory, image)) is None:
                continue
            image = cv2.imread(os.path.join(directory, image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (30, 30))
            data.append(image)
        except PermissionError:
            continue
    return data


def get_input(file=True):
    """
    gets input from user and validates it's a path
    :return:
    """
    while True:
        path = input('filename: ' if file else 'directory: ')
        try:
            validate_path(path, file)
        except OSError as error:
            print(error)
            continue
        return path


def get_path():
    if len(sys.argv) == 1:
        path = get_input()
    elif len(sys.argv) == 2:
        try:
            validate_path(sys.argv[1])
        except OSError:
            path = get_input()
        else:
            path = sys.argv[1]
    else:
        raise Exception('tsd.py takes only one parameter - path')

    return path
