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


def validate_path(input_text: str, file=True):
    if '\\' in input_text:
        path = tuple(input_text.split('\\'))
    elif '/' in input_text:
        path = tuple(input_text.split('/'))
    else:
        path = input_text

    if not isinstance(path, str):
        path = os.path.join(*path)
    else:
        path = path

    if file:
        if not os.path.isfile(path):
            raise OSError(f'{path} is not a file, please try again')
        else:
            return path
    else:
        if not os.path.isdir(path):
            raise OSError(f'{path} is not a directory, please try again')
        else:
            return path


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


def get_input(file=True):
    """
    gets input from user and validates it's a path
    :return:
    """
    while True:
        path = input('filename: ' if file else 'directory: ')
        try:
            string = validate_path(path)
        except OSError as error:
            print(error)
            continue
        return string
