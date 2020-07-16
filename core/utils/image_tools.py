import cv2
import os
import pickle
import numpy as np


def imwrite(filename, image):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except:
            pass
    cv2.imwrite(filename, image)

def npsave(filename, data):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except:
            pass
    np.save(filename, data)

def pkldump(filename, data):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except:
            pass
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def imhstack(images, height=None):
    images = as_list(images)
    images = list(map(image2C3, images))

    if height is None:
        height = np.array([img.shape[0] for img in images]).max()
    images = [resize_height(img, height) for img in images]

    if len(images) == 1:
        return images[0]

    images = [[img, np.full((height, 3, 3), 255, np.uint8)] for img in images]
    images = np.hstack(sum(images, []))
    return images

def imvstack(images, width=None):
    images = as_list(images)
    images = list(map(image2C3, images))

    if width is None:
        width = np.array([img.shape[1] for img in images]).max()
    images = [resize_width(img, width) for img in images]

    if len(images) == 1:
        return images[0]

    images = [[img, np.full((3, width, 3), 255, np.uint8)] for img in images]
    images = np.vstack(sum(images, []))
    return images

def as_list(data):
    if not isinstance(data, (list, tuple)):
        return [data]
    return list(data)

def image2C3(image):
    if image.ndim == 3:
        return image
    if image.ndim == 2:
        return np.repeat(image[..., np.newaxis], 3, axis=2)
    raise ValueError("image.ndim = {}, invalid image.".format(image.ndim))

def resize_height(image, height):
    if image.shape[0] == height:
        return image
    h, w = image.shape[:2]
    width = height * w // h
    image = cv2.resize(image, (width, height))
    return image

def resize_width(image, width):
    if image.shape[1] == width:
        return image
    h, w = image.shape[:2]
    height = width * h // w
    image = cv2.resize(image, (width, height))
    return image

def imtext(image, text, space=(3, 3), color=(0, 0, 0), thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.):
    assert isinstance(text, str), type(text)
    size = cv2.getTextSize(text, fontFace, fontScale, thickness)
    image = cv2.putText(image, text, (space[0], size[1]+space[1]), fontFace, fontScale, color, thickness)
    return image

