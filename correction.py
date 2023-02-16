'''
Модуль коррекции изображения
'''

import cv2

from project_typing import Image


def undistort(img: Image, object_points: list, image_points: list):
    '''
    Возвращает изображение без искажений объектива
    img: изображение или путь к нему
    object_points и image_points: параметры калибровки
    '''
    if isinstance(img, str):
        img = cv2.imread(img)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points,
                                                       img.shape[1:], None, None)
    return cv2.undistort(img, mtx, dist, None, mtx)

def remove_noise(img: Image, sturct_size: int) -> Image:
    '''
    Использует операции close и open для убирания шума
    img: изображение
    struct_size: размер структуры (большее значение = меньше шума и большее закругление)
    '''
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(sturct_size,sturct_size))
    img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, structure)
    img_smooth = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, structure)
    return img_smooth
