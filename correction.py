'''
Коррекция изображений
'''

import cv2


def undistort(img, mtx, dist):
    '''
    Возвращает изображение без искажений объектива
    img: изображение или путь к нему
    mtx и dist: параметры калибровки
    '''
    if isinstance(img, str):
        img = cv2.imread(img)
    return cv2.undistort(img, mtx, dist, None, mtx)

def remove_noise(img, sturct_size: int):
    '''
    Использует операции close и open для убирания шума
    img: изображение
    struct_size: размер структуры (большее значение = меньше шума и большее закругление)
    '''
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(sturct_size,sturct_size))
    img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, structure)
    img_smooth = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, structure)
    return img_smooth
