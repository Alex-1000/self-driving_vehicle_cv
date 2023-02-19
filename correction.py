'''
Коррекция изображений
'''

import cv2 as cv


def undistort(img, mtx, dist, new_mtx, roi):
    '''
    Возвращает изображение без искажений объектива
    img: путь к изображению
    mtx, dist и new_mtx: параметры калибровки
    roi: область интереса
    '''
    img = cv.undistort(img, mtx, dist, None, new_mtx)
    x, y, w, h = roi
    return img[y:y+h, x:x+w]

def remove_noise(img, sturct_size: int):
    '''
    Использует операции close и open для убирания шума
    img: изображение
    struct_size: размер структуры (большее значение = меньше шума и большее закругление)
    '''
    structure = cv.getStructuringElement(cv.MORPH_ELLIPSE,(sturct_size,sturct_size))
    img_close = cv.morphologyEx(img, cv.MORPH_CLOSE, structure)
    img_smooth = cv.morphologyEx(img_close, cv.MORPH_OPEN, structure)
    return img_smooth
