'''
Выделение цветов (квантизация)
'''
import numpy as np
import cv2 as cv

def quantization(img, color_count=3, scale=0.25):
    '''
    Приведение изображения к картинке из color_count цветов
    '''
    size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    pixels = cv.resize(img, size)
    shape = pixels.shape
    pixels = np.float32(pixels.reshape((-1,3)))
    # критерии прекращения итераций
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 3, 1.0)
    ret, label, center = cv.kmeans(pixels, color_count, None, criteria, 3, cv.KMEANS_PP_CENTERS)
    # обратное преобразование в изображение
    center = np.uint8(center)
    res = center[label.flatten()]
    return (
        cv.resize(res.reshape(shape),
                  (img.shape[1], img.shape[0]),
                  interpolation=cv.INTER_LINEAR),
        center
    )
