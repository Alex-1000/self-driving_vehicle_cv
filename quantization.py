import numpy as np
import cv2

def quantization(img, K=3):
    '''
    Приведение изображения к картинке из K цветов
    '''
    pixels = img.reshape((-1,3))
    pixels = np.float32(pixels)
    # критерии прекращения итераций
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # обратное преобразование в изображение
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape)), center
