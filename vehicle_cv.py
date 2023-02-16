import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from calibration import calibration
from correction import undistort
from display import show_difference
from quantization import quantization
from moments import find_crosswalk

chess_dir = Path('img/chess')     # папка с изображения для калибровки
image_dir = Path('img/road')      # папка с входными изображениями
out_dir = Path('img/out')         # папка для результатов

# размер-1 шахматной доски
nx = 6
ny = 8

# Разовая калибровка камеры (для повторной калибровки удалить файл cal_settings)
if Path('cal_settings').exists():
    with open('cal_settings', 'rb') as cal_settings:
        object_points = pickle.load(cal_settings)
        image_points = pickle.load(cal_settings)
else:
    object_points, image_points = calibration(chess_dir, out_dir, nx, ny)
    with open('cal_settings', 'wb') as cal_settings:
        pickle.dump(object_points, cal_settings)
        pickle.dump(image_points, cal_settings)


# вывод одной из шахматных досок после корректировки для проверки
# img = cv2.imread('img/chess/2.jpg')
# show_difference(img, undistort(img, object_points, image_points), '2.jpg')

for file in image_dir.iterdir():
# file = 'img/road/1.jpg'
    img = undistort(cv2.imread(str(file)), object_points, image_points)

    # выделение трёх основных цветов
    img_three, center = quantization(img, 3)
    c_black, c_gray, c_white = sorted(center, key=sum)

    # вычесление порога для перевода в B/W изображение
    threshold = (sum(c_gray) + sum(c_white)) // 6
    img_gray = cv2.cvtColor(img_three, cv2.COLOR_BGR2GRAY)
    ret, img_bw = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)

    # выделение контуров
    # contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # img_contours = img.copy()
    # cv2.drawContours(img_contours, contours, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1)

    # убрать шум на некачественных изображениях
    # img_close = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))
    # img_smooth = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))
    # ret, img_smooth_bw = cv2.threshold(img_smooth, threshold, 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(img_smooth_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # img_contours = img.copy()
    # cv2.drawContours(img_contours, contours, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1)

    # print(len(contours))
    # show_difference(img, img_contours, '3.jpg')

    # moments, areas = find_crosswalk(contours)

    img_contours = img.copy()
    crosswalk_lines = find_crosswalk(contours)
    for line in crosswalk_lines:
        cv2.drawContours(img_contours, contours[line:line+1], -1, (255,0,0), 3, cv2.LINE_AA, None, 1)
    show_difference(img, img_contours, file.name, f'Найдено: {len(crosswalk_lines)}')

    # for a in areas:
    #     img_contours = img.copy()
    #     cv2.drawContours(img_contours, contours[a[1]:a[1]+1], -1, (255,0,0), 3, cv2.LINE_AA, None, 1)
    #     print(a[1], moments[a[1]])
    #     print(a[1], moments[a[1]]['m00'] )
    #     show_difference(img_bw, img_contours, str(a[1]))
