import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from calibration import calibration
from coordinates import LinearFunction
from correction import undistort
from display import show_difference
from moments import find_crosswalk
from quantization import quantization

if __debug__:
    import time

chess_dir = Path('img/chess')     # папка с изображения для калибровки
image_dir = Path('img/road')      # папка с входными изображениями
out_dir = Path('img/out')         # папка для результатов

# размер-1 шахматной доски
nx = 6
ny = 8

# Разовая калибровка камеры (для повторной калибровки удалить файл cal_settings)
if __debug__:
    _timing_start = time.time()
if Path('cal_settings').exists():
    with open('cal_settings', 'rb') as cal_settings:
        mtx, dist, rvecs, tvecs, size = pickle.load(cal_settings)
else:
    mtx, dist, rvecs, tvecs, size = calibration(chess_dir, out_dir, nx, ny)
    with open('cal_settings', 'wb') as cal_settings:
        pickle.dump((mtx, dist, rvecs, tvecs, size), cal_settings)
if __debug__:
    print('Калибровка:', round((time.time() - _timing_start)*1000, 1))

# вывод одной из шахматных досок после корректировки для проверки
# img = cv2.imread('img/chess/2.jpg')
# show_difference(img, undistort(img, object_points, image_points), '2.jpg')

# test = [Path('img/road/12.jpg'), Path('img/road/14.jpg'), Path('img/road/19.jpg')]
for file in image_dir.iterdir():
# for file in test:
    if __debug__:
        _timing_start = time.time()
    img = undistort(cv2.imread(str(file)), mtx, dist)
    if img.shape[:2] != size:
        print(f'{file.name} имеет размер, отличный от остальных')
        continue
    if __debug__:
        print('Калибровка изображения:', round((time.time() - _timing_start) * 1000, 1))
        _timing_start = time.time()
    # выделение трёх основных цветов
    img_three, center = quantization(img, 3)
    if __debug__:
        print('Выделение цветов:', round((time.time() - _timing_start)*1000, 1))
    # show_difference(img, img_three, file.name)
    if __debug__:
        _timing_start = time.time()
    c_black, c_gray, c_white = sorted(center, key=sum)

    # вычесление порога для перевода в B/W изображение
    threshold = (sum(c_gray) + sum(c_white)) // 6
    img_gray = cv2.cvtColor(img_three, cv2.COLOR_BGR2GRAY)
    ret, img_bw = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
    if __debug__:
        print('Перевод в Ч/Б:', round((time.time() - _timing_start)*1000, 1))
        _timing_start = time.time()
    # выделение контуров
    # contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if __debug__:
        print('Нахождение контуров:', round((time.time() - _timing_start)*1000, 1))
        _timing_start = time.time()
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
    lines = find_crosswalk(contours)
    if __debug__:
        print('Нахождение полос пешеходного перехода:', round((time.time() - _timing_start)*1000, 1))
        _timing_start = time.time()
    # изображение с найденными полосами пешеходного перехода
    # for line in lines:
    #     cv2.drawContours(img_contours, contours[line:line+1], -1, (255,0,0), 3, cv2.LINE_AA, None, 1)
    # show_difference(img, img_contours, file.name, f'Найдено: {len(lines)}')

    # крайние левая и правая полосы
    edge_lines = [lines[0], lines[0]]
    edge_values = [ contours[lines[0]].min(axis=0)[0,0], contours[lines[0]].max(axis=0)[0,0] ]
    for line in lines[1:]:
        values = [ contours[line].min(axis=0)[0,0], contours[line].max(axis=0)[0,0] ]
        if edge_values[0] > values[0]:
            edge_lines[0], edge_values[0] = line, values[0]
        elif edge_values[1] < values[1]:
            edge_lines[1], edge_values[1] = line, values[1]
    if __debug__:
        print('Нахождение крайних полос:', round((time.time() - _timing_start)*1000, 1))
        _timing_start = time.time()
    cv2.drawContours(img_contours, [contours[edge_lines[0]]], -1, (255,0,0), 3, cv2.LINE_AA, None, 1)
    cv2.drawContours(img_contours, [contours[edge_lines[1]]], -1, (255,0,0), 3, cv2.LINE_AA, None, 1)

    # углы пешеходного перехода
    corners = [
        contours[edge_lines[0]].argmax(axis=0)[0,1],
        contours[edge_lines[1]].argmax(axis=0)[0,1],
        contours[edge_lines[0]].argmin(axis=0)[0,1],
        contours[edge_lines[1]].argmin(axis=0)[0,1]
    ]
    if __debug__:
        print('Нахождение углов:', round((time.time() - _timing_start)*1000, 1))
        _timing_start = time.time()
    # print(contours[edge_lines[0]][corners[0]], contours[edge_lines[1]][corners[1]],
    #     contours[edge_lines[0]][corners[2]], contours[edge_lines[1]][corners[3]])

    size = img.shape[:2]    # высота и ширина изображения

    # линии сверху и снизу пешеходного перехода
    f1 = LinearFunction.from_points(contours[edge_lines[0]][corners[0]][0], 
                                    contours[edge_lines[1]][corners[1]][0])
    f2 = LinearFunction.from_points(contours[edge_lines[0]][corners[2]][0],
                                    contours[edge_lines[1]][corners[3]][0])
    line_1 = f1.rectangle_line(size[::-1], round_digits=None)
    line_2 = f2.rectangle_line(size[::-1], round_digits=None)
    if __debug__:
        print('Выделение границ пешеходного перехода:', round((time.time() - _timing_start)*1000, 1), '\n')
    # img_contours = img.copy()
    # img_contours = np.zeros(size, dtype=np.uint8)
    cv2.line(img_contours, *line_1, [0, 255, 0], 3)
    cv2.line(img_contours, *line_2, [0, 0, 255], 3)
    show_difference(img, img_contours, file.name)

    # вывод контуров в порядке уменьшения площади
    # for a in areas:
    #     img_contours = img.copy()
    #     cv2.drawContours(img_contours, contours[a[1]:a[1]+1], -1, (255,0,0), 3, cv2.LINE_AA, None, 1)
    #     print(a[1], moments[a[1]])
    #     print(a[1], moments[a[1]]['m00'] )
    #     show_difference(img_bw, img_contours, str(a[1]))
