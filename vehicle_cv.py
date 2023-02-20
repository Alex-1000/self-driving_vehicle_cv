import pickle
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
# модель машинного обучения YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

from calibration import calibration
from coordinates import PolarLine
from correction import undistort
from display import show_difference
from moments import find_crosswalk
from quantization import quantization

if __debug__:
    import time

c1 = 0.077568134
c2 = 30.03752621

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
        mtx, dist, rvecs, tvecs, new_mtx, roi, input_size = pickle.load(cal_settings)
else:
    mtx, dist, rvecs, tvecs, new_mtx, roi, input_size = calibration(chess_dir, out_dir, nx, ny)
    with open('cal_settings', 'wb') as cal_settings:
        pickle.dump((mtx, dist, rvecs, tvecs, new_mtx, roi, input_size), cal_settings)
if __debug__:
    print('Калибровка:', round((time.time() - _timing_start)*1000, 1))

# вывод одной из шахматных досок после корректировки для проверки
# img = cv.imread('img/chess/2.jpg')
# show_difference(img, undistort(img, mtx, dist, new_mtx, roi), '2.jpg')

def test_check(lines):
    if lines is None or len(lines) < 4:
        return True
    return sum(
        int(0 < l[0][1] < np.pi * 5/16 or np.pi * 11/16 < l[0][1] < 2*np.pi) for l in lines
    ) < 4

# test = [Path('img/road/2.jpg'), Path('img/road/7.jpg'), Path('img/road/12.jpg')]
for file in image_dir.iterdir():
# for file in test:
    if __debug__:
        _timing_start = time.time()

    original = cv.imread(str(file))
    if original.shape[:2] != input_size:
        print(f'{file.name} имеет размер, отличный от остальных')
        continue
    original = undistort(original, mtx, dist, new_mtx, roi)
    size = original.shape[:2]    # высота и ширина изображения
    if __debug__:
        print('Калибровка изображения:', round((time.time() - _timing_start) * 1000, 1))
        _timing_start = time.time()

    # выделение трёх основных цветов
    img, center = quantization(original, 3)
    # закрываем горизонт
    # TODO
    # img = cv.rectangle(img, (0,0), (size[1], int(size[0]*0.3)), (0,0,0), -1)

    if __debug__:
        print('Выделение цветов:', round((time.time() - _timing_start)*1000, 1))
        show_difference(original, img, file.name)
        _timing_start = time.time()

    c_black, c_gray, c_white = sorted(center, key=sum)
    center = None

    # вычисление порога для перевода в B/W изображение
    threshold = (sum(c_gray) + sum(c_white)) // 6
    ret, img_bw = cv.threshold(
        cv.cvtColor(img, cv.COLOR_BGR2GRAY), threshold, 255, cv.THRESH_BINARY

    )

    if __debug__:
        print('Перевод в Ч/Б:', round((time.time() - _timing_start)*1000, 1))
        show_difference(original, img_bw, file.name)
        _timing_start = time.time()
    
    # выделение контуров
    contours, hierarchy = cv.findContours(img_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if __debug__:
        print('Нахождение контуров:', round((time.time() - _timing_start)*1000, 1))
        _timing_start = time.time()

    lines = find_crosswalk(contours)
    if __debug__:
        print('Нахождение полос пешеходного перехода:',
              round((time.time() - _timing_start)*1000, 1))
        img = original.copy()
        for l in lines:
            cv.drawContours(img, [contours[l]], -1, (255,0,0), 3, cv.LINE_AA, None, 1)
        show_difference(original, img, file.name)
        img1 = img.copy()
        _timing_start = time.time()

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

    # img = np.zeros(size, dtype=np.uint8)
    # img = original.copy()
    # for l in lines:
    #     cv.drawContours(img, [contours[l]], -1, (0,0,255), 4, cv.LINE_AA, None, 1)

    # cv.drawContours(img, [contours[edge_lines[0]]], -1, 255, 3, cv.LINE_AA, None, 1)
    # cv.drawContours(img, [contours[edge_lines[1]]], -1, 255, 3, cv.LINE_AA, None, 1)
    # lines = None
    # x = 400
    # while test_check(lines):
    #     cv.dilate(img, cv.getStructuringElement(cv.MORPH_RECT, (2,2)))
    #     # show_difference(original, img, file.name)
    #     lines = cv.HoughLines(img, 0.95, np.pi/90, x, None, 0, 0)
    #     x -= 10

    # print(len(lines), x)
    # # img = original.copy()
    # for l in lines:
    #     f = PolarLine(*l[0])
    #     cv.line(img1, *f.segment(*size), (0, 255, 0), 3)
    # # show_difference(original, img1, file.name)

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

    # линии сверху и снизу пешеходного перехода
    top_edge = PolarLine.from_points(
        contours[edge_lines[0]][corners[2]][0],
        contours[edge_lines[1]][corners[3]][0]
    )
    bottom_edge = PolarLine.from_points(
        contours[edge_lines[0]][corners[0]][0], 
        contours[edge_lines[1]][corners[1]][0]
    )
    if __debug__:
        print('Выделение верхних и нижних границ пешеходного перехода:',
              round((time.time() - _timing_start)*1000, 1), '\n')
    # img_contours = img.copy()
    # img_contours = np.zeros(size, dtype=np.uint8)
    img = original.copy()
    cv.line(img, *top_edge.segment(*size), [255], 3)
    cv.line(img, *bottom_edge.segment(*size), [255], 3)
    cv.drawContours(img, [contours[edge_lines[0]]], -1, (255,0,0), 3, cv.LINE_AA, None, 1)
    cv.drawContours(img, [contours[edge_lines[1]]], -1, (255,0,0), 3, cv.LINE_AA, None, 1)
    show_difference(original, img, file.name)

    result = model([img])
    img = result.render()[0]
    show_difference(original, img, file.name)

    # roi_top = PolarLine(top_edge.rho*1.5 - bottom_edge.rho*0.5, top_edge.theta)
    # cv.line(img, *roi_top.segment(*size), [255, 0, 0], 3)
    # pts = np.array([[0,0], *roi_top.segment(*size), [size[1],0]], np.int32).reshape((-1,1,2))
    # cv.fillPoly(img, [pts], (0))
    # show_difference(original, img, file.name)
    
    # roi_points = sorted(list(bottom_edge.segment(*size)), key=lambda x: x[1])
    # y_crop = roi_points[0][1]
    # if y_crop > 830:
    #     # TODO
    #     continue
    # pts = np.array([[0,0], *sorted(roi_points, key=lambda x: x[0]), [size[1],0]], np.int32).reshape((-1,1,2))
    # cv.fillPoly(img, [pts], 0)

    # img = cv.resize(img, (size[1]//10, size[0]))
    # print(y_crop)
    # img = img[y_crop:img.shape[0], 0:img.shape[1]]
    # lines = cv.HoughLines(img, 1.4, np.pi / 180, 250, None, 0, 0)
    # down_pts = []
    # bottom_center = img.shape[1]//2
    # if lines is not None:
    #     for i, l in enumerate(lines):
    #         f = PolarLine(*l[0])
    #         down_pts.append((bottom_center - f.x(img.shape[0]-1), i))
    #         # cv.line(img, *f.segment(*img.shape[:2]), 255, 1)
    # down_pts.sort(key=lambda x: abs(x[0]))
    # left, right = None, None
    # for d, i in down_pts:
    #     if d < 0 and left is None:
    #         left = lines[i]
    #     elif right is None:
    #         right = lines[i]
    #     if left is not None and right is not None:
    #         break
    # if right is None:
    #     right = left
    #     left = None
    # if left is not None:
    #     left = PolarLine(*left[0])
    #     cv.line(img, *left.segment(*img.shape[:2]), [255], 1)
    # if right is not None:
    #     right = PolarLine(*right[0])
    #     cv.line(img, *right.segment(*img.shape[:2]), [255], 1)
    # show_difference(original, img, file.name)
    # img = original.copy()
    # if left is not None:
    #     # left = PolarLine(*left[0])
    #     top = left.x(img.shape[0] - size[0]) * 10
    #     bot = left.x(img.shape[0]) * 10
    #     left = PolarLine.from_points((top, 0), (bot, size[0]))
    #     cv.line(img, *left.segment(*size), [0,0,255], 3)
    # if right is not None:
    #     # right = PolarLine(*right[0])
    #     top = right.x(img.shape[0] - size[0]) * 10
    #     bot = right.x(img.shape[0]) * 10
    #     right = PolarLine.from_points((top, 0), (bot, size[0]))
    #     cv.line(img, *right.segment(*size), [0,0,255], 3)
    # show_difference(original, img, file.name)



    # x = round(min(
    #     bottom_edge.distance((size[1]//4, size[0])),
    #     bottom_edge.distance((size[1]*3//4, size[0]))
    # ) * 0.2, 2)


    # x = bottom_edge.distance((size[1]*5//8, size[0]))
    # print(file.name, round(x * c1 + c2, 1))
    # show_difference(original, img, file.name, result_text=str(x))

    # ищем левую нижнюю и левую верхнюю точки левого контура
    # delta = 100
    # 0: левый контур, 1: правый контур
    # .,0: левая верхняя, .,1: правая верхняя, .,2: левая нижняя, .,3: правая нижняя
    # corners = [
    #     [ (size[1], None), (0, None), (size[1], None), (0, None) ], 
    #     [ (size[1], None), (0, None), (size[1], None), (0, None) ]
    # ]
    # for e, edge in enumerate(edge_lines):
    #     c_sorted = contours[edge][:,0][contours[edge][:,0,1].argsort()]
    #     for p in c_sorted:
    #         if (corners[e][0][1] is not None
    #            and abs((corners[e][0][0] - corners[e][1][0]) * np.cos(top_edge.theta))
    #            * 2 + 10 + c_sorted[0][1] < p[1]):
    #             break
    #         if top_edge.distance(p) > delta:
    #             continue
    #         corners[e][0] = min(corners[e][0], p, key=lambda x: x[0])
    #         corners[e][1] = max(corners[e][0], p, key=lambda x: x[0])



    # вывод контуров в порядке уменьшения площади
    # for a in areas:
    #     img_contours = img.copy()
    #     cv.drawContours(img_contours, contours[a[1]:a[1]+1], -1, (255,0,0), 3, cv.LINE_AA, None, 1)
    #     print(a[1], moments[a[1]])
    #     print(a[1], moments[a[1]]['m00'] )
    #     show_difference(img_bw, img_contours, str(a[1]))
