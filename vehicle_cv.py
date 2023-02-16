import pickle
# import os
import time
from pathlib import Path

import cv2
# import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from calibration import calibration
from correction import undistort
# from filters import test
# from transform import test
from filters import get_bin_img
from lane import fit_polynomial
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

# for file in image_dir.iterdir():
    # pass
file = 'img/road/3.jpg'
img = undistort(cv2.imread(file), object_points, image_points)

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
print(find_crosswalk(contours))

# for a in areas:
#     img_contours = img.copy()
#     cv2.drawContours(img_contours, contours[a[1]:a[1]+1], -1, (255,0,0), 3, cv2.LINE_AA, None, 1)
#     print(a[1], moments[a[1]])
#     print(a[1], moments[a[1]]['m00'] )
#     show_difference(img_bw, img_contours, str(a[1]))

exit()

# correction('img/chess/calibration3.jpg', out_dir, object_points, image_points)
# images = glob.glob('test_images/test*.jpg')
# images = glob.glob('test_images_my/*.jpg')
# for image in images:
#     correction(image, out_dir, object_points, image_points)
#     test(image, out_dir, object_points, image_points)
# image = 'square.png'
image = 'test_images_my/1.jpg'
kernel_size = 5
mag_thresh = (30, 100)
r_thresh = (235, 255)
s_thresh = (165, 255)
b_thresh = (160, 255)
g_thresh = (210, 255)

# img = undistort(image, object_points, image_points)
img = cv2.imread(image)

combined_binary = get_bin_img(img, kernel_size=kernel_size, sobel_thresh=mag_thresh, r_thresh=r_thresh, 
                            s_thresh=s_thresh, b_thresh = b_thresh, g_thresh=g_thresh)
combined_binary = cv2.cvtColor(combined_binary, cv2.COLOR_BGR2GRAY)
c1 = combined_binary.copy()
# left_fit, right_fit, left_fitx, right_fitx, left_lane_indices, right_lane_indices, out_img = fit_polynomial(warped, nwindows=20)
# left_fit, right_fit, left_fitx, right_fitx, left_lane_indices, right_lane_indices, out_img = fit_polynomial(combined_binary, nwindows=20)
# _, contours, hierarchy = cv2.findContours(combined_binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(6,6)))
# contours, hierarchy = cv2.findContours(combined_binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours( combined_binary, contours, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1 )
# plt.imshow(combined_binary, cmap='gray')
plt.imshow(combined_binary)
# plt.imshow(out_img)
plt.savefig(out_dir + "/op_" + str(time.time()) + ".jpg")
plt.imshow(c1)
plt.savefig(out_dir + "/op_" + str(time.time()) + ".jpg")
