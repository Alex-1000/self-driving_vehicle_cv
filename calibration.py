'''
Модуль калибровки камеры
'''

import cv2 as cv
import numpy as np

from display import show_difference


def calibration(chess_dir, out_dir, nx, ny, alpha=0.5):
    '''
    chess_dir: имя папки с изображениями шахматных досок
    out_dir: имя папки для результирующих изображений
    nx, ny: кол-во углов внутри шахматной доски
    alpha: параметр масштабирования для новой матрицы
    '''

    chess_files = chess_dir.iterdir()
    if out_dir.exists():
        for f in out_dir.iterdir():
            f.unlink()
    else:
        out_dir.mkdir()

    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:,:2] = np.mgrid[:nx, :ny].T.reshape(-1, 2)    # индексация углов [x,y,0]

    object_points = []  # массив точек калибровочного шаблона
    image_points = []   # массив проекций точек калибровочного шаблона

    failed = []     # файлы, для которых не найдены все углы

    # высота и ширина
    size = None

    for file in chess_files:
        img = cv.imread(str(file))

        if size is None:
            size = img.shape[:2]
        elif img.shape[:2] != size:
            print(f'Файл {file.name} имеет другой размер')
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    # в градации серого

        # критерии прекращения итераций
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # поиск углов шахматной доски
        ret, corners = cv.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            object_points.append(objp)

            # определение координат углов с субпиксельной точностью
            corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

            image_points.append(corners)

            # вывод углов на экран
            corners_image = img.copy()
            cv.drawChessboardCorners(corners_image, (nx, ny), corners, ret)
            # show_difference(img, corners_image, file.name, out_dir)
            show_difference(img, corners_image, file.name)
        else:
            failed.append(file.name)

    if failed:
        print('Неподходящие для калибровки изображения:', *sorted(failed), sep='\n')

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        object_points, image_points, size[::-1], None, None
    )
    new_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, size[::-1], alpha, size[::-1])
    return mtx, dist, rvecs, tvecs, new_mtx, roi, size
