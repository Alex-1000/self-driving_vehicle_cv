import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from correction import undistort
from filters import get_bin_img

def transform_image(img, offset=250, src=None, dst=None):
    '''
    Преобразование изображения в вид сверху
    '''
    img_size = (img.shape[1], img.shape[0])

    out_img_orig = np.copy(img)

    # TODO: магические числа?
    left_upper  = (585, 460)
    right_upper = (705, 460)
    left_lower  = (210, img.shape[0])
    right_lower = (1080, img.shape[0])


    warped_left_upper = (offset,0)
    warped_right_upper = (offset, img.shape[0])
    warped_left_lower = (img.shape[1] - offset, 0)
    warped_right_lower = (img.shape[1] - offset, img.shape[0])

    color_r = [0, 0, 255]
    color_g = [0, 255, 0]
    line_width = 5

    if src is None:
        src = np.float32([
            left_upper,
            left_lower,
            right_upper,
            right_lower
        ])

    if dst is None:
        dst = np.float32([
            warped_left_upper,
            warped_right_upper,
            warped_left_lower,
            warped_right_lower
        ])

    cv2.line(out_img_orig, left_lower, left_upper, color_r, line_width)
    cv2.line(out_img_orig, left_lower, right_lower, color_r , line_width * 2)
    cv2.line(out_img_orig, right_upper, right_lower, color_r, line_width)
    cv2.line(out_img_orig, right_upper, left_upper, color_g, line_width)

    # перспективное преобразование
    M = cv2.getPerspectiveTransform(src, dst)
    minv = cv2.getPerspectiveTransform(dst, src)

    # преобразовать изображение
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)
    out_warped_img = np.copy(warped)

    cv2.line(out_warped_img, warped_right_upper, warped_left_upper, color_r, line_width)
    cv2.line(out_warped_img, warped_right_upper, warped_right_lower, color_r , line_width * 2)
    cv2.line(out_warped_img, warped_left_lower, warped_right_lower, color_r, line_width)
    cv2.line(out_warped_img, warped_left_lower, warped_left_upper, color_g, line_width)

    return warped, M, minv, out_img_orig, out_warped_img

def test(image, out_dir, object_points, image_points):
    # Testing the threshholding with warp
    kernel_size = 5
    mag_thresh = (30, 100)
    r_thresh = (235, 255)
    s_thresh = (165, 255)
    b_thresh = (160, 255)
    g_thresh = (210, 255)

    # for image_name in images:
    # img = undistort(image_name, object_points, image_points)
    img = undistort(image, object_points, image_points)

    # combined_binary = get_bin_img(img, kernel_size=kernel_size, sobel_thresh=mag_thresh,
                                #   r_thresh=r_thresh, s_thresh=s_thresh, b_thresh = b_thresh, g_thresh=g_thresh)
    # warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary)
    warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + image , fontsize=18)
    ax2.imshow(warped, cmap='gray')
    ax2.set_title("Warped:: "+ image, fontsize=18)
    # f.savefig(out_dir + "/op_" + str(time.time()) + ".jpg")
    f.savefig(out_dir + "/warp_" + image.replace('/', '_')[:-4] + ".jpg")
    plt.close()
