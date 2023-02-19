from coordinates import PolarLine as pl, dec2pol
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math


# img = np.zeros((200, 300), dtype=np.uint8)
# p = (90, 0)
# l1 = pl.from_points((100, 0), (0, 200))
# print(l1.distance(p))
# # l2 = pl(0, math.pi*0.7)
# # print(l1.intersection(l2))
# cv.line(img, *l1.segment(200, 300), (255), 1, cv.LINE_AA)
# # cv.line(img, *l2.segment(200, 300), (255), 1, cv.LINE_AA)
# cv.circle(img, p, 3, 255, -1)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.show()
a = np.arange(6).reshape((2,3))
print(a)
print(a.max(axis=1))

