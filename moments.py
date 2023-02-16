import cv2
import numpy as np

from itertools import product

k1 = 0.15
k2 = 0.15
k3 = 0.1

def find_crosswalk(contours):
    moments = [cv2.moments(c) for c in contours]
    areas = sorted([(m['m00'], i) for i, m in enumerate(moments)], reverse=True)

    # return moments, areas
    # выбор лучшего момента для поиска среди контуров полных полос пешеходного перехода
    # weights = {}
    # for i in [13, 16, 20, 23]:
    #     m = moments[i]
    #     w = {}
    #     for k, n in m.items():
    #         w[k] = n / m['m00'] ** (int(k[-1]) + int(k[-2]))
    #         # w[k] = n
    #     weights[i] = w

    # diff = {}
    # for k in moments[13].keys():
    #     avr = 0
    #     for i in [13, 16, 20, 23]:
    #         avr += weights[i][k]
    #     avr /= 4

    #     m = 0
    #     for i in [13, 16, 20, 23]:
    #         m = max(m, weights[i][k] / avr - 1)
    #     diff[k] = round(m, 6)
    # print(diff)
    # print(moments[13], moments[47], sep='\n')
    # exit()

    lines = set()
    for i, a in enumerate(areas):
        for b in areas[i + 1:]:
            if a[0] + b[0] == 0 or abs(a[0] - b[0]) / (a[0] + b[0]) > k1:
                break
            if (abs((moments[a[1]]['nu02'] - moments[b[1]]['nu02'])
                / (moments[a[1]]['nu02'] + moments[b[1]]['nu02'])) > k3 or
                abs((moments[a[1]]['m01'] / a[0] - moments[b[1]]['m01'] / b[0])
                / (moments[a[1]]['m01'] / a[0] + moments[b[1]]['m01'] / b[0])) > k2):
                continue
            lines.add(a)
            lines.add(b)
            # if len(lines) > 3:
            #     break
        if lines:
            break
    return lines
