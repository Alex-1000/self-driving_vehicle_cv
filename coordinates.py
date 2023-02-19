'''
Работа с координатами
'''

from __future__ import annotations

import math
from math import atan2, cos, pi, sin
from typing import Tuple

# pi = math.pi

# def sin(x, precision=7) -> float:
#     return round(math.sin(x), precision)
# def cos(x, precision=7) -> float:
#     return round(math.cos(x), precision)
# def arctan(x, y, precision=7) -> float:
#     return round(math.atan2(x, y), precision)

def pol2dec(rho: float, theta: float, round_digits=0) -> Tuple[float, float]:
    '''
    Декартовы координаты точки из полярных
    '''
    return (round(rho * sin(theta), round_digits), round(rho * cos(theta), round_digits))

def dec2pol(x: float, y: float) -> Tuple[float, float]:
    '''
    Полярные координаты точки из декартовых
    '''
    # return ((x ** 2 + y ** 2) ** 0.5, arctan(y, x))
    return ((x ** 2 + y ** 2) ** 0.5, atan2(y, x))

class PolarLine:
    '''
    Линия в полярной системе координат
    '''
    rho: float
    theta: float

    def __init__(self, rho: float, theta: float):
        self.rho = rho
        self.theta = theta

    @staticmethod
    def from_points(p1: Tuple[int, int], p2: Tuple[int, int]):
        '''
        Прямая по двум точкам
        '''
        theta = -atan2(p2[0] - p1[0], p2[1] - p1[1])
        rho = p1[0] * cos(theta) + p1[1] * sin(theta)
        return PolarLine(rho, theta)

    def y(self, x: float, default=None) -> float:
        '''
        y от x в декартовых координатах
        '''
        if sin(self.theta) == 0:
            return default
        return (self.rho - x * cos(self.theta)) / sin(self.theta)

    def x(self, y: float, default=None) -> float:
        '''
        x от y в декартовых координатах
        '''
        if cos(self.theta) == 0:
            return default
        return (self.rho - y * sin(self.theta)) / cos(self.theta)

    def intersection(self, line: PolarLine, to_int=True) -> Tuple[float, float]:
        '''
        Пересечение с прямой в декартовых координатах
        '''
        if sin(line.theta - self.theta) == 0:
            return None
        x = ((line.rho * sin(self.theta) - self.rho * sin(line.theta))
             / sin(self.theta - line.theta))
        y = self.y(x, default=line.rho)
        return (round(x), round(y)) if to_int else (x, y)

    def segment(self, height: int, width: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        '''
        Возвращает сегмент в пределах прямоугольника (0;0)x(width;height)
        '''
        result = set()
        x = 0
        y = self.y(x)
        if y is not None and 0 <= y < height:
            result.add((x, round(y)))

        x = width - 1
        y = self.y(x)
        if y is not None and 0 <= y < height:
            result.add((x, round(y)))

        if len(result) < 2:
            y = 0
            x = self.x(y)
            if x is not None and 0 <= x < width:
                result.add((round(x), y))

            if len(result) < 2:
                y = height - 1
                x = self.x(y)
                if x is not None and 0 <= x < width:
                    result.add((round(x), y))
        return result

    def distance(self, point: Tuple[int, int]) -> float:
        '''
        Расстояние от точки до прямой
        '''
        rho, theta = dec2pol(*point)
        return abs(self.rho - rho * cos(abs(self.theta - theta)))
        