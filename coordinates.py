'''
Работа с координатами
'''

from typing import Tuple

class LinearFunction:
    '''
    Линейная функция y = kx + b
    Параметр round_digits=None преобразует float в int
    '''
    k: float
    b: float

    def __init__(self, k: float, b: float):
        self.k = k
        self.b = b

    @staticmethod
    def from_points(p1: Tuple[int, int], p2: Tuple[int, int]):
        '''
        Определение линейной функции по двум точкам
        '''
        k = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b = p1[1] - k * p1[0]
        return LinearFunction(k, b)

    def y(self, x: float, round_digits=0) -> float:
        '''
        Значение y из x
        '''
        return round(self.k * x + self.b, round_digits)

    def x(self, y: float, round_digits=0) -> float:
        '''
        Значение x из y
        '''
        return round((y - self.b) / self.k, round_digits)

    def rectangle_line(self, size: Tuple[int, int], round_digits=0) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        '''
        Пересечение линии с прямоугольником (с координатами (0;0);size)
        '''
        w, h = size[0] - 1, size[1] - 1

        x0, y0 = 0, self.y(0)
        if y0 < 0:
            x0, y0 = self.x(0), 0
        elif y0 > h:
            x0, y0 = self.x(h), h

        x1, y1 = w, self.y(w)
        if y1 < 0:
            x1, y1 = self.x(w), w
        elif y1 > h:
            x1, y1 = self.x(h), h

        return (
                (round(x0, round_digits), round(y0, round_digits)),
                (round(x1, round_digits), round(y1, round_digits))
            )
