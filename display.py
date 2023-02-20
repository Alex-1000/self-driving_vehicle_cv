'''
Отображение изображений
'''
from pathlib import Path
from string import ascii_letters

import cv2 as cv
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

figsize = (12.8, 7.2)
dpi = 150
fontsize = 12

def show_difference(original, result, filename: str, result_text='Результат', out_dir=None):
    '''
    Вывод на экран или в файл изображения с двумя картинками
    '''
    # создание картинки 1920x1080 с двумя изображениями
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    f.tight_layout()    # включение авто-вписывания
    ax1.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
    ax1.set_title(f'Оригинал: {filename}', fontsize=fontsize)
    ax2.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    ax2.set_title(result_text, fontsize=fontsize)
    if out_dir:
        for letter in ascii_letters:
            new_file = Path(filename).stem + letter + Path(filename).suffix
            if not (out_dir / new_file).exists():
                f.savefig(str(out_dir / new_file))
                break
    else:
        plt.show()
    # plt.close()
