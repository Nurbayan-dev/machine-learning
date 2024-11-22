# edge detection - обнаружение границ
import numpy as np
   # 1. Open source Computer Vision. - library
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('grayscale-image.jpg', cv.IMREAD_GRAYSCALE)
# cv.IMREAD_GRAYSCAL гарантирует, что изображение будет считано как одно-канальное (черно-белое) изображение.

# assert проверяет, удалось ли загрузить изображение. Если нет, выводится сообщение об ошибке, предлагающее проверить путь к файлу.
assert img is not None, "file could not be read, check with os.path.exists()"
#2.  Применение алгоритма Канни для обнаружения границ:
edges = cv.Canny(img, 100, 200)

# 3. Edge detection methods
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])