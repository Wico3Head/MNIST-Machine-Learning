import numpy as np
import cv2 as cv
from setting import *

pic = np.zeros((20, 20), dtype=np.uint8)
pic[1:9, 1:9] = 1
img = cv.cvtColor(pic, cv.COLOR_GRAY2BGR)
cv.imshow('img', img)
cv.waitKey()