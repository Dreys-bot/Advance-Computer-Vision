import cv2
import numpy as np
from matplotlib import pyplot as plt

img_br = cv2.imread("background images/IMG-20200708-WA0081.jpg")
img_rgb = cv2.cvtColor(img_br, cv2.COLOR_BGR2RGB)

rectangle = (0,0,300,380)
mask = np.zeros(img_rgb.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

cv2.grapCut(img_rgb, mask, rectangle, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)