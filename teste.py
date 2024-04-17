import cv2
import numpy as np


img = np.zeros((100, 100, 3))

while True:
    cv2.imshow('teste', img)
    if cv2.waitKey(1) == ord('q'):
            break