import cv2
import numpy as np

for i in range(2, 100):
    d = np.load(f'game_play/{i}.npy')
    print(d.shape)

    cv2.imshow('', cv2.resize(d, (0, 0), fx=25, fy=25))
    cv2.waitKey(100)

