import cv2
import numpy as np
import matplotlib.pyplot as plt

def getPoint(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # x y
    return np.array(plt.ginput(1), dtype=np.int32).flatten()

if __name__ == "__main__":
    img1 = cv2.imread('images/ex2a/opencv_frame_0.png')
    img2 = cv2.imread('images/ex2a/opencv_frame_1.png')
    Z = 85  # 100 cm
    b = 10  # 10 cm
    sensor_width = 8.5  # 8.5 mm

    xr = getPoint(img1)[0]
    xl = getPoint(img2)[0]

    f_pixels = Z * (xl - xr) / b
    print(f_pixels)

    f_mm = f_pixels * sensor_width / img1.shape[1]
    print(f_mm)
