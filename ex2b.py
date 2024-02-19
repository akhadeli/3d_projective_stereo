import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# 2a images: baseline is 2cm
# Z_toBook = 60cm
POINTS = 6

def getPoints(frame, num_points):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # x y
    points = plt.ginput(num_points)
    # plt.close()
    return np.array(points, dtype=np.int32)

if __name__ == "__main__":
    img1 = cv2.imread('images/ex2b/opencv_frame_0.png')
    img2 = cv2.imread('images/ex2b/opencv_frame_1.png')
    # img1 = cv2.imread('images/ex2b/ex2b_1.jpg')
    # img2 = cv2.imread('images/ex2b/ex2b_2.jpg')

    b = 0.02       # 2cm
    f = 600       # 600 pixels

    points_1 = getPoints(img1, POINTS)
    points_2 = getPoints(img2, POINTS)
    # print(points_1)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    count = 1
    X_arr = []
    Y_arr = []
    Z_arr = []
    for p_set in zip(points_1, points_2):
        xl = p_set[0][0]
        xr = p_set[1][0]

        yl = p_set[0][1]

        Z = b * f / (xl - xr)
        X = xl * Z / f
        Y = yl * Z / f
        X_arr.append(X)
        Y_arr.append(Y)
        Z_arr.append(Z)
        ax.scatter(X, Z, Y)
        ax.text(X, Z, Y, str(count))
        count += 1
    
    # ax.plot(X_arr, Z_arr, Y_arr)
    print(X_arr, Y_arr, Z_arr)
    for i in range(1, len(X_arr), 2):
        # ax.plot(X_arr[i-1:])
        ax.plot(X_arr[i-1:i+1], Z_arr[i-1:i+1], Y_arr[i-1:i+1])
    
    plt.show()