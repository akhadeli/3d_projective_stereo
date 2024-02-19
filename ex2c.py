import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# ex2c images: starting from 50cm to 60cm in 2cm intervals
# Z_toBook = 60cm

POINTS = 6
IMAGES = 3

def getPoints(frame, num_points):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # x y
    points = plt.ginput(num_points)
    # plt.close()
    return np.array(points, dtype=np.int32)

if __name__ == "__main__":
    img_array = []
    points_array = []
    X_l = []
    X_r = []
    Y_l = []
    Y_r = []
    
    b = 0.02       # 2cm
    B = []

    f = 600       # 600 pixels

    for i in range(IMAGES):
        img_array.append(cv2.imread(f'images/ex2c/opencv_frame_{i}.png'))
        points_array.append(getPoints(img_array[i], POINTS))
    
    # plt.close()
    points_array = np.array(points_array)

    for i in range(IMAGES-1):
        for j in range(i+1, IMAGES):
            inner_xl = []
            inner_xr = []
            inner_yl = []
            inner_yr = []
            for k in range(POINTS):
                inner_xl.append(points_array[i][k][0])
                inner_xr.append(points_array[j][k][0])
                inner_yl.append(points_array[i][k][1])
                inner_yr.append(points_array[j][k][1])
            B.append(b*(j-i))
            X_l.append(inner_xl)
            X_r.append(inner_xr)
            Y_l.append(inner_yl)
            Y_r.append(inner_yr)
    # print(points_array)
    X_l = np.array(X_l)
    X_r = np.array(X_r)
    Y_l = np.array(Y_l)
    Y_r = np.array(Y_r)

    print(X_l)
    print(X_r)

    D = X_l -X_r

    # print(D)

    B = np.array(B)
    B = B[..., None]
    # print(D)
    # print(B)

    X_arr = []
    Y_arr = []
    Z_arr = []

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # ax.set_xlim(-1, 0)
    # ax.set_ylim(-1, 0)
    # ax.set_zlim(-1, 0)

    for i in range(POINTS):
        D_i = D[:, i, None]
        Z_i = np.linalg.lstsq(D_i, f * B)[0][0][0]
        X = X_l[:, i][0] * Z_i / f
        Y = Y_l[:, i][0] * Z_i / f
        # print(X, Y, Z_i)
        X_arr.append(X)
        Y_arr.append(Y)
        Z_arr.append(Z_i)
        ax.scatter(X, Z_i, Y)
        ax.text(X, Z_i, Y, str(i))
    
    print(X_arr, Y_arr, Z_arr)
    for i in range(1, POINTS, 2):
        # ax.plot(X_arr[i-1:])
        ax.plot(X_arr[i-1:i+1], Z_arr[i-1:i+1], Y_arr[i-1:i+1])
        # if i+1 < POINTS:

    for i in range(0, int(POINTS/2)+1):
        ax.plot([X_arr[i], X_arr[i+2]], [Z_arr[i], Z_arr[i+2]], [Y_arr[i], Y_arr[i+2]])


    plt.show()

    # D = X_l - X_r
    # D = D[..., None]
    # B = B[..., None]

    # print(D)
    # print(f*B)

    # Z = np.linalg.lstsq(D, f * B)

    # print(B,'\n', X_l - X_r)
    # print(Z)

    # img1 = cv2.imread('images/ex2b/ex2b_1.jpg')
    # img2 = cv2.imread('images/ex2b/ex2b_2.jpg')
