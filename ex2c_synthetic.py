import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from ex1 import draw_cloud, cube_point_cloud, buildH

POINTS = 8
IMAGES = 3

def getPoints(fig, num_points):
    points = fig.ginput(num_points)
    return np.array(points, dtype=np.int32)


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot()

    cube = cube_point_cloud(100, 1)

    tx = 1
    ty = 0
    tz = 2

    points_array = []
    X_l = []
    X_r = []
    Y_l = []
    Y_r = []

    b= 0.3
    B = []
    
    f = 600

    for i in range(IMAGES):
        p_persp = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]]) @ np.concatenate((np.eye(3), np.array([[tx, ty, tz]]).T), axis=1)
        t_cube = p_persp @ cube
        draw_cloud(ax2, t_cube, '2d')
        tx += b
        points_array.append(getPoints(fig2, POINTS))

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
    
    X_l = np.array(X_l)
    X_r = np.array(X_r)
    Y_l = np.array(Y_l)
    Y_r = np.array(Y_r)

    D = X_l -X_r

    B = np.array(B)
    B = B[..., None]

    X_arr = []
    Y_arr = []
    Z_arr = []

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

    fig.savefig('output/ex2c/synthetic_reconstruction.png')

    # OPTIONAL
    # for i in range(1, POINTS, 2):
    #     # ax.plot(X_arr[i-1:])
    #     ax.plot(X_arr[i-1:i+1], Z_arr[i-1:i+1], Y_arr[i-1:i+1])
    #     # if i+1 < POINTS:

    # for i in range(0, int(POINTS/2)+1):
    #     ax.plot([X_arr[i], X_arr[i+2]], [Z_arr[i], Z_arr[i+2]], [Y_arr[i], Y_arr[i+2]])

    plt.show()