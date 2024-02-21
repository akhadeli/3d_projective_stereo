import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def translate_and_rotate(p, t, theta):
    pass

def transform(p, H):
    return H @ p

def buildH(tx, ty, tz, roll_x, pitch_y, yaw_z):
    Rx = np.array([[1, 0, 0, 0], 
                   [0, np.cos(roll_x), -np.sin(roll_x), 0], 
                   [0, np.sin(roll_x), np.cos(roll_x), 0], 
                   [0, 0, 0, 1]])
    
    Ry = np.array([[np.cos(pitch_y), 0, -np.sin(pitch_y), 0], 
                   [0, 1, 0, 0], 
                   [np.sin(pitch_y), 0, np.cos(pitch_y), 0], 
                   [0, 0, 0, 1]])
    
    Rz = np.array([[np.cos(yaw_z), -np.sin(yaw_z), 0, 0], 
                   [np.sin(yaw_z), np.cos(yaw_z), 0, 0], 
                   [0, 0, 1, 0], 
                   [0, 0, 0, 1]])
    
    T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
    H = Rz @ (Ry @ (Rx @ T))
    return H

def line_point_cloud(num_points, length = 1):
    return cube_line(num_points, length=length)

def draw_cloud(ax, cloud, projection='3d'):
    if projection == '3d':
        ax.scatter(cloud[0, :]/cloud[3, :], cloud[1, :]/cloud[3, :], cloud[2, :]/cloud[3, :])
    elif projection == '2d':
        ax.scatter(cloud[0, :]/cloud[2, :], cloud[1, :]/cloud[2, :])

def circle_point_cloud(num_points, radius = 1):
    t = 2 * np.pi * np.random.rand(1, num_points).flatten()
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.zeros((1, num_points)).flatten()
    circle = np.array([x, y, z, np.ones((1, num_points)).flatten()])
    return circle

def cube_line(num_points, x_g = 1, y_g = 0, z_g = 0, length = 1, x_t = 0, y_t = 0, z_t = 0):
    t = length * np.random.rand(1, num_points).flatten()

    x = t * x_g + x_t * np.ones((1, num_points)).flatten()
    y = t * y_g + y_t * np.ones((1, num_points)).flatten()
    z = t * z_g + z_t * np.ones((1, num_points)).flatten()

    line = np.array([x, y, z, np.ones((1, num_points)).flatten()])
    return line
    
def cube_point_cloud(num_points, side_length=1):
    b1 = cube_line(num_points, length=side_length)
    b2 = cube_line(num_points, 0, 1, 0, length=side_length)
    b3 = cube_line(num_points, length=side_length, y_t=side_length)
    b4 = cube_line(num_points, 0, 1, 0, length=side_length, x_t=side_length)

    v1 = cube_line(num_points, 0, 0, 1, length=side_length)
    v2 = cube_line(num_points, 0, 0, 1, length=side_length, x_t=side_length)
    v3 = cube_line(num_points, 0, 0, 1, length=side_length, y_t=side_length)
    v4 = cube_line(num_points, 0, 0, 1, length=side_length, x_t=side_length, y_t=side_length)

    t1 = cube_line(num_points, length=side_length, z_t=side_length)
    t2 = cube_line(num_points, 0, 1, 0, length=side_length, z_t=side_length)
    t3 = cube_line(num_points, length=side_length, y_t=side_length, z_t=side_length)
    t4 = cube_line(num_points, 0, 1, 0, length=side_length, x_t=side_length, z_t=side_length)

    cube = np.concatenate((b1, b2, b3, b4, v1, v2, v3, v4, t1, t2, t3, t4), axis=1)
    return cube

def save(filename):
    plt.savefig(filename)
    plt.clf()

    return fig.add_subplot(121, projection="3d"), fig.add_subplot(222)

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(222)

    # Line
    line = line_point_cloud(100, 3)
    draw_cloud(ax, line)
    ax, ax2d = save('output/ex1/line.png')

    # Circle
    circle = circle_point_cloud(200, 3)
    draw_cloud(ax, circle)
    ax, ax2d = save('output/ex1/circle.png')

    # Cube
    cube = cube_point_cloud(100, 1)
    draw_cloud(ax, cube)
    ax, ax2d = save('output/ex1/cube.png')

    # Orthographic Projection
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]])
    
    draw_cloud(ax, cube)
    t_cube = transform(cube, H)
    draw_cloud(ax2d, t_cube, projection='2d')
    ax, ax2d = save('output/ex1/orthographic/orthographic_cube.png')

    draw_cloud(ax, line)
    t_line = transform(line, H)
    draw_cloud(ax2d, t_line, projection='2d')
    ax, ax2d = save('output/ex1/orthographic/orthographic_line.png')

    draw_cloud(ax, circle)
    t_circle = transform(circle, H)
    draw_cloud(ax2d, t_circle, projection='2d')
    ax, ax2d = save('output/ex1/orthographic/orthographic_circle.png')


    # Perspective Projection
    H = buildH(1, 0, 2, 0, 0, 0)

    draw_cloud(ax, cube)
    t_cube = transform(cube, H)
    draw_cloud(ax2d, t_cube, projection='2d')
    ax, ax2d = save('output/ex1/perspective/perspective_cube.png')

    draw_cloud(ax, line)
    t_line = transform(line, H)
    draw_cloud(ax2d, t_line, projection='2d')
    ax, ax2d = save('output/ex1/perspective/perspective_line.png')

    draw_cloud(ax, circle)
    t_circle = transform(circle, H)
    draw_cloud(ax2d, t_circle, projection='2d')
    ax, ax2d = save('output/ex1/perspective/perspective_circle.png')


    # Perspective Projection (with rot)
    H = buildH(-0.5, -1, 2, np.pi, np.pi, np.pi)

    draw_cloud(ax, cube)
    t_cube = transform(cube, H)
    draw_cloud(ax2d, t_cube, projection='2d')
    ax, ax2d = save('output/ex1/perspective_rot/perspective_rot_cube.png')

    draw_cloud(ax, line)
    t_line = transform(line, H)
    draw_cloud(ax2d, t_line, projection='2d')
    ax, ax2d = save('output/ex1/perspective_rot/perspective_rot_line.png')

    draw_cloud(ax, circle)
    t_circle = transform(circle, H)
    draw_cloud(ax2d, t_circle, projection='2d')
    ax, ax2d = save('output/ex1/perspective_rot/perspective_rot_circle.png')

    # How does this exercise relate to camera calibration?
    # Projection use depth information in order to accurately a 3D on a 2D place
    # Camera calibaration allows us to extract depth information from images
    # Using both of these allows us to take an object in the world and
    # reconstruct it as a 3D model

    # parallel lines cannot be modeled properly as they converge to a vanishing point