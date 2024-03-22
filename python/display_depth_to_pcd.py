import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_point_cloud(rgb_image, depth_image, fx, fy, cx, cy, scale):
    points_3d = []
    points_color = []
    # Downsample to 1/scale
    for row in range(0, rgb_image.shape[0], scale):
        for col in range(0, rgb_image.shape[1], scale):
            point_vector = np.array([col - cx, row - cy, fx])  # [x, y, z]
            point_vector = point_vector / fx
            depth = depth_image[row, col]
            if depth:
                point_color = rgb_image[row, col] / 255

                points_3d.append(list(point_vector * depth))
                points_color.append(list(point_color))

    points_3d = np.array(points_3d)
    return points_3d, points_color

def main():
    color_image = cv2.imread("test_images/color_image484.png")
    depth_image = cv2.imread("test_images/depth_image213.png", cv2.IMREAD_ANYDEPTH)
    depth_estimate_image = cv2.imread("test_images/depth_estimate_image213.png", cv2.IMREAD_ANYDEPTH)
    # Convert BGR image to RGB
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    depth_estimate_image = 1 / depth_estimate_image  # disparity -> depth

    # Camera parameters
    fx = 620
    fy = 618
    cx = 325
    cy = 258

    # point cloud from Depth camera
    points_3d, points_color = get_point_cloud(color_image, depth_image, fx, fy, cx, cy, 10)
    points_x = points_3d[:, 0]
    points_y = points_3d[:, 1]
    points_z = points_3d[:, 2]
    # point cloud from Depth estimation
    points_3d_est, points_color_est = get_point_cloud(color_image, depth_estimate_image, fx, fy, cx, cy, 10)
    points_3d_est = points_3d_est * 100
    points_x_est = points_3d_est[:, 0]
    points_y_est = points_3d_est[:, 1]
    points_z_est = points_3d_est[:, 2]

    # 2D RGB image plot
    fig = plt.figure(figsize=(15, 8))
    plt_row = 2
    plt_col = 3
    ax = fig.add_subplot(plt_row, plt_col, 1)
    ax.imshow(color_image)
    ax.set_title('RGB image')

    ### Plot point cloud from depth camera ###
    # 3D scatter plot 생성
    ax = fig.add_subplot(plt_row, plt_col, 2, projection='3d')
    scatter = ax.scatter(points_x, points_y, points_z, c=points_color)
    # 플롯 축 레이블 설정
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('Depth camera result')

    # 2D depth image (actually, it's inverse of the depth image)
    ax = fig.add_subplot(plt_row, plt_col, 5)
    ax.imshow(1 / depth_image, cmap='plasma')
    ax.set_title('depth from depth camera')

    ### Plot point cloud from Depth Anything model estimation ###
    # 3D scatter plot 생성
    ax = fig.add_subplot(plt_row, plt_col, 3, projection='3d')
    scatter = ax.scatter(points_x_est, points_y_est, points_z_est, c=points_color_est)
    # 플롯 축 레이블 설정
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('Depth estimation result')

    # 2D depth image (actually, it's inverse of the depth image)
    ax = fig.add_subplot(plt_row, plt_col, 6)
    ax.imshow(1 / depth_estimate_image, cmap='plasma')
    ax.set_title('depth from Depth Anything')

    plt.show()

if __name__ == "__main__":
    main()