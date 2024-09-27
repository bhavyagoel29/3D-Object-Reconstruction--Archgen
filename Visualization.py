import open3d as o3d
import numpy as np
import copy
import random
import math
def main():

    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud("/Users/pratham/Desktop/voxel file/yeehaw.ply")
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])
    # norm_pointcloud = pcd - np.mean(pcd, axis=0)
    # norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

    # # rotation around z-axis
    # theta = random.random() * 2. * math.pi  # rotation angle
    # rot_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
    #                        [math.sin(theta), math.cos(theta), 0],
    #                        [0, 0, 1]])
    #
    # rot_pointcloud = rot_matrix.dot(pcd.T).T
    #
    # # add some noise
    # noise = np.random.normal(0, 0.02, (pcd.shape))
    # noisy_pointcloud = rot_pointcloud + noise
    # o3d.visualization.draw_geometries(noisy_pointcloud)

    # diameter = np.linalg.norm(np.asarray(pcd.get_min_bound()) - np.asarray(pcd.get_max_bound()))
    # camera = [0, 0, diameter]
    # radius = diameter * 100
    #
    # # Performing the hidden point removal operation on the point cloud using the
    # # camera and radius parameters defined above.
    # # The output is a list of indexes of points that are visible.
    # _, pt_map = pcd.hidden_point_removal(camera, radius)
    # # Painting all the visible points in the point cloud in blue, and all the hidden points in red.
    #
    # pcd_visible = pcd.select_by_index(pt_map)
    # pcd_visible.paint_uniform_color([0, 0, 1])  # Blue points are visible points (to be kept).
    # print("No. of visible points : ", pcd_visible)
    #
    # pcd_hidden = pcd.select_by_index(pt_map, invert=True)
    # pcd_hidden.paint_uniform_color([1, 0, 0])  # Red points are hidden points (to be removed).
    # print("No. of hidden points : ", pcd_hidden)
    #
    # # Visualizing the visible (blue) and hidden (red) points in the point cloud.
    # draw_geoms_list = [pcd_visible, pcd_hidden]
    # o3d.visualization.draw_geometries(draw_geoms_list)
    # # Defining a function to convert degrees to radians.
    # def deg2rad(deg):
    #     return deg * np.pi / 180
    #
    # # Rotating the point cloud about the X-axis by 90 degrees.
    # x_theta = deg2rad(90)
    # y_theta = deg2rad(0)
    # z_theta = deg2rad(0)
    # tmp_pcd_r = copy.deepcopy(pcd)
    # R = tmp_pcd_r.get_rotation_matrix_from_axis_angle([x_theta, y_theta, z_theta])
    # tmp_pcd_r.rotate(R, center=(0, 0, 0))
    #
    # # Visualizing the rotated point cloud.
    # draw_geoms_list = [tmp_pcd_r]
    # o3d.visualization.draw_geometries(draw_geoms_list)



if __name__ == "__main__":
    main()
