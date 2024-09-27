import os
import plyfile

# # Define the path to your PLY file
# ply_file_path = '/Users/pratham/Desktop/3D reconstruction/Modified/T1 modified/y3.ply'
#
# # Read the PLY file without a context manager
# ply_data = plyfile.PlyData.read(ply_file_path)
#
# # Print the PLY file's data structure to identify the field names
# print(ply_data.elements)

# import open3d as o3d
# import numpy as np
#
# # Load your 3D point cloud data
# point_cloud = o3d.io.read_point_cloud("/Users/pratham/Desktop/3D reconstruction/Dataset 2/T1 modified/y_gt.ply")


# import os
# import open3d as o3d
#
# # Specify the input directory containing the PLY files
# input_directory = "/Users/pratham/Desktop/3D reconstruction/Dataset 2/T1"
#
# # Specify the output directory for saving modified PLY files
# output_directory = "/Users/pratham/Desktop/3D reconstruction/Dataset 2/T1 modified"
#
# # Create the output directory if it doesn't exist
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)
#
# # List all PLY files in the input directory
# ply_files = [f for f in os.listdir(input_directory) if f.endswith(".ply")]
#
# # Loop through each PLY file, remove RGB channels, and save to the output directory
# for ply_file in ply_files:
#     # Load the PLY file
#     point_cloud = o3d.io.read_point_cloud(os.path.join(input_directory, ply_file))
#
#     # Remove RGB channels (if they exist)
#     if point_cloud.has_colors():
#         point_cloud.colors = o3d.utility.Vector3dVector([])  # Empty colors
#
#     # Save the modified PLY file to the output directory
#     output_file = os.path.join(output_directory, ply_file)
#     o3d.io.write_point_cloud(output_file, point_cloud)
#
#     print(f"Removed RGB channels from {ply_file} and saved as {output_file} in the output directory")
# from torch.utils.data import TensorDataset, DataLoader
# import numpy as np
# import open3d as o3d
#
# def get_model(file_path):
#     pcd = o3d.io.read_point_cloud(file_path)
#     print(pcd)
#     array = np.asarray(pcd.points)
#     return array
#
# folder_name = '/Users/pratham/Desktop/Dataset 2/T1'
# def load_all(folder_name, contains = None):
#     file_names = [f for f in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, f))]
#     if (contains != None):
#         file_names = [s for s in file_names if contains in s]
#
#         models = []
# #
#         for m in range(len(file_names)):
#             file_path = (folder_name + '/' + file_names[m])
#             models.append(get_model(file_path))
# #
#         print(np.array(models))
#
# #
# all_models1 = load_all(folder_name, contains = "y1.ply") # names ends with a rotation number for 12 rotations, 30 degrees each
# all_models7 = load_all(folder_name, contains = "y2.ply")