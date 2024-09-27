import os
import torch
from torch.utils.data import Dataset, DataLoader
from pyntcloud import PyntCloud  # You may need to install this library
import open3d as o3d


class PointCloudDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load the PLY files (input and label)
        file_name = self.file_list[idx]
        input_ply_path = os.path.join(self.data_dir, "y1.ply")
        label_ply_path = os.path.join(self.data_dir, "y_gt.ply")

        # Load PLY data using PyntCloud (or other PLY loading library)
        input_point_cloud = PyntCloud.from_file(input_ply_path).xyz
        label_point_cloud = PyntCloud.from_file(label_ply_path).xyz

        # Convert to PyTorch tensors
        input_point_cloud = torch.FloatTensor(input_point_cloud)
        label_point_cloud = torch.FloatTensor(label_point_cloud)

        return input_point_cloud, label_point_cloud


# Example usage:
data_dir = "/Users/pratham/Desktop/3D reconstruction/Modified/T1 modified/input"

# Create the dataset
dataset = PointCloudDataset(data_dir)

# Create a DataLoader with batch size and other options
batch_size = 32  # Adjust this according to your needs
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for batch in dataloader:
    input_point_clouds, label_point_clouds = batch

    print(input_point_clouds.dtype)

    # Convert PyTorch tensors to NumPy arrays
    input_point_clouds = input_point_clouds.numpy()
    label_point_clouds = label_point_clouds.numpy()

    # Create Open3D PointCloud objects
    input_pcd = o3d.geometry.PointCloud()
    input_pcd.points = o3d.utility.Vector3dVector(input_point_clouds[0])
    input_pcd.paint_uniform_color([1, 0, 0])  # Blue for input

    label_pcd = o3d.geometry.PointCloud()
    label_pcd.points = o3d.utility.Vector3dVector(label_point_clouds[0])
    label_pcd.paint_uniform_color([0, 1, 0])  # Red for label

    # Visualize the point clouds using Open3D
    o3d.visualization.draw_geometries([label_pcd])








