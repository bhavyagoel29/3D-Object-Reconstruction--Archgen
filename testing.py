import os
import open3d as o3d
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomPointCloudDataset(Dataset):
    def __init__(self, input_dir, label_dir):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.input_files = sorted(os.listdir(self.input_dir))
        self.label_file = os.path.join(self.label_dir, "y_gt.ply")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file_path = os.path.join(self.input_dir, self.input_files[idx])

        # Read PLY files using open3d
        input_cloud = o3d.io.read_point_cloud(input_file_path)
        label_cloud = o3d.io.read_point_cloud(self.label_file)

        # Convert point clouds to NumPy arrays
        input_points = torch.tensor(input_cloud.points, dtype=torch.float32)
        label_points = torch.tensor(label_cloud.points, dtype=torch.float32)

        # Pad or truncate the point clouds to have the same number of points
        max_points = max(input_points.size(0), label_points.size(0))
        input_points = torch.cat([input_points, torch.zeros(max_points - input_points.size(0), 3)], dim=0)
        label_points = torch.cat([label_points, torch.zeros(max_points - label_points.size(0), 3)], dim=0)

        return input_points, label_points

# Specify your input directory and label directory
input_directory = '/Users/pratham/Desktop/3D reconstruction/Modified/T1 modified/input/'
label_directory = '/Users/pratham/Desktop/3D reconstruction/Modified/T1 modified/labels/'

# Create a custom dataset
custom_dataset = CustomPointCloudDataset(input_directory, label_directory)

# Create a DataLoader
batch_size = 4
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

for batch in dataloader:
    input_batch, label_batch = batch
    # Process the batch as needed for your task
    print("Input batch shape:", input_batch.shape)
    print("Label batch shape:", label_batch.shape)

for batch in dataloader:
    input_batch, label_batch = batch
    for i in range(batch_size):
        input_cloud = o3d.geometry.PointCloud()
        input_cloud.points = o3d.utility.Vector3dVector(input_batch[i].numpy())

        label_cloud = o3d.geometry.PointCloud()
        label_cloud.points = o3d.utility.Vector3dVector(label_batch[i].numpy())

        # Visualize the input and label point clouds
        o3d.visualization.draw_geometries([input_cloud])






