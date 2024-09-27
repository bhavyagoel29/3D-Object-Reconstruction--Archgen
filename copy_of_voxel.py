# -*- coding: utf-8 -*-

#from plyfile import PlyData
import numpy as np
import os
import numpy as np

os.environ['QT_API'] = 'pyqt5'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

def load_ply_file(file_path):
    with open(file_path, 'rb') as ply_file:
        ply_data = PlyData.read(ply_file)
        vertices = ply_data['vertex']
        point_cloud = np.array([(v['x'], v['y'], v['z']) for v in vertices], dtype=np.float32)

    return point_cloud

# Voxelization process
def voxels(file_path):
    with open(file_path, 'rb') as ply_file:
        ply_data = PlyData.read(ply_file)
        vertices = ply_data['vertex']
        point_cloud = np.array([(v['x'], v['y'], v['z']) for v in vertices], dtype=np.float32)
        for point in point_cloud:
            voxel_coords = ((point - min_coords) / voxel_resolution).astype(int)
            voxel_grid[voxel_coords[0], voxel_coords[1], voxel_coords[2]] = True

        return voxel_grid

# Example usage:
file_path = "C:\\Users\\DeLL\\Downloads\\y1.ply"
# Read the 3D point cloud from a PLY file (assumes you have a function to load PLY files)
point_cloud = load_ply_file(file_path)
print(point_cloud)
# Define voxel grid parameters (voxel size and bounding box)
voxel_resolution = 0.1  # Adjust this based on your requirements
min_coords = np.min(point_cloud, axis=0)
max_coords = np.max(point_cloud, axis=0)
grid_shape = ((max_coords - min_coords) / voxel_resolution).astype(int) + 5

# Initialize the voxel grid
voxel_grid = np.zeros(grid_shape, dtype=bool)

for point in point_cloud:
    voxel_coords = ((point - min_coords) / voxel_resolution).astype(int)
    voxel_grid[voxel_coords[0], voxel_coords[1], voxel_coords[2]] = True
print("meow")
print(voxel_coords)
voxel_shape = voxel_grid.shape
print(voxel_shape)

'''class Generator(nn.Module):
    def __init__(self, latent_dim, num_points, point_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points
        self.point_dim = point_dim

#        # Define the generator architecture
#        self.fc1 = nn.Linear(latent_dim, 256)
#        self.fc2 = nn.Linear(256, 512)
#        self.fc3 = nn.Linear(512, num_points * point_dim)
#        # Project random noise to an intermediate representation
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 133 * 109 * 120),  # Adjust for the desired output shape
            nn.Tanh()
        )
    def forward(self, z):
        #x = torch.relu(self.fc1(z))
        #x = torch.relu(self.fc2(x))
        #x = torch.tanh(self.fc3(x))  # Output is in the range [-1, 1]

        # Reshape the output to match the desired point cloud format
        #x = x.view(-1, self.num_points, self.point_dim)

        # Generate 3D point cloud from noise
        point_cloud = self.fc(z)
        # Reshape to the desired output shape
        point_cloud = point_cloud.view(-1, 133, 109, 120)
        return point_cloud

        #return x
'''

'''
class Generator(nn.Module):
    def __init__(self, noise_size=101, cube_resolution=32):
        super(Generator, self).__init__()

        self.noise_size = noise_size
        self.cube_resolution = cube_resolution

        self.gen_conv1 = torch.nn.ConvTranspose3d(self.noise_size, 256, kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=1)
        self.gen_conv2 = torch.nn.ConvTranspose3d(256, 128, kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=1)
        self.gen_conv3 = torch.nn.ConvTranspose3d(128, 64, kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=1)
        self.gen_conv4 = torch.nn.ConvTranspose3d(64, 32, kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=1)
        self.gen_conv5 = torch.nn.ConvTranspose3d(32, 1, kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=1)

        self.gen_bn1 = nn.BatchNorm3d(256)
        self.gen_bn2 = nn.BatchNorm3d(128)
        self.gen_bn3 = nn.BatchNorm3d(64)
        self.gen_bn4 = nn.BatchNorm3d(32)

    def forward(self, x, condition):
        condition_tensor = condition * torch.ones([x.shape[0], 1], device=x.device)
        x = torch.cat([x, condition_tensor], dim=1)
        x = x.view(x.shape[0], self.noise_size, 1, 1, 1)

        x = F.relu(self.gen_bn1(self.gen_conv1(x)))
        x = F.relu(self.gen_bn2(self.gen_conv2(x)))
        x = F.relu(self.gen_bn3(self.gen_conv3(x)))
        x = F.relu(self.gen_bn4(self.gen_conv4(x)))
        x = self.gen_conv5(x)
        x = torch.sigmoid(x)

        return x.squeeze()
'''

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_size=101, cube_resolution=32):
        super(Generator, self).__init__()

        self.noise_size = noise_size
        self.cube_resolution = cube_resolution

        # Adjust the ConvTranspose3d layers to achieve the desired output shape
        self.gen_conv1 = torch.nn.ConvTranspose3d(self.noise_size, 256, kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=1)
        self.gen_conv2 = torch.nn.ConvTranspose3d(256, 128, kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=1)
        self.gen_conv3 = torch.nn.ConvTranspose3d(128, 64, kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=1)
        self.gen_conv4 = torch.nn.ConvTranspose3d(64, 32, kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=1)

        # Adjust the final ConvTranspose3d layer for the desired output shape
        self.gen_conv5 = torch.nn.ConvTranspose3d(32, 1, kernel_size=[4, 4, 4], stride=[1, 1, 1], padding=0)

        self.gen_bn1 = nn.BatchNorm3d(256)
        self.gen_bn2 = nn.BatchNorm3d(128)
        self.gen_bn3 = nn.BatchNorm3d(64)
        self.gen_bn4 = nn.BatchNorm3d(32)

    def forward(self, x, condition):
        condition_tensor = condition * torch.ones([x.shape[0], 1, 133, 109, 120], device=x.device)
        x = torch.cat([x, condition_tensor], dim=1)
        x = x.view(x.shape[0], self.noise_size, 1, 1, 1)

        x = torch.relu(self.gen_bn1(self.gen_conv1(x)))
        x = torch.relu(self.gen_bn2(self.gen_conv2(x)))
        x = torch.relu(self.gen_bn3(self.gen_conv3(x)))
        x = torch.relu(self.gen_bn4(self.gen_conv4(x)))
        x = self.gen_conv5(x)
        x = torch.sigmoid(x)

        return x.squeeze()


# Example usage:
#latent_dim = 100
#num_points = 1024  # Number of points in the point cloud
#point_dim = 3  # Dimensionality of each point (x, y, z)
#batch_size = 32

# Define hyperparameters
latent_dim = 100
num_points = 1024
point_dim = 3
batch_size = 32
num_epochs = 1000  # Adjust as needed
learning_rate = 0.0002

# Instantiate the generator and move it to the GPU if available
generator = Generator(101,32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)

# Define a loss function (e.g., mean squared error)
criterion = nn.MSELoss()

# Define an optimizer (e.g., Adam)
optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

# Specify your input directory and label directory
input_directory = "C:\\Users\\DeLL\\Downloads\\y1.ply"
#label_directory = '/Users/pratham/Desktop/3D reconstruction/Modified/T1 modified/labels/'
custom_dataset = voxels(input_directory)
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    generator.train()  # Set the generator in training mode
    for batch in dataloader:
        batch = batch.to(device)

        # Generate random noise
        z = torch.randn(batch_size, latent_dim).to(device)
        condition = torch.zeros((batch_size,109), device=device)  # Modify this to your specific condition


        # Generate synthetic point clouds
        #generated_point_clouds = generator(z)
        # Generate synthetic 3D data using the generator
        generated_3d_data = generator(z, condition)


    #     # Ensure the generated shape matches the input shape
        generated_point_clouds = generated_3d_data.view(batch.shape)

    #     # Compute the loss (e.g., mean squared error)
    #     loss = criterion(generated_point_clouds, batch.view(-1,1))

    #     # Backpropagation and optimization
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     # Print training progress
    #     print(f"Epoch [{epoch}/{num_epochs}] Loss: {loss.item():.4f}")

    # # Optionally, you can save the generator's weights at the end of each epoch
    # torch.save(generator.state_dict(), f"generator_epoch_{epoch}.pth")





