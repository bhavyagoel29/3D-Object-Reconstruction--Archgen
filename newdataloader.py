import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from plyfile import PlyData

def load_ply_file(file_path):
    with open(file_path, 'rb') as ply_file:
        ply_data = PlyData.read(ply_file)
        vertices = ply_data['vertex']
        point_cloud = np.array([(v['x'], v['y'], v['z']) for v in vertices], dtype=np.float32)
    return point_cloud

class VoxelDataset(Dataset):
    def __init__(self, file_path, voxel_resolution=0.1):
        self.point_cloud = load_ply_file(file_path)
        self.voxel_resolution = voxel_resolution
        self.voxel_grid, self.grid_shape = self.create_voxel_grid()

    def create_voxel_grid(self):
        min_coords = np.min(self.point_cloud, axis=0)
        max_coords = np.max(self.point_cloud, axis=0)
        grid_shape = np.ceil((max_coords - min_coords) / self.voxel_resolution).astype(int) + 1

        voxel_dict = {}
        for point in self.point_cloud:
            voxel_coords = tuple(np.floor((point - min_coords) / self.voxel_resolution).astype(int))
            voxel_dict[voxel_coords] = True

        return voxel_dict, grid_shape

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        voxel_grid = np.zeros(self.grid_shape, dtype=np.float32)
        for coords in self.voxel_grid.keys():
            voxel_grid[coords] = 1.0
        voxel_tensor = torch.from_numpy(voxel_grid)
        return voxel_tensor

# Usage
file_path = "C:/Users/DeLL/Desktop/CGAN/yeehaw.ply"
dataset = VoxelDataset(file_path, voxel_resolution=1)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for voxel_grid in dataloader:
    voxel_shape= voxel_grid.shape
    
# Define voxel grid parameters (voxel size and bounding box)
voxel_resolution = 0.1  # Adjust this based on your requirements
print(voxel_shape)

'''
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
input_directory = "/content/y1.ply"
#label_directory = '/Users/pratham/Desktop/3D reconstruction/Modified/T1 modified/labels/'
custom_dataset = VoxelDataset(input_directory)
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)'''