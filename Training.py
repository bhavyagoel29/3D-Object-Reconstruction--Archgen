import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

import numpy as np
import os

# def load_single_voxel(file_path):
#     # Implement your logic to load a single voxel file
#     # This depends on your voxel data format
#     # You may use libraries like numpy or custom parsers
#
#     # For example, if your voxel data is stored as a numpy array
#     try:
#         voxel = np.load(file_path)
#         return voxel
#     except Exception as e:
#         print(f"Error loading voxel from {file_path}: {str(e)}")
#         return None
#
# for root, dirs, files in os.walk(data_path):
#     for file in files:
#         if file.endswith(".ply"):  # Adjust the file extension to match your format
#             file_path = os.path.join(root, file)
#             voxel = load_single_voxel(file_path)
#             voxel_data.append(voxel)
data_path = "/Users/pratham/Desktop/voxel file/"
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".ply"):# Adjust the file extension to match your format
            print(root)

#
# def load_voxel_data(data_path):
#     voxel_data = []
#
#     # Walk through the 'data_path' directory and load each voxel file
#     for root, dirs, files in os.walk(data_path):
#         for file in files:
#             if file.endswith(".ply"):  # Adjust the file extension to match your format
#                 file_path = os.path.join(root, file)
#                 voxel = load_single_voxel(file_path)
#                 voxel_data.append(voxel)
#
#     return voxel_data
#
#
#
# # Usage
# data_path = 'c'
# voxel_data = load_voxel_data(data_path)
# print(voxel_data)

# # Define a custom dataset to load voxel data
# class VoxelDataset(Dataset):
#     def __init__(self, data_path, transform=None):
#         # Load your voxel data from 'data_path' (provide your own loading logic)
#         self.data = load_voxel_data(data_path)
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         sample = self.data[idx]
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample

# # Define the generator
# class Generator(nn.Module):
#     def __init__(self, noise_dim, condition_dim, output_dim):
#         super(Generator, self).__init__()
#
#         # Define your generator architecture
#         self.noise_dim = noise_dim
#         self.condition_dim = condition_dim
#         self.output_dim = output_dim
#
#         self.fc = nn.Sequential(
#             nn.Linear(noise_dim + condition_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, output_dim),
#             nn.Tanh()
#         )
#
#     def forward(self, noise, condition):
#         x = torch.cat((noise, condition), dim=1)
#         x = self.fc(x)
#         return x
#
# # Define hyperparameters
# noise_dim = 100
# condition_dim = 3  # Change this to match the dimension of your voxel data
# output_dim = 64 * 64 * 64  # Adjust according to your desired output size
#
# # Initialize the generator and define other components (e.g., discriminator)
# generator = Generator(noise_dim, condition_dim, output_dim)
# # Define your discriminator here
#
# # Define optimizers
# generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# # Define discriminator optimizer
#
# # Create dataloader for the voxel data
# batch_size = 32
# dataset = VoxelDataset(data_path='/Users/pratham/Desktop')
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#
# # Training loop (you need to complete this part)
# num_epochs = 100
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         # Update the generator
#         generator.zero_grad()
#         # Generate noise and condition vectors
#         noise = torch.randn(batch_size, noise_dim)
#         condition = batch  # You may need to reshape/modify your condition data
#         fake_data = generator(noise, condition)
#         # Calculate generator loss and backpropagate
#         # Update the generator optimizer
#
#         # Update the discriminator
#         # Calculate discriminator loss and backpropagate
#         # Update the discriminator optimizer
#
# # You need to save the generator and discriminator models after training
# torch.save(generator.state_dict(), 'generator.pth')
# Save the discriminator model as well




