from plyfile import PlyData, PlyElement
import numpy as np
import os
from mayavi import mlab
import numpy as np

os.environ['QT_API'] = 'pyqt5'


def load_ply_file(file_path):
    with open(file_path, 'rb') as ply_file:
        ply_data = PlyData.read(ply_file)
        vertices = ply_data['vertex']
        point_cloud = np.array([(v['x'], v['y'], v['z']) for v in vertices], dtype=np.float32)

    return point_cloud

# Example usage:
file_path = "/Users/pratham/Desktop/Dataset 2/T1/y1.ply"
# Read the 3D point cloud from a PLY file (assumes you have a function to load PLY files)
point_cloud = load_ply_file(file_path)

# Define voxel grid parameters (voxel size and bounding box)
voxel_resolution = 0.01  # Adjust this based on your requirements
min_coords = np.min(point_cloud, axis=0)
max_coords = np.max(point_cloud, axis=0)
grid_shape = ((max_coords - min_coords) / voxel_resolution).astype(int) + 1

# Initialize the voxel grid
voxel_grid = np.zeros(grid_shape, dtype=bool)

# Voxelization process
for point in point_cloud:
    voxel_coords = ((point - min_coords) / voxel_resolution).astype(int)
    voxel_grid[voxel_coords[0], voxel_coords[1], voxel_coords[2]] = True

# Get the positions of occupied voxels
occupied_voxels = np.argwhere(voxel_grid)

# Create PlyElement for the voxelized data
vertex_data = np.array([(x, y, z) for x, y, z in occupied_voxels], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
vertex_element = PlyElement.describe(vertex_data, 'vertex')

# Define PlyData with header information
ply_data = PlyData([vertex_element])

# Create a PlyData object and write it to a PLY file
output_file_path = "/Users/pratham/Desktop/yeehaw.ply"
ply_data.write(output_file_path)
print("Successful")

# Create a Mayavi figure
mlab.figure(size=(800, 600))

# Create a scalar field based on voxel presence
scalar_field = voxel_grid.astype(float)  # Convert True/False to 1.0/0.0

# Generate an isosurface from the scalar field
contour = mlab.contour3d(scalar_field, contours=[0.5], colormap='viridis')

# Adjust the opacity for better visualization
contour.actor.property.opacity = 0.5

# Show the figure
mlab.show()


# Now, voxel_grid contains the voxelized representation of the point cloud
# You can save it or use it for further processing.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load or create your voxel grid (for example, using the previous code)
# voxel_grid = ...

# # Create a 3D figure
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Find the coordinates of the filled (True) voxels
# filled_voxels = np.transpose(voxel_grid.nonzero())
#
# # Plot the filled voxels
# ax.scatter(filled_voxels[:, 0], filled_voxels[:, 1], filled_voxels[:, 2], c='b', marker='o')
#
# # Set the aspect ratio to be equal in all dimensions
# ax.set_box_aspect([1, 1, 1])
#
# # Set labels for the axes
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# # Show the plot
# plt.show()


