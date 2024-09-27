from plyfile import PlyData

# Load the PLY file
ply_data = PlyData.read("/Users/pratham/Desktop/yeehaw.ply")
# Access the voxel data
voxel_data = ply_data['vertex']
x = voxel_data['x']
y = voxel_data['y']
z = voxel_data['z']

# Voxel dimensions
voxel_dimensions = (x.max(), y.max(), z.max())

# Voxel count
voxel_count = len(voxel_data)

# Voxel density (occupied voxels / total voxels)
voxel_resolution = 0.1  # Adjust to match your data
total_voxels = (x.max() / voxel_resolution) * (y.max() / voxel_resolution) * (z.max() / voxel_resolution)
voxel_density = voxel_count / total_voxels

# Center of mass
center_of_mass = (x.mean(), y.mean(), z.mean())

# Data range and scale
min_coords = (x.min(), y.min(), z.min())
max_coords = (x.max(), y.max(), z.max())

# Boundary detection
x_boundary = (x.min(), x.max())
y_boundary = (y.min(), y.max())
z_boundary = (z.min(), z.max())

# Output the characteristics
print(f"Voxel Dimensions: {voxel_dimensions}")
print(f"Voxel Count: {voxel_count}")
print(f"Voxel Density: {voxel_density}")
print(f"Center of Mass: {center_of_mass}")
print(f"Data Range: X: {min_coords[0]} to {max_coords[0]}, Y: {min_coords[1]} to {max_coords[1]}, Z: {min_coords[2]} to {max_coords[2]}")
print(f"Boundary: X: {x_boundary}, Y: {y_boundary}, Z: {z_boundary}")

