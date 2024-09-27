import numpy as np
import open3d as o3d


pcd = o3d.io.read_point_cloud('/Users/pratham/Desktop/3D reconstruction/Modified/T1 modified/y_gt.ply')
print(pcd)
# # #
# # Convert float64 numpy array of shape (n, 3) to Open3D format.
# points = o3d.utility.Vector3dVector(point_cloud[:, :])
# pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, :])
# pcd.normals = o3d.utility.Vector3dVector(point_cloud[:, :])
# o3d.visualization.draw_geometries([pcd])
# # # #
# # # perform surface reconstruction
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, linear_fit=False)[0]
# #
# #
o3d.io.write_triangle_mesh("/Users/pratham/Desktop/"+"p_mesh_c.ply", poisson_mesh)
# # #
# # # # generate Levels of Details (LoD)
# # lods = a target number of triangles
def lod_mesh_export(mesh, lods, extension, path):
    mesh_lods = {}
    for i in lods:
        mesh_lod = mesh.simplify_quadric_decimation(i)
        o3d.io.write_triangle_mesh(path+"lod_"+str(i)+extension, mesh_lod)
        mesh_lods[i] = mesh_lod
    print("generation of "+str(i)+" LoD successful")
    return mesh_lods
# # #
# # #
my_lods = lod_mesh_export(poisson_mesh, [3000,5000,130000], ".ply", "/Users/pratham/Desktop/")

input_file = "/Users/pratham/Desktop/lod_130000.ply"
pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud

# Visualize the point cloud within open3d
o3d.visualization.draw_geometries([pcd])