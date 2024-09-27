import open3d as o3d
import numpy as np

#Visualize a single 3D object
def point_cloud():

    pcd = o3d.io.read_point_cloud("/Users/pratham/Desktop/Dataset 2/T2/y_gt.ply")
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])

def lod_mesh_export(mesh, lods, extension, path):
    mesh_lods = {}
    for i in lods:
        mesh_lod = mesh.simplify_quadric_decimation(i)
        o3d.io.write_triangle_mesh(path+"lod_"+str(i)+extension, mesh_lod)
        mesh_lods[i] = mesh_lod
    print("generation of "+str(i)+" LoD successful")
    return mesh_lods

def mesh():
    pcd = o3d.io.read_point_cloud('/Users/pratham/Desktop/3D reconstruction/Modified/T1 modified/input/y_gt.ply')
    print(pcd)
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, linear_fit=False)[0]
    o3d.io.write_triangle_mesh("/Users/pratham/Desktop/" + "p_mesh_c.ply", poisson_mesh)
    my_lods = lod_mesh_export(poisson_mesh, [3000, 5000, 130000], ".ply", "/Users/pratham/Desktop/")

    input_file = "/Users/pratham/Desktop/lod_130000.ply"
    pcd = o3d.io.read_point_cloud(input_file)  # Read the point cloud

    # Visualize the point cloud within open3d
    o3d.visualization.draw_geometries([pcd])



if __name__ == "__main__":
    point_cloud()